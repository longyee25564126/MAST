# Copyright (c) Ruopeng Gao. All Rights Reserved.

"""
MAST Asymmetric Decoder.

Core idea:
  - Detect queries see images (cross-attention to encoder memory).
  - Track queries do NOT see images; they observe detect + track queries only.
  - Both query types have independent FFNs.
  - Only detect queries participate in iterative bbox refinement.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.misc import inverse_sigmoid
from models.ffn import FFN
from models.ops.modules import MSDeformAttn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MastDecoderLayer(nn.Module):
    """
    Single MAST decoder layer — three steps:
      1. Asymmetric Self-Attention
      2. Deformable Cross-Attention (detect only)
      3. Independent FFNs
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float, activation: str,
                 n_levels: int, n_heads: int, n_points: int):
        super().__init__()

        # ── Detect-to-Detect self-attention ──────────────────────────────────
        self.detect_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                      batch_first=True)
        self.detect_self_attn_norm = nn.LayerNorm(d_model)
        self.detect_self_attn_dropout = nn.Dropout(dropout)

        # ── Track-to-All self-attention ───────────────────────────────────────
        self.track_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                     batch_first=True)
        self.track_self_attn_norm = nn.LayerNorm(d_model)
        self.track_self_attn_dropout = nn.Dropout(dropout)

        # ── Detect cross-attention (deformable) ───────────────────────────────
        self.detect_cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.detect_cross_attn_norm = nn.LayerNorm(d_model)
        self.detect_cross_attn_dropout = nn.Dropout(dropout)

        # ── Track skip (no cross-attn; LayerNorm for numerical stability) ─────
        self.track_skip_norm = nn.LayerNorm(d_model)

        # ── Independent FFNs ──────────────────────────────────────────────────
        self.detect_ffn = FFN(d_model, d_ffn, activation=_get_activation(activation))
        self.detect_ffn_norm = nn.LayerNorm(d_model)
        self.detect_ffn_dropout = nn.Dropout(dropout)

        self.track_ffn = FFN(d_model, d_ffn, activation=_get_activation(activation))
        self.track_ffn_norm = nn.LayerNorm(d_model)
        self.track_ffn_dropout = nn.Dropout(dropout)

    def forward(self, detect_queries, track_queries, detect_pos, track_spatial_info,
                reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask):
        """
        Args:
            detect_queries:     [B, N_detect, d_model]
            track_queries:      [B, N_track, d_model]  or None
            detect_pos:         [B, N_detect, d_model]  learnable query pos for detect
            track_spatial_info: [B, N_track, d_model]  sinusoidal encoding of last bbox centre
            reference_points:   [B, N_detect, n_levels, 2]  for deformable cross-attn
            src:                encoder memory
            src_spatial_shapes, level_start_index, src_padding_mask: for deformable attn
        Returns:
            updated detect_queries [B, N_detect, d_model]
            updated track_queries  [B, N_track, d_model] or None
        """

        has_tracks = track_queries is not None and track_queries.shape[1] > 0

        # ── Step 1: Asymmetric Self-Attention ─────────────────────────────────

        # 1a. Detect-to-Detect (batch_first=True)
        q = k = detect_queries + detect_pos
        v = detect_queries
        attn_out, _ = self.detect_self_attn(q, k, v)
        detect_queries = detect_queries + self.detect_self_attn_dropout(attn_out)
        detect_queries = self.detect_self_attn_norm(detect_queries)

        # 1b. Track-to-All (track sees both detect and track queries)
        if has_tracks:
            track_q = track_queries + track_spatial_info
            all_k = torch.cat([detect_queries + detect_pos,
                                track_queries + track_spatial_info], dim=1)
            all_v = torch.cat([detect_queries, track_queries], dim=1)
            attn_out, _ = self.track_self_attn(track_q, all_k, all_v)
            track_queries = track_queries + self.track_self_attn_dropout(attn_out)
            track_queries = self.track_self_attn_norm(track_queries)

        # ── Step 2: Cross-Attention (detect only) ────────────────────────────

        # Detect: deformable cross-attention with image features
        # MSDeformAttn expects query with pos added; src query = content + pos
        detect_cross_in = detect_queries + detect_pos
        cross_out = self.detect_cross_attn(
            detect_cross_in, reference_points, src,
            src_spatial_shapes, level_start_index, src_padding_mask,
        )
        detect_queries = detect_queries + self.detect_cross_attn_dropout(cross_out)
        detect_queries = self.detect_cross_attn_norm(detect_queries)

        # Track: skip cross-attention, only LayerNorm
        if has_tracks:
            track_queries = self.track_skip_norm(track_queries)

        # ── Step 3: Independent FFNs ──────────────────────────────────────────

        detect_queries = detect_queries + self.detect_ffn_dropout(self.detect_ffn(detect_queries))
        detect_queries = self.detect_ffn_norm(detect_queries)

        if has_tracks:
            track_queries = track_queries + self.track_ffn_dropout(self.track_ffn(track_queries))
            track_queries = self.track_ffn_norm(track_queries)

        return detect_queries, track_queries


class MastDecoder(nn.Module):
    """
    Stack of MastDecoderLayer.
    Handles iterative bbox refinement for detect queries only.
    """

    def __init__(self, decoder_layer: MastDecoderLayer, num_layers: int,
                 return_intermediate: bool = True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # Set externally by the MAST model after detection head is built
        self.bbox_embed = None

    def forward(self, detect_queries, track_queries, detect_pos, track_spatial_info,
                reference_points, src, src_spatial_shapes, src_level_start_index,
                src_valid_ratios, src_padding_mask):
        """
        Args:
            detect_queries:     [B, N_detect, d_model]
            track_queries:      [B, N_track, d_model] or None
            detect_pos:         [B, N_detect, d_model]
            track_spatial_info: [B, N_track, d_model] or None
            reference_points:   [B, N_detect, 2]
            src:                encoder memory [B, S, d_model]
            src_spatial_shapes: [n_levels, 2]
            src_level_start_index: [n_levels]
            src_valid_ratios:   [B, n_levels, 2]
            src_padding_mask:   [B, S]
        Returns:
            detect_outputs: [num_layers, B, N_detect, d_model]  (if return_intermediate)
            track_output:   [B, N_track, d_model]  (final layer only)
            inter_reference_points: [num_layers, B, N_detect, 2]
        """
        detect_output = detect_queries
        track_output = track_queries

        intermediate_detect = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # Scale reference points by valid ratios
            if reference_points.shape[-1] == 4:
                ref_pts_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                ref_pts_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            detect_output, track_output = layer(
                detect_queries=detect_output,
                track_queries=track_output,
                detect_pos=detect_pos,
                track_spatial_info=track_spatial_info,
                reference_points=ref_pts_input,
                src=src,
                src_spatial_shapes=src_spatial_shapes,
                level_start_index=src_level_start_index,
                src_padding_mask=src_padding_mask,
            )

            # Iterative bbox refinement (detect only)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](detect_output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    new_reference_points = tmp.clone()
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate_detect.append(detect_output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return (
                torch.stack(intermediate_detect),
                track_output,
                torch.stack(intermediate_reference_points),
            )

        return detect_output, track_output, reference_points


def _get_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
