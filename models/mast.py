# Copyright (c) Ruopeng Gao. All Rights Reserved.

"""
MAST — Multiple-object Association Selective Transformer.

Architecture:
  - Detect queries see images and find objects.
  - Track queries observe detect + track queries (no image access),
    then claim their target via the Assignment Head.

Weight-loading compatibility note
──────────────────────────────────
load_detr_pretrain() in misc.py adds "detr." prefix to all pretrained keys and
expects strict loading.  To keep those key paths intact the MAST model exposes
all DeformableDETR-equivalent parameters under self.detr with the same
sub-module names used by DeformableDETR:

    detr.backbone.*
    detr.transformer.encoder.layers.*
    detr.transformer.decoder.layers.*
    detr.transformer.level_embed
    detr.transformer.reference_points.*
    detr.class_embed.*
    detr.bbox_embed.*
    detr.query_embed.weight
    detr.input_proj.*

New MAST-only parameters sit outside self.detr and are initialised randomly.
"""

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_

from utils.nested_tensor import NestedTensor, nested_tensor_from_tensor_list
from models.misc import inverse_sigmoid, _get_clones
from models.mlp import MLP
from models.deformable_backbone import (
    Backbone, Joiner, PositionEmbeddingSine, PositionEmbeddingLearned,
)
from models.deformable_encoder import DeformableEncoder, DeformableTransformerEncoderLayer
from models.deformable_decoder import (
    DeformableTransformerDecoder, DeformableTransformerDecoderLayer,
)
from models.deformable_mast_decoder import MastDecoder, MastDecoderLayer
from models.deformable_detection_head import compute_detections, set_aux_loss
from models.deformable_matcher import HungarianMatcher
from models.deformable_criterion import SetCriterion
from models.assignment_head import AssignmentHead
from models.init_mlp import InitMLP
from models.tia import TIA
from structures.args import Args


# ─── Internal sub-modules ─────────────────────────────────────────────────────

class _Transformer(nn.Module):
    """
    Container for encoder + decoder with key names matching the pretrained
    DeformableDETR weight structure:
        transformer.encoder.layers.*
        transformer.decoder.layers.*
        transformer.level_embed
        transformer.reference_points.*
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 num_feature_levels: int, d_model: int):
        super().__init__()
        self.encoder = encoder          # DeformableEncoder  (.layers.*)
        self.decoder = decoder          # DeformableTransformerDecoder or None
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        normal_(self.level_embed)
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)

    @staticmethod
    def get_valid_ratio(mask: torch.Tensor) -> torch.Tensor:
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_W.float() / W, valid_H.float() / H], -1)

    def prepare_encoder_input(self, srcs, masks, pos_embeds):
        """
        Flatten multi-scale features and build spatial metadata.

        Returns:
            src_flatten:           [B, sum(H*W), d_model]
            mask_flatten:          [B, sum(H*W)]
            lvl_pos_embed_flatten: [B, sum(H*W), d_model]
            spatial_shapes:        [n_levels, 2]  long tensor
            level_start_index:     [n_levels]     long tensor
            valid_ratios:          [B, n_levels, 2]
        """
        src_flat, mask_flat, lvl_pos_flat = [], [], []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flat.append(src.flatten(2).transpose(1, 2))
            mask_flat.append(mask.flatten(1))
            pos_embed_flat = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_flat.append(pos_embed_flat + self.level_embed[lvl].view(1, 1, -1))

        src_flatten = torch.cat(src_flat, 1)
        mask_flatten = torch.cat(mask_flat, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_flat, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long,
                                         device=src_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        return src_flatten, mask_flatten, lvl_pos_embed_flatten, \
               spatial_shapes, level_start_index, valid_ratios


class _DetrCore(nn.Module):
    """
    Container for DETR components.
    Key paths under 'detr.*' match the pretrained DeformableDETR checkpoint,
    enabling load_detr_pretrain() in misc.py to work without modification.
    """

    def __init__(self, backbone: nn.Module, transformer: _Transformer,
                 class_embed: nn.ModuleList, bbox_embed: nn.ModuleList,
                 query_embed: nn.Embedding, input_proj: nn.ModuleList):
        super().__init__()
        self.backbone = backbone          # Joiner
        self.transformer = transformer    # _Transformer
        self.class_embed = class_embed    # nn.ModuleList[nn.Linear]
        self.bbox_embed = bbox_embed      # nn.ModuleList[MLP]
        self.query_embed = query_embed    # nn.Embedding(num_queries, d_model*2)
        self.input_proj = input_proj      # nn.ModuleList


# ─── Main Model ───────────────────────────────────────────────────────────────

class MAST(nn.Module):
    """
    MAST: Multiple-object Association Selective Transformer.

    Forward interface:
        model(samples)                                   → pretrain / first frame
        model(samples, track_queries, track_spatial_info) → tracking mode
    """

    def __init__(
        self,
        detr: _DetrCore,
        mast_decoder: MastDecoder,      # None in only_detr mode
        assignment_head: AssignmentHead, # None in only_detr mode
        init_mlp: InitMLP,               # None in only_detr mode
        tia: TIA,                        # None in only_detr mode
        num_feature_levels: int,
        num_queries: int,
        aux_loss: bool,
        only_detr: bool,
        d_model: int = 256,
    ):
        super().__init__()
        self.detr = detr
        self.mast_decoder = mast_decoder
        self.assignment_head = assignment_head
        self.init_mlp = init_mlp
        self.tia = tia
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.only_detr = only_detr
        self.d_model = d_model

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_srcs_masks_pos(self, features, samples, pos):
        """Apply input projections; extend srcs/masks/pos for extra feature levels."""
        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.detr.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                src = (self.detr.input_proj[l](features[-1].tensors)
                       if l == _len_srcs else self.detr.input_proj[l](srcs[-1]))
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        return srcs, masks, pos

    def _compute_detections(self, hs, init_reference, inter_references):
        """Run detection head and package outputs."""
        outputs_class, outputs_coord = compute_detections(
            hs, init_reference, inter_references,
            self.detr.class_embed, self.detr.bbox_embed,
        )
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
        }
        if self.aux_loss:
            out['aux_outputs'] = set_aux_loss(outputs_class, outputs_coord)
        return out

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, samples, track_queries=None, track_spatial_info=None):
        """
        Args:
            samples:            NestedTensor  [B, 3, H, W]
            track_queries:      [B, N_track, d_model] or None
            track_spatial_info: [B, N_track, d_model] or None
                                (sinusoidal encoding of last matched bbox centre)
        Returns:
            dict with keys:
                pred_logits:        [B, N_detect, num_classes]
                pred_boxes:         [B, N_detect, 4]
                aux_outputs:        list of dicts  (if aux_loss)
                detect_outputs:     [B, N_detect, d_model]
                track_outputs:      [B, N_track, d_model]  or None
                assignment_logits:  [B, N_track, N_detect+1]  or None
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        # 1. Backbone
        features, pos = self.detr.backbone(samples)

        # 2. Input projections + extra feature levels
        srcs, masks, pos = self._build_srcs_masks_pos(features, samples, pos)

        # 3. Encoder input preparation + encoding
        transformer = self.detr.transformer
        src_flat, mask_flat, lvl_pos_flat, spatial_shapes, lvl_start_idx, valid_ratios = \
            transformer.prepare_encoder_input(srcs, masks, pos)

        memory = transformer.encoder(
            src_flat, spatial_shapes, lvl_start_idx, valid_ratios,
            lvl_pos_flat, mask_flat,
        )
        bs = memory.shape[0]

        # 4. Prepare detect query input
        q_embed = self.detr.query_embed.weight       # [N_queries, d_model*2]
        query_pos, tgt = torch.split(q_embed, self.d_model, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)   # [B, N, d_model]
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)               # [B, N, d_model]
        reference_points = transformer.reference_points(query_pos).sigmoid()  # [B, N, 2]
        init_reference = reference_points

        # 5. Decoder + detection head
        if self.only_detr:
            # ── Pretrain mode: standard decoder ───────────────────────────────
            hs, inter_references = transformer.decoder(
                tgt, reference_points, memory,
                spatial_shapes, lvl_start_idx, valid_ratios,
                query_pos, mask_flat,
            )
            out = self._compute_detections(hs, init_reference, inter_references)
            out["detect_outputs"] = hs[-1]
            out["track_outputs"] = None
            out["assignment_logits"] = None

        else:
            # ── Tracking mode: MAST asymmetric decoder ────────────────────────
            detect_hs, track_output, inter_references = self.mast_decoder(
                detect_queries=tgt,
                track_queries=track_queries,
                detect_pos=query_pos,
                track_spatial_info=track_spatial_info,
                reference_points=reference_points,
                src=memory,
                src_spatial_shapes=spatial_shapes,
                src_level_start_index=lvl_start_idx,
                src_valid_ratios=valid_ratios,
                src_padding_mask=mask_flat,
            )

            out = self._compute_detections(detect_hs, init_reference, inter_references)
            out["detect_outputs"] = detect_hs[-1]
            out["track_outputs"] = track_output

            if track_queries is not None and track_queries.shape[1] > 0:
                out["assignment_logits"] = self.assignment_head(track_output, detect_hs[-1])
            else:
                out["assignment_logits"] = None

        return out


# ─── Build ────────────────────────────────────────────────────────────────────

def build(config: dict):
    """
    Construct a MAST model + SetCriterion from a config dict.

    The config dict uses the same keys as MOTIP's config (BACKBONE, DETR_*, etc.)
    so existing .yaml files work without modification.

    Returns:
        model:     MAST
        criterion: SetCriterion
    """
    device = torch.device(config["DEVICE"])

    # ── Shared hyperparameters ────────────────────────────────────────────────
    d_model = config["DETR_HIDDEN_DIM"]
    num_classes = config["NUM_CLASSES"]
    num_queries = config["DETR_NUM_QUERIES"]
    num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    aux_loss = config["DETR_AUX_LOSS"]
    with_box_refine = config["DETR_WITH_BOX_REFINE"]
    two_stage = config["DETR_TWO_STAGE"]
    only_detr = config.get("ONLY_DETR", False)

    assert not two_stage, "two_stage is not supported in MAST"

    # ── Args object (reuses existing backbone/pos-encoding builders) ──────────
    detr_args = Args()
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    detr_args.hidden_dim = d_model
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.num_feature_levels = num_feature_levels

    # ── 1. Backbone ───────────────────────────────────────────────────────────
    from models.deformable_backbone import build_backbone
    backbone = build_backbone(detr_args)      # Joiner

    # ── 2. Encoder ────────────────────────────────────────────────────────────
    enc_layer = DeformableTransformerEncoderLayer(
        d_model=d_model,
        d_ffn=config["DETR_DIM_FEEDFORWARD"],
        dropout=config["DETR_DROPOUT"],
        activation="relu",
        n_levels=num_feature_levels,
        n_heads=config["DETR_NUM_HEADS"],
        n_points=config["DETR_ENC_N_POINTS"],
    )
    encoder = DeformableEncoder(enc_layer, config["DETR_ENC_LAYERS"])

    # ── 3. Decoder (standard; used in pretrain OR kept as detr.transformer.decoder
    #                for pretrained weight key compatibility) ───────────────────
    dec_layer = DeformableTransformerDecoderLayer(
        d_model=d_model,
        d_ffn=config["DETR_DIM_FEEDFORWARD"],
        dropout=config["DETR_DROPOUT"],
        activation="relu",
        n_levels=num_feature_levels,
        n_heads=config["DETR_NUM_HEADS"],
        n_points=config["DETR_DEC_N_POINTS"],
    )
    std_decoder = DeformableTransformerDecoder(
        dec_layer, config["DETR_DEC_LAYERS"], return_intermediate=True,
    )

    # ── 4. Transformer wrapper ────────────────────────────────────────────────
    transformer = _Transformer(encoder, std_decoder, num_feature_levels, d_model)

    # Reset transformer params (MSDeformAttn, etc.)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    from models.ops.modules import MSDeformAttn
    for m in transformer.modules():
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()
    # Re-init level_embed and reference_points after xavier sweep
    normal_(transformer.level_embed)
    xavier_uniform_(transformer.reference_points.weight.data, gain=1.0)
    constant_(transformer.reference_points.bias.data, 0.)

    # ── 5. Detection head components (class_embed, bbox_embed) ────────────────
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)

    class_embed_base = nn.Linear(d_model, num_classes)
    class_embed_base.bias.data = torch.ones(num_classes) * bias_value

    bbox_embed_base = MLP(d_model, d_model, 4, 3)
    nn.init.constant_(bbox_embed_base.layers[-1].weight.data, 0)
    nn.init.constant_(bbox_embed_base.layers[-1].bias.data, 0)

    num_dec_layers = config["DETR_DEC_LAYERS"]
    num_pred = num_dec_layers  # two_stage=False

    if with_box_refine:
        class_embed_list = _get_clones(class_embed_base, num_pred)
        bbox_embed_list = _get_clones(bbox_embed_base, num_pred)
        nn.init.constant_(bbox_embed_list[0].layers[-1].bias.data[2:], -2.0)
        # Wire bbox_embed into the standard decoder for iterative refinement
        std_decoder.bbox_embed = bbox_embed_list
    else:
        class_embed_list = nn.ModuleList([class_embed_base for _ in range(num_pred)])
        bbox_embed_list = nn.ModuleList([bbox_embed_base for _ in range(num_pred)])
        nn.init.constant_(bbox_embed_list[0].layers[-1].bias.data[2:], -2.0)
        std_decoder.bbox_embed = None

    # ── 6. Query embedding + input projections ────────────────────────────────
    query_embed = nn.Embedding(num_queries, d_model * 2)

    num_backbone_outs = len(backbone.strides)
    input_proj_list = []
    in_channels = backbone.num_channels[-1]
    for i in range(num_backbone_outs):
        in_ch = backbone.num_channels[i]
        input_proj_list.append(nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        ))
    for _ in range(num_feature_levels - num_backbone_outs):
        input_proj_list.append(nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, d_model),
        ))
        in_channels = d_model
    input_proj = nn.ModuleList(input_proj_list)
    for proj in input_proj:
        nn.init.xavier_uniform_(proj[0].weight, gain=1)
        nn.init.constant_(proj[0].bias, 0)

    # ── 7. _DetrCore ──────────────────────────────────────────────────────────
    detr_core = _DetrCore(
        backbone=backbone,
        transformer=transformer,
        class_embed=class_embed_list,
        bbox_embed=bbox_embed_list,
        query_embed=query_embed,
        input_proj=input_proj,
    )

    # ── 8. MAST-specific components ───────────────────────────────────────────
    if only_detr:
        mast_decoder = None
        assignment_head_module = None
        init_mlp_module = None
        tia_module = None
    else:
        mast_dec_layer = MastDecoderLayer(
            d_model=d_model,
            d_ffn=config["DETR_DIM_FEEDFORWARD"],
            dropout=config["DETR_DROPOUT"],
            activation="relu",
            n_levels=num_feature_levels,
            n_heads=config["DETR_NUM_HEADS"],
            n_points=config["DETR_DEC_N_POINTS"],
        )
        mast_decoder = MastDecoder(mast_dec_layer, num_dec_layers, return_intermediate=True)

        # Wire bbox_embed for iterative refinement in MAST decoder too
        if with_box_refine:
            mast_decoder.bbox_embed = bbox_embed_list  # shared with std_decoder

        assignment_head_module = AssignmentHead(d_model)
        init_mlp_module = InitMLP(d_model)
        tia_module = TIA(d_model)

    # ── 9. Assemble MAST ──────────────────────────────────────────────────────
    model = MAST(
        detr=detr_core,
        mast_decoder=mast_decoder,
        assignment_head=assignment_head_module,
        init_mlp=init_mlp_module,
        tia=tia_module,
        num_feature_levels=num_feature_levels,
        num_queries=num_queries,
        aux_loss=aux_loss,
        only_detr=only_detr,
        d_model=d_model,
    )

    # ── 10. Criterion ─────────────────────────────────────────────────────────
    matcher = HungarianMatcher(
        cost_class=config["DETR_SET_COST_CLASS"],
        cost_bbox=config["DETR_SET_COST_BBOX"],
        cost_giou=config["DETR_SET_COST_GIOU"],
    )
    weight_dict = {
        'loss_ce': config["DETR_CLS_LOSS_COEF"],
        'loss_bbox': config["DETR_BBOX_LOSS_COEF"],
        'loss_giou': config["DETR_GIOU_LOSS_COEF"],
    }
    if aux_loss:
        aux_weight_dict = {}
        for i in range(num_dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=['labels', 'boxes', 'cardinality'],
        focal_alpha=config["DETR_FOCAL_ALPHA"],
    )
    criterion.to(device)

    return model, criterion
