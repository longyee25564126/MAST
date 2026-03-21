# MAST Implementation Guide for Claude Code

## Overview

Refactor the existing MOTIP codebase to implement the MAST (Multiple-object Association Selective Transformer) architecture. The core idea: **detect queries see images and find objects; track queries don't see images, they only observe detect queries and other track queries, then claim their target via an Assignment Head.**

## Important: Read Before Coding

1. **Read the entire existing codebase first** before making any changes. Understand how MOTIP currently works.
2. **This is a model architecture refactoring only.** Do NOT modify training (`train.py`), inference (`submit_and_evaluate.py`), data loading (`data/`), or other non-model code in this phase.
3. **Preserve the pretrain (ONLY_DETR) mode.** The system must still support pure DETR detection pretraining.
4. **All existing CUDA ops in `models/ops/` remain untouched.**

---

## Step 1: Restructure `models/` Directory

### Delete
- `models/deformable_detr/` (entire directory)
- `models/motip/` (entire directory)

### Create new files by extracting from old code

The source files to extract from are:
- `models/deformable_detr/backbone.py`
- `models/deformable_detr/position_encoding.py`
- `models/deformable_detr/deformable_transformer.py`
- `models/deformable_detr/deformable_detr.py`
- `models/deformable_detr/matcher.py`
- `models/deformable_detr/segmentation.py` (only need `sigmoid_focal_loss`, `dice_loss`)

### New file structure under `models/`

```
models/
├── __init__.py                         # keep, update imports
├── deformable_backbone.py              # backbone + position encoding + Joiner
├── deformable_encoder.py               # DeformableTransformerEncoderLayer + Encoder
├── deformable_decoder.py               # Original decoder (for DETR pretrain mode)
├── deformable_mast_decoder.py          # NEW: MAST asymmetric decoder
├── deformable_detection_head.py        # class_embed + bbox_embed + iterative refinement
├── deformable_matcher.py               # HungarianMatcher (copy from matcher.py)
├── deformable_criterion.py             # SetCriterion for detection + assignment loss
├── mast.py                             # NEW: Main model class
├── assignment_head.py                  # NEW: Assignment Head
├── init_mlp.py                         # NEW: detect → track space mapping
├── tia.py                              # NEW: Temporal Identity Aggregation
├── misc.py                             # keep as-is
├── mlp.py                              # keep as-is
├── ffn.py                              # keep as-is
├── runtime_tracker.py                  # keep as-is for now (will be rewritten later)
├── ops/                                # keep as-is, do not touch
│   └── ...
```

---

## Step 2: Extract Existing Code into New Files

### `deformable_backbone.py`
- Copy `Backbone`, `BackboneBase`, `FrozenBatchNorm2d`, `Joiner` from `backbone.py`
- Copy `PositionEmbeddingSine`, `PositionEmbeddingLearned`, `build_position_encoding` from `position_encoding.py`
- Keep the `build_backbone` function

### `deformable_encoder.py`
- Copy `DeformableTransformerEncoderLayer`, `DeformableTransformerEncoder` from `deformable_transformer.py`
- Include helper functions: `_get_clones`, `_get_activation_fn`
- Include the `level_embed` parameter and `get_valid_ratio` method (these are needed for preparing encoder input)
- Create a wrapper class `DeformableEncoder` that holds:
  - `encoder_layers`
  - `level_embed`
  - The input preparation logic (src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, etc.)
  - The `reference_points` linear layer (for generating initial reference points for the decoder)
  - This wrapper currently lives scattered in `DeformableTransformer.forward()`. Consolidate it here.

### `deformable_decoder.py`
- Copy `DeformableTransformerDecoderLayer`, `DeformableTransformerDecoder` from `deformable_transformer.py`
- This is only used during DETR pretrain (`ONLY_DETR: True`). Keep it identical to the original.

### `deformable_detection_head.py`
- Extract from `DeformableDETR.__init__()` and `DeformableDETR.forward()`:
  - `class_embed` (nn.Linear or cloned list)
  - `bbox_embed` (MLP or cloned list)
  - The iterative bbox refinement logic
  - `_set_aux_loss`
- Create a class `DeformableDetectionHead` that:
  - Takes decoder output embeddings + reference points as input
  - Returns `pred_logits`, `pred_boxes`, and optionally `aux_outputs`
  - Handles `with_box_refine` mode

### `deformable_matcher.py`
- Directly copy `HungarianMatcher` and `build_matcher` from `matcher.py`

### `deformable_criterion.py`
- Copy `SetCriterion` from `deformable_detr.py`
- Copy `sigmoid_focal_loss` from `segmentation.py`
- Later, add `AssignmentCriterion` for the assignment loss (focal loss on assignment logits)

---

## Step 3: Implement New MAST Components

### `deformable_mast_decoder.py` — The Core of MAST

This is the most critical new file. It replaces the standard decoder with an asymmetric decoder.

```python
class MastDecoderLayer(nn.Module):
    """
    Single MAST decoder layer with three steps:
    1. Asymmetric Self-Attention
    2. Deformable Cross-Attention (detect only)
    3. Independent FFNs
    """
    def __init__(self, d_model, d_ffn, dropout, activation, n_levels, n_heads, n_points):
        # --- Detect-to-Detect Self-Attention ---
        self.detect_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.detect_self_attn_norm = nn.LayerNorm(d_model)

        # --- Track-to-All Self-Attention ---
        self.track_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.track_self_attn_norm = nn.LayerNorm(d_model)

        # --- Cross-Attention (detect only) ---
        self.detect_cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.detect_cross_attn_norm = nn.LayerNorm(d_model)

        # --- Track skip norm (no cross-attn, just layernorm for numerical stability) ---
        self.track_skip_norm = nn.LayerNorm(d_model)

        # --- Independent FFNs ---
        self.detect_ffn = FFN(d_model, d_ffn)  # use the existing FFN class
        self.detect_ffn_norm = nn.LayerNorm(d_model)
        self.track_ffn = FFN(d_model, d_ffn)
        self.track_ffn_norm = nn.LayerNorm(d_model)

    def forward(self, detect_queries, track_queries, detect_pos, track_spatial_info,
                reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask):
        """
        Args:
            detect_queries: [B, N_detect, d_model]
            track_queries: [B, N_track, d_model]  (can be None or empty if no tracks yet)
            detect_pos: [B, N_detect, d_model] (learnable query pos for detect)
            track_spatial_info: [B, N_track, d_model] (sinusoidal encoding of last matched bbox center)
            reference_points: [B, N_detect, n_levels, 2] (for deformable cross-attn)
            src: encoder memory
            src_spatial_shapes, level_start_index, src_padding_mask: for deformable attn
        Returns:
            updated detect_queries, updated track_queries
        """

        # === Step 1: Asymmetric Self-Attention ===

        # 1a. Detect-to-Detect self-attention (standard)
        # Q = K = detect_content + detect_pos, V = detect_content
        q = k = detect_queries + detect_pos
        v = detect_queries
        # Use nn.MultiheadAttention (need transpose for batch_first=False, or set batch_first=True)
        detect_queries = detect_queries + self.detect_self_attn(q, k, v)
        detect_queries = self.detect_self_attn_norm(detect_queries)

        # 1b. Track-to-All self-attention (track sees detect + track)
        # Only if there are track queries
        if track_queries is not None and track_queries.shape[1] > 0:
            # Q = track_content + track_spatial_info
            # K = concat(detect_content, track_content) + concat(detect_pos, track_spatial_info)
            # V = concat(detect_content, track_content)
            track_q = track_queries + track_spatial_info
            all_k = torch.cat([detect_queries + detect_pos, track_queries + track_spatial_info], dim=1)
            all_v = torch.cat([detect_queries, track_queries], dim=1)
            track_queries = track_queries + self.track_self_attn(track_q, all_k, all_v)
            track_queries = self.track_self_attn_norm(track_queries)

        # === Step 2: Cross-Attention (detect only) ===

        # Detect: standard deformable cross-attention with image features
        detect_queries = detect_queries + self.detect_cross_attn(
            detect_queries + detect_pos, reference_points, src,
            src_spatial_shapes, level_start_index, src_padding_mask
        )
        detect_queries = self.detect_cross_attn_norm(detect_queries)

        # Track: skip cross-attention, only LayerNorm
        if track_queries is not None and track_queries.shape[1] > 0:
            track_queries = self.track_skip_norm(track_queries)

        # === Step 3: Independent FFNs ===

        detect_queries = detect_queries + self.detect_ffn(detect_queries)
        detect_queries = self.detect_ffn_norm(detect_queries)

        if track_queries is not None and track_queries.shape[1] > 0:
            track_queries = track_queries + self.track_ffn(track_queries)
            track_queries = self.track_ffn_norm(track_queries)

        return detect_queries, track_queries


class MastDecoder(nn.Module):
    """
    Stack of MastDecoderLayer.
    Handles iterative bbox refinement for detect queries only.
    """
    def __init__(self, decoder_layer, num_layers, return_intermediate=True):
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # bbox_embed will be set externally (from detection head) for iterative refinement
        self.bbox_embed = None

    def forward(self, detect_queries, track_queries, detect_pos, track_spatial_info,
                reference_points, src, src_spatial_shapes, src_level_start_index,
                src_valid_ratios, src_padding_mask):
        """
        Returns:
            detect_outputs: [num_layers, B, N_detect, d_model] if return_intermediate
            track_output: [B, N_track, d_model] (only final layer)
            inter_reference_points: for detection head
        """
        detect_output = detect_queries
        track_output = track_queries

        intermediate_detect = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # Prepare reference_points_input (same as original decoder)
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            detect_output, track_output = layer(
                detect_queries=detect_output,
                track_queries=track_output,
                detect_pos=detect_pos,
                track_spatial_info=track_spatial_info,
                reference_points=reference_points_input,
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
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate_detect.append(detect_output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate_detect), track_output, torch.stack(intermediate_reference_points)

        return detect_output, track_output, reference_points
```

**Important implementation notes for MastDecoderLayer:**
- Use `batch_first=True` for `nn.MultiheadAttention` to avoid constant transposing
- The `detect_pos` is the same learnable positional embedding as in standard Deformable DETR (`query_embed`)
- `track_spatial_info` is sinusoidal encoding of the last matched bbox center (detached)
- Track queries have NO reference points and do NOT participate in deformable cross-attention
- Track queries have NO iterative bbox refinement


### `assignment_head.py`

```python
class AssignmentHead(nn.Module):
    def __init__(self, d_model):
        self.track_proj = nn.Linear(d_model, d_model)
        self.detect_proj = nn.Linear(d_model, d_model)
        self.dustbin = nn.Parameter(torch.randn(d_model))
        self.scale = d_model ** -0.5

    def forward(self, track_queries, detect_queries):
        """
        Args:
            track_queries: [B, N_track, d_model]
            detect_queries: [B, N_detect, d_model]
        Returns:
            logits: [B, N_track, N_detect + 1]  (last column = dustbin)
        """
        T = self.track_proj(track_queries)    # [B, N_track, d_model]
        D = self.detect_proj(detect_queries)  # [B, N_detect, d_model]
        sim = torch.bmm(T, D.transpose(1, 2)) * self.scale  # [B, N_track, N_detect]
        dustbin_score = (T * self.dustbin).sum(-1, keepdim=True)  # [B, N_track, 1]
        logits = torch.cat([sim, dustbin_score], dim=-1)  # [B, N_track, N_detect+1]
        return logits
```

### `init_mlp.py`

```python
class InitMLP(nn.Module):
    """Maps detect query output to track query space when a new target is born."""
    def __init__(self, d_model):
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, detect_output):
        """
        Args:
            detect_output: [N_new, d_model] — detect query outputs for newly born targets
        Returns:
            track_query_init: [N_new, d_model]
        """
        return self.mlp(detect_output)
```

### `tia.py`

```python
class TIA(nn.Module):
    """Temporal Identity Aggregation: update track query after successful assignment."""
    def __init__(self, d_model):
        self.observe_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, track_query, matched_detect, confidence):
        """
        Args:
            track_query: [N, d_model] — current track queries
            matched_detect: [N, d_model] — matched detect query outputs
            confidence: [N] — sigmoid of assignment logits
        Returns:
            updated_track_query: [N, d_model]
        """
        observation = self.observe_proj(matched_detect)
        gate_input = torch.cat([track_query, observation, confidence.unsqueeze(-1)], dim=-1)
        gate = self.gate_net(gate_input)
        return gate * observation + (1 - gate) * track_query
```

### `mast.py` — Main Model

```python
class MAST(nn.Module):
    def __init__(
        self,
        backbone,           # Joiner (backbone + position encoding)
        encoder,            # DeformableEncoder
        decoder,            # MastDecoder (for tracking) or DeformableTransformerDecoder (for pretrain)
        detection_head,     # DeformableDetectionHead
        assignment_head,    # AssignmentHead (None if only_detr)
        init_mlp,           # InitMLP (None if only_detr)
        tia,                # TIA (None if only_detr)
        num_feature_levels,
        num_queries,
        only_detr=False,
    ):
        ...
        # input_proj layers (same as original DeformableDETR)
        # query_embed (learnable, same as original)

    def forward(self, samples, track_queries=None, track_spatial_info=None):
        """
        Args:
            samples: NestedTensor (images)
            track_queries: [B, N_track, d_model] or None (no tracks yet / pretrain mode)
            track_spatial_info: [B, N_track, d_model] or None
        Returns:
            dict with:
                'pred_logits': [B, N_detect, num_classes]
                'pred_boxes': [B, N_detect, 4]
                'aux_outputs': list of dicts (if aux_loss)
                'detect_outputs': [B, N_detect, d_model]  (last layer detect embeddings)
                'track_outputs': [B, N_track, d_model]  (last layer track embeddings, None if pretrain)
                'assignment_logits': [B, N_track, N_detect+1]  (None if pretrain or no tracks)
        """
        # 1. Backbone
        features, pos = self.backbone(samples)

        # 2. Prepare multi-scale features (same as original DeformableDETR.forward)
        srcs, masks = self._prepare_input_proj(features, samples)

        # 3. Encoder
        memory, spatial_shapes, level_start_index, valid_ratios = self.encoder(srcs, masks, pos)

        # 4. Prepare decoder input
        query_embed, tgt = torch.split(self.query_embed.weight, self.d_model, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()

        # 5. Decoder
        if self.only_detr:
            # Use standard decoder for pretrain
            hs, inter_references = self.decoder(tgt, reference_points, memory, ...)
            # Detection head
            out = self.detection_head(hs, reference_points, inter_references)
            out["detect_outputs"] = hs[-1]
            return out
        else:
            # Use MAST decoder
            detect_hs, track_output, inter_references = self.decoder(
                detect_queries=tgt,
                track_queries=track_queries,
                detect_pos=query_embed,
                track_spatial_info=track_spatial_info,
                reference_points=reference_points,
                src=memory,
                src_spatial_shapes=spatial_shapes,
                src_level_start_index=level_start_index,
                src_valid_ratios=valid_ratios,
                src_padding_mask=mask_flatten,
            )
            # Detection head (only uses detect outputs)
            out = self.detection_head(detect_hs, reference_points, inter_references)
            out["detect_outputs"] = detect_hs[-1]
            out["track_outputs"] = track_output

            # Assignment head
            if track_queries is not None and track_queries.shape[1] > 0:
                out["assignment_logits"] = self.assignment_head(track_output, detect_hs[-1])
            else:
                out["assignment_logits"] = None

            return out
```

**The `__init__.py` should have a `build` function** similar to `models/motip/__init__.py` that:
1. Builds backbone, encoder, decoder(s), detection head, assignment head, init_mlp, tia
2. Returns the MAST model and the criterion
3. Respects the `ONLY_DETR` config flag

---

## Step 4: Implementation Checklist

In order of implementation:

- [ ] 1. Create `deformable_backbone.py` (extract from existing)
- [ ] 2. Create `deformable_encoder.py` (extract from existing)
- [ ] 3. Create `deformable_decoder.py` (extract from existing, for pretrain)
- [ ] 4. Create `deformable_detection_head.py` (extract from existing)
- [ ] 5. Create `deformable_matcher.py` (copy from existing)
- [ ] 6. Create `deformable_criterion.py` (extract from existing)
- [ ] 7. Implement `assignment_head.py` (new)
- [ ] 8. Implement `init_mlp.py` (new)
- [ ] 9. Implement `tia.py` (new)
- [ ] 10. Implement `deformable_mast_decoder.py` (new, most critical)
- [ ] 11. Implement `mast.py` (new, main model)
- [ ] 12. Update `models/__init__.py`
- [ ] 13. Delete `models/deformable_detr/` and `models/motip/`
- [ ] 14. **Verify**: The build function can construct the model without errors
- [ ] 15. **Verify**: Pretrain mode (ONLY_DETR=True) still works — forward pass produces same output structure

---

## Step 5: Key Constraints

1. **d_model = 256** throughout (DETR_HIDDEN_DIM)
2. **num_queries = 300** detect queries (DETR_NUM_QUERIES)
3. **6 decoder layers** (DETR_DEC_LAYERS)
4. **8 attention heads** (DETR_NUM_HEADS)
5. **4 feature levels** (DETR_NUM_FEATURE_LEVELS)
6. **with_box_refine = True** (DETR_WITH_BOX_REFINE)
7. **two_stage = False** (DETR_TWO_STAGE)
8. Track queries are VARIABLE in number (0 to N_track), NOT fixed like detect queries
9. Track queries have NO reference points
10. Track queries do NOT go through cross-attention
11. Track queries do NOT go through iterative bbox refinement
12. The sinusoidal encoding for track_spatial_info should use the same function as position encoding (similar to `pos_to_pos_embed` in `misc.py`)

---

## Step 6: What NOT to Do

1. Do NOT modify `models/ops/` at all
2. Do NOT modify `train.py`, `submit_and_evaluate.py`, `data/`, `configs/`, or any non-model code
3. Do NOT implement the training loop or inference logic for MAST — only the model architecture
4. Do NOT remove `runtime_tracker.py` — keep it as-is even though it won't work with MAST yet
5. Do NOT add any new dependencies
6. Do NOT change the pretrained weight loading logic in `misc.py` — ensure backward compatibility

---

## Step 7: Verification

After implementation, verify with a simple test:

```python
import torch
from models.mast import build

# Minimal config for building the model
config = {
    "BACKBONE": "resnet50",
    "DILATION": False,
    "NUM_CLASSES": 1,
    "DEVICE": "cuda",
    "DETR_NUM_QUERIES": 300,
    "DETR_NUM_FEATURE_LEVELS": 4,
    "DETR_AUX_LOSS": True,
    "DETR_WITH_BOX_REFINE": True,
    "DETR_TWO_STAGE": False,
    "DETR_HIDDEN_DIM": 256,
    "DETR_MASKS": False,
    "DETR_POSITION_EMBEDDING": "sine",
    "DETR_NUM_HEADS": 8,
    "DETR_ENC_LAYERS": 6,
    "DETR_DEC_LAYERS": 6,
    "DETR_DIM_FEEDFORWARD": 1024,
    "DETR_DROPOUT": 0.0,
    "DETR_DEC_N_POINTS": 4,
    "DETR_ENC_N_POINTS": 4,
    "DETR_CLS_LOSS_COEF": 2.0,
    "DETR_BBOX_LOSS_COEF": 5.0,
    "DETR_GIOU_LOSS_COEF": 2.0,
    "DETR_FOCAL_ALPHA": 0.25,
    "DETR_SET_COST_CLASS": 2.0,
    "DETR_SET_COST_BBOX": 5.0,
    "DETR_SET_COST_GIOU": 2.0,
    "ONLY_DETR": False,
    "LR": 1e-4,
    "LR_BACKBONE_SCALE": 0.1,
}

model, criterion = build(config)
model = model.cuda()

# Test forward pass (pretrain mode first, then tracking mode)
from utils.nested_tensor import nested_tensor_from_tensor_list
dummy_image = torch.randn(1, 3, 800, 1200).cuda()
samples = nested_tensor_from_tensor_list([dummy_image[0]])

# Without track queries (first frame or pretrain)
out = model(samples)
assert "pred_logits" in out
assert "pred_boxes" in out
assert "detect_outputs" in out
assert out["pred_logits"].shape == (1, 300, 1)
assert out["pred_boxes"].shape == (1, 300, 4)
print("Forward pass without track queries: OK")

# With track queries (subsequent frames)
track_q = torch.randn(1, 5, 256).cuda()  # 5 tracked objects
track_si = torch.randn(1, 5, 256).cuda()
out = model(samples, track_queries=track_q, track_spatial_info=track_si)
assert out["assignment_logits"].shape == (1, 5, 301)  # 300 detect + 1 dustbin
print("Forward pass with track queries: OK")
```
