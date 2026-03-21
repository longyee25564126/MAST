# Copyright (c) Ruopeng Gao. All Rights Reserved.

"""
Detection head helper — computes pred_logits / pred_boxes from decoder outputs.

The class_embed and bbox_embed parameters are owned by _DetrCore (in mast.py)
to keep them at the key paths detr.class_embed.* and detr.bbox_embed.* so that
pretrained weight loading via load_detr_pretrain() (misc.py) works unchanged.
This module therefore provides only the forward computation logic.
"""

import torch
from models.misc import inverse_sigmoid


def compute_detections(hs, init_reference, inter_references, class_embed, bbox_embed):
    """
    Convert decoder outputs to detection predictions.

    Args:
        hs:               [num_layers, B, N_detect, d_model]
        init_reference:   [B, N_detect, 2]  (initial reference points)
        inter_references: [num_layers, B, N_detect, 2]  (per-layer updated refs)
        class_embed:      nn.ModuleList[nn.Linear]  length = num_layers
        bbox_embed:       nn.ModuleList[MLP]        length = num_layers

    Returns:
        outputs_class: [num_layers, B, N_detect, num_classes]
        outputs_coord: [num_layers, B, N_detect, 4]
    """
    outputs_classes = []
    outputs_coords = []

    for lvl in range(hs.shape[0]):
        reference = init_reference if lvl == 0 else inter_references[lvl - 1]
        reference = inverse_sigmoid(reference)

        outputs_class = class_embed[lvl](hs[lvl])
        tmp = bbox_embed[lvl](hs[lvl])

        if reference.shape[-1] == 4:
            tmp = tmp + reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] = tmp[..., :2] + reference

        outputs_coord = tmp.sigmoid()
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)

    outputs_class = torch.stack(outputs_classes)
    outputs_coord = torch.stack(outputs_coords)
    return outputs_class, outputs_coord


def set_aux_loss(outputs_class, outputs_coord):
    """Build the aux_outputs list (all layers except the last)."""
    return [
        {'pred_logits': a, 'pred_boxes': b}
        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
    ]
