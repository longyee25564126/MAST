# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Deformable DETR / MAST
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Detection criterion (SetCriterion) and loss helpers.
Extracted from deformable_detr.py and segmentation.py.
"""

import copy
import torch
import torch.nn.functional as F
from torch import nn

from utils import box_ops
from utils.nested_tensor import nested_tensor_from_tensor_list
from models.misc import accuracy, interpolate, inverse_sigmoid
from utils.misc import is_distributed, distributed_world_size


# ─── Loss helpers ─────────────────────────────────────────────────────────────

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Focal loss used in RetinaNet: https://arxiv.org/abs/1708.02002.
    Args:
        inputs:    float tensor, predictions
        targets:   float tensor, same shape as inputs, binary class labels
        num_boxes: normalisation constant
        alpha:     weighting factor for positive vs negative examples
        gamma:     exponent of the modulating factor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs, targets, num_boxes):
    """DICE loss, similar to generalised IoU for masks."""
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


# ─── SetCriterion ─────────────────────────────────────────────────────────────

class SetCriterion(nn.Module):
    """
    Computes the detection loss for DETR / MAST (DETR component).
    Steps:
        1) Hungarian matching between GT boxes and predictions.
        2) Supervised loss on class, bbox for matched pairs.
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=2,
        ) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)
        src_masks = src_masks[src_idx]
        src_masks = interpolate(
            src_masks[:, None], size=target_masks.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks[tgt_idx].flatten(1)
        return {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        assert "batch_len" in kwargs, "batch_len is required in kwargs"
        batch_len = kwargs["batch_len"]
        kwargs = {}

        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        loss_dict = {}
        iter_idxs = torch.tensor(
            list(range(len(targets))), dtype=torch.int64,
            device=outputs['pred_logits'].device,
        )
        from train import batch_iterator, tensor_dict_index_select
        for batch_iter_idxs, batch_targets, batch_indices in batch_iterator(
            batch_len, iter_idxs, targets, indices
        ):
            batch_outputs = tensor_dict_index_select(outputs, batch_iter_idxs, dim=0)
            batch_loss_dict = loss_map[loss](batch_outputs, batch_targets, batch_indices, 1, **kwargs)
            for k, v in batch_loss_dict.items():
                loss_dict[k] = loss_dict.get(k, 0) + v

        if loss in ("labels", "boxes", "masks"):
            for k in loss_dict:
                loss_dict[k] /= num_boxes
        return loss_dict

    def forward(self, outputs, targets, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items()
                               if k != 'aux_outputs' and k != 'enc_outputs'}

        if "batch_len" not in kwargs:
            indices = self.matcher(outputs_without_aux, targets)
        else:
            indices = []
            iter_idxs = torch.tensor(
                list(range(len(targets))), dtype=torch.int64,
                device=outputs_without_aux['pred_logits'].device,
            )
            from train import batch_iterator, tensor_dict_index_select
            for batch_iter_idxs, batch_targets in batch_iterator(
                kwargs["batch_len"], iter_idxs, targets
            ):
                batch_out = tensor_dict_index_select(outputs_without_aux, batch_iter_idxs, dim=0)
                indices += self.matcher(batch_out, batch_targets)

        batch_len = kwargs["batch_len"]
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device,
        )
        if is_distributed():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / distributed_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes,
                                        batch_len=batch_len))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kw = {'log': False} if loss == 'labels' else {}
                    kw["batch_len"] = batch_len
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_boxes, **kw)
                    losses.update({k + f'_{i}': v for k, v in l_dict.items()})

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            enc_indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    continue
                kw = {'log': False} if loss == 'labels' else {}
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, enc_indices, num_boxes, **kw)
                losses.update({k + '_enc': v for k, v in l_dict.items()})

        return losses, indices
