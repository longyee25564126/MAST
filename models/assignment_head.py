# Copyright (c) Ruopeng Gao. All Rights Reserved.

"""
Assignment Head for MAST.
Computes similarity logits between track queries and detect queries,
including a learnable dustbin column for unmatched tracks.
"""

import torch
import torch.nn as nn


class AssignmentHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.track_proj = nn.Linear(d_model, d_model)
        self.detect_proj = nn.Linear(d_model, d_model)
        self.dustbin = nn.Parameter(torch.randn(d_model))
        self.scale = d_model ** -0.5

    def forward(self, track_queries: torch.Tensor, detect_queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            track_queries:  [B, N_track, d_model]
            detect_queries: [B, N_detect, d_model]
        Returns:
            logits: [B, N_track, N_detect + 1]   (last column = dustbin)
        """
        T = self.track_proj(track_queries)    # [B, N_track, d_model]
        D = self.detect_proj(detect_queries)  # [B, N_detect, d_model]
        sim = torch.bmm(T, D.transpose(1, 2)) * self.scale   # [B, N_track, N_detect]
        dustbin_score = (T * self.dustbin).sum(-1, keepdim=True)  # [B, N_track, 1]
        logits = torch.cat([sim, dustbin_score], dim=-1)           # [B, N_track, N_detect+1]
        return logits
