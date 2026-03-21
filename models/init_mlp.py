# Copyright (c) Ruopeng Gao. All Rights Reserved.

"""
InitMLP for MAST.
Maps a detect query output to track query space when a new target is born.
"""

import torch
import torch.nn as nn


class InitMLP(nn.Module):
    """Maps detect query output to track query space when a new target is born."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, detect_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            detect_output: [N_new, d_model] — detect query outputs for newly born targets
        Returns:
            track_query_init: [N_new, d_model]
        """
        return self.mlp(detect_output)
