# Copyright (c) Ruopeng Gao. All Rights Reserved.

"""
TIA — Temporal Identity Aggregation.
Updates a track query after a successful assignment by gating the
matched detect observation into the track state.
"""

import torch
import torch.nn as nn


class TIA(nn.Module):
    """Temporal Identity Aggregation: update track query after successful assignment."""

    def __init__(self, d_model: int):
        super().__init__()
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

    def forward(self, track_query: torch.Tensor, matched_detect: torch.Tensor,
                confidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            track_query:    [N, d_model] — current track queries
            matched_detect: [N, d_model] — matched detect query outputs
            confidence:     [N]          — sigmoid of assignment logits
        Returns:
            updated_track_query: [N, d_model]
        """
        observation = self.observe_proj(matched_detect)
        gate_input = torch.cat([track_query, observation, confidence.unsqueeze(-1)], dim=-1)
        gate = self.gate_net(gate_input)
        return gate * observation + (1 - gate) * track_query
