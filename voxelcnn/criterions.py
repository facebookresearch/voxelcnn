#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """ Compute CrossEntropyLoss for coordinates and block types predictions

        Args:
            outputs (dict): A dict of
                ```
                {
                    "coords": float tensor of shape (N, 1, D, D, D),
                    "types": float tensor of shape (N, C, D, D, D),
                }
                ```
                where N is the batch size, C is the number of block types, and D is the
                local size.

            targets (dict): A dict of
                ```
                {
                    "coords": int tensor of shape (N, A), the encoded target coordinates
                    "types": int tensor of shape (N, A), the target block types
                }
                ```
                where N is the batch size and A is the number of next groundtruth
                actions. However, we only use the next one action to compute the loss.

        Returns:
            A dict of losses: coords_loss, types_loss, overall_loss, where each is a
            tensor of float scalar
        """
        N, C, D, D, D = outputs["types"].shape
        assert outputs["coords"].shape == (N, 1, D, D, D)

        coords_targets = targets["coords"][:, 0].view(-1)
        types_targets = targets["types"][:, 0].view(-1)

        coords_outputs = outputs["coords"].view(N, -1)

        # Gather the type prediction on ground truth coordinate
        types_outputs = (
            outputs["types"]
            .view(N, C, D * D * D)
            .gather(dim=2, index=coords_targets.view(N, 1, 1).expand(N, C, 1))
            .view(N, -1)
        )

        coords_loss = self.loss(coords_outputs, coords_targets)
        types_loss = self.loss(types_outputs, types_targets)

        return {
            "coords_loss": coords_loss,
            "types_loss": types_loss,
            "overall_loss": coords_loss + types_loss,
        }
