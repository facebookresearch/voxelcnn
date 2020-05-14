#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
from torch import nn


def conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> List[nn.Module]:
    conv = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding,
        stride=stride,
        bias=False,
    )
    bn = nn.BatchNorm3d(out_channels)
    relu = nn.ReLU()
    return [conv, bn, relu]


class VoxelCNN(nn.Module):
    def __init__(
        self,
        local_size: int = 7,
        global_size: int = 21,
        history: int = 3,
        num_block_types: int = 256,
        num_features: int = 16,
    ):
        """ VoxelCNN model

        Args:
            local_size (int): Local context size. Default: 7
            global_size (int): Global context size. Default: 21
            history (int): Number of previous steps considered as inputs. Default: 3
            num_block_types (int): Total number of different block types. Default: 256
            num_features (int): Number of channels output by the encoders. Default: 16
        """
        super().__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.history = history
        self.num_block_types = num_block_types
        self.num_features = num_features

        self.local_encoder = self._build_local_encoder()
        self.global_encoder = self._build_global_encoder()

        self.feature_extractor = nn.Sequential(
            *conv3d(self.num_features * 2, self.num_features, kernel_size=1, padding=0)
        )
        self.coords_predictor = nn.Conv3d(
            self.num_features, 1, kernel_size=1, padding=0
        )
        self.types_predictor = nn.Conv3d(
            self.num_features, self.num_block_types, kernel_size=1, padding=0
        )

        self._init_params()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs (dict): A dict of inputs
                ```
                {
                    "local": float tensor of shape (N, C * H, D, D, D),
                    "global": float tensor of shape (N, 1, G, G, G),
                    "center": int tensor of shape (N, 3), the coordinate of the last
                        blocks, optional
                }
                ```
                where N is the batch size, C is the number of block types, H is the
                history length, D is the local size, and G is the global size.

        Returns:
            A dict of coordinates and types scores
            ```
            {
                "coords": float tensor of shape (N, 1, D, D, D),
                "types": float tensor of shape (N, C, D, D, D),
                "center": int tensor of shape (N, 3), the coordinate of the last blocks.
                    Output only when inputs have "center"
            }
            ```
        """
        outputs = torch.cat(
            [
                self.local_encoder(inputs["local"]),
                self.global_encoder(inputs["global"]),
            ],
            dim=1,
        )
        outputs = self.feature_extractor(outputs)
        ret = {
            "coords": self.coords_predictor(outputs),
            "types": self.types_predictor(outputs),
        }
        if "center" in inputs:
            ret["center"] = inputs["center"]
        return ret

    def _build_local_encoder(self) -> nn.Module:
        layers = conv3d(self.num_block_types * self.history, self.num_features)
        for _ in range(3):
            layers.extend(conv3d(self.num_features, self.num_features))
        return nn.Sequential(*layers)

    def _build_global_encoder(self) -> nn.Module:
        layers = conv3d(1, self.num_features)
        layers.extend(conv3d(self.num_features, self.num_features))
        layers.append(
            nn.AdaptiveMaxPool3d((self.local_size, self.local_size, self.local_size))
        )
        layers.extend(conv3d(self.num_features, self.num_features))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is None:
                    # Normal Conv3d layers
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                else:
                    # Last layers of coords and types predictions
                    nn.init.normal_(m.weight, mean=0, std=0.001)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
