#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch
from voxelcnn.criterions import CrossEntropyLoss


class TestCrossEntropyLoss(unittest.TestCase):
    def test_forward(self):
        targets = {
            "coords": torch.tensor([[0], [13]]),
            "types": torch.tensor([[0], [1]]),
        }

        coords_outputs = torch.zeros((2, 1, 3, 3, 3))
        coords_outputs[0, 0, 0, 0, 0] = 1.0
        coords_outputs[1, 0, 1, 1, 1] = 1.0

        types_outputs = torch.zeros((2, 2, 3, 3, 3))
        types_outputs[0, 0, 0, 0, 0] = 1.0
        types_outputs[1, 1, 1, 1, 1] = 1.0
        outputs = {"coords": coords_outputs, "types": types_outputs}

        criterion = CrossEntropyLoss()
        losses = criterion(outputs, targets)

        p_coords = math.exp(1.0) / (math.exp(1.0) + 26)
        p_types = math.exp(1.0) / (math.exp(1.0) + 1)
        self.assertAlmostEqual(
            float(losses["coords_loss"]), -math.log(p_coords), places=3
        )
        self.assertAlmostEqual(
            float(losses["types_loss"]), -math.log(p_types), places=3
        )


if __name__ == "__main__":
    unittest.main()
