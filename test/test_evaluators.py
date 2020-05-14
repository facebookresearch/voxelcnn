#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from voxelcnn.evaluators import Accuracy


class TestAccuracy(unittest.TestCase):
    def test_forward(self):
        coords_outputs = torch.zeros((2, 1, 3, 3, 3))
        coords_outputs[0, 0, 0, 0, 0] = 1.0
        coords_outputs[1, 0, 1, 1, 1] = 1.0

        types_outputs = torch.zeros((2, 2, 3, 3, 3))
        types_outputs[0, 0, 0, 0, 0] = 1.0
        types_outputs[1, 1, 1, 1, 1] = 1.0

        outputs = {"coords": coords_outputs, "types": types_outputs}
        targets = {
            "coords": torch.tensor([[0, 1], [12, 13]]),
            "types": torch.tensor([[0, 0], [1, 1]]),
        }

        acc1 = Accuracy(next_steps=1)(outputs, targets).item()
        self.assertEqual(acc1, 0.5)

        acc2 = Accuracy(next_steps=2)(outputs, targets).item()
        self.assertEqual(acc2, 1.0)
