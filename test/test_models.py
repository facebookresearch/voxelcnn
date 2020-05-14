#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from voxelcnn.models import VoxelCNN


class TestVoxelCNN(unittest.TestCase):
    def test_forward(self):
        model = VoxelCNN()
        inputs = {
            "local": torch.rand(5, 256 * 3, 7, 7, 7),
            "global": torch.rand(5, 1, 21, 21, 21),
            "center": torch.randint(128, size=(5, 3)),
        }
        outputs = model(inputs)
        self.assertEqual(set(outputs.keys()), {"coords", "types", "center"})
        self.assertEqual(outputs["coords"].shape, (5, 1, 7, 7, 7))
        self.assertEqual(outputs["types"].shape, (5, 256, 7, 7, 7))
        self.assertEqual(outputs["center"].shape, (5, 3))
