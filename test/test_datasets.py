#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from voxelcnn.datasets import Craft3DDataset


class TestCraft3DDataset(unittest.TestCase):
    def setUp(self):
        self.annotation = torch.tensor(
            [[0, 0, 0, 0], [3, 3, 3, 3], [1, 1, 1, 1], [2, 2, 2, 2]]
        )

    def test_convert_to_voxels(self):
        voxels = Craft3DDataset._convert_to_voxels(
            self.annotation, size=3, occupancy_only=False
        )
        self.assertEqual(voxels.shape, (256, 3, 3, 3))
        self.assertEqual(voxels[1, 0, 0, 0], 1)
        self.assertEqual(voxels[2, 1, 1, 1], 1)
        self.assertEqual(voxels[3, 2, 2, 2], 1)

        voxels = Craft3DDataset._convert_to_voxels(
            self.annotation, size=5, occupancy_only=True
        )
        self.assertEqual(voxels.shape, (1, 5, 5, 5))
        self.assertEqual(voxels[0, 0, 0, 0], 1)
        self.assertEqual(voxels[0, 1, 1, 1], 1)
        self.assertEqual(voxels[0, 2, 2, 2], 1)
        self.assertEqual(voxels[0, 3, 3, 3], 1)

    def test_prepare_inputs(self):
        inputs = Craft3DDataset.prepare_inputs(
            self.annotation, local_size=3, global_size=5, history=2
        )
        self.assertEqual(set(inputs.keys()), {"local", "global", "center"})
        self.assertTrue(torch.all(inputs["center"] == torch.tensor([2, 2, 2])))
        self.assertEqual(inputs["local"].shape, (512, 3, 3, 3))
        self.assertEqual(inputs["local"][1, 0, 0, 0], 1)
        self.assertEqual(inputs["local"][2, 1, 1, 1], 1)
        self.assertEqual(inputs["local"][3, 2, 2, 2], 1)
        self.assertEqual(inputs["local"][257, 0, 0, 0], 1)
        self.assertEqual(inputs["local"][258, 1, 1, 1], 0)
        self.assertEqual(inputs["local"][259, 2, 2, 2], 1)
        self.assertEqual(inputs["global"].shape, (1, 5, 5, 5))
        self.assertEqual(inputs["global"][0, 0, 0, 0], 1)
        self.assertEqual(inputs["global"][0, 1, 1, 1], 1)
        self.assertEqual(inputs["global"][0, 2, 2, 2], 1)
        self.assertEqual(inputs["global"][0, 3, 3, 3], 1)

    def test_prepare_targets(self):
        targets = Craft3DDataset.prepare_targets(
            self.annotation, next_steps=2, local_size=3
        )
        self.assertEqual(set(targets.keys()), {"coords", "types"})
        self.assertTrue(torch.all(targets["coords"] == torch.tensor([-100, 26])))
        self.assertTrue(torch.all(targets["types"] == torch.tensor([-100, 1])))


if __name__ == "__main__":
    unittest.main()
