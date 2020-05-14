#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from voxelcnn.predictor import Predictor


class TestPredictor(unittest.TestCase):
    def test_decode(self):
        coords_outputs = torch.zeros((2, 1, 3, 3, 3))
        coords_outputs[0, 0, 0, 0, 0] = 1.0
        coords_outputs[1, 0, 1, 1, 1] = 1.0

        types_outputs = torch.zeros((2, 2, 3, 3, 3))
        types_outputs[0, 0, 0, 0, 0] = 1.0
        types_outputs[1, 1, 1, 1, 1] = 1.0

        center = torch.tensor([[3, 3, 3], [10, 11, 12]])
        outputs = {"coords": coords_outputs, "types": types_outputs, "center": center}

        predictions = Predictor.decode(outputs)
        self.assertTrue(
            torch.all(predictions == torch.tensor([[0, 2, 2, 2], [1, 10, 11, 12]]))
        )
