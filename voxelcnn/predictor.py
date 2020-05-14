#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from torch import nn

from .datasets import Craft3DDataset
from .utils import OrderedSet


class Predictor(object):
    def __init__(
        self,
        model: nn.Module,
        local_size: int = 7,
        global_size: int = 21,
        history: int = 3,
    ):
        """ Predictor for inference and evaluation

        Args:
            model (nn.Module): VoxelCNN model
            local_size (int): Local context size. Default: 7
            global_size (int): Global context size. Default: 21
            history (int): Number of previous steps considered as inputs. Default: 3
        """
        super().__init__()
        self.model = model
        self.local_size = local_size
        self.global_size = global_size
        self.history = history

    @torch.no_grad()
    def predict(self, annotation: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """ Continuous prediction for given steps starting from a prebuilt house

        Args:
            annotation (torch.Tensor): M x 4 int tensor, where M is the number of
                prebuilt blocks. The first column is the block type, followed by the
                absolute block coordinates.
            steps (int): How many steps to predict. Default: 1

        Returns:
            An int tensor of (steps, 4) if steps > 1, otherwise (4,). Denoting the
            predicted blocks. The first column is the block type, followed by the
            absolute block coordinates.
        """
        predictions = []
        for _ in range(steps):
            inputs = Craft3DDataset.prepare_inputs(
                annotation,
                local_size=self.local_size,
                global_size=self.global_size,
                history=self.history,
            )
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
            if next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.model(inputs)
            prediction = self.decode(outputs).cpu()
            predictions.append(prediction)
            annotation = torch.cat([annotation, prediction], dim=0)
        predictions = torch.cat(predictions, dim=0)
        return predictions.squeeze()

    @torch.no_grad()
    def predict_until_wrong(
        self, annotation: torch.Tensor, start: int = 0
    ) -> torch.Tensor:
        """ Starting from a house, predict until a wrong prediction occurs

        Args:
            annotation (torch.Tensor): M x 4 int tensor, where M is the number of
                prebuilt blocks. The first column is the block type, followed by the
                absolute block coordinates.
            start (int): Starting from this number of blocks prebuilt

        Returns:
            An int tensor of (steps, 4). Denoting the correctly predicted blocks. The
                first column is the block type, followed by the absolute block
                coordinates.
        """
        built = annotation[:start].tolist()
        to_build = {tuple(x) for x in annotation[start:].tolist()}
        predictions = []
        while len(to_build) > 0:
            block = self.predict(torch.tensor(built, dtype=torch.int64)).tolist()
            if tuple(block) not in to_build:
                break
            predictions.append(block)
            built.append(block)
            to_build.remove(tuple(block))
        return torch.tensor(predictions)

    @torch.no_grad()
    def predict_until_end(
        self, annotation: torch.Tensor, start: int = 0
    ) -> Dict[str, torch.Tensor]:
        """ Starting from a house, predict until a the house is completed

        Args:
            annotation (torch.Tensor): M x 4 int tensor, where M is the number of
                prebuilt blocks. The first column is the block type, followed by the
                absolute block coordinates.
            start (int): Starting from this number of blocks prebuilt

        Returns:
            A dict of
            ```
            {
                "predictions": int tensor of shape (steps, 4),
                "targets": int tensor of shape (steps, 4),
                "is_correct": bool tensor of shape (steps,)
            }
            ```
            where ``steps`` is the number of blocks predicted, in order to complete the
            house. ``predictions`` is the model predictions, which could be wrong at
            some steps. ``targets`` contains the corrected blocks. ``is_correct``
            denotes whether the prediction is correct at certain step.
        """
        built = annotation[:start].tolist()
        to_build = OrderedSet(tuple(x) for x in annotation[start:].tolist())

        predictions = []
        targets = []
        is_correct = []
        while len(to_build) > 0:
            block = self.predict(torch.tensor(built, dtype=torch.int64)).tolist()
            predictions.append(block)
            if tuple(block) in to_build:
                # Prediction is correct. Add it to the house
                is_correct.append(True)
                targets.append(block)
                built.append(block)
                to_build.remove(tuple(block))
            else:
                # Prediction is wrong. Correct it by the ground truth block
                is_correct.append(False)
                correction = to_build.pop(last=False)
                targets.append(correction)
                built.append(correction)

        return {
            "predictions": torch.tensor(predictions),
            "targets": torch.tensor(targets),
            "is_correct": torch.tensor(is_correct),
        }

    @staticmethod
    def decode(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """ Convert model output scores to absolute block coordinates and types

        Args:
            outputs (dict): A dict of coordinates and types scores
                ```
                {
                    "coords": float tensor of shape (N, 1, D, D, D),
                    "types": float tensor of shape (N, C, D, D, D),
                    "center": int tensor of shape (N, 3), the coordinate of the last
                        blocks
                }
                ```
                where N is the batch size, C is the number of block types, D is the
                local context size
        Returns:
            An int tensor of shape (N, 4), where the first column is the block type,
            followed by absolute block coordinates
        """
        N, C, D, D, D = outputs["types"].shape
        assert outputs["coords"].shape == (N, 1, D, D, D)
        assert outputs["center"].shape == (N, 3)

        coords_predictions = outputs["coords"].view(N, -1).argmax(dim=1)

        types_predictions = (
            outputs["types"]
            .view(N, C, D * D * D)
            .gather(dim=2, index=coords_predictions.view(N, 1, 1).expand(N, C, 1))
            .argmax(dim=1)
            .view(-1)
        )

        z = coords_predictions % D
        x = coords_predictions // (D * D)
        y = (coords_predictions - z - x * (D * D)) // D
        ret = torch.stack([types_predictions, x, y, z], dim=1)
        ret[:, 1:] += outputs["center"] - ret.new_tensor([D, D, D]) // 2

        return ret


if __name__ == "__main__":
    from .models import VoxelCNN

    model = VoxelCNN()
    predictor = Predictor(model.eval())
    annotation = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
    results = predictor.predict_until_end(annotation)
    print(results)
