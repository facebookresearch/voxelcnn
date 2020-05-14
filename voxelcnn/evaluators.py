#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, Tuple

import torch
from torch import nn
from tqdm import tqdm

from .datasets import Craft3DDataset
from .predictor import Predictor


class Accuracy(nn.Module):
    def __init__(self, next_steps: int = 1):
        """ Compute the accuracy of coordinates and types predictions

        Args:
            next_steps (int): The number of future ground truth steps to be considered.
                Default: 1
        """
        super().__init__()
        self.next_steps = next_steps

    def step(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """ Compute sample-wise accuracy in a minibatch

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
                actions.
        """
        N, C, D, D, D = outputs["types"].shape
        assert outputs["coords"].shape == (N, 1, D, D, D)
        assert targets["coords"].shape == targets["types"].shape

        K = self.next_steps if self.next_steps > 0 else targets["coords"].shape[1]
        if targets["coords"].shape[1] < K:
            raise RuntimeError(f"Targets do not contain next {K} steps")

        coords_targets = targets["coords"][:, :K].view(N, -1)
        types_targets = targets["types"][:, :K].view(N, -1)

        coords_predictions = outputs["coords"].view(N, -1).argmax(dim=1, keepdim=True)
        coords_correct = (coords_predictions == coords_targets).any(dim=1)

        types_predictions = (
            outputs["types"]
            .view(N, C, D * D * D)
            .gather(dim=2, index=coords_predictions.view(N, 1, 1).expand(N, C, 1))
            .argmax(dim=1)
            .view(N, -1)
        )
        types_correct = (types_predictions == types_targets).any(dim=1)

        both_correct = coords_correct & types_correct
        return both_correct

    def stop(self, correct: torch.Tensor) -> torch.Tensor:
        """ Average over batched results

        Args:
            correct (torch.Tensor): (N,) bool vector, whether each sample is correct

        Returns:
            A float scalar tensor of averaged accuracy
        """
        return correct.float().mean()

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.stop(self.step(outputs, targets))


class CCA(object):
    def __init__(
        self,
        percentages: Tuple[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
        local_size: int = 7,
        global_size: int = 21,
        history: int = 3,
    ):
        """ Consecutive Correct Actions

        Args:
            percentages (tuple): Evaluate based on these percentages of blocks
                prebuilt for each house. Average the CCA over these results
            local_size (int): Local context size. Default: 7
            global_size (int): Global context size. Default: 21
            history (int): Number of previous steps considered as inputs. Default: 3
        """
        super().__init__()
        self.percentages = percentages
        self.local_size = local_size
        self.global_size = global_size
        self.history = history

    @torch.no_grad()
    def evaluate(self, dataset: Craft3DDataset, model: nn.Module) -> Dict[str, float]:
        """ Evaluate a model by CCA over a given dataset

        Args:
            dataset (Craft3DDataset): A dataset
            model (nn.Module): A VoxelCNN model

        Returns:
            A dict of string to float
            ```
                {
                    "cca_x%": where x is the percentage of prebuilt house,
                    "cca_avg": averaged CCA across all the percentages,
                }
            ```
        """
        predictor = Predictor(
            model.eval(),
            local_size=self.local_size,
            global_size=self.global_size,
            history=self.history,
        )
        all_results = defaultdict(list)
        for i in tqdm(range(dataset.get_num_houses())):
            house = dataset.get_house(i)
            for p in self.percentages:
                start = int(len(house) * p)
                predictions = predictor.predict_until_wrong(house, start=start)
                all_results[p].append(len(predictions))

        results = {
            f"cca_{k:.0%}": float(torch.tensor(v).float().mean())
            for k, v in all_results.items()
        }
        results["cca_avg"] = float(torch.tensor(list(results.values())).float().mean())
        return results


class MTC(object):
    def __init__(
        self,
        percentage: float = 0.1,
        local_size: int = 7,
        global_size: int = 21,
        history: int = 3,
    ):
        """ Mistakes to Complete

        Args:
            percentage (float): Evaluate based on this percentage of blocks
                prebuilt for each house
            local_size (int): Local context size. Default: 7
            global_size (int): Global context size. Default: 21
            history (int): Number of previous steps considered as inputs. Default: 3
        """
        super().__init__()
        self.percentage = percentage
        self.local_size = local_size
        self.global_size = global_size
        self.history = history

    @torch.no_grad()
    def evaluate(self, dataset: Craft3DDataset, model: nn.Module) -> Dict[str, float]:
        """ Evaluate a model by MTC over a given dataset

        Args:
            dataset (Craft3DDataset): A dataset
            model (nn.Module): A VoxelCNN model

        Returns:
            A dict of string to float
            ```
                {
                    "mtc": Unnormalized MTC,
                    "mtc_normed": Normalized MTC by the total blocks of each house,
                }
            ```
        """
        predictor = Predictor(
            model.eval(),
            local_size=self.local_size,
            global_size=self.global_size,
            history=self.history,
        )
        unnormed = []
        normed = []
        for i in tqdm(range(dataset.get_num_houses())):
            house = dataset.get_house(i)
            start = int(len(house) * self.percentage)
            is_correct = predictor.predict_until_end(house, start=start)["is_correct"]
            num_mistakes = len(is_correct) - int(is_correct.sum())
            total = len(is_correct)
            unnormed.append(num_mistakes)
            normed.append(num_mistakes / total)
        return {
            "mtc": float(torch.tensor(unnormed).float().mean()),
            "mtc_normed": float(torch.tensor(normed).float().mean()),
        }

    def _compute(self, predictor: Predictor, annotation: torch.Tensor) -> int:
        start = int(len(annotation) * self.percentage)
        is_correct = predictor.predict_until_end(annotation, start=start)["is_correct"]
        num_mistakes = len(is_correct) - is_correct.sum()
        return num_mistakes
