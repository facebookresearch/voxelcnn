#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import tarfile
import warnings
from os import path as osp
from typing import Dict, Optional, Tuple

import numpy as np
import requests
import torch
from torch.utils.data import Dataset


class Craft3DDataset(Dataset):
    NUM_BLOCK_TYPES = 256
    URL = "https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz"

    def __init__(
        self,
        data_dir: str,
        subset: str,
        local_size: int = 7,
        global_size: int = 21,
        history: int = 3,
        next_steps: int = -1,
        max_samples: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """ Download and construct 3D-Craft dataset

        data_dir (str): Directory to save/load the dataset
        subset (str): 'train' | 'val' | 'test'
        local_size (int): Local context size. Default: 7
        global_size (int): Global context size. Default: 21
        history (int): Number of previous steps considered as inputs. Default: 3
        next_steps (int): Number of next steps considered as targets. Default: -1,
            meaning till the end
        max_samples (int, optional): Limit the maximum number of samples. Used for
            faster debugging. Default: None, meaning no limit
        logger (logging.Logger, optional): A logger. Default: None, meaning will print
            to stdout
        """
        super().__init__()
        self.data_dir = data_dir
        self.subset = subset
        self.local_size = local_size
        self.global_size = global_size
        self.history = history
        self.max_local_distance = self.local_size // 2
        self.max_global_distance = self.global_size // 2
        self.next_steps = next_steps
        self.max_samples = max_samples
        self.logger = logger

        if self.subset not in ("train", "val", "test"):
            raise ValueError(f"Unknown subset: {self.subset}")

        if not self._has_raw_data():
            self._download()

        self._load_dataset()
        self._find_valid_items()

        self.print_stats()

    def print_stats(self):
        num_blocks_per_house = [len(x) for x in self._valid_indices.values()]
        ret = "\n"
        ret += f"3D Craft Dataset\n"
        ret += f"================\n"
        ret += f"  data_dir: {self.data_dir}\n"
        ret += f"  subset: {self.subset}\n"
        ret += f"  local_size: {self.local_size}\n"
        ret += f"  global_size: {self.global_size}\n"
        ret += f"  history: {self.history}\n"
        ret += f"  next_steps: {self.next_steps}\n"
        ret += f"  max_samples: {self.max_samples}\n"
        ret += f"  --------------\n"
        ret += f"  num_houses: {len(self._valid_indices)}\n"
        ret += f"  avg_blocks_per_house: {np.mean(num_blocks_per_house):.3f}\n"
        ret += f"  min_blocks_per_house: {min(num_blocks_per_house)}\n"
        ret += f"  max_blocks_per_house: {max(num_blocks_per_house)}\n"
        ret += f"  total_valid_blocks: {len(self._flattened_valid_indices)}\n"
        ret += "\n"
        self._log(ret)

    def __len__(self) -> int:
        """ Get number of valid blocks """
        ret = len(self._flattened_valid_indices)
        if self.max_samples is not None:
            ret = min(ret, self.max_samples)
        return ret

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """ Get the index-th valid block

        Returns:
            A tuple of inputs and targets, where inputs is a dict of
            ```
            {
                "local": float tensor of shape (C * H, D, D, D),
                "global": float tensor of shape (1, G, G, G),
                "center": int tensor of shape (3,), the coordinate of the last block
            }
            ```
            where C is the number of block types, H is the history length, D is the
            local size, and G is the global size.

            targets is a dict of
            ```
            {
                "coords": int tensor of shape (A,)
                "types": int tensor of shape (A,)
            }
            ```
            where A is the number of next steps to be considered as targets.
        """
        house_id, block_id = self._flattened_valid_indices[index]
        annotation = self._all_houses[house_id]
        inputs = Craft3DDataset.prepare_inputs(
            annotation[: block_id + 1],
            local_size=self.local_size,
            global_size=self.global_size,
            history=self.history,
        )
        targets = Craft3DDataset.prepare_targets(
            annotation[block_id:],
            next_steps=self.next_steps,
            local_size=self.local_size,
        )
        return inputs, targets

    def get_house(self, index: int) -> torch.Tensor:
        """ Get the annotation for the index-th house. Use for thorough evaluation """
        return self._all_houses[index]

    def get_num_houses(self) -> int:
        """ Get the total number of houses. Use for thorough evaluation """
        return len(self._all_houses)

    @staticmethod
    @torch.no_grad()
    def prepare_inputs(
        annotation: torch.Tensor,
        local_size: int = 7,
        global_size: int = 21,
        history: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """ Convert annotation to input tensors

        Args:
            annotation (torch.Tensor): M x 4 int tensor, where M is the number of
                prebuilt blocks. The first column is the block type, followed by the
                block coordinates.

        Returns:
            ```
            {
                "local": float tensor of shape (C * H, D, D, D),
                "global": float tensor of shape (1, G, G, G),
                "center": int tensor of shape (3,), the coordinate of the last block
            }
            ```
            where C is the number of block types, H is the history length, D is the
            local size, and G is the global size.
        """
        global_inputs = Craft3DDataset._convert_to_voxels(
            annotation, size=global_size, occupancy_only=True
        )
        local_inputs = Craft3DDataset._convert_to_voxels(
            annotation, size=local_size, occupancy_only=False
        )
        if len(annotation) == 0:
            return {
                "local": local_inputs.repeat(history, 1, 1, 1),
                "global": global_inputs,
                "center": torch.zeros((3,), dtype=torch.int64),
            }

        last_coord = annotation[-1, 1:]
        center_coord = last_coord.new_full((3,), local_size // 2)
        local_history = [local_inputs]
        for i in range(len(annotation) - 1, len(annotation) - history, -1):
            if i < 0:
                local_history.append(torch.zeros_like(local_inputs))
            else:
                prev_inputs = local_history[-1].clone()
                prev_coord = annotation[i, 1:] - last_coord + center_coord
                if all((prev_coord >= 0) & (prev_coord < local_size)):
                    x, y, z = prev_coord
                    prev_inputs[:, x, y, z] = 0
                local_history.append(prev_inputs)
        local_inputs = torch.cat(local_history, dim=0)
        return {"local": local_inputs, "global": global_inputs, "center": last_coord}

    @staticmethod
    @torch.no_grad()
    def prepare_targets(
        annotation: torch.Tensor, next_steps: int = 1, local_size: int = 7
    ) -> Dict[str, torch.Tensor]:
        """ Convert annotation to target tensors

        Args:
            annotation (torch.Tensor): (M + 1) x 4 int tensor, where M is the number of
                blocks to build, plus one for the last built block. The first column
                is the block type, followed by the block coordinates.

        Returns:
            ```
            {
                "coords": int tensor of shape (A,)
                "types": int tensor of shape (A,)
            }
            ```
            where A is the number of next steps to be considered as targets
        """
        coords_targets = torch.full((next_steps,), -100, dtype=torch.int64)
        types_targets = coords_targets.clone()

        if len(annotation) <= 1:
            return {"coords": coords_targets, "types": types_targets}

        offsets = torch.tensor([local_size * local_size, local_size, 1])
        last_coord = annotation[0, 1:]
        center_coord = last_coord.new_full((3,), local_size // 2)

        N = min(1 + next_steps, len(annotation))
        next_types = annotation[1:N, 0].clone()
        next_coords = annotation[1:N, 1:] - last_coord + center_coord
        mask = (next_coords < 0) | (next_coords >= local_size)
        mask = mask.any(dim=1)
        next_coords = (next_coords * offsets).sum(dim=1)
        next_coords[mask] = -100
        next_types[mask] = -100

        coords_targets[: len(next_coords)] = next_coords
        types_targets[: len(next_types)] = next_types

        return {"coords": coords_targets, "types": types_targets}

    @staticmethod
    def _convert_to_voxels(
        annotation: torch.Tensor, size: int, occupancy_only: bool = False
    ) -> torch.Tensor:
        voxels_shape = (
            (1, size, size, size)
            if occupancy_only
            else (Craft3DDataset.NUM_BLOCK_TYPES, size, size, size)
        )
        if len(annotation) == 0:
            return torch.zeros(voxels_shape, dtype=torch.float32)

        annotation = annotation.clone()
        if occupancy_only:
            # No block types. Just coordinate occupancy
            annotation[:, 0] = 0
        # Shift the coordinates to make the last block centered
        last_coord = annotation[-1, 1:]
        center_coord = last_coord.new_tensor([size // 2, size // 2, size // 2])
        annotation[:, 1:] += center_coord - last_coord
        # Find valid annotation that inside the cube
        valid_mask = (annotation[:, 1:] >= 0) & (annotation[:, 1:] < size)
        valid_mask = valid_mask.all(dim=1)
        annotation = annotation[valid_mask]
        # Use sparse tensor to construct the voxels cube
        return torch.sparse.FloatTensor(
            annotation.t(), torch.ones(len(annotation)), voxels_shape
        ).to_dense()

    def _log(self, msg: str):
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)

    def _has_raw_data(self) -> bool:
        return osp.isdir(osp.join(self.data_dir, "houses"))

    def _download(self):
        os.makedirs(self.data_dir, exist_ok=True)

        tar_path = osp.join(self.data_dir, "houses.tar.gz")
        if not osp.isfile(tar_path):
            self._log(f"Downloading dataset from {Craft3DDataset.URL}")
            response = requests.get(Craft3DDataset.URL, allow_redirects=True)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to retrieve image from url: {Craft3DDataset.URL}. "
                    f"Status: {response.status_code}"
                )
            with open(tar_path, "wb") as f:
                f.write(response.content)

        extracted_dir = osp.join(self.data_dir, "houses")
        if not osp.isdir(extracted_dir):
            self._log(f"Extracting dataset to {extracted_dir}")
            tar = tarfile.open(tar_path, "r")
            tar.extractall(self.data_dir)

    def _load_dataset(self):
        splits_path = osp.join(self.data_dir, "splits.json")
        if not osp.isfile(splits_path):
            raise RuntimeError(f"Split file not found at: {splits_path}")

        with open(splits_path, "r") as f:
            splits = json.load(f)

        self._all_houses = []
        max_len = 0
        for filename in splits[self.subset]:
            annotation = osp.join(self.data_dir, "houses", filename, "placed.json")
            if not osp.isfile(annotation):
                warnings.warn(f"No annotation file for: {annotation}")
                continue
            annotation = self._load_annotation(annotation)
            if len(annotation) >= 100:
                self._all_houses.append(annotation)
                max_len = max(max_len, len(annotation))

        if self.next_steps <= 0:
            self.next_steps = max_len

    def _load_annotation(self, annotation_path: str) -> torch.Tensor:
        with open(annotation_path, "r") as f:
            annotation = json.load(f)
        final_house = {}
        types_and_coords = []
        last_timestamp = -1
        for i, item in enumerate(annotation):
            timestamp, annotator_id, coordinate, block_info, action = item
            assert timestamp >= last_timestamp
            last_timestamp = timestamp
            coordinate = tuple(np.asarray(coordinate).astype(np.int64).tolist())
            block_type = np.asarray(block_info, dtype=np.uint8).astype(np.int64)[0]
            if action == "B":
                final_house.pop(coordinate, None)
            else:
                final_house[coordinate] = i
            types_and_coords.append((block_type,) + coordinate)
        indices = sorted(final_house.values())
        types_and_coords = [types_and_coords[i] for i in indices]
        return torch.tensor(types_and_coords, dtype=torch.int64)

    def _find_valid_items(self):
        self._valid_indices = {}
        for i, annotation in enumerate(self._all_houses):
            diff_coord = annotation[:-1, 1:] - annotation[1:, 1:]
            valids = abs(diff_coord) <= self.max_local_distance
            valids = valids.all(dim=1).nonzero(as_tuple=True)[0]
            self._valid_indices[i] = valids.tolist()

        self._flattened_valid_indices = []
        for i, indices in self._valid_indices.items():
            for j in indices:
                self._flattened_valid_indices.append((i, j))


if __name__ == "__main__":
    work_dir = osp.join(osp.dirname(osp.abspath(__file__)), "..")
    dataset = Craft3DDataset(osp.join(work_dir, "data"), "val")
    for i in range(5):
        inputs, targets = dataset[i]
        print(targets)
