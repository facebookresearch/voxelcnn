#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
from glob import glob
from os import path as osp
from typing import Any, Dict, Optional

import torch
from torch import nn, optim


class Checkpointer(object):
    def __init__(self, root_dir: str):
        """ Save and load checkpoints. Maintain best metrics

        Args:
            root_dir (str): Directory to save the checkpoints
        """
        super().__init__()
        self.root_dir = root_dir
        self.best_metric = -1
        self.best_epoch = None

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        epoch: int,
        metric: float,
    ):
        if self.best_metric < metric:
            self.best_metric = metric
            self.best_epoch = epoch
            is_best = True
        else:
            is_best = False

        os.makedirs(self.root_dir, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
            },
            osp.join(self.root_dir, f"{epoch:02d}.pth"),
        )

        if is_best:
            shutil.copy(
                osp.join(self.root_dir, f"{epoch:02d}.pth"),
                osp.join(self.root_dir, "best.pth"),
            )

    def load(
        self,
        load_from: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, Any]:
        ckp = torch.load(self._get_path(load_from))
        if model is not None:
            model.load_state_dict(ckp["model"])
        if optimizer is not None:
            optimizer.load_state_dict(ckp["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ckp["scheduler"])
        return ckp

    def resume(
        self,
        resume_from: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> int:
        ckp = self.load(
            resume_from, model=model, optimizer=optimizer, scheduler=scheduler
        )
        self.best_epoch = ckp["best_epoch"]
        self.best_metric = ckp["best_metric"]
        return ckp["epoch"]

    def _get_path(self, load_from: str) -> str:
        if load_from == "best":
            return osp.join(self.root_dir, "best.pth")
        if load_from == "latest":
            return sorted(glob(osp.join(self.root_dir, "[0-9]*.pth")))[-1]
        if load_from.isnumeric():
            return osp.join(self.root_dir, f"{int(load_from):02d}.pth")
        return load_from
