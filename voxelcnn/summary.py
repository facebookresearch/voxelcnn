#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch


class Summary(object):
    Datum = Dict[str, Union[float, torch.Tensor]]

    def __init__(self, window_size: int = 20, logger: Optional[logging.Logger] = None):
        """ Training summary helper

        Args:
            window_size (int): Compute the moving average of scalars within this number
                of history values
            logger (logging.Logger, optional): A logger. Default: None, meaning will
                print to stdout
        """
        super().__init__()
        self.window_size = window_size
        self.logger = logger
        self._times = defaultdict(list)
        self._lrs = defaultdict(list)
        self._losses = defaultdict(list)
        self._metrics = defaultdict(list)

    def add_times(self, times: Datum):
        for k, v in times.items():
            if isinstance(v, torch.Tensor):
                v = float(v)
            self._times[k].append(v)

    def add_lrs(self, lrs: Datum):
        for k, v in lrs.items():
            if isinstance(v, torch.Tensor):
                v = float(v)
            self._lrs[k].append(v)

    def add_losses(self, losses: Datum):
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                v = float(v)
            self._losses[k].append(v)

    def add_metrics(self, metrics: Datum):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = float(v)
            self._metrics[k].append(v)

    def add(
        self,
        times: Optional[Datum] = None,
        lrs: Optional[Datum] = None,
        losses: Optional[Datum] = None,
        metrics: Optional[Datum] = None,
    ):
        if times is not None:
            self.add_times(times)
        if lrs is not None:
            self.add_lrs(lrs)
        if losses is not None:
            self.add_losses(losses)
        if metrics is not None:
            self.add_metrics(metrics)

    def print_current(self, prefix: Optional[str] = None):
        items = [] if prefix is None else [prefix]
        items += [f"{k}: {v[-1]:.6f}" for k, v in list(self._lrs.items())]
        items += [
            f"{k}: {v[-1]:.3f} ({self._moving_average(v):.3f})"
            for k, v in list(self._times.items())
            + list(self._losses.items())
            + list(self._metrics.items())
        ]
        self._log("  ".join(items))

    def _moving_average(self, values: List[float]) -> float:
        return np.mean(values[max(0, len(values) - self.window_size) :])

    def _log(self, msg: str):
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
