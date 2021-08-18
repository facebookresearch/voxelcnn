#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import operator
import os
import sys
from collections import OrderedDict
from os import path as osp
from time import time as tic
from typing import Dict, List, Tuple, Union

import torch


StructuredData = Union[
    Dict[str, "StructuredData"],
    List["StructuredData"],
    Tuple["StructuredData"],
    torch.Tensor,
]


def to_cuda(data: StructuredData) -> StructuredData:
    if isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)
    if isinstance(data, dict):
        return {k: to_cuda(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_cuda(x) for x in data)
    raise ValueError(f"Unknown data type: {type(data)}")


def collate_batches(batches):
    """ Collate a list of batches into a batch

    Args:
        batches (list): a list of batches. Each batch could be a tensor, dict, tuple,
            list, string, number, namedtuple

    Returns:
        batch: collated batches where tensors are concatenated along the outer dim.
            For example, given samples `[torch.empty(3, 5), torch.empty(3, 5)]`, the
            result will be a tensor of shape `(6, 5)`.
    """
    batch = batches[0]
    if isinstance(batch, torch.Tensor):
        return batch if len(batches) == 1 else torch.cat(batches, 0)
    if isinstance(batch, (list, tuple)):
        transposed = zip(*batches)
        return type(batch)([collate_batches(b) for b in transposed])
    if isinstance(batch, dict):
        return {k: collate_batches([d[k] for d in batches]) for k in batch}
    # Otherwise, just return the input as it is
    return batches


def setup_logger(name=None, save_file=None, rank=0, level=logging.DEBUG):
    """ Setup logger once for each process

    Logging messages will be printed to stdout, with DEBUG level. If save_file is set,
    messages will be logged to disk files as well.

    When multiprocessing, this function must be called inside each process, but only
    the main process (rank == 0) will log to stdout and files.

    Args:
        name (str, optional): Name of the logger. Default: None, will use the root
            logger
        save_file (str, optional): Path to a file where log messages are saved into.
            Default: None, do not log to file
        rank (int): Rank of the process. Default: 0, the main process
        level (int): An integer of logging level. Recommended to be one of
            logging.DEBUG / INFO / WARNING / ERROR / CRITICAL. Default: logging.DEBUG
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Don't log results for the non-main process
    if rank > 0:
        logging.disable(logging.CRITICAL)
        return logger

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s:%(lineno)4d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = False  # prevent double logging

    if save_file is not None:
        os.makedirs(osp.dirname(osp.abspath(save_file)), exist_ok=True)
        fh = logging.FileHandler(save_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class Section(object):
    """
    Examples
    --------
    >>> with Section('Loading Data'):
    >>>     num_samples = load_data()
    >>>     print(f'=> {num_samples} samples loaded')

    will print out something like
    => Loading data ...
    => 42 samples loaded
    => Done!
    """

    def __init__(self, description, newline=True, logger=None, timing="auto"):
        super(Section, self).__init__()
        self.description = description
        self.newline = newline
        self.logger = logger
        self.timing = timing
        self.t0 = None
        self.t1 = None

    def __enter__(self):
        self.t0 = tic()
        self.print_message("=> " + str(self.description) + " ...")

    def __exit__(self, type, value, traceback):
        self.t1 = tic()
        msg = "=> Done"
        if self.timing != "none":
            t, unit = self._get_time_and_unit(self.t1 - self.t0)
            msg += f" in {t:.3f} {unit}"
        self.print_message(msg)
        if self.newline:
            self.print_message()

    def print_message(self, message=""):
        if self.logger is None:
            try:
                print(message, flush=True)
            except TypeError:
                print(message)
                sys.stdout.flush()
        else:
            self.logger.info(message)

    def _get_time_and_unit(self, time):
        if self.timing == "auto":
            return self._auto_determine_time_and_unit(time)
        elif self.timing == "us":
            return time * 1000000, "us"
        elif self.timing == "ms":
            return time * 1000, "ms"
        elif self.timing == "s":
            return time, "s"
        elif self.timing == "m":
            return time / 60, "m"
        elif self.timing == "h":
            return time / 3600, "h"
        else:
            raise ValueError(f"Unknown timing mode: {self.timing}")

    def _auto_determine_time_and_unit(self, time):
        if time < 0.001:
            return time * 1000000, "us"
        elif time < 1.0:
            return time * 1000, "ms"
        elif time < 60:
            return time, "s"
        elif time < 3600:
            return time / 60, "m"
        else:
            return time / 3600, "h"


class OrderedSet(object):
    """ A set that remembers insertion order """

    def __init__(self, iterable=None):
        self.__dict = OrderedDict()
        for value in iterable or []:
            self.add(value)

    def add(self, value):
        if value not in self.__dict:
            self.__dict[value] = None

    def remove(self, value):
        if value in self.__dict:
            del self.__dict[value]
        else:
            raise KeyError(value)

    def pop(self, last=True):
        """ Pop the last or first element

        Args:
            last (bool): If True, pop the last element (most recently inserted).
                Otherwise pop the first element (oldest).
        """
        return self.__dict.popitem(last=last)[0]

    def __iter__(self):
        return self.__dict.__iter__()

    def __len__(self):
        return self.__dict.__len__()

    def __contains__(self, value):
        return value in self.__dict

    def intersection(self, other):
        new = OrderedSet()
        for value in self:
            if value in other:
                new.add(value)
        return new

    def __and__(self, other):
        return self.intersection(other)

    def union(self, other):
        new = OrderedSet()
        for value in itertools.chain(self, other):
            new.add(value)
        return new

    def __or__(self, other):
        return self.union(other)

    def __ge__(self, other):
        return set(self.__dict).__ge__(set(other.__dict))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and all(map(operator.eq, self, other))
        return set.__eq__(set(self), other)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f"{self.__class__.__name__}([{', '.join(repr(x) for x in self)}])"

    def __str__(self):
        return self.__repr__()
