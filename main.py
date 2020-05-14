#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import warnings
from datetime import datetime
from os import path as osp
from time import time as tic

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from voxelcnn.checkpoint import Checkpointer
from voxelcnn.criterions import CrossEntropyLoss
from voxelcnn.datasets import Craft3DDataset
from voxelcnn.evaluators import CCA, MTC, Accuracy
from voxelcnn.models import VoxelCNN
from voxelcnn.summary import Summary
from voxelcnn.utils import Section, collate_batches, setup_logger, to_cuda


def global_setup(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    if not args.cpu_only:
        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Fallback to using CPU only")
            args.cpu_only = True
        else:
            torch.cuda.benchmark = True


def build_data_loaders(args, logger):
    data_loaders = {}
    for subset in ("train", "val", "test"):
        dataset = Craft3DDataset(
            args.data_dir,
            subset,
            max_samples=args.max_samples,
            next_steps=10,
            logger=logger,
        )
        data_loaders[subset] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=subset == "train",
            num_workers=args.num_workers,
            pin_memory=not args.cpu_only,
        )
    return data_loaders


def build_model(args, logger):
    model = VoxelCNN()
    if not args.cpu_only:
        model.cuda()
    logger.info("Model architecture:\n" + str(model))
    return model


def build_criterion(args):
    criterion = CrossEntropyLoss()
    if not args.cpu_only:
        criterion.cuda()
    return criterion


def build_optimizer(args, model):
    no_decay = []
    decay = []
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{"params": no_decay, "weight_decay": 0}, {"params": decay}]
    return optim.SGD(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=True,
    )


def build_scheduler(args, optimizer):
    return optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )


def build_evaluators(args):
    return {
        "acc@1": Accuracy(next_steps=1),
        "acc@5": Accuracy(next_steps=5),
        "acc@10": Accuracy(next_steps=10),
    }


def train(
    args, epoch, data_loader, model, criterion, optimizer, scheduler, evaluators, logger
):
    summary = Summary(logger=logger)
    model.train()
    timestamp = tic()
    for i, (inputs, targets) in enumerate(data_loader):
        times = {"data": tic() - timestamp}
        if not args.cpu_only:
            inputs = to_cuda(inputs)
            targets = to_cuda(targets)
        outputs = model(inputs)
        losses = criterion(outputs, targets)
        with torch.no_grad():
            metrics = {k: float(v(outputs, targets)) for k, v in evaluators.items()}

        optimizer.zero_grad()
        losses["overall_loss"].backward()
        optimizer.step()
        try:
            lr = scheduler.get_last_lr()[0]
        except Exception:
            # For backward compatibility
            lr = scheduler.get_lr()[0]

        times["time"] = tic() - timestamp
        summary.add(times=times, lrs={"lr": lr}, losses=losses, metrics=metrics)
        summary.print_current(
            prefix=f"[{epoch}/{args.num_epochs}][{i + 1}/{len(data_loader)}]"
        )
        timestamp = tic()
    scheduler.step()


@torch.no_grad()
def evaluate(args, epoch, data_loader, model, evaluators, logger):
    summary = Summary(logger=logger)
    model.eval()
    timestamp = tic()
    batch_results = []
    for i, (inputs, targets) in enumerate(data_loader):
        times = {"data": tic() - timestamp}
        if not args.cpu_only:
            inputs = to_cuda(inputs)
            targets = to_cuda(targets)
        outputs = model(inputs)
        batch_results.append(
            {k: v.step(outputs, targets) for k, v in evaluators.items()}
        )

        times["time"] = tic() - timestamp
        summary.add(times=times)
        summary.print_current(
            prefix=f"[{epoch}/{args.num_epochs}][{i + 1}/{len(data_loader)}]"
        )
        timestamp = tic()
    results = collate_batches(batch_results)
    metrics = {k: float(v.stop(results[k])) for k, v in evaluators.items()}
    return metrics


def main(args):
    # Set log file name based on current date and time
    cur_datetime = datetime.now().strftime("%Y%m%d.%H%M%S")
    log_path = osp.join(args.save_dir, f"log.{cur_datetime}.txt")
    logger = setup_logger(save_file=log_path)
    logger.info(f"Save logs to: {log_path}")

    with Section("Global setup", logger=logger):
        global_setup(args)

    with Section("Building data loaders", logger=logger):
        data_loaders = build_data_loaders(args, logger)

    with Section("Building model", logger=logger):
        model = build_model(args, logger)

    with Section("Building criterions, optimizer, scheduler", logger=logger):
        criterion = build_criterion(args)
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)

    with Section("Building evaluators", logger=logger):
        evaluators = build_evaluators(args)

    checkpointer = Checkpointer(args.save_dir)
    last_epoch = 0
    if args.resume is not None:
        with Section(f"Resuming from model: {args.resume}", logger=logger):
            last_epoch = checkpointer.resume(
                args.resume, model=model, optimizer=optimizer, scheduler=scheduler
            )

    for epoch in range(last_epoch + 1, args.num_epochs + 1):
        with Section(f"Training epoch {epoch}", logger=logger):
            train(
                args,
                epoch,
                data_loaders["train"],
                model,
                criterion,
                optimizer,
                scheduler,
                evaluators,
                logger,
            )
        with Section(f"Validating epoch {epoch}", logger=logger):
            # Evaluate on the validation set by the lightweight accuracy metrics
            metrics = evaluate(
                args, epoch, data_loaders["val"], model, evaluators, logger
            )
            # Use acc@10 as the key metric to select best model
            checkpointer.save(model, optimizer, scheduler, epoch, metrics["acc@10"])
            metrics_str = "  ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
            best_mark = "*" if epoch == checkpointer.best_epoch else ""
            logger.info(f"Finish  epoch: {epoch}  {metrics_str} {best_mark}")

    best_epoch = checkpointer.best_epoch
    with Section(f"Final test with best model from epoch: {best_epoch}", logger=logger):
        # Load the best model and evaluate all the metrics on the test set
        checkpointer.load("best", model=model)
        metrics = evaluate(
            args, best_epoch, data_loaders["test"], model, evaluators, logger
        )

        # Additional evaluation metrics. Takes quite long time to evaluate
        dataset = data_loaders["test"].dataset
        params = {
            "local_size": dataset.local_size,
            "global_size": dataset.global_size,
            "history": dataset.history,
        }
        metrics.update(CCA(**params).evaluate(dataset, model))
        metrics.update(MTC(**params).evaluate(dataset, model))

        metrics_str = "  ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
        logger.info(f"Final test from best epoch: {best_epoch}\n{metrics_str}")


if __name__ == "__main__":
    work_dir = osp.dirname(osp.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Train and evaluate VoxelCNN model on 3D-Craft dataset"
    )
    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default=osp.join(work_dir, "data"),
        help="Path to the data directory",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for preprocessing",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="When debugging, set this option to limit the number of training samples",
    )
    # Optimizer
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    # Scheduler
    parser.add_argument("--step_size", type=int, default=5, help="StepLR step size")
    parser.add_argument("--gamma", type=int, default=0.1, help="StepLR gamma")
    parser.add_argument("--num_epochs", type=int, default=12, help="Total train epochs")
    # Misc
    parser.add_argument(
        "--save_dir",
        type=str,
        default=osp.join(work_dir, "logs"),
        help="Path to a directory to save log file and checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="'latest' | 'best' | '<epoch number>' | '<path to a checkpoint>'. "
        "Default: None, will not resume",
    )
    parser.add_argument("--cpu_only", action="store_true", help="Only using CPU")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    main(parser.parse_args())
