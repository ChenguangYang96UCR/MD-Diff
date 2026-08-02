#!/usr/bin/env python3
"""Evaluate a trained MG-Diff checkpoint under test-time data corruptions.

The checkpoint is never retrained. Corruptions are applied only to model inputs;
future ground truth and its evaluation mask remain unchanged.
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.logger import Logger


class RobustnessTestOptions(TestOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument(
            "--corruption",
            default="none",
            choices=["none", "missing_events", "timestamp_rounding", "geocoding_error"],
        )
        parser.add_argument(
            "--corruption_rate",
            type=float,
            default=0.1,
            help="Missing-entry rate or fraction of nodes with corrupted locations.",
        )
        parser.add_argument(
            "--round_steps",
            type=int,
            default=3,
            help="Number of consecutive observations merged into one time bucket.",
        )
        parser.add_argument("--corruption_seed", type=int, default=42)
        parser.add_argument(
            "--robustness_csv",
            default="logs/data_robustness.csv",
            help="CSV file to which the final metrics are appended.",
        )
        return parser


def _validate_options(opt):
    if not 0.0 <= opt.corruption_rate <= 1.0:
        raise ValueError("--corruption_rate must be in [0, 1].")
    if opt.round_steps < 1:
        raise ValueError("--round_steps must be at least 1.")


def _history_length(data, opt):
    # morsediffusionfore_model uses the first half as historical context.
    return data["pred"].shape[2] // 2


def add_missing_events(data, opt, batch_index):
    """Remove observed historical events and replace them by normalized zero.

    `missing_mask == 1` denotes a missing value in this repository. Future
    observations are deliberately untouched so all settings share the same GT.
    """
    pred = data["pred"].clone()
    missing = data["missing_mask"].clone()
    t_his = _history_length(data, opt)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(opt.corruption_seed + batch_index)

    history_missing = missing[:, :, :t_his]
    originally_observed = history_missing == 0
    selected = (
        torch.rand(history_missing.shape, generator=generator)
        < opt.corruption_rate
    ) & originally_observed

    pred_history = pred[:, :, :t_his]
    pred_history[selected] = 0.0  # dataset is standardized: zero is its mean
    history_missing[selected] = 1.0

    pred[:, :, :t_his] = pred_history
    missing[:, :, :t_his] = history_missing
    data["pred"] = pred
    data["missing_mask"] = missing
    return int(selected.sum().item())


def _bucket_average_in_place(values, missing, width):
    """Coarsen regularly sampled history using observed-value bucket means."""
    length = values.shape[2]
    changed = 0
    for start in range(0, length, width):
        end = min(start + width, length)
        block = values[:, :, start:end]
        block_missing = missing[:, :, start:end]
        observed = 1.0 - block_missing
        count = observed.sum(dim=2, keepdim=True)
        mean = (block * observed).sum(dim=2, keepdim=True) / count.clamp_min(1.0)
        replacement = mean.expand_as(block)
        valid_bucket = (count > 0).expand_as(block)
        values[:, :, start:end] = torch.where(
            valid_bucket, replacement, block
        )
        changed += int(valid_bucket.sum().item())
    return changed


def round_timestamps(data, opt):
    """Simulate timestamp rounding by aggregation into coarser regular bins.

    The model ignores the raw `time` tensor, so merely rounding it would not
    perturb its input. We therefore apply the standard consequence of rounding:
    observations mapped to the same timestamp are mean-aggregated.
    """
    pred = data["pred"].clone()
    missing = data["missing_mask"]
    t_his = _history_length(data, opt)
    changed = _bucket_average_in_place(
        pred[:, :, :t_his], missing[:, :, :t_his], opt.round_steps
    )
    data["pred"] = pred

    # PEMS has time-of-day covariates; coarsen historical covariates consistently.
    if "feat" in data:
        feat = data["feat"].clone()
        dummy_missing = torch.zeros_like(feat[:, :, :t_his])
        _bucket_average_in_place(
            feat[:, :, :t_his], dummy_missing, opt.round_steps
        )
        data["feat"] = feat
    return changed


def add_geocoding_errors(data, opt):
    """Assign a fraction of nodes incorrect spatial identities in adjacency.

    The released datasets contain adjacency matrices rather than raw latitude
    and longitude. Permuting selected adjacency rows/columns simulates stations
    whose signals are associated with incorrect geocoded locations while
    preserving symmetry and global graph statistics.
    """
    adj = data["adj"].clone()
    num_nodes = adj.shape[1]
    num_bad = int(round(opt.corruption_rate * num_nodes))
    if opt.corruption_rate > 0 and num_bad < 2:
        num_bad = 2
    num_bad = min(num_bad, num_nodes)
    if num_bad < 2:
        return 0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(opt.corruption_seed)
    selected = torch.randperm(num_nodes, generator=generator)[:num_bad]
    shuffled = selected[torch.randperm(num_bad, generator=generator)]

    permutation = torch.arange(num_nodes)
    permutation[selected] = shuffled
    # Each batch contains the same graph, but retain the batch dimension.
    data["adj"] = adj[:, permutation][:, :, permutation]
    return num_bad


def corrupt_batch(data, opt, batch_index):
    if opt.corruption == "none":
        return 0
    if opt.corruption == "missing_events":
        return add_missing_events(data, opt, batch_index)
    if opt.corruption == "timestamp_rounding":
        return round_timestamps(data, opt)
    if opt.corruption == "geocoding_error":
        return add_geocoding_errors(data, opt)
    raise ValueError(f"Unsupported corruption: {opt.corruption}")


def append_metrics(path, opt, metrics, seconds, affected):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    row = {
        "dataset": opt.dataset_mode,
        "file_time": opt.file_time,
        "epoch": opt.epoch,
        "corruption": opt.corruption,
        "corruption_rate": opt.corruption_rate,
        "round_steps": opt.round_steps,
        "corruption_seed": opt.corruption_seed,
        "affected": affected,
        "seconds": seconds,
        **{key: float(value) for key, value in metrics.items()},
    }
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    opt, model_config = RobustnessTestOptions().parse()
    _validate_options(opt)
    dataset = create_dataset(opt)
    print(f"The number of testing batches = {len(dataset)}")

    model = create_model(opt, model_config)
    model.setup(opt)
    visualizer = Logger(opt)
    model.eval()

    affected = 0
    start = time.time()
    for batch_index, data in tqdm(enumerate(dataset), total=len(dataset)):
        affected += corrupt_batch(data, opt, batch_index)
        model.set_input(data)
        model.test()
        model.cache_results()

    seconds = time.time() - start
    model.compute_visuals()
    model.compute_metrics()
    metrics = model.get_current_metrics()
    visualizer.print_current_metrics(-1, 0, metrics, seconds)
    append_metrics(opt.robustness_csv, opt, metrics, seconds, affected)
    print(f"[Robustness] affected={affected}, saved={opt.robustness_csv}")


if __name__ == "__main__":
    main()
