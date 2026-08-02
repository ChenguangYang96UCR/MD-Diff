#!/usr/bin/env python3
"""Run test.py with multiple random seeds and summarize mean ± std."""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


METRIC_PATTERN = re.compile(
    r"MAE:\s*([0-9.eE+-]+)\s+"
    r"RMSE:\s*([0-9.eE+-]+)\s+"
    r"MAPE:\s*([0-9.eE+-]+)\s+"
    r"CRPS:\s*([0-9.eE+-]+)"
)
METRIC_NAMES = ("MAE", "RMSE", "MAPE", "CRPS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[2030, 2031, 2032, 2033, 2034])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--test_script", default="test.py")
    parser.add_argument("--output_dir", default="logs/test_5seeds_GZAir")

    # Everything after "--" is passed directly to test.py.
    parser.add_argument("test_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.test_args and args.test_args[0] == "--":
        args.test_args = args.test_args[1:]
    return args


def run_one_seed(args, seed, output_dir):
    command = [
        args.python,
        args.test_script,
        *args.test_args,
        "--seed",
        str(seed),
    ]
    log_path = output_dir / f"test_seed_{seed}.log"

    print(f"\n{'=' * 72}")
    print(f"Testing seed {seed}")
    print("Command:", " ".join(command))
    print(f"{'=' * 72}")

    metric_matches = []
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            match = METRIC_PATTERN.search(line)
            if match:
                metric_matches.append(tuple(float(x) for x in match.groups()))

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Seed {seed} failed with exit code {return_code}. "
            f"See {log_path}."
        )
    if not metric_matches:
        raise RuntimeError(f"No metrics found for seed {seed}. See {log_path}.")

    # The final metric line is the test result.
    return metric_matches[-1]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        metrics = run_one_seed(args, seed, output_dir)
        results.append((seed, *metrics))

    csv_path = output_dir / "metrics_5seeds.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("seed", *METRIC_NAMES))
        writer.writerows(results)

    values = np.asarray([row[1:] for row in results], dtype=np.float64)
    means = values.mean(axis=0)
    # Sample standard deviation across independent runs.
    stds = values.std(axis=0, ddof=1) if len(results) > 1 else np.zeros(4)

    print("\nIndividual results")
    print("| Seed | MAE | RMSE | MAPE | CRPS |")
    print("|-|-|-|-|-|")
    for row in results:
        print(
            f"| {row[0]} | {row[1]:.5f} | {row[2]:.5f} | "
            f"{row[3]:.5f} | {row[4]:.5f} |"
        )

    print("\nMean ± std")
    print("| Method | MAE | RMSE | MAPE | CRPS |")
    print("|-|-|-|-|-|")
    formatted = [
        f"{mean:.5f} ± {std:.5f}" for mean, std in zip(means, stds)
    ]
    print(f"| Model | {' | '.join(formatted)} |")
    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
