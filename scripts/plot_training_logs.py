"""Aggregate today's Belief-RAC, IPPO, and RAC logs and plot test return."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RECENT_STATS = re.compile(r"Recent Stats\s*\|\s*t_env:\s*(\d+)")
TEST_RETURN = re.compile(r"test_return_mean:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
SEED = re.compile(r"seed_(\d+)")


@dataclass(frozen=True)
class Run:
    algorithm: str
    seed: int
    path: Path
    steps: np.ndarray
    values: np.ndarray


def classify(text: str) -> str | None:
    lowered = text.lower()
    if "belief_dual_iql_ree" in lowered:
        return "Belief-RAC"
    if "ippo_seed_" in lowered or "/ippo/" in lowered:
        return "IPPO"
    if "dual_iql_ree" in lowered:
        return "RAC"
    return None


def parse_run(path: Path) -> Run | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    algorithm = classify(text)
    if algorithm is None:
        return None
    seed_match = SEED.search(text)
    if seed_match is None:
        raise ValueError(f"Cannot determine seed from {path.name}")

    stats_matches = list(RECENT_STATS.finditer(text))
    points: list[tuple[int, float]] = []
    for index, match in enumerate(stats_matches):
        end = stats_matches[index + 1].start() if index + 1 < len(stats_matches) else len(text)
        block = text[match.end() : end]
        value_match = TEST_RETURN.search(block)
        if value_match:
            points.append((int(match.group(1)), float(value_match.group(1))))
    if not points:
        raise ValueError(f"No t_env/test_return_mean pairs found in {path.name}")

    # Keep the last value if a logger emitted the same timestep more than once.
    unique = dict(points)
    ordered = sorted(unique.items())
    return Run(
        algorithm,
        int(seed_match.group(1)),
        path,
        np.asarray([item[0] for item in ordered], dtype=float),
        np.asarray([item[1] for item in ordered], dtype=float),
    )


def aggregate(runs: list[Run], grid_points: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    common_start = max(float(run.steps[0]) for run in runs)
    common_end = min(float(run.steps[-1]) for run in runs)
    if common_end <= common_start:
        raise ValueError("Runs have no overlapping timestep interval")
    grid = np.linspace(common_start, common_end, grid_points)
    result: dict[str, np.ndarray] = {}
    for algorithm in ("Belief-RAC", "IPPO", "RAC"):
        selected = [run for run in runs if run.algorithm == algorithm]
        if len(selected) < 2:
            raise ValueError(f"Expected at least two seeds for {algorithm}, found {len(selected)}")
        matrix = np.vstack([np.interp(grid, run.steps, run.values) for run in selected])
        result[f"{algorithm}_mean"] = matrix.mean(axis=0)
        result[f"{algorithm}_std"] = matrix.std(axis=0, ddof=1)
        result[f"{algorithm}_variance"] = matrix.var(axis=0, ddof=1)
        result[f"{algorithm}_n"] = np.full(grid.shape, len(selected))
    return grid, result


def save_csv(path: Path, grid: np.ndarray, data: dict[str, np.ndarray]) -> None:
    columns = ["t_env", *data]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, step in enumerate(grid):
            writer.writerow([int(round(step)), *(float(data[name][index]) for name in data)])


def save_plot(path: Path, grid: np.ndarray, data: dict[str, np.ndarray]) -> None:
    colors = {"Belief-RAC": "#0072B2", "IPPO": "#D55E00", "RAC": "#009E73"}
    fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
    for algorithm in ("Belief-RAC", "IPPO", "RAC"):
        mean = data[f"{algorithm}_mean"]
        std = data[f"{algorithm}_std"]
        ax.plot(grid, mean, label=algorithm, color=colors[algorithm], linewidth=2.4)
        ax.fill_between(
            grid,
            mean - std,
            mean + std,
            color=colors[algorithm],
            alpha=0.18,
            linewidth=0,
        )
    ax.set_title("Test Return: Mean ± 1 Standard Deviation (3 Seeds)", fontsize=14, pad=12)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Test Return")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.legend(frameon=True, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--date", default=datetime.now().date().isoformat())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--grid-points", type=int, default=200)
    args = parser.parse_args()

    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    candidates = sorted(
        path
        for path in args.downloads.rglob("*.log")
        if datetime.fromtimestamp(path.stat().st_mtime).date() == target_date
    )
    runs = [run for path in candidates if (run := parse_run(path)) is not None]
    counts = {name: sum(run.algorithm == name for run in runs) for name in ("Belief-RAC", "IPPO", "RAC")}
    if any(count != 3 for count in counts.values()):
        raise ValueError(f"Expected exactly three seeds per algorithm, found {counts}")

    grid, data = aggregate(runs, args.grid_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    save_plot(args.output, grid, data)
    save_csv(args.csv, grid, data)
    print(f"Runs: {counts}")
    for run in sorted(runs, key=lambda item: (item.algorithm, item.seed)):
        print(
            f"{run.algorithm:10s} seed={run.seed} points={len(run.steps):3d} "
            f"range={int(run.steps[0])}-{int(run.steps[-1])} file={run.path.name}"
        )
    print(f"Common range: {int(grid[0])}-{int(grid[-1])}")
    print(f"Plot: {args.output}")
    print(f"CSV:  {args.csv}")


if __name__ == "__main__":
    main()
