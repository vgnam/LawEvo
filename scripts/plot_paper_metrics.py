"""Create a publication-ready 2x2 comparison figure from training logs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

RECENT_STATS = re.compile(r"Recent Stats\s*\|\s*t_env:\s*(\d+)")
STAT = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*)\s*:\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)%?"
)
SEED = re.compile(r"seed_(\d+)")
METRICS = (
    ("return_mean", "Train Return", "Return", "(a)"),
    ("test_return_mean", "Test Return", "Return", "(b)"),
    ("coverage_rate_mean", "Train Coverage", "Coverage (%)", "(c)"),
    ("test_coverage_rate_mean", "Test Coverage", "Coverage (%)", "(d)"),
)
ALGORITHMS = ("Belief-RAC", "IPPO", "RAC")
COLORS = {"Belief-RAC": "#0072B2", "IPPO": "#D55E00", "RAC": "#009E73"}


@dataclass(frozen=True)
class Run:
    algorithm: str
    seed: int
    path: Path
    series: dict[str, tuple[np.ndarray, np.ndarray]]


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

    matches = list(RECENT_STATS.finditer(text))
    collected: dict[str, list[tuple[int, float]]] = {metric[0]: [] for metric in METRICS}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        stats = dict(STAT.findall(text[match.end() : end]))
        step = int(match.group(1))
        for metric, value in stats.items():
            if metric in collected:
                collected[metric].append((step, float(value)))

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric, points in collected.items():
        if not points:
            raise ValueError(f"No {metric} values found in {path.name}")
        ordered = sorted(dict(points).items())
        series[metric] = (
            np.asarray([point[0] for point in ordered], dtype=float),
            np.asarray([point[1] for point in ordered], dtype=float),
        )
    return Run(algorithm, int(seed_match.group(1)), path, series)


def common_grid(runs: list[Run], points: int) -> np.ndarray:
    starts = [run.series[metric][0][0] for run in runs for metric, *_ in METRICS]
    ends = [run.series[metric][0][-1] for run in runs for metric, *_ in METRICS]
    start, end = max(starts), min(ends)
    if end <= start:
        raise ValueError("Runs do not share an overlapping timestep range")
    return np.linspace(start, end, points)


def aggregate(runs: list[Run], grid: np.ndarray) -> dict[tuple[str, str, str], np.ndarray]:
    output: dict[tuple[str, str, str], np.ndarray] = {}
    for metric, *_ in METRICS:
        for algorithm in ALGORITHMS:
            selected = sorted(
                (run for run in runs if run.algorithm == algorithm), key=lambda run: run.seed
            )
            if len(selected) != 3:
                raise ValueError(f"Expected 3 seeds for {algorithm}, found {len(selected)}")
            values = np.vstack([np.interp(grid, *run.series[metric]) for run in selected])
            output[(metric, "mean", algorithm)] = values.mean(axis=0)
            output[(metric, "std", algorithm)] = values.std(axis=0, ddof=1)
            output[(metric, "variance", algorithm)] = values.var(axis=0, ddof=1)
    return output


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figure(
    png_path: Path,
    pdf_path: Path,
    grid: np.ndarray,
    data: dict[tuple[str, str, str], np.ndarray],
) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    axes = axes.ravel()

    legend_handles = []
    for axis, (metric, title, ylabel, panel) in zip(axes, METRICS):
        for algorithm in ALGORITHMS:
            mean = data[(metric, "mean", algorithm)]
            std = data[(metric, "std", algorithm)]
            (line,) = axis.plot(
                grid,
                mean,
                color=COLORS[algorithm],
                linewidth=2.15,
                label=algorithm,
                zorder=3,
            )
            axis.fill_between(
                grid,
                mean - std,
                mean + std,
                color=COLORS[algorithm],
                alpha=0.16,
                linewidth=0,
                zorder=2,
            )
            if len(legend_handles) < len(ALGORITHMS):
                legend_handles.append(line)
        axis.set_title(f"{panel} {title}", loc="left", fontweight="bold", pad=7)
        axis.set_ylabel(ylabel)
        axis.grid(True, which="major", color="#B8B8B8", alpha=0.35, linewidth=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0)
        if "Coverage" in title:
            lower = min(
                np.min(data[(metric, "mean", algorithm)] - data[(metric, "std", algorithm)])
                for algorithm in ALGORITHMS
            )
            upper = max(
                np.max(data[(metric, "mean", algorithm)] + data[(metric, "std", algorithm)])
                for algorithm in ALGORITHMS
            )
            padding = 0.08 * (upper - lower)
            axis.set_ylim(max(0, lower - padding), min(100, upper + padding))

    for axis in axes[2:]:
        axis.set_xlabel("Environment Steps")
    for axis in axes:
        axis.ticklabel_format(axis="x", style="sci", scilimits=(5, 5), useMathText=True)

    fig.legend(
        legend_handles,
        ALGORITHMS,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=3.2,
        columnspacing=2.5,
    )
    fig.suptitle(
        "Performance Comparison on MATE",
        fontsize=15,
        fontweight="bold",
        y=0.945,
    )
    fig.text(
        0.5,
        0.012,
        "Solid lines: mean across 3 seeds; shaded regions: ±1 standard deviation.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.87, hspace=0.30, wspace=0.20)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_csv(
    path: Path,
    grid: np.ndarray,
    data: dict[tuple[str, str, str], np.ndarray],
) -> None:
    columns: list[tuple[str, str, str]] = []
    for metric, *_ in METRICS:
        for algorithm in ALGORITHMS:
            for statistic in ("mean", "std", "variance"):
                columns.append((metric, statistic, algorithm))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t_env",
                *(f"{algorithm}_{metric}_{statistic}" for metric, statistic, algorithm in columns),
            ]
        )
        for index, step in enumerate(grid):
            writer.writerow([round(step), *(float(data[column][index]) for column in columns)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--date", default=datetime.now().astimezone().date().isoformat())
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--grid-points", type=int, default=240)
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date)
    log_paths = sorted(
        path
        for path in args.downloads.rglob("*.log")
        if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone().date()
        == target_date
    )
    runs = [run for path in log_paths if (run := parse_run(path)) is not None]
    counts = {
        algorithm: sum(run.algorithm == algorithm for run in runs) for algorithm in ALGORITHMS
    }
    if any(count != 3 for count in counts.values()):
        raise ValueError(f"Expected exactly 3 seeds per algorithm, found {counts}")

    grid = common_grid(runs, args.grid_points)
    data = aggregate(runs, grid)
    for path in (args.png, args.pdf, args.csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    plot_figure(args.png, args.pdf, grid, data)
    save_csv(args.csv, grid, data)
    print(f"Runs: {counts}")
    print(f"Common timestep range: {int(grid[0])}-{int(grid[-1])}")
    print(f"PNG: {args.png}")
    print(f"PDF: {args.pdf}")
    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
