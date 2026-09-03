from __future__ import annotations

import argparse
import getpass
import json
import re
from dataclasses import asdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from lawevo.evolve.nvidia_nim import (
    DEFAULT_NVIDIA_BASE_URL,
    DEFAULT_NVIDIA_MODEL,
    NVIDIAChatClient,
    env_setting,
    load_env_file,
    resolve_endpoint,
)
from lawevo.pid import (
    PIDBounds,
    PIDGains,
    evaluate_gains,
    generate_scenarios,
    grid_tune,
    simulate_pid,
)

BASELINES = {
    "P": PIDGains(1.2, 0.0, 0.0, 2.5, 0.0, 0.0),
    "PI": PIDGains(1.0, 0.06, 0.0, 2.2, 0.04, 0.0),
    "PD": PIDGains(1.3, 0.0, 0.12, 2.8, 0.0, 0.18),
    "PID (manual)": PIDGains(1.1, 0.05, 0.10, 2.5, 0.03, 0.15),
}


def extract_candidates(response: str, bounds: PIDBounds) -> list[PIDGains]:
    fence = re.sub(r"```(?:json)?|```", "", response, flags=re.IGNORECASE).strip()
    starts = [index for index in (fence.find("["), fence.find("{")) if index >= 0]
    if not starts:
        return []
    start = min(starts)
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(fence[start:])
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("candidates", [payload])
    if not isinstance(payload, list):
        return []
    output: list[PIDGains] = []
    fields = tuple(PIDGains.__dataclass_fields__)
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            gains = PIDGains(*(float(item[field]) for field in fields))
        except (KeyError, TypeError, ValueError):
            continue
        if bounds.contains(gains) and gains not in output:
            output.append(gains)
    return output


def proposal_prompt(
    generation: int,
    parents: list[tuple[PIDGains, dict[str, float]]],
    count: int,
    bounds: PIDBounds,
) -> str:
    parent_payload = [{"gains": gains.to_dict(), "metrics": metrics} for gains, metrics in parents]
    return f"""Generation {generation}: propose {count} new gain vectors for a unicycle PID controller.

The controller uses one PID for goal distance (linear velocity) and one PID for wrapped
heading error (angular velocity). Controls are clipped to v in [-1, 1.5] and omega in
[-2, 2]. Integral terms use anti-windup. Fitness rewards success and penalizes final
distance, settling time, energy, jerk, and heading error. Higher score is better.

Bounds:
{json.dumps(asdict(bounds), indent=2)}

Current elite candidates measured by the simulator:
{json.dumps(parent_payload, indent=2)}

Explore locally around strong elites, but include at least two diverse candidates to avoid
premature convergence. Respect every bound. Return ONLY a JSON array of exactly {count}
objects. Every object must contain exactly these numeric keys:
kp_distance, ki_distance, kd_distance, kp_heading, ki_heading, kd_heading.
"""


def random_population(count: int, bounds: PIDBounds, seed: int) -> list[PIDGains]:
    rng = np.random.default_rng(seed)
    ranges = tuple(asdict(bounds).values())
    return [PIDGains(*(rng.uniform(lower, upper) for lower, upper in ranges)) for _ in range(count)]


def per_scenario_values(gains: PIDGains, scenarios) -> dict[str, np.ndarray]:
    trajectories = [simulate_pid(gains, scenario) for scenario in scenarios]
    success = np.asarray([item.success for item in trajectories], dtype=float)
    final_distance = np.asarray([item.final_distance for item in trajectories])
    settling = np.asarray([item.settling_time for item in trajectories])
    energy = np.asarray([item.energy for item in trajectories])
    jerk = np.asarray([item.jerk for item in trajectories])
    heading = np.asarray([item.mean_heading_error for item in trajectories])
    score = (
        120.0 * success
        - 10.0 * final_distance
        - 1.2 * settling
        - 0.08 * energy
        - 0.0003 * jerk
        - 2.0 * heading
    )
    return {
        "score": score,
        "success": 100 * success,
        "final_distance": final_distance,
        "settling": settling,
        "energy": energy,
    }


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def plot_comparison(results, png_path: Path, pdf_path: Path) -> None:
    configure_plot_style()
    metrics = (
        ("score", "Objective Score ↑", "(a)"),
        ("final_distance", "Final Distance ↓", "(b)"),
        ("settling", "Settling Time (s) ↓", "(c)"),
        ("energy", "Control Energy ↓", "(d)"),
    )
    names = list(results)
    display_names = {
        "P": "P",
        "PI": "PI",
        "PD": "PD",
        "PID (manual)": "Manual PID",
        "PID (grid-tuned)": "Grid PID",
        "PID (NIM GPT-OSS)": "NIM PID",
    }
    colors = ["#999999", "#BDBDBD", "#7F7F7F", "#595959", "#56B4E9", "#0072B2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
    for axis, (metric, title, panel) in zip(axes.ravel(), metrics):
        arrays = [results[name][metric] for name in names]
        means = [float(np.mean(array)) for array in arrays]
        sems = [float(np.std(array, ddof=1) / np.sqrt(len(array))) for array in arrays]
        positions = np.arange(len(names))
        axis.bar(
            positions,
            means,
            yerr=sems,
            color=colors[: len(names)],
            edgecolor="white",
            linewidth=0.7,
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        axis.set_title(f"{panel} {title}", loc="left", fontweight="bold")
        axis.set_xticks(
            positions,
            [display_names[name] for name in names],
            rotation=18,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
    fig.suptitle("PID Controller Comparison on Held-out Scenarios", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Bars show mean ± SEM across 30 held-out scenarios.",
        ha="center",
        color="#444444",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.14, hspace=0.34, wspace=0.22)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_evolution(history: list[dict[str, float]], path: Path) -> None:
    configure_plot_style()
    generations = [item["generation"] for item in history]
    best = [item["best_score"] for item in history]
    mean = [item["population_mean"] for item in history]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(generations, best, marker="o", linewidth=2.2, color="#0072B2", label="Best-so-far")
    axis.plot(
        generations, mean, marker="s", linewidth=1.8, color="#009E73", label="Population mean"
    )
    axis.set_xlabel("Generation")
    axis.set_ylabel("Training Objective Score")
    axis.set_title("NVIDIA NIM GPT-OSS PID Evolution", fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/pid_nim"))
    parser.add_argument(
        "--model",
        default=env_setting("OPENAI_MODEL", "NVIDIA_MODEL", default=DEFAULT_NVIDIA_MODEL),
    )
    parser.add_argument(
        "--base-url",
        default=resolve_endpoint(
            env_setting("OPENAI_BASE_URL", "NVIDIA_BASE_URL", default=DEFAULT_NVIDIA_BASE_URL)
        ),
    )
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--proposals", type=int, default=10)
    parser.add_argument("--train-scenarios", type=int, default=16)
    parser.add_argument("--test-scenarios", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    api_key = env_setting("OPENAI_API_KEY", "NVIDIA_API_KEY") or getpass.getpass("API key: ")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY (or NVIDIA_API_KEY) in .env")
    client = NVIDIAChatClient(api_key=api_key, model=args.model, endpoint=args.base_url)
    bounds = PIDBounds()
    train = generate_scenarios(args.train_scenarios, seed=20260826)
    test = generate_scenarios(args.test_scenarios, seed=20260917)

    grid_gains, grid_metrics = grid_tune(train)
    population = list(BASELINES.values()) + [grid_gains] + random_population(7, bounds, seed=9157)
    archive: dict[PIDGains, dict[str, float]] = {}
    history: list[dict[str, float]] = []
    raw_responses: list[dict[str, object]] = []

    for generation in range(args.generations + 1):
        for gains in population:
            if gains not in archive:
                archive[gains] = evaluate_gains(gains, train).to_dict()
        ranked = sorted(archive.items(), key=lambda item: item[1]["score"], reverse=True)
        population_scores = [archive[gains]["score"] for gains in population]
        history.append(
            {
                "generation": generation,
                "best_score": ranked[0][1]["score"],
                "population_mean": float(np.mean(population_scores)),
            }
        )
        print(
            f"generation={generation} best={ranked[0][1]['score']:.4f} "
            f"mean={np.mean(population_scores):.4f} gains={ranked[0][0].to_dict()}",
            flush=True,
        )
        if generation == args.generations:
            break
        parents = ranked[:5]
        prompt = proposal_prompt(generation + 1, parents, args.proposals, bounds)
        response = client.complete(
            "You are an expert in nonlinear control and robust PID tuning. Output strict JSON only.",
            prompt,
            temperature=0.75,
            reasoning_effort="medium",
        )
        proposals = extract_candidates(response, bounds)
        raw_responses.append(
            {"generation": generation + 1, "response": response, "valid": len(proposals)}
        )
        if len(proposals) < args.proposals:
            proposals.extend(
                random_population(args.proposals - len(proposals), bounds, 1000 + generation)
            )
        population = [item[0] for item in parents[:2]] + proposals[: args.proposals]

    best_gains, best_train_metrics = max(archive.items(), key=lambda item: item[1]["score"])
    methods = dict(BASELINES)
    methods["PID (grid-tuned)"] = grid_gains
    methods["PID (NIM GPT-OSS)"] = best_gains
    test_summary = {name: evaluate_gains(gains, test).to_dict() for name, gains in methods.items()}
    scenario_results = {name: per_scenario_values(gains, test) for name, gains in methods.items()}

    payload = {
        "model": args.model,
        "train_scenarios": len(train),
        "test_scenarios": len(test),
        "objective": "120*success - 10*final_distance - 1.2*settling_time - 0.08*energy - 0.0003*jerk - 2*heading_error",
        "baselines": {name: gains.to_dict() for name, gains in methods.items()},
        "grid_train_metrics": grid_metrics.to_dict(),
        "nim_best_train_metrics": best_train_metrics,
        "test_metrics": test_summary,
        "evolution_history": history,
        "api_calls": len(raw_responses),
        "valid_proposals_per_call": [item["valid"] for item in raw_responses],
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output / "nim_responses.json").write_text(
        json.dumps(raw_responses, indent=2), encoding="utf-8"
    )
    plot_comparison(
        scenario_results,
        args.output / "pid_baseline_comparison.png",
        args.output / "pid_baseline_comparison.pdf",
    )
    plot_evolution(history, args.output / "nim_pid_evolution.png")
    print(json.dumps({"best_gains": best_gains.to_dict(), "test_metrics": test_summary}, indent=2))


if __name__ == "__main__":
    main()
