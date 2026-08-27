from __future__ import annotations

import argparse
import getpass
import json
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from lawevo.evolve.nvidia_nim import (
    DEFAULT_NVIDIA_BASE_URL,
    DEFAULT_NVIDIA_MODEL,
    NVIDIAChatClient,
    load_env_file,
)
from lawevo.pid import (
    DISTANCE_TERMS,
    HEADING_TERMS,
    VELOCITY_GATES,
    ControllerStructure,
    generate_scenarios,
    simulate_structure,
    tune_cem,
)

FIXED_PID = ControllerStructure(
    "Fixed PID",
    ("error", "integral", "derivative"),
    ("error", "integral", "derivative"),
    "positive_cosine",
)
INITIAL_STRUCTURES = (
    ControllerStructure("P", ("error",), ("error",), "positive_cosine"),
    ControllerStructure(
        "PI", ("error", "integral"), ("error", "integral"), "positive_cosine"
    ),
    ControllerStructure(
        "PD", ("error", "derivative"), ("error", "derivative"), "positive_cosine"
    ),
    FIXED_PID,
)


def structure_key(structure: ControllerStructure) -> tuple[object, ...]:
    return structure.distance_terms, structure.heading_terms, structure.velocity_gate


def extract_structures(response: str) -> list[ControllerStructure]:
    text = re.sub(r"```(?:json)?|```", "", response, flags=re.IGNORECASE).strip()
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("structures", [payload])
    if not isinstance(payload, list):
        return []
    output: list[ControllerStructure] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        try:
            structure = ControllerStructure(
                str(item.get("name", f"proposal_{index}"))[:60],
                tuple(item["distance_terms"]),
                tuple(item["heading_terms"]),
                str(item.get("velocity_gate", "positive_cosine")),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if structure_key(structure) not in {structure_key(existing) for existing in output}:
            output.append(structure)
    return output


def evolution_prompt(generation: int, elites: list[dict[str, object]], count: int) -> str:
    return f"""Generation {generation}: evolve {count} interpretable controller structures.

The controller has two outputs. Each output is a weighted sum of selected basis terms;
all weights K are tuned AFTER structure generation by Cross-Entropy Method with an equal
simulation budget. You select terms only, never numeric gains.

Allowed distance terms: {json.dumps(DISTANCE_TERMS)}
Allowed heading terms: {json.dumps(HEADING_TERMS)}
Allowed velocity gates: {json.dumps(VELOCITY_GATES)}

Semantics:
- error, integral, derivative: conventional PID signals.
- tanh_error and tanh_derivative: bounded nonlinear feedback/damping.
- sqrt_error: sign(e)*sqrt(abs(e)); quadratic_error: e*abs(e).
- heading_coupling in v: distance_error*max(0,cos(heading_error)).
- distance_coupling in omega: heading_error*min(distance,3).
- positive_cosine gate suppresses forward motion while pointing away from goal.

Fitness rewards success, accuracy, and fast settling; it penalizes energy, jerk, heading
error, and number of terms. Current structures after inner-loop CEM tuning:
{json.dumps(elites, indent=2)}

Propose useful structural mutations and crossovers. Include 2-6 unique terms per channel.
Avoid redundant combinations that express nearly the same signal. Return ONLY a JSON array
of exactly {count} objects with keys name, distance_terms, heading_terms, velocity_gate.
"""


def scenario_arrays(
    structure: ControllerStructure, gains: np.ndarray, scenarios
) -> dict[str, np.ndarray]:
    runs = [simulate_structure(structure, gains, scenario) for scenario in scenarios]
    success = np.asarray([run.success for run in runs], dtype=float)
    distance = np.asarray([run.final_distance for run in runs])
    settling = np.asarray([run.settling_time for run in runs])
    energy = np.asarray([run.energy for run in runs])
    jerk = np.asarray([run.jerk for run in runs])
    heading = np.asarray([run.mean_heading_error for run in runs])
    score = (
        120 * success
        - 10 * distance
        - 1.2 * settling
        - 0.10 * energy
        - 0.003 * jerk
        - 2 * heading
        - 0.08 * structure.parameter_count
    )
    return {
        "score": score,
        "final_distance": distance,
        "settling_time": settling,
        "energy": energy,
        "jerk": jerk,
        "success": success,
    }


def plot_results(results: dict[str, dict[str, np.ndarray]], output: Path) -> None:
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
    panels = (
        ("score", "Objective Score ↑", "(a)"),
        ("final_distance", "Final Distance ↓", "(b)"),
        ("settling_time", "Settling Time (s) ↓", "(c)"),
        ("jerk", "Control Jerk ↓", "(d)"),
    )
    names = list(results)
    colors = ["#8C8C8C", "#56B4E9", "#0072B2", "#009E73"][: len(names)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for axis, (metric, title, panel) in zip(axes.ravel(), panels):
        values = [results[name][metric] for name in names]
        means = [np.mean(value) for value in values]
        sems = [np.std(value, ddof=1) / np.sqrt(len(value)) for value in values]
        x = np.arange(len(names))
        axis.bar(x, means, yerr=sems, color=colors, capsize=3, edgecolor="white")
        axis.set_title(f"{panel} {title}", loc="left", fontweight="bold")
        axis.set_xticks(x, names, rotation=16, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle("Structure Evolution with Inner-loop CEM Tuning", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, "Mean ± SEM over 30 held-out scenarios.", ha="center", color="#444444")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.15, hspace=0.35, wspace=0.22)
    fig.savefig(output / "structure_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "structure_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/pid_structure_nim"))
    parser.add_argument("--model", default=os.environ.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL))
    parser.add_argument(
        "--base-url", default=os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)
    )
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--proposals", type=int, default=5)
    parser.add_argument("--cem-iterations", type=int, default=7)
    parser.add_argument("--cem-population", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("NVIDIA_API_KEY") or getpass.getpass("NVIDIA API key: ")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is required")

    client = NVIDIAChatClient(api_key, model=args.model, endpoint=args.base_url)
    train = generate_scenarios(10, seed=20260827)
    test = generate_scenarios(30, seed=20261001)
    evaluated: dict[tuple[object, ...], dict[str, object]] = {}
    generation_structures = list(INITIAL_STRUCTURES)
    responses: list[dict[str, object]] = []

    for generation in range(args.generations + 1):
        for structure in generation_structures:
            key = structure_key(structure)
            if key in evaluated:
                continue
            gains, metrics, cem_history = tune_cem(
                structure,
                train,
                iterations=args.cem_iterations,
                population_size=args.cem_population,
            )
            evaluated[key] = {
                "structure": structure,
                "gains": gains,
                "metrics": metrics,
                "cem_history": cem_history,
                "generation": generation,
            }
            print(
                f"generation={generation} structure={structure.name!r} "
                f"score={metrics.score:.4f} terms={structure.parameter_count}",
                flush=True,
            )
        ranked = sorted(evaluated.values(), key=lambda item: item["metrics"].score, reverse=True)
        if generation == args.generations:
            break
        elite_payload = [
            {
                "structure": item["structure"].to_dict(),
                "tuned_metrics": item["metrics"].to_dict(),
            }
            for item in ranked[:6]
        ]
        response = client.complete(
            "You are an expert control researcher evolving compact symbolic feedback laws. Return JSON only.",
            evolution_prompt(generation + 1, elite_payload, args.proposals),
            temperature=0.8,
            reasoning_effort="medium",
        )
        proposed = extract_structures(response)
        responses.append(
            {"generation": generation + 1, "valid": len(proposed), "response": response}
        )
        generation_structures = proposed[: args.proposals]
        if not generation_structures:
            raise RuntimeError("NIM returned no valid structures")

    ranked = sorted(evaluated.values(), key=lambda item: item["metrics"].score, reverse=True)
    best = ranked[0]
    fixed = next(
        item
        for item in evaluated.values()
        if structure_key(item["structure"]) == structure_key(FIXED_PID)
    )

    previous_grid = np.array([1.6, 0.1, 0.0, 3.4, 0.04, 0.0])
    previous_nim = np.array([1.95, 0.39, 0.21, 4.8, 0.13, 0.66])
    methods = {
        "Grid PID": (FIXED_PID, previous_grid),
        "Gain-only NIM": (FIXED_PID, previous_nim),
        "PID + CEM": (FIXED_PID, fixed["gains"]),
        "Structure + CEM": (best["structure"], best["gains"]),
    }
    heldout = {
        name: scenario_arrays(structure, gains, test)
        for name, (structure, gains) in methods.items()
    }
    plot_results(heldout, args.output)

    payload = {
        "model": args.model,
        "outer_generations": args.generations,
        "api_calls": len(responses),
        "cem": {"iterations": args.cem_iterations, "population": args.cem_population},
        "train_scenarios": 10,
        "test_scenarios": 30,
        "objective": "120*success - 10*distance - 1.2*settling - 0.10*energy - 0.003*jerk - 2*heading_error - 0.08*term_count",
        "best_structure": best["structure"].to_dict(),
        "best_gains": best["gains"].tolist(),
        "best_formula": best["structure"].formula(best["gains"]),
        "best_train_metrics": best["metrics"].to_dict(),
        "fixed_pid_cem_gains": fixed["gains"].tolist(),
        "fixed_pid_train_metrics": fixed["metrics"].to_dict(),
        "heldout": {
            name: {metric: float(np.mean(values)) for metric, values in result.items()}
            for name, result in heldout.items()
        },
        "valid_structures_per_call": [response["valid"] for response in responses],
        "all_structures": [
            {
                "structure": item["structure"].to_dict(),
                "gains": item["gains"].tolist(),
                "train_metrics": item["metrics"].to_dict(),
                "generation": item["generation"],
            }
            for item in ranked
        ],
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output / "nim_responses.json").write_text(
        json.dumps(responses, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: payload[key] for key in ("best_formula", "best_train_metrics", "heldout")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
