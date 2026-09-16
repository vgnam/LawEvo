"""Reproducible direct and gain-retuned transfer of saved evolved laws."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lawevo.pid import ADAPTERS, CONTROLLED_VARIANT_ADAPTERS, PANDA_GYM_ADAPTERS
from lawevo.pid.gym_benchmark import (
    GymStructure,
    evaluate_gym_structure,
    inverted_pendulum_lqr,
    tune_gym_cem,
)

TARGETS = [
    "InvertedPendulum-v5",
    "LawevoInvertedPendulumPulse-v0",
    "Reacher-v5",
    "LawevoReacherPayload-v0",
    "PandaReachDense-v3",
    "LawevoPandaReachMovingSlow-v0",
    "PandaPushDense-v3",
    "LawevoPandaPushLowFriction-v0",
    "PandaSlideDense-v3",
    "LawevoPandaSlideFrictionShift-v0",
]


def compatibility(source_contract, target_contract, signals):
    """Conservative identity transfer; never invent or rename task signals."""
    if not source_contract.get("complete") or not target_contract.get("complete"):
        return "incomplete signal contract"
    for field in ("action_shape", "action"):
        if source_contract[field] != target_contract[field]:
            return f"incompatible {field}"
    for signal in signals:
        a = source_contract["signals"].get(signal)
        b = target_contract["signals"].get(signal)
        if a is None or b is None:
            return f"unavailable signal: {signal}"
        if a != b:
            return f"signal contract differs: {signal}"
    return None


def discover(root, targets):
    sources = []
    for env in targets:
        for path in sorted((root / env).glob("*/data/summary/results.json"), reverse=True):
            data = json.loads(path.read_text(encoding="utf-8"))
            result = data.get("result", {})
            law = result.get("best_evolved")
            if data.get("search_method") != "llm" or not law or not law.get("test"):
                continue
            sources.append(
                {
                    "environment": env,
                    "file": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "contract": data["protocol"]["signal_contract"],
                    "structure": law["structure"],
                    "gains": law["gains"],
                    "train_metrics": law["train_metrics"],
                }
            )
            break
    return sources


def export(out, report):
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    names = report["targets"]
    sources = [s["environment"] for s in report["sources"]]
    baseline = {b["target"]: b for b in report["baselines"]}
    with (out / "transfer_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "target",
                "mode",
                "status",
                "reason",
                "SR",
                "SG",
                "target_baseline_SR",
                "delta_baseline_pp",
                "effort",
                "slew",
            ]
        )
        for c in report["cells"]:
            m = c.get("metrics", {})
            b = baseline.get(c["target"], {}).get("metrics", {})
            delta = 100 * (m["success_rate"] - b["success_rate"]) if m and b else None
            writer.writerow(
                [
                    c["source"],
                    c["target"],
                    c["mode"],
                    c["status"],
                    c.get("reason"),
                    m.get("success_rate"),
                    m.get("sg"),
                    b.get("success_rate"),
                    delta,
                    m.get("energy"),
                    m.get("jerk"),
                ]
            )
    for mode in ("zero_shot", "retuned"):
        lookup = {(c["source"], c["target"]): c for c in report["cells"] if c["mode"] == mode}
        rows = []
        values = np.full((len(sources), len(names)), np.nan)
        for i, source in enumerate(sources):
            row = [source]
            for j, target in enumerate(names):
                cell = lookup.get((source, target), {})
                if cell.get("status") == "ok":
                    m = cell["metrics"]
                    values[i, j] = 100 * m["success_rate"]
                    sg = "NA" if m["sg"] is None else f"{m['sg']:.3f}"
                    row.append(f"{values[i, j]:.1f} / {sg}")
                else:
                    row.append("N/A" if cell.get("status") == "incompatible" else "pending/error")
            rows.append(row)
        with (out / f"{mode}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source / target (SR % / SG)", *names])
            writer.writerows(rows)
            writer.writerow(
                [
                    "Target classical baseline (training-selected)",
                    *[
                        f"{100 * baseline[t]['metrics']['success_rate']:.1f} / "
                        f"{baseline[t]['metrics']['sg']:.3f}"
                        if t in baseline and baseline[t]["metrics"]["sg"] is not None
                        else "N/A"
                        for t in names
                    ],
                ]
            )
        tex = [
            "% Cells: success rate (%) / success gap. N/A: incompatible contracts.",
            r"\begin{tabular}{l" + "c" * len(names) + "}",
            r"\hline",
            "Source / Target & " + " & ".join(f"T{i + 1}" for i in range(len(names))) + r" \\",
        ]
        tex += [f"S{i + 1} & " + " & ".join(row[1:]) + r" \\" for i, row in enumerate(rows)]
        tex += [r"\hline", r"\end{tabular}"]
        tex += [f"% S{i + 1}: {s}" for i, s in enumerate(sources)]
        tex += [f"% T{i + 1}: {s}" for i, s in enumerate(names)]
        (out / f"{mode}.tex").write_text("\n".join(tex), encoding="utf-8")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 8))
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=100, cmap=cmap)
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(sources)), sources, fontsize=8)
        for i, row in enumerate(rows):
            for j, label in enumerate(row[1:]):
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if values[i, j] > 65 else "black",
                )
        ax.set_title(f"{mode.replace('_', ' ').title()} transfer: SR (%) / SG; pilot")
        fig.colorbar(im, ax=ax, label="Success rate (%)")
        fig.tight_layout()
        fig.savefig(out / f"{mode}.png", dpi=180)
        fig.savefig(out / f"{mode}.pdf")
        plt.close(fig)
    write_summary(out, report)


def write_summary(out, report):
    protocol = report["protocol"]
    good = [c for c in report["cells"] if c["status"] == "ok"]
    pairs = {(c["source"], c["target"]) for c in good}
    lines = [
        "# Cross-task transfer pilot",
        "",
        (
            f"{len(report['sources'])} source laws, {len(report['targets'])} target tasks, "
            f"{len(pairs)} compatible pairs, two transfer modes."
        ),
        "",
        (
            "Each matrix cell is success rate (%) / success gap. Grey N/A cells mean "
            "incompatible or incomplete contracts, not zero success. This is primarily "
            "base-to-variant transfer; no claim of arbitrary cross-domain portability is made."
        ),
        "",
        "## Protocol",
        "",
        (
            "- Source: training-selected best evolved law from the latest completed LLM run "
            "for each available task. Source files and hashes are recorded in results.json."
        ),
        "- Zero-shot: frozen expression and saved source gains.",
        "- Retuned: frozen expression, CEM gains initialized from scratch on target training seeds.",
        (
            f"- Target training seeds: {protocol['train_seeds']}; "
            f"test seeds: {protocol['test_seeds'][0]}--{protocol['test_seeds'][-1]}."
        ),
        (
            f"- CEM: {protocol['iterations']} iterations x {protocol['population']} samples, "
            "plus the initial zero-gain evaluation. The same budget applies to each tunable "
            "classical baseline. Analytic LQR is additionally considered for pendulum tasks."
        ),
        (
            "- Target classical reference is selected on training SR, then SG, then secondary "
            "utility. No test-based controller selection is performed."
        ),
        (
            "- Task semantics come from the current adapters. Even source-task diagonal cells "
            "are rerun; historical test results are not inserted into this matrix."
        ),
        (
            "- One search repetition per source and a small retuning budget: exploratory results, "
            "not a final benchmark or evidence of statistical significance. Source search cost "
            "is not included in the target retuning budget."
        ),
        (
            "- Shared target helpers are retained. Effort is integrated squared command, "
            "and slew is integrated squared command rate, not joules or mechanical jerk."
        ),
        "",
        "## Cross-task results",
        "",
        "| Source | Target | Zero-shot SR % | Retuned SR % | Retuned - zero (pp) | Target baseline SR % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lookup = {(c["source"], c["target"], c["mode"]): c for c in good}
    baselines = {b["target"]: b for b in report["baselines"]}
    for source, target in sorted(pairs):
        if source == target:
            continue
        z = lookup.get((source, target, "zero_shot"))
        r = lookup.get((source, target, "retuned"))
        if not z or not r or target not in baselines:
            continue
        zs, rs = 100 * z["metrics"]["success_rate"], 100 * r["metrics"]["success_rate"]
        bs = 100 * baselines[target]["metrics"]["success_rate"]
        lines.append(f"| {source} | {target} | {zs:.1f} | {rs:.1f} | {rs - zs:+.1f} | {bs:.1f} |")
    lines += [
        "",
        "## Source coverage",
        "",
        "Targets without an available completed source run have columns but no source rows: "
        + ", ".join(
            t for t in report["targets"] if t not in {s["environment"] for s in report["sources"]}
        )
        + ".",
        "",
        "## Artifacts",
        "",
        "- zero_shot.png / .pdf / .csv / .tex: frozen-gain matrix.",
        "- retuned.png / .pdf / .csv / .tex: target-retuned matrix.",
        "- transfer_details.csv: all cells, incompatibility reasons, baseline deltas and costs.",
        "- results.json: source provenance, contracts, gains and per-seed trajectories metrics.",
        "",
        "## Reproduce this pilot",
        "",
        "```powershell",
        (
            "py -m experiments.cross_task_transfer --output results/cross_task_transfer_pilot "
            "--episodes 30 --train-episodes 2 --cem-iterations 1 --cem-population 4"
        ),
        "```",
        "",
        (
            "Default tuning uses 5 x 24 CEM samples and 6 training episodes. Use a new output "
            "directory and fresh final test seeds for a larger follow-up; do not pool these "
            "pilot tests with a redesigned experiment. --resume requires an identical manifest."
        ),
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--train-episodes", type=int, default=6)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument("--cem-population", type=int, default=24)
    parser.add_argument("--test-seed", type=int, default=190000)
    parser.add_argument("--train-seed", type=int, default=180000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.episodes, args.train_episodes, args.cem_iterations) < 1 or args.cem_population < 2:
        parser.error("positive episode/iteration counts and population >= 2 required")
    train = list(range(args.train_seed, args.train_seed + args.train_episodes))
    test = list(range(args.test_seed, args.test_seed + args.episodes))
    if set(train) & set(test):
        parser.error("train/test seeds must be disjoint")
    adapters = {
        a.env_id: a
        for a in {**ADAPTERS, **CONTROLLED_VARIANT_ADAPTERS, **PANDA_GYM_ADAPTERS}.values()
    }
    sources = discover(args.results_root, TARGETS)
    if not sources:
        parser.error("no completed evolved sources found")
    config = {
        "train_seeds": train,
        "test_seeds": test,
        "iterations": args.cem_iterations,
        "population": args.cem_population,
        "selection": "latest completed LLM run; training-selected best_evolved",
        "compatibility": "exact saved/current signal and action contracts; no remapping",
        "interpretation": "one source search per task; current simulator; pilot, no significance claim",
        "tuning_evaluations_per_structure": 1 + args.cem_iterations * args.cem_population,
    }
    contracts = {e: json.loads(json.dumps(adapters[e].signal_contract.to_dict())) for e in TARGETS}
    report = {
        "protocol": config,
        "sources": sources,
        "targets": TARGETS,
        "target_contracts": contracts,
        "cells": [],
        "baselines": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.json"
    if result_path.exists():
        if not args.resume:
            parser.error("output exists; use another directory or --resume")
        old = json.loads(result_path.read_text())
        if any(old[k] != report[k] for k in ("protocol", "sources", "targets", "target_contracts")):
            parser.error("resume manifest differs")
        report = old
    done = {(c["source"], c["target"], c["mode"]) for c in report["cells"]}
    for source in sources:
        law = GymStructure.from_dict(source["structure"])
        for target in TARGETS:
            reason = compatibility(source["contract"], contracts[target], law.signals)
            for mode in ("zero_shot", "retuned"):
                key = (source["environment"], target, mode)
                if key in done:
                    continue
                cell = {"source": key[0], "target": target, "mode": mode}
                if reason:
                    cell.update(status="incompatible", reason=reason)
                else:
                    print("Evaluating", *key, flush=True)
                    try:
                        gains = np.asarray(source["gains"])
                        if mode == "retuned":
                            gains, metrics = tune_gym_cem(
                                adapters[target],
                                law,
                                train,
                                iterations=args.cem_iterations,
                                population_size=args.cem_population,
                            )
                            cell["train_metrics"] = metrics.to_dict()
                        metrics, episodes = evaluate_gym_structure(
                            adapters[target], law, gains, test
                        )
                        cell.update(
                            status="ok",
                            gains=gains.tolist(),
                            metrics=metrics.to_dict(),
                            episodes=[dict(seed=s, **asdict(e)) for s, e in zip(test, episodes)],
                        )
                    except Exception as exc:  # noqa: BLE001 -- retain per-cell simulator failures
                        cell.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                report["cells"].append(cell)
                result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for target in TARGETS:
        if any(b["target"] == target for b in report["baselines"]):
            continue
        print("Tuning target baselines", target, flush=True)
        candidates = []
        for law in adapters[target].classical:
            gains, metrics = tune_gym_cem(
                adapters[target],
                law,
                train,
                iterations=args.cem_iterations,
                population_size=args.cem_population,
            )
            candidates.append((law, gains, metrics))
        if "InvertedPendulum" in target:
            law, gains = inverted_pendulum_lqr()
            metrics, _ = evaluate_gym_structure(adapters[target], law, gains, train)
            candidates.append((law, gains, metrics))
        law, gains, training = max(candidates, key=lambda c: c[2].selection_key)
        metrics, episodes = evaluate_gym_structure(adapters[target], law, gains, test)
        report["baselines"].append(
            {
                "target": target,
                "structure": law.to_dict(),
                "gains": gains.tolist(),
                "train_metrics": training.to_dict(),
                "metrics": metrics.to_dict(),
                "candidates": [
                    {"structure": l.to_dict(), "train_metrics": m.to_dict()}
                    for l, g, m in candidates
                ],
                "episodes": [dict(seed=s, **asdict(e)) for s, e in zip(test, episodes)],
            }
        )
        result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    export(args.output, report)
    print("Complete:", args.output, flush=True)


if __name__ == "__main__":
    main()
