from __future__ import annotations

import argparse
import getpass
import json
import os
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
from lawevo.morplaw import (
    TEMPLATE_ADAPTERS,
    TEMPLATES,
    MorpLawConfig,
    MorpLawRunner,
    PairMetrics,
    PairRecord,
    evaluate_pair,
    extract_morphologies,
    extract_structures,
    law_mutation_prompt,
    morphology_mutation_prompt,
    pair_formula,
    tune_pair_cem,
)
from lawevo.morplaw.morphology import MorphologySpec, MorphologyTemplate
from lawevo.pid.gym_benchmark import ADAPTERS, LOCOMOTION_ADAPTERS

SYSTEM_PROMPT = (
    "You are an expert control researcher co-designing robot morphologies and compact "
    "symbolic feedback laws. Return JSON only."
)

ENV_ADAPTERS = {**ADAPTERS, **LOCOMOTION_ADAPTERS}

VARIANT_NAMES = (
    "classical",
    "law_only",
    "morph_only",
    "none",
    "m_to_l",
    "l_to_m",
    "both",
)


def make_law_generator(
    client: NVIDIAChatClient,
    task_key: str,
    adapter,
    cache: dict[tuple, PairRecord],
    log: list[dict[str, object]],
    variant: str,
):
    def law_generator(incumbent, belief, count, generation):
        elites = sorted(cache.values(), key=lambda item: item.metrics.score, reverse=True)[:6]
        prompt = law_mutation_prompt(
            task_key, adapter, incumbent, belief, elites, count, generation
        )
        response = client.complete(SYSTEM_PROMPT, prompt, temperature=0.8, reasoning_effort="medium")
        proposed = extract_structures(response, adapter.allowed_terms)
        log.append(
            {
                "variant": variant,
                "generation": generation,
                "side": "law",
                "valid": len(proposed),
                "response": response,
            }
        )
        if len(proposed) < count:
            raise RuntimeError(f"LLM returned only {len(proposed)} valid laws, expected {count}")
        return proposed[:count]

    return law_generator


def make_morph_generator(
    client: NVIDIAChatClient,
    task_key: str,
    template: MorphologyTemplate,
    cache: dict[tuple, PairRecord],
    log: list[dict[str, object]],
    variant: str,
):
    def morph_generator(incumbent, belief, count, generation):
        elites = sorted(cache.values(), key=lambda item: item.metrics.score, reverse=True)[:6]
        prompt = morphology_mutation_prompt(
            task_key, template, incumbent, belief, elites, count, generation
        )
        response = client.complete(SYSTEM_PROMPT, prompt, temperature=0.8, reasoning_effort="medium")
        proposed = extract_morphologies(response, template)
        log.append(
            {
                "variant": variant,
                "generation": generation,
                "side": "morphology",
                "valid": len(proposed),
                "response": response,
            }
        )
        if len(proposed) < count:
            raise RuntimeError(
                f"LLM returned only {len(proposed)} valid morphologies, expected {count}"
            )
        return proposed[:count]

    return morph_generator


def record_to_json(record: PairRecord, env_id: str) -> dict[str, object]:
    payload = record.to_dict()
    payload["env"] = env_id
    return payload


def record_from_json(payload: dict[str, object]) -> PairRecord:
    from lawevo.pid.gym_benchmark import GymStructure

    structure_payload = payload["structure"]
    structure = GymStructure(structure_payload["name"], tuple(structure_payload["terms"]))
    return PairRecord(
        spec=MorphologySpec.of(payload["spec"]),
        structure=structure,
        gains=np.asarray(payload["gains"], dtype=float),
        metrics=PairMetrics(**payload["metrics"]),
        generation=payload["generation"],
        provenance=payload["provenance"],
        episode_budget=payload["episode_budget"],
    )


def load_cache(path: Path, env_id: str) -> dict[tuple, PairRecord]:
    cache: dict[tuple, PairRecord] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("env") != env_id:
            continue
        record = record_from_json(payload)
        cache[record.key()] = record
    return cache


def save_cache(path: Path, cache: dict[tuple, PairRecord], env_id: str) -> None:
    lines = [
        json.dumps(record_to_json(record, env_id)) + "\n" for record in cache.values()
    ]
    path.write_text("".join(lines), encoding="utf-8")


def plot_results(
    names: list[str],
    heldout: dict[str, PairMetrics],
    per_episode: dict[str, dict[str, np.ndarray]],
    output: Path,
) -> None:
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
        ("score", "Pair Score (held-out) ↑"),
        ("episode_return", "Episode Return ↑"),
        ("success_rate", "Success Rate ↑"),
        ("energy_norm", "Normalized Energy ↓"),
    )
    colors = ["#8C8C8C", "#D55E00", "#F0E442", "#56B4E9", "#009E73", "#CC79A7", "#0072B2"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    for axis, (metric, title) in zip(axes.ravel(), panels):
        means = [getattr(heldout[name], metric) for name in names]
        sems = []
        for name in names:
            values = per_episode[name].get(metric)
            sems.append(
                float(np.std(values, ddof=1) / np.sqrt(len(values))) if values is not None else 0.0
            )
        x = np.arange(len(names))
        axis.bar(x, means, yerr=sems, color=colors, capsize=3, edgecolor="white")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(x, names, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle(
        "MorpLaw: Morphology × Law Co-evolution (held-out episodes)",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(0.5, 0.01, "Mean ± SEM over held-out episodes.", ha="center", color="#444444")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.14, hspace=0.35, wspace=0.22)
    fig.savefig(output / "morplaw_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "morplaw_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        choices=sorted(TEMPLATES),
        default="reacher",
        help="morphable MuJoCo environment (template must exist)",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="defaults to results/morplaw_<environment>"
    )
    parser.add_argument("--model", default=os.environ.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL))
    parser.add_argument(
        "--base-url", default=os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)
    )
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--proposals-per-side", type=int, default=4)
    parser.add_argument("--joint-top-k", type=int, default=2)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument("--cem-population", type=int, default=24)
    parser.add_argument("--train-episodes", type=int, default=6)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--variants", nargs="+", choices=VARIANT_NAMES, default=list(VARIANT_NAMES))
    parser.add_argument("--resume", action="store_true", help="reuse the records.jsonl cache")
    args = parser.parse_args()
    output = args.output or Path(f"results/morplaw_{args.environment}")
    output.mkdir(parents=True, exist_ok=True)
    needs_llm = any(name != "classical" for name in args.variants)
    api_key = ""
    if needs_llm:
        api_key = os.environ.get("NVIDIA_API_KEY") or getpass.getpass("NVIDIA API key: ")
        if not api_key:
            raise SystemExit("NVIDIA_API_KEY is required for LLM variants")

    adapter = ENV_ADAPTERS[TEMPLATE_ADAPTERS[args.environment]]
    template = TEMPLATES[args.environment]
    default_spec = template.default_spec()
    train_seeds = list(range(args.train_episodes))
    test_seeds = list(range(1000, 1000 + args.test_episodes))
    cache_path = output / "records.jsonl"
    cache = load_cache(cache_path, adapter.env_id) if args.resume else {}
    responses: list[dict[str, object]] = []

    client = (
        NVIDIAChatClient(api_key, model=args.model, endpoint=args.base_url)
        if needs_llm
        else None
    )
    config_kwargs = {
        "generations": args.generations,
        "proposals_per_side": args.proposals_per_side,
        "joint_top_k": args.joint_top_k,
        "cem_iterations": args.cem_iterations,
        "cem_population": args.cem_population,
    }

    initial_pairs = [(default_spec, structure) for structure in adapter.classical]
    results: dict[str, dict[str, object]] = {}
    for name in args.variants:
        print(f"=== variant {name} ===", flush=True)
        if name == "classical":
            for spec, structure in initial_pairs:
                key = (structure.key(), spec.key())
                if key in cache:
                    continue
                gains, metrics, budget = tune_pair_cem(
                    adapter,
                    template,
                    spec,
                    structure,
                    train_seeds,
                    iterations=args.cem_iterations,
                    population_size=args.cem_population,
                )
                cache[key] = PairRecord(spec, structure, gains, metrics, 0, "seed", budget)
            classical = [
                cache[(structure.key(), default_spec.key())] for structure in adapter.classical
            ]
            best = max(classical, key=lambda item: (item.metrics.score, str(item.key())))
            results[name] = {
                "best": record_to_json(best, adapter.env_id),
                "reports": [],
                "api_calls": 0,
                "episodes_spent": sum(item.episode_budget for item in classical),
            }
        else:
            direction_flags = {
                "law_only": {"morphology_frozen": True},
                "morph_only": {"law_frozen": True},
                "none": {"cross_direction": "none"},
                "m_to_l": {"cross_direction": "m_to_l"},
                "l_to_m": {"cross_direction": "l_to_m"},
                "both": {"cross_direction": "both"},
            }
            config = MorpLawConfig(**config_kwargs, **direction_flags[name])
            law_gen = make_law_generator(
                client, args.environment, adapter, cache, responses, name
            )
            morph_gen = make_morph_generator(
                client, args.environment, template, cache, responses, name
            )
            runner = MorpLawRunner(
                adapter,
                template,
                train_seeds,
                law_gen,
                morph_gen,
                config,
                archive=cache,
            )
            best, reports = runner.run(initial_pairs)
            results[name] = {
                "best": record_to_json(best, adapter.env_id),
                "reports": [report.to_dict() for report in reports],
                "api_calls": len([entry for entry in responses if entry["variant"] == name]),
                "episodes_spent": runner.episodes_spent,
                "belief": {
                    "morph_to_law": [
                        {"statement": item.statement, "context": item.context}
                        for item in runner.belief.morph_to_law
                    ],
                    "law_to_morph": [
                        {"statement": item.statement, "context": item.context}
                        for item in runner.belief.law_to_morph
                    ],
                },
            }
        save_cache(cache_path, cache, adapter.env_id)
        best_payload = record_from_json(results[name]["best"])
        print(
            f"variant {name} done: best={best_payload.structure.name}"
            f"@{best_payload.spec.describe()} "
            f"score={best_payload.metrics.score:.4g} "
            f"api_calls={results[name]['api_calls']} "
            f"episodes_spent={results[name]['episodes_spent']}",
            flush=True,
        )

    # Held-out evaluation of every variant's best pair.
    heldout: dict[str, PairMetrics] = {}
    per_episode: dict[str, dict[str, np.ndarray]] = {}
    for name, entry in results.items():
        best = record_from_json(entry["best"])
        metrics, episodes = evaluate_pair(
            adapter, template, best.spec, best.structure, best.gains, test_seeds
        )
        heldout[name] = metrics
        per_episode[name] = {
            "episode_return": np.asarray([item.episode_return for item in episodes]),
            "success_rate": np.asarray([float(item.success) for item in episodes]),
            "energy_norm": np.asarray([item.energy for item in episodes]) / metrics.total_mass,
        }
        entry["heldout"] = metrics.to_dict()

    plot_results(list(results), heldout, per_episode, output)
    payload = {
        "model": args.model,
        "environment": adapter.env_id,
        "morphology_fields": template.field_descriptions(),
        "config": {
            **config_kwargs,
            "train_episodes": args.train_episodes,
            "test_episodes": args.test_episodes,
            "variants": args.variants,
        },
        "variants": results,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "nim_responses.json").write_text(
        json.dumps(responses, indent=2), encoding="utf-8"
    )
    summary = {}
    for name, entry in results.items():
        best = record_from_json(entry["best"])
        summary[name] = {
            "best_pair": f"{best.structure.name}@{best.spec.describe()}",
            "formula": pair_formula(best.structure, best.gains),
            "train_score": best.metrics.score,
            "heldout": entry["heldout"],
        }
    print(json.dumps(summary, indent=2))
    (output / "README.md").write_text(
        "\n".join(
            [
                "# MorpLaw run",
                "",
                f"- environment: `{adapter.env_id}`",
                f"- model: `{args.model}`",
                f"- config: {json.dumps(payload['config'], indent=2)}",
                f"- variants: {', '.join(args.variants)}",
                "",
                "## Best pair per variant (train-tuned; held-out means in `results.json`)",
                "",
                "| variant | best pair | train score | held-out score |",
                "|---|---|---:|---:|",
            ]
            + [
                f"| {name} | `{summary[name]['best_pair']}` | "
                f"{summary[name]['train_score']:.4g} | "
                f"{summary[name]['heldout']['score']:.4g} |"
                for name in summary
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
