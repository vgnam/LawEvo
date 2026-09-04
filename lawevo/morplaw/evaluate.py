from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lawevo.morplaw.morphology import MorphologyGenome, MorphologySpec, MorphologyTemplate
from lawevo.pid.gym_benchmark import (
    BenchmarkAdapter,
    GymEpisode,
    GymMetrics,
    GymStructure,
    evaluate_gym_structure,
)

if TYPE_CHECKING:
    import gymnasium as gym


@dataclass(frozen=True)
class PairMetrics:
    """Metrics of one (morphology, structure) pair, normalized for cross-body comparison."""

    score: float
    episode_return: float
    success_rate: float
    energy: float
    energy_norm: float
    jerk: float
    complexity: int
    morph_cost: float
    total_mass: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "score": self.score,
            "episode_return": self.episode_return,
            "success_rate": self.success_rate,
            "energy": self.energy,
            "energy_norm": self.energy_norm,
            "jerk": self.jerk,
            "complexity": self.complexity,
            "morph_cost": self.morph_cost,
            "total_mass": self.total_mass,
        }


def make_morph_env(
    adapter: BenchmarkAdapter, template: MorphologyTemplate, spec: MorphologyGenome
) -> gym.Env:
    """Create the adapter's environment with the morphology applied.

    MJCF templates pass the rendered XML. URDF templates (the parametric
    Panda) forward ``urdf_path`` plus their environment parameters. Pure
    parameter templates forward their fields as ``gym.make`` kwargs.
    """
    import gymnasium as gym

    extra_kwargs = adapter.morph_env_kwargs() if hasattr(adapter, "morph_env_kwargs") else {}
    if getattr(template, "mjcf_template", True):
        return gym.make(
            adapter.env_id,
            xml_file=str(template.xml_path(spec)),
            max_episode_steps=adapter.horizon,
            **extra_kwargs,
        )
    adapter.make_env()  # lazy env registration for PyBullet variants
    make_kwargs = dict(extra_kwargs)
    make_kwargs["max_episode_steps"] = adapter.horizon
    if getattr(template, "urdf_template", False):
        assert isinstance(spec, MorphologySpec)
        make_kwargs["urdf_path"] = str(template.urdf_path(spec))
        make_kwargs["motor_force"] = spec.get("motor_force")
        make_kwargs.update(template.environment_parameters(spec))
    else:
        parameters = spec.to_dict() if isinstance(spec, MorphologySpec) else dict(spec.to_dict())
        make_kwargs.update(parameters)
    return gym.make(adapter.env_id, **make_kwargs)


def morph_cost(template: MorphologyTemplate, spec: MorphologyGenome, weight: float = 0.05) -> float:
    """Template-specific physical/structural complexity penalty."""
    return template.cost(spec, weight)


def pair_adjust(
    adapter: BenchmarkAdapter,
    template: MorphologyTemplate,
    spec: MorphologyGenome,
    metrics: GymMetrics,
    morph_cost_weight: float = 0.05,
) -> PairMetrics:
    mass = template.total_mass(spec)
    energy_norm = metrics.energy / max(mass, 1e-9)
    cost = morph_cost(template, spec, morph_cost_weight)
    score = adapter.score(metrics.episode_return, energy_norm, metrics.jerk, metrics.complexity)
    return PairMetrics(
        score - cost,
        metrics.episode_return,
        metrics.success_rate,
        metrics.energy,
        energy_norm,
        metrics.jerk,
        metrics.complexity,
        cost,
        mass,
    )


def evaluate_pair(
    adapter: BenchmarkAdapter,
    template: MorphologyTemplate,
    spec: MorphologyGenome,
    structure: GymStructure,
    gains: np.ndarray,
    seeds: list[int],
    *,
    morph_cost_weight: float = 0.05,
) -> tuple[PairMetrics, list[GymEpisode]]:
    """Evaluate a tuned pair on the given episode seeds."""
    envs = [make_morph_env(adapter, template, spec) for _ in seeds]
    try:
        metrics, episodes = evaluate_gym_structure(adapter, structure, gains, seeds, envs=envs)
        return pair_adjust(adapter, template, spec, metrics, morph_cost_weight), episodes
    finally:
        for env in envs:
            env.close()


def tune_pair_cem(
    adapter: BenchmarkAdapter,
    template: MorphologyTemplate,
    spec: MorphologyGenome,
    structure: GymStructure,
    seeds: list[int],
    *,
    iterations: int = 5,
    population_size: int = 24,
    morph_cost_weight: float = 0.05,
) -> tuple[np.ndarray, PairMetrics, int]:
    """CEM gain tuning for one (morphology, structure) pair.

    The CEM seed derives from the full pair identity, so every pair gets a
    deterministic, equal simulation budget. The returned budget counts episodes.
    """
    digest = hashlib.sha256(
        json.dumps(
            {
                "env": adapter.env_id,
                "template": template.cache_namespace(),
                "law": structure.to_expression_string(),
                "morph": spec.key(),
            },
            sort_keys=True,
        ).encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))
    dimension = structure.parameter_count
    mean, sigma = np.zeros(dimension), np.full(dimension, 3.0)
    envs = [make_morph_env(adapter, template, spec) for _ in seeds]
    episode_count = 0

    def evaluate(gains: np.ndarray) -> PairMetrics:
        nonlocal episode_count
        metrics, _ = evaluate_gym_structure(adapter, structure, gains, seeds, envs=envs)
        episode_count += len(seeds)
        return pair_adjust(adapter, template, spec, metrics, morph_cost_weight)

    try:
        best_gains = mean.copy()
        best_metrics = evaluate(best_gains)
        elite_count = max(2, round(0.2 * population_size))
        for _ in range(iterations):
            samples = np.clip(rng.normal(mean, sigma, size=(population_size, dimension)), -20, 20)
            scored = sorted(
                ((sample, evaluate(sample)) for sample in samples),
                key=lambda item: item[1].score,
                reverse=True,
            )
            elites = np.vstack([item[0] for item in scored[:elite_count]])
            mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
            sigma = np.maximum(0.05, 0.25 * sigma + 0.75 * elites.std(axis=0))
            if scored[0][1].score > best_metrics.score:
                best_gains, best_metrics = scored[0][0].copy(), scored[0][1]
        return best_gains, best_metrics, episode_count
    finally:
        for env in envs:
            env.close()


def pair_formula(structure: GymStructure, gains: np.ndarray) -> str:
    return structure.formula(gains)
