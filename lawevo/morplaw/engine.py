from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lawevo.evolve.belief import BeliefSpace, Experience
from lawevo.morplaw.evaluate import PairMetrics, tune_pair_cem
from lawevo.morplaw.morphology import (
    MorphologyError,
    MorphologySpec,
    MorphologyTemplate,
)
from lawevo.pid.gym_benchmark import BenchmarkAdapter, GymStructure

VALID_DIRECTIONS = ("both", "m_to_l", "l_to_m", "none")


@dataclass(frozen=True)
class MorpLawConfig:
    generations: int = 5
    proposals_per_side: int = 4
    joint_top_k: int = 2
    cem_iterations: int = 5
    cem_population: int = 24
    cross_direction: str = "both"
    morphology_frozen: bool = False
    law_frozen: bool = False
    morph_cost_weight: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "generations",
            "proposals_per_side",
            "joint_top_k",
            "cem_iterations",
            "cem_population",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.cross_direction not in VALID_DIRECTIONS:
            raise ValueError(f"cross_direction must be one of {VALID_DIRECTIONS}")
        if self.morphology_frozen and self.law_frozen:
            raise ValueError("degenerate config: both morphology and law frozen")


@dataclass
class PairRecord:
    spec: MorphologySpec
    structure: GymStructure
    gains: np.ndarray
    metrics: PairMetrics
    generation: int
    provenance: str = "seed"
    episode_budget: int = 0

    def key(self) -> tuple[tuple[str, ...], tuple[tuple[str, float], ...]]:
        return (self.structure.key(), self.spec.key())

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "structure": self.structure.to_dict(),
            "gains": self.gains.tolist(),
            "metrics": self.metrics.to_dict(),
            "generation": self.generation,
            "provenance": self.provenance,
            "episode_budget": self.episode_budget,
        }


class LawGenerator(Protocol):
    def __call__(
        self, incumbent: PairRecord, belief: BeliefSpace, count: int, generation: int
    ) -> Sequence[GymStructure]: ...


class MorphologyGenerator(Protocol):
    def __call__(
        self, incumbent: PairRecord, belief: BeliefSpace, count: int, generation: int
    ) -> Sequence[MorphologySpec]: ...


@dataclass(frozen=True)
class MorpLawGenerationReport:
    generation: int
    incumbent: str
    best_score: float
    evaluated: int
    rejected: int
    cross_table: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "incumbent": self.incumbent,
            "best_score": self.best_score,
            "evaluated": self.evaluated,
            "rejected": self.rejected,
            "cross_table": self.cross_table,
        }


class MorpLawRunner:
    """Co-evolution of MJCF morphology and symbolic law structures.

    Each generation proposes laws (on the incumbent morphology) and morphologies
    (under the incumbent law), evaluates the one-sided cross pairs for clean
    attribution, evaluates the top-k joint pairs, and keeps the best pair
    (elitist 1+lambda). Cross-evaluation results feed two context-tagged
    experience channels: morph_to_law (measured facts) and law_to_morph
    (hypotheses for the morphology side).
    """

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        template: MorphologyTemplate,
        train_seeds: Sequence[int],
        law_generator: LawGenerator,
        morph_generator: MorphologyGenerator,
        config: MorpLawConfig | None = None,
        archive: dict[tuple, PairRecord] | None = None,
    ) -> None:
        if not train_seeds:
            raise ValueError("at least one training seed is required")
        if template.env_id != adapter.env_id:
            raise ValueError(
                f"template env {template.env_id!r} does not match adapter {adapter.env_id!r}"
            )
        self.adapter = adapter
        self.template = template
        self.train_seeds = tuple(train_seeds)
        self.law_generator = law_generator
        self.morph_generator = morph_generator
        self.config = config or MorpLawConfig()
        self.archive = archive if archive is not None else {}
        self.belief = BeliefSpace()
        self.reports: list[MorpLawGenerationReport] = []
        self.calls: dict[str, int] = {"law": 0, "morph": 0}
        self.episodes_spent = 0

    def _evaluate(
        self, spec: MorphologySpec, structure: GymStructure, provenance: str, generation: int
    ) -> PairRecord | None:
        key = (structure.key(), spec.key())
        existing = self.archive.get(key)
        if existing is not None:
            return existing
        try:
            gains, metrics, budget = tune_pair_cem(
                self.adapter,
                self.template,
                spec,
                structure,
                list(self.train_seeds),
                iterations=self.config.cem_iterations,
                population_size=self.config.cem_population,
                morph_cost_weight=self.config.morph_cost_weight,
            )
        except MorphologyError as exc:
            self.belief.update(
                [
                    Experience(
                        "failure",
                        f"{structure.name} on {spec.describe()} rejected: {exc}",
                        context={"morphology": _spec_context(spec)},
                    )
                ]
            )
            return None
        record = PairRecord(spec, structure, gains, metrics, generation, provenance, budget)
        self.archive[key] = record
        self.episodes_spent += budget
        return record

    def run(
        self, initial_pairs: Sequence[tuple[MorphologySpec, GymStructure]]
    ) -> tuple[PairRecord, list[MorpLawGenerationReport]]:
        if not initial_pairs:
            raise ValueError("initial pairs must not be empty")
        initial_records: list[PairRecord] = []
        for spec, structure in initial_pairs:
            record = self._evaluate(spec, structure, "seed", 0)
            if record is not None:
                initial_records.append(record)
        if not initial_records:
            raise RuntimeError("no initial pair could be evaluated")
        incumbent = max(
            initial_records, key=lambda item: (item.metrics.score, str(item.key()))
        )
        for generation in range(1, self.config.generations + 1):
            law_proposals: list[GymStructure] = []
            morph_proposals: list[MorphologySpec] = []
            if not self.config.law_frozen:
                print(
                    f"[generation {generation}] requesting {self.config.proposals_per_side} "
                    "law proposals from the LLM ...",
                    flush=True,
                )
                law_proposals = list(
                    self.law_generator(incumbent, self.belief, self.config.proposals_per_side, generation)
                )
                self._check_count(law_proposals, "law", generation)
                self.calls["law"] += 1
                print(f"[generation {generation}] received {len(law_proposals)} law proposals", flush=True)
            if not self.config.morphology_frozen:
                print(
                    f"[generation {generation}] requesting {self.config.proposals_per_side} "
                    "morphology proposals from the LLM ...",
                    flush=True,
                )
                morph_proposals = list(
                    self.morph_generator(
                        incumbent, self.belief, self.config.proposals_per_side, generation
                    )
                )
                self._check_count(morph_proposals, "morphology", generation)
                self.calls["morph"] += 1
                print(
                    f"[generation {generation}] received {len(morph_proposals)} morphology proposals",
                    flush=True,
                )
            print(
                f"[generation {generation}] evaluating {len(law_proposals)} law + "
                f"{len(morph_proposals)} morphology cross pairs ...",
                flush=True,
            )
            law_records = [
                self._evaluate(incumbent.spec, structure, "law_cross", generation)
                for structure in law_proposals
            ]
            morph_records = [
                self._evaluate(spec, incumbent.structure, "morph_cross", generation)
                for spec in morph_proposals
            ]
            joint_records = self._evaluate_joint_pairs(
                law_proposals, morph_proposals, law_records, morph_records, generation
            )
            self._emit_experiences(
                incumbent, law_proposals, morph_proposals, law_records, morph_records
            )
            candidates = [
                record
                for record in [*law_records, *morph_records, *joint_records]
                if record is not None
            ]
            incumbent = max(
                [incumbent, *candidates],
                key=lambda item: (item.metrics.score, str(item.key())),
            )
            rejected = (
                len(law_proposals)
                + len(morph_proposals)
                + len(joint_records)
                - sum(record is not None for record in [*law_records, *morph_records, *joint_records])
            )
            evaluated = sum(
                record is not None for record in [*law_records, *morph_records, *joint_records]
            )
            report = MorpLawGenerationReport(
                generation,
                f"{incumbent.structure.name}@{incumbent.spec.describe()}",
                incumbent.metrics.score,
                evaluated,
                rejected,
                {
                    "law_cross": sum(record is not None for record in law_records),
                    "morph_cross": sum(record is not None for record in morph_records),
                    "joint": sum(record is not None for record in joint_records),
                    "rejected": rejected,
                },
            )
            self.reports.append(report)
            print(
                f"[generation {generation}] incumbent={incumbent.structure.name}"
                f"@{incumbent.spec.describe()} score={incumbent.metrics.score:.4g} "
                f"evaluated={evaluated} rejected={rejected}",
                flush=True,
            )
        print(
            f"run complete: best={incumbent.structure.name}@{incumbent.spec.describe()} "
            f"score={incumbent.metrics.score:.4g} episodes_spent={self.episodes_spent}",
            flush=True,
        )
        return incumbent, self.reports

    def _evaluate_joint_pairs(
        self,
        law_proposals: list[GymStructure],
        morph_proposals: list[MorphologySpec],
        law_records: list[PairRecord | None],
        morph_records: list[PairRecord | None],
        generation: int,
    ) -> list[PairRecord | None]:
        if not law_proposals or not morph_proposals:
            return []
        morph_scores = {
            spec.key(): record.metrics.score
            for spec, record in zip(morph_proposals, morph_records, strict=True)
            if record is not None
        }
        law_scores = {
            structure.key(): record.metrics.score
            for structure, record in zip(law_proposals, law_records, strict=True)
            if record is not None
        }
        candidates = [
            (spec, structure)
            for spec in morph_proposals
            if spec.key() in morph_scores
            for structure in law_proposals
            if structure.key() in law_scores
        ]
        ranked = sorted(
            candidates,
            key=lambda pair: -(morph_scores[pair[0].key()] + law_scores[pair[1].key()]),
        )
        return [
            self._evaluate(spec, structure, "joint", generation)
            for spec, structure in ranked[: self.config.joint_top_k]
        ]

    def _emit_experiences(
        self,
        baseline: PairRecord,
        law_proposals: list[GymStructure],
        morph_proposals: list[MorphologySpec],
        law_records: list[PairRecord | None],
        morph_records: list[PairRecord | None],
    ) -> None:
        experiences: list[Experience] = []
        if self.config.cross_direction in ("both", "m_to_l"):
            for structure, record in zip(law_proposals, law_records, strict=True):
                if record is None:
                    continue
                delta = record.metrics.score - baseline.metrics.score
                experiences.append(
                    Experience(
                        "morph_to_law",
                        (
                            f"M[{baseline.spec.describe()}]: {structure.name} "
                            f"({', '.join(structure.terms)}) score {record.metrics.score:.4g} vs "
                            f"incumbent {baseline.structure.name} {baseline.metrics.score:.4g} "
                            f"(delta {delta:+.4g}); success {record.metrics.success_rate:.3g} vs "
                            f"{baseline.metrics.success_rate:.3g}; energy_norm "
                            f"{record.metrics.energy_norm:.4g} vs {baseline.metrics.energy_norm:.4g}"
                        ),
                        delta,
                        context={"morphology": _spec_context(baseline.spec)},
                    )
                )
        if self.config.cross_direction in ("both", "l_to_m"):
            for spec, record in zip(morph_proposals, morph_records, strict=True):
                if record is None:
                    continue
                delta = record.metrics.score - baseline.metrics.score
                experiences.append(
                    Experience(
                        "law_to_morph",
                        (
                            f"under {baseline.structure.name} ({', '.join(baseline.structure.terms)}): "
                            f"morphology change [{self.template.field_deltas(spec)}] -> score "
                            f"{record.metrics.score:.4g} vs {baseline.metrics.score:.4g} "
                            f"(delta {delta:+.4g}); success {record.metrics.success_rate:.3g} vs "
                            f"{baseline.metrics.success_rate:.3g}; energy_norm "
                            f"{record.metrics.energy_norm:.4g} vs {baseline.metrics.energy_norm:.4g}. "
                            f"{_bottleneck_note(record, baseline)}"
                        ),
                        delta,
                        context={"structure": json.dumps(list(baseline.structure.terms))},
                        hypothesis=True,
                    )
                )
        self.belief.update(experiences)

    def _check_count(self, items: Sequence[object], side: str, generation: int) -> None:
        if len(items) != self.config.proposals_per_side:
            raise ValueError(
                f"{side} generator returned {len(items)} proposals at generation "
                f"{generation}, expected {self.config.proposals_per_side}"
            )


def _spec_context(spec: MorphologySpec) -> str:
    return json.dumps(spec.to_dict(), sort_keys=True)


def _bottleneck_note(record: PairRecord, baseline: PairRecord) -> str:
    if (
        record.metrics.energy_norm > 1.15 * baseline.metrics.energy_norm
        and record.metrics.score < baseline.metrics.score
    ):
        return "Hypothesis: actuator gear is the bottleneck, not geometry — raise gear before mass."
    if record.metrics.success_rate < baseline.metrics.success_rate:
        return (
            "Hypothesis: this body change exceeds the current law's stability margin — "
            "reconsider the changed fields."
        )
    return "Hypothesis: this direction looks compatible with the incumbent law — continue exploring it."
