from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from inspect import signature
from typing import Protocol

import numpy as np

from lawevo.evolve.belief import BeliefSpace, Experience
from lawevo.morplaw.evaluate import PairMetrics, tune_pair_cem
from lawevo.morplaw.knowledge import (
    KNOWLEDGE_MODES,
    DirectedKnowledgeBase,
    EvidenceRecord,
    KnowledgeHypothesis,
)
from lawevo.morplaw.morphology import (
    MorphologyError,
    MorphologyGenome,
    MorphologyTemplate,
)
from lawevo.morplaw.navigator import MorpLawNavigator, SearchDirective
from lawevo.morplaw.proposals import LawProposal, MorphologyProposal
from lawevo.pid.gym_benchmark import BenchmarkAdapter, GymStructure


@dataclass(frozen=True)
class MorpLawConfig:
    generations: int = 5
    proposals_per_side: int = 4
    responsive_per_side: int = 1
    joint_top_k: int = 2
    cem_iterations: int = 5
    cem_population: int = 24
    knowledge_mode: str = "full"
    morphology_frozen: bool = False
    law_frozen: bool = False
    morph_cost_weight: float = 0.05
    knowledge_capacity: int = 24
    retrieve_per_polarity: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "generations",
            "proposals_per_side",
            "responsive_per_side",
            "joint_top_k",
            "cem_iterations",
            "cem_population",
            "retrieve_per_polarity",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.knowledge_mode not in KNOWLEDGE_MODES:
            raise ValueError(f"knowledge_mode must be one of {KNOWLEDGE_MODES}")
        if self.knowledge_capacity <= 0:
            raise ValueError("knowledge_capacity must be positive")
        if self.morphology_frozen and self.law_frozen:
            raise ValueError("degenerate config: both morphology and law frozen")


@dataclass
class PairRecord:
    spec: MorphologyGenome
    structure: GymStructure
    gains: np.ndarray
    metrics: PairMetrics
    generation: int
    provenance: str = "seed"
    episode_budget: int = 0

    def key(self) -> tuple[object, object]:
        return (self.structure.key(), self.spec.key())

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "spec_type": self.spec.spec_type,
            "structure": self.structure.to_dict(),
            "gains": self.gains.tolist(),
            "metrics": self.metrics.to_dict(),
            "generation": self.generation,
            "provenance": self.provenance,
            "episode_budget": self.episode_budget,
        }


class LawGenerator(Protocol):
    def __call__(
        self,
        incumbent: PairRecord,
        knowledge: DirectedKnowledgeBase,
        count: int,
        generation: int,
        directive: SearchDirective,
        responsive: bool,
    ) -> Sequence[LawProposal | GymStructure]: ...


class MorphologyGenerator(Protocol):
    def __call__(
        self,
        incumbent: PairRecord,
        knowledge: DirectedKnowledgeBase,
        count: int,
        generation: int,
        directive: SearchDirective,
        responsive: bool,
    ) -> Sequence[MorphologyProposal | MorphologyGenome]: ...


@dataclass(frozen=True)
class InteractionRecord:
    generation: int
    morphology: dict[str, object]
    law_terms: tuple[str, ...]
    baseline_score: float
    morph_score: float
    law_score: float
    joint_score: float
    interaction: float
    provenance: str

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "morphology": self.morphology,
            "law_terms": list(self.law_terms),
            "baseline_score": self.baseline_score,
            "morph_score": self.morph_score,
            "law_score": self.law_score,
            "joint_score": self.joint_score,
            "interaction": self.interaction,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class MorpLawGenerationReport:
    generation: int
    incumbent: str
    best_score: float
    evaluated: int
    rejected: int
    cross_table: dict[str, int]
    navigator: dict[str, object]
    interactions: tuple[InteractionRecord, ...] = ()

    @property
    def max_abs_interaction(self) -> float:
        return max((abs(item.interaction) for item in self.interactions), default=0.0)

    def to_dict(self) -> dict[str, object]:
        values = [item.interaction for item in self.interactions]
        return {
            "generation": self.generation,
            "incumbent": self.incumbent,
            "best_score": self.best_score,
            "evaluated": self.evaluated,
            "rejected": self.rejected,
            "cross_table": self.cross_table,
            "navigator": self.navigator,
            "interaction_count": len(values),
            "mean_interaction": sum(values) / len(values) if values else 0.0,
            "max_abs_interaction": self.max_abs_interaction,
            "interactions": [item.to_dict() for item in self.interactions],
        }


@dataclass(frozen=True)
class _JointEvaluation:
    morph_proposal: MorphologyProposal
    law_proposal: LawProposal
    morph_record: PairRecord
    law_record: PairRecord
    joint_record: PairRecord
    provenance: str


class MorpLawRunner:
    """Knowledge-augmented bidirectional morphology-law co-evolution.

    Every ablation uses the same navigator, proposal counts, one-sided probes,
    responsive counterfactuals, CEM budget, and factorial interaction protocol.
    ``knowledge_mode`` only gates retrieval and accumulation for the two directed
    knowledge channels, keeping the ablation scientifically comparable.
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
        evaluation_cache: dict[tuple, PairRecord] | None = None,
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
        self.evaluation_cache = evaluation_cache if evaluation_cache is not None else self.archive
        self.knowledge = DirectedKnowledgeBase(
            self.config.knowledge_mode,
            self.config.knowledge_capacity,
            self.config.retrieve_per_polarity,
        )
        self.failures = BeliefSpace()
        self.navigator = MorpLawNavigator()
        self.reports: list[MorpLawGenerationReport] = []
        self.calls: dict[str, int] = {"law": 0, "morph": 0}
        self.episodes_spent = 0
        self.episodes_requested = 0

    def _evaluate(
        self, spec: MorphologyGenome, structure: GymStructure, provenance: str, generation: int
    ) -> PairRecord | None:
        key = (structure.key(), spec.key())
        existing = self.archive.get(key)
        if existing is None:
            existing = self.evaluation_cache.get(key)
        if existing is not None:
            self.episodes_requested += existing.episode_budget
            record = PairRecord(
                existing.spec,
                existing.structure,
                existing.gains,
                existing.metrics,
                generation,
                provenance,
                existing.episode_budget,
            )
            self.archive[key] = record
            return record
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
            self.failures.update(
                [
                    Experience(
                        "failure",
                        f"{structure.name} on {spec.describe()} rejected: {exc}",
                        context={"morphology": json.dumps(spec.to_dict(), sort_keys=True)},
                    )
                ]
            )
            return None
        record = PairRecord(spec, structure, gains, metrics, generation, provenance, budget)
        self.archive[key] = record
        self.evaluation_cache[key] = record
        self.episodes_spent += budget
        self.episodes_requested += budget
        return record

    def run(
        self, initial_pairs: Sequence[tuple[MorphologyGenome, GymStructure]]
    ) -> tuple[PairRecord, list[MorpLawGenerationReport]]:
        if not initial_pairs:
            raise ValueError("initial pairs must not be empty")
        initial_records = [
            record
            for spec, structure in initial_pairs
            if (record := self._evaluate(spec, structure, "seed", 0)) is not None
        ]
        if not initial_records:
            raise RuntimeError("no initial pair could be evaluated")
        incumbent = max(initial_records, key=lambda item: (item.metrics.score, str(item.key())))
        best_history = [incumbent.metrics.score]

        for generation in range(1, self.config.generations + 1):
            law_diversity, morph_diversity = _archive_diversity(self.archive.values())
            last_interaction = self.reports[-1].max_abs_interaction if self.reports else 0.0
            directive = self.navigator.decide(
                generation,
                best_history,
                law_diversity=law_diversity,
                morph_diversity=morph_diversity,
                last_max_abs_interaction=last_interaction,
            )
            print(
                f"[generation {generation}] navigator={directive.mode}: {directive.reason}",
                flush=True,
            )
            law_proposals = (
                []
                if self.config.law_frozen
                else self._request_laws(
                    incumbent,
                    self.config.proposals_per_side,
                    generation,
                    directive,
                    responsive=False,
                )
            )
            morph_proposals = (
                []
                if self.config.morphology_frozen
                else self._request_morphologies(
                    incumbent,
                    self.config.proposals_per_side,
                    generation,
                    directive,
                    responsive=False,
                )
            )
            law_records = [
                self._evaluate(incumbent.spec, proposal.structure, "law_cross", generation)
                for proposal in law_proposals
            ]
            morph_records = [
                self._evaluate(proposal.spec, incumbent.structure, "morph_cross", generation)
                for proposal in morph_proposals
            ]
            self._observe_law_effects(incumbent, law_proposals, law_records, generation)
            self._observe_morph_effects(incumbent, morph_proposals, morph_records, generation)
            base_joints, base_joint_records = self._evaluate_base_joints(
                incumbent,
                law_proposals,
                morph_proposals,
                law_records,
                morph_records,
                generation,
                directive,
            )
            base_interactions = tuple(
                self._observe_joint(incumbent, evaluation, generation) for evaluation in base_joints
            )
            responsive_joints, responsive_records = self._evaluate_responsive_joints(
                incumbent,
                law_proposals,
                morph_proposals,
                law_records,
                morph_records,
                generation,
                directive,
            )
            responsive_interactions = tuple(
                self._observe_joint(incumbent, evaluation, generation)
                for evaluation in responsive_joints
            )
            interactions = (*base_interactions, *responsive_interactions)
            all_slots: list[PairRecord | None] = [
                *law_records,
                *morph_records,
                *base_joint_records,
                *responsive_records,
            ]
            candidates = [record for record in all_slots if record is not None]
            incumbent = max(
                [incumbent, *candidates],
                key=lambda item: (item.metrics.score, str(item.key())),
            )
            best_history.append(incumbent.metrics.score)
            evaluated = sum(record is not None for record in all_slots)
            rejected = len(all_slots) - evaluated
            report = MorpLawGenerationReport(
                generation,
                f"{incumbent.structure.name}@{incumbent.spec.describe()}",
                incumbent.metrics.score,
                evaluated,
                rejected,
                {
                    "law_cross": sum(record is not None for record in law_records),
                    "morph_cross": sum(record is not None for record in morph_records),
                    "base_joint": len(base_joints),
                    "responsive": sum(record is not None for record in responsive_records),
                    "rejected": rejected,
                },
                {
                    **directive.to_dict(),
                    "operator_stats": self.navigator.stats_dict(),
                },
                interactions,
            )
            self.reports.append(report)
            print(
                f"[generation {generation}] incumbent={incumbent.structure.name}"
                f"@{incumbent.spec.describe()} score={incumbent.metrics.score:.4g} "
                f"evaluated={evaluated} rejected={rejected} "
                f"interactions={len(interactions)}",
                flush=True,
            )
        print(
            f"run complete: best={incumbent.structure.name}@{incumbent.spec.describe()} "
            f"score={incumbent.metrics.score:.4g} episodes_spent={self.episodes_spent} "
            f"episodes_requested={self.episodes_requested}",
            flush=True,
        )
        return incumbent, self.reports

    def _request_laws(
        self,
        incumbent: PairRecord,
        count: int,
        generation: int,
        directive: SearchDirective,
        *,
        responsive: bool,
    ) -> list[LawProposal]:
        if count == 0:
            return []
        phase = "responsive" if responsive else "primary"
        print(f"[generation {generation}] requesting {count} {phase} law proposals", flush=True)
        raw = list(
            _invoke_generator(
                self.law_generator,
                incumbent,
                self.knowledge,
                count,
                generation,
                directive,
                responsive,
            )
        )
        proposals = [_as_law_proposal(item) for item in raw]
        self._check_count(proposals, f"{phase} law", generation, count)
        self.calls["law"] += 1
        return proposals

    def _request_morphologies(
        self,
        incumbent: PairRecord,
        count: int,
        generation: int,
        directive: SearchDirective,
        *,
        responsive: bool,
    ) -> list[MorphologyProposal]:
        if count == 0:
            return []
        phase = "responsive" if responsive else "primary"
        print(
            f"[generation {generation}] requesting {count} {phase} morphology proposals",
            flush=True,
        )
        raw = list(
            _invoke_generator(
                self.morph_generator,
                incumbent,
                self.knowledge,
                count,
                generation,
                directive,
                responsive,
            )
        )
        proposals = [_as_morph_proposal(item) for item in raw]
        self._check_count(proposals, f"{phase} morphology", generation, count)
        self.calls["morph"] += 1
        return proposals

    def _evaluate_base_joints(
        self,
        baseline: PairRecord,
        law_proposals: list[LawProposal],
        morph_proposals: list[MorphologyProposal],
        law_records: list[PairRecord | None],
        morph_records: list[PairRecord | None],
        generation: int,
        directive: SearchDirective,
    ) -> tuple[list[_JointEvaluation], list[PairRecord | None]]:
        candidates = [
            (morph_proposal, law_proposal, morph_record, law_record)
            for morph_proposal, morph_record in zip(morph_proposals, morph_records, strict=True)
            if morph_record is not None
            for law_proposal, law_record in zip(law_proposals, law_records, strict=True)
            if law_record is not None
        ]
        selected = _select_joint_candidates(
            candidates, baseline.metrics.score, self.config.joint_top_k, directive.mode
        )
        output: list[_JointEvaluation] = []
        slots: list[PairRecord | None] = []
        for morph_proposal, law_proposal, morph_record, law_record in selected:
            joint = self._evaluate(
                morph_proposal.spec, law_proposal.structure, "base_joint", generation
            )
            slots.append(joint)
            if joint is not None:
                output.append(
                    _JointEvaluation(
                        morph_proposal,
                        law_proposal,
                        morph_record,
                        law_record,
                        joint,
                        "base_joint",
                    )
                )
        return output, slots

    def _evaluate_responsive_joints(
        self,
        baseline: PairRecord,
        law_proposals: list[LawProposal],
        morph_proposals: list[MorphologyProposal],
        law_records: list[PairRecord | None],
        morph_records: list[PairRecord | None],
        generation: int,
        directive: SearchDirective,
    ) -> tuple[list[_JointEvaluation], list[PairRecord | None]]:
        count = self.config.responsive_per_side
        if count == 0 or not law_proposals or not morph_proposals:
            return [], []
        valid_laws = [
            (proposal, record)
            for proposal, record in zip(law_proposals, law_records, strict=True)
            if record is not None
        ]
        valid_morphs = [
            (proposal, record)
            for proposal, record in zip(morph_proposals, morph_records, strict=True)
            if record is not None
        ]
        if not valid_laws or not valid_morphs:
            return [], []
        law_anchor_proposal, law_anchor = max(
            valid_laws, key=lambda pair: (pair[1].metrics.score, str(pair[1].key()))
        )
        morph_anchor_proposal, morph_anchor = max(
            valid_morphs, key=lambda pair: (pair[1].metrics.score, str(pair[1].key()))
        )
        responsive_laws = self._request_laws(
            morph_anchor, count, generation, directive, responsive=True
        )
        responsive_morphs = self._request_morphologies(
            law_anchor, count, generation, directive, responsive=True
        )
        evaluations: list[_JointEvaluation] = []
        slots: list[PairRecord | None] = []
        for proposal in responsive_laws:
            counterfactual = self._evaluate(
                baseline.spec, proposal.structure, "responsive_law_counterfactual", generation
            )
            joint = self._evaluate(
                morph_anchor.spec, proposal.structure, "responsive_law_joint", generation
            )
            slots.extend((counterfactual, joint))
            if counterfactual is not None and joint is not None:
                evaluations.append(
                    _JointEvaluation(
                        morph_anchor_proposal,
                        proposal,
                        morph_anchor,
                        counterfactual,
                        joint,
                        "responsive_morph_to_law",
                    )
                )
        for proposal in responsive_morphs:
            counterfactual = self._evaluate(
                proposal.spec, baseline.structure, "responsive_morph_counterfactual", generation
            )
            joint = self._evaluate(
                proposal.spec, law_anchor.structure, "responsive_morph_joint", generation
            )
            slots.extend((counterfactual, joint))
            if counterfactual is not None and joint is not None:
                evaluations.append(
                    _JointEvaluation(
                        proposal,
                        law_anchor_proposal,
                        counterfactual,
                        law_anchor,
                        joint,
                        "responsive_law_to_morph",
                    )
                )
        return evaluations, slots

    def _observe_law_effects(
        self,
        baseline: PairRecord,
        proposals: list[LawProposal],
        records: list[PairRecord | None],
        generation: int,
    ) -> None:
        for proposal, record in zip(proposals, records, strict=True):
            if record is None:
                self.navigator.record(proposal.operator, proposed=1, valid=0, improved=0)
                continue
            self._observe_effect(
                baseline,
                record,
                proposal.hypothesis,
                proposal.retrieved_ids,
                generation,
                _law_modification(baseline.structure, proposal.structure),
                proposal.operator,
            )

    def _observe_morph_effects(
        self,
        baseline: PairRecord,
        proposals: list[MorphologyProposal],
        records: list[PairRecord | None],
        generation: int,
    ) -> None:
        for proposal, record in zip(proposals, records, strict=True):
            if record is None:
                self.navigator.record(proposal.operator, proposed=1, valid=0, improved=0)
                continue
            self._observe_effect(
                baseline,
                record,
                proposal.hypothesis,
                proposal.retrieved_ids,
                generation,
                self.template.field_deltas(proposal.spec, baseline.spec),
                proposal.operator,
            )

    def _observe_effect(
        self,
        parent: PairRecord,
        offspring: PairRecord,
        hypothesis: KnowledgeHypothesis,
        retrieved_ids: Sequence[str],
        generation: int,
        modification: str,
        operator: str,
        *,
        provenance: str = "one_sided",
        interaction: float | None = None,
    ) -> str | None:
        delta = offspring.metrics.score - parent.metrics.score
        evidence = EvidenceRecord(
            hypothesis.direction,
            generation,
            _record_key(parent),
            _record_key(offspring),
            modification,
            delta,
            _metric_deltas(offspring.metrics, parent.metrics),
            _knowledge_context(self.template.knowledge_key(), parent),
            hypothesis.id,
            provenance,
            interaction,
        )
        self.knowledge.credit(retrieved_ids, delta > 1e-12)
        item_id = self.knowledge.observe(hypothesis, evidence)
        self.navigator.record(operator, proposed=1, valid=1, improved=int(delta > 1e-12))
        return item_id

    def _observe_joint(
        self, baseline: PairRecord, evaluation: _JointEvaluation, generation: int
    ) -> InteractionRecord:
        interaction = (
            evaluation.joint_record.metrics.score
            - evaluation.morph_record.metrics.score
            - evaluation.law_record.metrics.score
            + baseline.metrics.score
        )
        self._observe_effect(
            evaluation.morph_record,
            evaluation.joint_record,
            evaluation.law_proposal.hypothesis,
            evaluation.law_proposal.retrieved_ids,
            generation,
            _law_modification(evaluation.morph_record.structure, evaluation.law_proposal.structure),
            evaluation.law_proposal.operator,
            provenance=evaluation.provenance,
            interaction=interaction,
        )
        self._observe_effect(
            evaluation.law_record,
            evaluation.joint_record,
            evaluation.morph_proposal.hypothesis,
            evaluation.morph_proposal.retrieved_ids,
            generation,
            self.template.field_deltas(evaluation.morph_proposal.spec, evaluation.law_record.spec),
            evaluation.morph_proposal.operator,
            provenance=evaluation.provenance,
            interaction=interaction,
        )
        return InteractionRecord(
            generation,
            evaluation.morph_proposal.spec.to_dict(),
            evaluation.law_proposal.structure.terms,
            baseline.metrics.score,
            evaluation.morph_record.metrics.score,
            evaluation.law_record.metrics.score,
            evaluation.joint_record.metrics.score,
            interaction,
            evaluation.provenance,
        )

    @staticmethod
    def _check_count(items: Sequence[object], side: str, generation: int, expected: int) -> None:
        if len(items) != expected:
            raise ValueError(
                f"{side} generator returned {len(items)} proposals at generation "
                f"{generation}, expected {expected}"
            )


def _as_law_proposal(item: LawProposal | GymStructure) -> LawProposal:
    if isinstance(item, LawProposal):
        return item
    modification = f"select law terms {', '.join(item.terms)}"
    return LawProposal(
        item,
        KnowledgeHypothesis(
            "morph_to_law",
            f"Test whether {modification} matches the current morphology.",
            modification,
            "under a similar morphology and task",
            {"score": "increase"},
            modification,
        ),
    )


def _invoke_generator(
    generator: object,
    incumbent: PairRecord,
    knowledge: DirectedKnowledgeBase,
    count: int,
    generation: int,
    directive: SearchDirective,
    responsive: bool,
) -> Sequence[object]:
    """Support the original four-argument callback while exposing richer context."""
    parameters = signature(generator).parameters
    if len(parameters) >= 6:
        return generator(incumbent, knowledge, count, generation, directive, responsive)
    return generator(incumbent, knowledge, count, generation)


def _as_morph_proposal(item: MorphologyProposal | MorphologyGenome) -> MorphologyProposal:
    if isinstance(item, MorphologyProposal):
        return item
    modification = f"set morphology {item.describe()}"
    return MorphologyProposal(
        item,
        KnowledgeHypothesis(
            "law_to_morph",
            f"Test whether {modification} supports the current law.",
            modification,
            "under a similar law and task",
            {"score": "increase"},
            modification,
        ),
    )


def _select_joint_candidates(
    candidates: list[tuple[MorphologyProposal, LawProposal, PairRecord, PairRecord]],
    baseline_score: float,
    count: int,
    mode: str,
) -> list[tuple[MorphologyProposal, LawProposal, PairRecord, PairRecord]]:
    if count <= 0 or not candidates:
        return []

    def additive(item: tuple[MorphologyProposal, LawProposal, PairRecord, PairRecord]) -> float:
        return item[2].metrics.score + item[3].metrics.score - baseline_score

    ranked = sorted(candidates, key=lambda item: (additive(item), str(item[:2])), reverse=True)
    selected = [ranked[0]]
    remaining = ranked[1:]
    while remaining and len(selected) < count:
        if mode in ("explore", "joint_confirm"):
            next_item = max(
                remaining,
                key=lambda item: (
                    _candidate_novelty(item, selected),
                    -abs(
                        (item[2].metrics.score - baseline_score)
                        - (item[3].metrics.score - baseline_score)
                    ),
                    additive(item),
                ),
            )
        else:
            next_item = remaining[0]
        selected.append(next_item)
        remaining.remove(next_item)
    return selected


def _candidate_novelty(
    candidate: tuple[MorphologyProposal, LawProposal, PairRecord, PairRecord],
    selected: Sequence[tuple[MorphologyProposal, LawProposal, PairRecord, PairRecord]],
) -> int:
    return min(
        int(candidate[0].spec.key() != item[0].spec.key())
        + int(candidate[1].structure.key() != item[1].structure.key())
        for item in selected
    )


def _archive_diversity(records: Sequence[PairRecord]) -> tuple[float, float]:
    values = list(records)
    if not values:
        return 1.0, 1.0
    normalizer = max(1.0, math.sqrt(len(values)))
    laws = len({record.structure.key() for record in values})
    morphs = len({record.spec.key() for record in values})
    return min(1.0, laws / normalizer), min(1.0, morphs / normalizer)


def _knowledge_context(env_id: str, record: PairRecord) -> dict[str, object]:
    return {
        "task": env_id,
        "morphology": record.spec.to_dict(),
        "law_terms": list(record.structure.terms),
        "metrics": record.metrics.to_dict(),
    }


def _record_key(record: PairRecord) -> str:
    return json.dumps(
        {"law_terms": record.structure.terms, "morphology": record.spec.to_dict()},
        sort_keys=True,
    )


def _metric_deltas(child: PairMetrics, parent: PairMetrics) -> dict[str, float]:
    return {
        "score": child.score - parent.score,
        "episode_return": child.episode_return - parent.episode_return,
        "success_rate": child.success_rate - parent.success_rate,
        "energy_norm": child.energy_norm - parent.energy_norm,
        "jerk": child.jerk - parent.jerk,
        "morph_cost": child.morph_cost - parent.morph_cost,
    }


def _law_modification(parent: GymStructure, child: GymStructure) -> str:
    added = [term for term in child.terms if term not in parent.terms]
    removed = [term for term in parent.terms if term not in child.terms]
    return (
        f"law terms {list(parent.terms)} -> {list(child.terms)}; "
        f"added={added or ['none']}, removed={removed or ['none']}"
    )
