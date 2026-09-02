from __future__ import annotations

from dataclasses import dataclass

from lawevo.morplaw.knowledge import KnowledgeHypothesis
from lawevo.morplaw.morphology import MorphologySpec
from lawevo.pid.gym_benchmark import GymStructure


@dataclass(frozen=True)
class LawProposal:
    structure: GymStructure
    hypothesis: KnowledgeHypothesis
    retrieved_ids: tuple[str, ...] = ()
    operator: str = "law_mutation"


@dataclass(frozen=True)
class MorphologyProposal:
    spec: MorphologySpec
    hypothesis: KnowledgeHypothesis
    retrieved_ids: tuple[str, ...] = ()
    operator: str = "morph_mutation"
