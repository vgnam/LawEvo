from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

KnowledgeDirection = Literal["morph_to_law", "law_to_morph"]
KnowledgePolarity = Literal["positive", "negative"]

KNOWLEDGE_DIRECTIONS: tuple[KnowledgeDirection, ...] = (
    "morph_to_law",
    "law_to_morph",
)
KNOWLEDGE_MODES = ("no_knowledge", "m_to_l", "l_to_m", "full")


def mode_enables(mode: str, direction: KnowledgeDirection) -> bool:
    if mode not in KNOWLEDGE_MODES:
        raise ValueError(f"knowledge mode must be one of {KNOWLEDGE_MODES}")
    return mode == "full" or (mode == "m_to_l" and direction == "morph_to_law") or (
        mode == "l_to_m" and direction == "law_to_morph"
    )


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class KnowledgeHypothesis:
    """A falsifiable, directed design hypothesis proposed before evaluation."""

    direction: KnowledgeDirection
    summary: str
    recommendation: str
    condition: str
    prediction: dict[str, str] = field(default_factory=dict)
    modification: str = ""

    @property
    def id(self) -> str:
        return _stable_id(
            "kh",
            {
                "direction": self.direction,
                "summary": self.summary,
                "recommendation": self.recommendation,
                "condition": self.condition,
                "prediction": self.prediction,
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "direction": self.direction,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "condition": self.condition,
            "prediction": self.prediction,
            "modification": self.modification,
        }


@dataclass(frozen=True)
class EvidenceRecord:
    """Immutable parent-to-offspring evidence used to test a hypothesis."""

    direction: KnowledgeDirection
    generation: int
    parent_key: str
    offspring_key: str
    modification: str
    delta_score: float
    metric_deltas: dict[str, float]
    context: dict[str, object]
    hypothesis_id: str | None = None
    provenance: str = "one_sided"
    interaction: float | None = None

    @property
    def id(self) -> str:
        return _stable_id(
            "ev",
            {
                "direction": self.direction,
                "generation": self.generation,
                "parent_key": self.parent_key,
                "offspring_key": self.offspring_key,
                "modification": self.modification,
                "provenance": self.provenance,
                "hypothesis_id": self.hypothesis_id,
            },
        )

    @property
    def improved(self) -> bool:
        return self.delta_score > 1e-12

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "direction": self.direction,
            "generation": self.generation,
            "parent_key": self.parent_key,
            "offspring_key": self.offspring_key,
            "modification": self.modification,
            "delta_score": self.delta_score,
            "metric_deltas": self.metric_deltas,
            "context": self.context,
            "hypothesis_id": self.hypothesis_id,
            "provenance": self.provenance,
            "interaction": self.interaction,
        }


@dataclass
class KnowledgeItem:
    hypothesis: KnowledgeHypothesis
    polarity: KnowledgePolarity
    utility: float = 1.0
    uses: int = 0
    support: int = 0
    contradictions: int = 0
    last_used: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    contexts: list[dict[str, object]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.hypothesis.id}:{self.polarity}"

    @property
    def status(self) -> str:
        if self.contradictions >= 2 and self.contradictions > self.support:
            return "refuted"
        if self.support >= 2 and self.support > self.contradictions:
            return "supported"
        if self.evidence_ids:
            return "tested"
        return "proposed"

    def to_dict(self) -> dict[str, object]:
        return {
            **self.hypothesis.to_dict(),
            "knowledge_id": self.id,
            "polarity": self.polarity,
            "utility": self.utility,
            "uses": self.uses,
            "support": self.support,
            "contradictions": self.contradictions,
            "last_used": self.last_used,
            "status": self.status,
            "evidence_ids": self.evidence_ids,
            "contexts": self.contexts,
        }


class DirectedKnowledgeBase:
    """Four-bank memory: two directions times positive insights/negative pitfalls."""

    def __init__(
        self,
        mode: str = "full",
        capacity_per_bank: int = 24,
        retrieve_per_polarity: int = 3,
    ) -> None:
        if mode not in KNOWLEDGE_MODES:
            raise ValueError(f"knowledge mode must be one of {KNOWLEDGE_MODES}")
        if capacity_per_bank <= 0:
            raise ValueError("capacity_per_bank must be positive")
        if retrieve_per_polarity < 0:
            raise ValueError("retrieve_per_polarity must be non-negative")
        self.mode = mode
        self.capacity_per_bank = capacity_per_bank
        self.retrieve_per_polarity = retrieve_per_polarity
        self.items: dict[str, KnowledgeItem] = {}
        self.evidence: list[EvidenceRecord] = []

    def enabled(self, direction: KnowledgeDirection) -> bool:
        return mode_enables(self.mode, direction)

    def observe(
        self,
        hypothesis: KnowledgeHypothesis,
        evidence: EvidenceRecord,
    ) -> str | None:
        """Admit evidence and turn the tested hypothesis into an insight or pitfall."""
        if hypothesis.direction != evidence.direction:
            raise ValueError("hypothesis and evidence directions must match")
        self.evidence.append(evidence)
        if not self.enabled(hypothesis.direction):
            return None
        polarity: KnowledgePolarity = "positive" if evidence.improved else "negative"
        item_id = f"{hypothesis.id}:{polarity}"
        item = self.items.get(item_id)
        if item is None:
            item = KnowledgeItem(hypothesis=hypothesis, polarity=polarity)
            self.items[item_id] = item
        opposite_polarity = "negative" if polarity == "positive" else "positive"
        opposite = self.items.get(f"{hypothesis.id}:{opposite_polarity}")
        if opposite is not None and max(
            (
                _mapping_similarity(evidence.context, context)
                for context in opposite.contexts
            ),
            default=0.0,
        ) >= 0.9:
            opposite.contradictions += 1
            opposite.utility -= 1.0
        if evidence.id not in item.evidence_ids:
            item.evidence_ids.append(evidence.id)
            item.support += 1
            item.utility += min(1.0, abs(evidence.delta_score))
        if evidence.context not in item.contexts:
            item.contexts.append(evidence.context)
            item.contexts[:] = item.contexts[-6:]
        self._prune(hypothesis.direction, polarity)
        return item_id

    def retrieve(
        self,
        direction: KnowledgeDirection,
        query: Mapping[str, object],
        *,
        generation: int,
        per_polarity: int | None = None,
    ) -> list[KnowledgeItem]:
        per_polarity = (
            self.retrieve_per_polarity if per_polarity is None else per_polarity
        )
        if not self.enabled(direction) or per_polarity <= 0:
            return []
        selected: list[KnowledgeItem] = []
        for polarity in ("positive", "negative"):
            candidates = [
                item
                for item in self.items.values()
                if item.hypothesis.direction == direction
                and item.polarity == polarity
                and item.status != "refuted"
            ]
            ranked = sorted(
                candidates,
                key=lambda item: (
                    self._retrieval_score(item, query, generation),
                    item.id,
                ),
                reverse=True,
            )
            selected.extend(ranked[:per_polarity])
        deduplicated: dict[str, KnowledgeItem] = {}
        for item in selected:
            existing = deduplicated.get(item.hypothesis.id)
            if existing is None or self._retrieval_score(
                item, query, generation
            ) > self._retrieval_score(existing, query, generation):
                deduplicated[item.hypothesis.id] = item
        output = list(deduplicated.values())
        for item in output:
            item.uses += 1
            item.last_used = generation
        return output

    def credit(self, item_ids: Sequence[str], success: bool) -> None:
        """Credit retrieved knowledge from the downstream candidate outcome."""
        for item_id in dict.fromkeys(item_ids):
            item = self.items.get(item_id)
            if item is None:
                continue
            item.utility += 1.0 if success else -1.0
            if success:
                item.support += 1
            else:
                item.contradictions += 1

    def summary(self, items: Sequence[KnowledgeItem]) -> str:
        if not items:
            return "- No relevant validated knowledge yet."
        lines: list[str] = []
        positives = [item for item in items if item.polarity == "positive"]
        negatives = [item for item in items if item.polarity == "negative"]
        if positives:
            lines.append("[insights_to_follow]")
            lines.extend(self._render(item, avoid=False) for item in positives)
        if negatives:
            lines.append("[pitfalls_to_avoid]")
            lines.extend(self._render(item, avoid=True) for item in negatives)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        banks: dict[str, list[dict[str, object]]] = {}
        for direction in KNOWLEDGE_DIRECTIONS:
            for polarity in ("positive", "negative"):
                key = f"{direction}:{polarity}"
                banks[key] = [
                    item.to_dict()
                    for item in sorted(self.items.values(), key=lambda item: item.id)
                    if item.hypothesis.direction == direction and item.polarity == polarity
                ]
        return {
            "mode": self.mode,
            "banks": banks,
            "evidence": [record.to_dict() for record in self.evidence],
        }

    def _prune(self, direction: KnowledgeDirection, polarity: KnowledgePolarity) -> None:
        bank = [
            item
            for item in self.items.values()
            if item.hypothesis.direction == direction and item.polarity == polarity
        ]
        if len(bank) <= self.capacity_per_bank:
            return
        ranked = sorted(
            bank,
            key=lambda item: (
                item.status != "refuted",
                item.utility,
                item.support - item.contradictions,
                item.last_used,
                item.id,
            ),
            reverse=True,
        )
        for item in ranked[self.capacity_per_bank :]:
            del self.items[item.id]

    @staticmethod
    def _render(item: KnowledgeItem, *, avoid: bool) -> str:
        action = (
            f"avoid/reconsider {item.hypothesis.recommendation}"
            if avoid
            else item.hypothesis.recommendation
        )
        prediction = json.dumps(item.hypothesis.prediction, sort_keys=True)
        return (
            f"- ({item.id}, {item.status}, utility={item.utility:.2f}) "
            f"When {item.hypothesis.condition}: {action}. "
            f"Rationale: {item.hypothesis.summary}. Predicted effects: {prediction}."
        )

    @staticmethod
    def _retrieval_score(
        item: KnowledgeItem, query: Mapping[str, object], generation: int
    ) -> float:
        similarity = max(
            (_mapping_similarity(query, context) for context in item.contexts),
            default=0.0,
        )
        utility = 0.15 * math.tanh(item.utility / 4.0)
        recency = 0.05 / (1.0 + max(0, generation - item.last_used))
        return similarity + utility + recency


def _mapping_similarity(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    shared = sorted(set(left) & set(right))
    if not shared:
        return 0.0
    scores = [_value_similarity(left[key], right[key]) for key in shared]
    return sum(scores) / len(scores)


def _value_similarity(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return _mapping_similarity(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        left_set = {str(value) for value in left}
        right_set = {str(value) for value in right}
        union = left_set | right_set
        return len(left_set & right_set) / len(union) if union else 1.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        scale = max(abs(float(left)), abs(float(right)), 1e-9)
        return math.exp(-3.0 * abs(float(left) - float(right)) / scale)
    left_tokens = set(str(left).lower().split())
    right_tokens = set(str(right).lower().split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0
