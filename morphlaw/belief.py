from __future__ import annotations

from dataclasses import dataclass, field

STRUCTURED_CATEGORIES = ("morph_to_law", "law_to_morph")


@dataclass(frozen=True)
class Experience:
    category: str
    statement: str
    score: float = 0.0
    context: dict[str, str] | None = None
    hypothesis: bool = False


@dataclass
class BeliefSpace:
    primitive: list[str] = field(default_factory=list)
    failure: list[str] = field(default_factory=list)
    code_idiom: list[str] = field(default_factory=list)
    morph_to_law: list[Experience] = field(default_factory=list)
    law_to_morph: list[Experience] = field(default_factory=list)
    max_items_per_category: int = 12

    def update(self, experiences: list[Experience]) -> None:
        """Bounded deterministic consolidation fallback.

        A caller may replace this with an LLM consolidator. Exact duplicates are
        merged, newer high-score statements win, and memory remains bounded.
        The legacy string categories keep their historical behavior; the two
        structured cross-channel categories store Experiences with context.
        """
        string_groups = {
            "primitive": self.primitive,
            "failure": self.failure,
            "code_idiom": self.code_idiom,
        }
        for category, target in string_groups.items():
            ranked = sorted(
                (item for item in experiences if item.category == category),
                key=lambda item: item.score,
                reverse=True,
            )
            merged = list(dict.fromkeys([item.statement for item in ranked] + target))
            target[:] = merged[: self.max_items_per_category]
        structured_groups = {
            "morph_to_law": self.morph_to_law,
            "law_to_morph": self.law_to_morph,
        }
        for category, target in structured_groups.items():
            ranked = sorted(
                (item for item in experiences if item.category == category),
                key=lambda item: item.score,
                reverse=True,
            )
            seen: set[tuple[str, tuple[tuple[str, str], ...] | None]] = set()
            merged: list[Experience] = []
            for item in [*ranked, *target]:
                context_key = (
                    tuple(sorted(item.context.items())) if item.context is not None else None
                )
                dedup_key = (item.statement, context_key)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                merged.append(item)
            target[:] = merged[: self.max_items_per_category]

    def summary(
        self,
        categories: tuple[str, ...] | None = None,
        context_match: dict[str, str] | None = None,
    ) -> str:
        """Render selected categories; `context_match` keeps only Experiences whose
        context contains all given key-value pairs (stale-knowledge filtering)."""
        selected = categories or ("primitive", "failure", "code_idiom")
        lines: list[str] = []
        for category in selected:
            lines.append(f"[{category}]")
            if category in STRUCTURED_CATEGORIES:
                values = getattr(self, category)
                if context_match is not None:
                    values = [
                        item
                        for item in values
                        if item.context is not None
                        and all(item.context.get(key) == value for key, value in context_match.items())
                    ]
                if not values:
                    lines.append("- No consolidated experience yet.")
                    continue
                for item in values:
                    prefix = "[hypothesis] " if item.hypothesis else ""
                    lines.append(f"- {prefix}{item.statement}")
            else:
                values = getattr(self, category)
                if not values:
                    lines.append("- No consolidated experience yet.")
                    continue
                lines.extend(f"- {value}" for value in values)
        return "\n".join(lines)
