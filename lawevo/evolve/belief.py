from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Experience:
    category: str
    statement: str
    score: float = 0.0


@dataclass
class BeliefSpace:
    primitive: list[str] = field(default_factory=list)
    failure: list[str] = field(default_factory=list)
    code_idiom: list[str] = field(default_factory=list)
    max_items_per_category: int = 12

    def update(self, experiences: list[Experience]) -> None:
        """Bounded deterministic consolidation fallback.

        A caller may replace this with an LLM consolidator. Exact duplicates are
        merged, newer high-score statements win, and memory remains bounded.
        """
        groups = {
            "primitive": self.primitive,
            "failure": self.failure,
            "code_idiom": self.code_idiom,
        }
        for category, target in groups.items():
            ranked = sorted(
                (item for item in experiences if item.category == category),
                key=lambda item: item.score,
                reverse=True,
            )
            merged = list(dict.fromkeys([item.statement for item in ranked] + target))
            target[:] = merged[: self.max_items_per_category]

    def summary(self, categories: tuple[str, ...] | None = None) -> str:
        selected = categories or ("primitive", "failure", "code_idiom")
        lines: list[str] = []
        for category in selected:
            values = getattr(self, category)
            lines.append(f"[{category}]")
            lines.extend(f"- {value}" for value in values)
            if not values:
                lines.append("- No consolidated experience yet.")
        return "\n".join(lines)
