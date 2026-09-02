from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OperatorStats:
    attempts: int = 0
    proposed: int = 0
    valid: int = 0
    improved: int = 0

    @property
    def validity_rate(self) -> float:
        return self.valid / self.proposed if self.proposed else 0.0

    @property
    def improvement_rate(self) -> float:
        return self.improved / self.valid if self.valid else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "attempts": self.attempts,
            "proposed": self.proposed,
            "valid": self.valid,
            "improved": self.improved,
            "validity_rate": self.validity_rate,
            "improvement_rate": self.improvement_rate,
        }


@dataclass(frozen=True)
class SearchDirective:
    generation: int
    mode: str
    reason: str
    law_guidance: str
    morph_guidance: str
    joint_guidance: str

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "mode": self.mode,
            "reason": self.reason,
            "law_guidance": self.law_guidance,
            "morph_guidance": self.morph_guidance,
            "joint_guidance": self.joint_guidance,
        }


@dataclass
class MorpLawNavigator:
    """Transparent state-aware controller shared by all knowledge ablations."""

    stagnation_threshold: int = 2
    diversity_threshold: float = 0.5
    interaction_threshold: float = 0.1
    operator_stats: dict[str, OperatorStats] = field(default_factory=dict)

    def decide(
        self,
        generation: int,
        best_history: list[float],
        *,
        law_diversity: float,
        morph_diversity: float,
        last_max_abs_interaction: float = 0.0,
    ) -> SearchDirective:
        stagnation = _stagnation_count(best_history)
        recent_progress = _recent_progress(best_history)
        min_diversity = min(law_diversity, morph_diversity)
        if last_max_abs_interaction >= self.interaction_threshold:
            mode = "joint_confirm"
            reason = (
                f"large morphology-law interaction ({last_max_abs_interaction:.3g}) needs "
                "counterfactual confirmation"
            )
        elif stagnation >= self.stagnation_threshold or min_diversity < self.diversity_threshold:
            mode = "explore"
            reason = (
                f"stagnation={stagnation}, law_diversity={law_diversity:.2f}, "
                f"morph_diversity={morph_diversity:.2f}"
            )
        elif recent_progress >= 2:
            mode = "exploit"
            reason = f"best score improved for {recent_progress} consecutive generations"
        else:
            mode = "balance"
            reason = "search is progressing without a confirmed dominant interaction"
        guidance = _guidance(mode)
        return SearchDirective(generation, mode, reason, *guidance)

    def record(
        self,
        operator: str,
        *,
        proposed: int,
        valid: int,
        improved: int,
    ) -> None:
        stats = self.operator_stats.setdefault(operator, OperatorStats())
        stats.attempts += 1
        stats.proposed += proposed
        stats.valid += valid
        stats.improved += improved

    def stats_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            name: stats.to_dict() for name, stats in sorted(self.operator_stats.items())
        }


def _stagnation_count(history: list[float]) -> int:
    if len(history) < 2:
        return 0
    count = 0
    for newer, older in zip(reversed(history[1:]), reversed(history[:-1]), strict=True):
        if newer > older + 1e-9:
            break
        count += 1
    return count


def _recent_progress(history: list[float]) -> int:
    if len(history) < 2:
        return 0
    count = 0
    for newer, older in zip(reversed(history[1:]), reversed(history[:-1]), strict=True):
        if newer <= older + 1e-9:
            break
        count += 1
    return count


def _guidance(mode: str) -> tuple[str, str, str]:
    if mode == "explore":
        return (
            "Prefer structurally different signal motifs; do not rename an existing law.",
            "Probe physically distinct fields or topology while keeping each edit interpretable.",
            "Favor novel cross-combinations and hypotheses with little evidence.",
        )
    if mode == "exploit":
        return (
            "Refine or simplify proven law motifs while preserving their useful mechanism.",
            "Make small parent-relative body changes along empirically successful directions.",
            "Pair the strongest compatible one-sided improvements.",
        )
    if mode == "joint_confirm":
        return (
            "Adapt the law explicitly to the selected body's changed mechanics.",
            "Adapt the body explicitly to the selected law's observed failure mode.",
            "Generate a falsifiable counterfactual for the observed interaction.",
        )
    return (
        "Mix one novel law motif with evidence-guided refinements.",
        "Mix one physical probe with evidence-guided local morphology changes.",
        "Balance predicted fitness, diversity, and hypothesis confirmation.",
    )
