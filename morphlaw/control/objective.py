"""Fixed-scale secondary utility; success and success gap are ranked separately."""

import math
from dataclasses import asdict, dataclass

PROTOCOL_VERSION = "success-first-typed-sg-v2"


@dataclass(frozen=True)
class ObjectiveConfig:
    return_scale: float = 10.0
    effort_scale: float = 1.0
    slew_scale: float = 100.0
    complexity_scale: float = 16.0
    weights: tuple[float, float, float, float] = (0.60, 0.20, 0.15, 0.05)

    def __post_init__(self):
        scales = (self.return_scale, self.effort_scale, self.slew_scale, self.complexity_scale)
        if any(not math.isfinite(x) or x <= 0 for x in scales):
            raise ValueError("objective scales must be positive and finite")
        if len(self.weights) != 4 or any(not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError("four nonnegative finite weights are required")
        if not math.isclose(sum(self.weights), 1.0):
            raise ValueError("objective weights must sum to one")

    def score(self, episode_return, effort, slew, complexity):
        if any(not math.isfinite(x) for x in (episode_return, effort, slew, complexity)):
            return -math.inf
        if min(effort, slew, complexity) < 0:
            return -math.inf
        r = 0.5 + math.atan(episode_return / self.return_scale) / math.pi
        e = effort / (effort + self.effort_scale)
        j = slew / (slew + self.slew_scale)
        c = complexity / (complexity + self.complexity_scale)
        wr, we, wj, wc = self.weights
        return wr * r - we * e - wj * j - wc * c

    def to_dict(self):
        return {
            **asdict(self),
            "protocol_version": PROTOCOL_VERSION,
            "rank_order": ["success_rate descending", "sg ascending when available",
                           "secondary_score descending"],
            "normalization": "R: 0.5+atan(R/scale)/pi; costs: x/(x+scale)",
            "energy_semantics": "integrated squared action (command effort, not joules)",
            "jerk_semantics": "integrated squared action slew (not mechanical jerk)",
        }


def objective_for(env_id):
    # Fixed references shared by each pair, never fitted to the evolving archive.
    if "Reacher" in env_id:
        return ObjectiveConfig(5.0, 1.0, 100.0)
    if "PandaReach" in env_id:
        return ObjectiveConfig(7.5, 1.0, 100.0)
    if "PandaPush" in env_id or "PandaSlide" in env_id:
        return ObjectiveConfig(10.0, 1.0, 100.0)
    return ObjectiveConfig()
