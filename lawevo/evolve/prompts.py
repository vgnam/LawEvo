from __future__ import annotations

import json
from collections.abc import Sequence

from lawevo.evolve.belief import BeliefSpace


def policy_mutation_prompt(
    belief: BeliefSpace,
    parents: Sequence[dict[str, object]],
    latest_failure: str = "none",
) -> str:
    return f"""You evolve a nominal robot controller. Safety is handled separately by a CBF filter.

ACCUMULATED EXPERIENCE
{belief.summary(("code_idiom",))}

PARENTS (source and measured metrics)
{json.dumps(list(parents), indent=2, default=str)}

LATEST FAILURE MODE
{latest_failure}

Return exactly two sections:
1. A short CHANGE explanation.
2. One Python function named pi_nominal(state, goal, obstacles) returning a finite numpy
   vector with the same control dimension. Do not change the signature, import modules,
   access files/network, or include top-level executable code.
"""


def barrier_mutation_prompt(
    belief: BeliefSpace,
    parent_barriers: Sequence[dict[str, object]],
    available_primitives: Sequence[str],
    latest_rejection: str = "none",
) -> str:
    return f"""You evolve a control-barrier expression for a robot.

ACCUMULATED EXPERIENCE
{belief.summary(("primitive", "failure"))}

PARENT BARRIER TREES AND METRICS
{json.dumps(list(parent_barriers), indent=2, default=str)}

LATEST VERIFIER REJECTION
{latest_rejection}

Allowed primitives for this robot: {", ".join(available_primitives)}.
Return one JSON object only, with no markdown fence. It must have one of these shapes:
{{"op":"min","terms":[{{"primitive":"dist_to_obstacle","args":[0,0.3]}}]}}
{{"op":"wsum","terms":[{{"weight":0.5,"term":{{"primitive":"...","args":[]}}}}]}}
Weights must be in (0,1], margins must be non-negative, and every primitive must be in
the allowed list. Output is parser-checked before verification.
"""
