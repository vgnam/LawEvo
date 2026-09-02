from __future__ import annotations

import json
import re
from collections.abc import Sequence

from lawevo.evolve.belief import BeliefSpace
from lawevo.morplaw.engine import PairRecord
from lawevo.morplaw.morphology import MorphologyError, MorphologySpec, MorphologyTemplate
from lawevo.morplaw.tasks import (
    CONTROL_GOALS,
    EFFICIENCY_GUIDANCE,
    MORPHOLOGY_GUIDANCE,
    TASK_DESCRIPTIONS,
    TERM_SEMANTICS,
)
from lawevo.pid.gym_benchmark import BenchmarkAdapter, GymStructure

_TERM_KEYS = {
    "reacher": "reacher",
    "walker2d": "locomotion",
    "hopper": "locomotion",
    "half_cheetah": "locomotion",
    "swimmer": "swimmer",
    "ant": "ant",
    "swimmer_topology": "swimmer",
    "ant_topology": "ant",
}


def elite_payload(records: Sequence[PairRecord]) -> list[dict[str, object]]:
    return [
        {
            "spec": record.spec.to_dict(),
            "structure": record.structure.to_dict(),
            "gains": record.gains.tolist(),
            "metrics": record.metrics.to_dict(),
        }
        for record in records
    ]


def _context_lookup(task_key: str, kind: str) -> str:
    if kind == "terms":
        return TERM_SEMANTICS[_TERM_KEYS[task_key]]
    return {
        "task": TASK_DESCRIPTIONS,
        "goal": CONTROL_GOALS,
        "morph": MORPHOLOGY_GUIDANCE,
    }[kind][task_key]


def law_mutation_prompt(
    task_key: str,
    adapter: BenchmarkAdapter,
    incumbent: PairRecord,
    belief: BeliefSpace,
    elites: Sequence[PairRecord],
    count: int,
    generation: int,
) -> str:
    morph_context = json.dumps(incumbent.spec.to_dict(), sort_keys=True)
    return f"""Generation {generation}: evolve {count} compact controller structures for the
task described below, evaluated on the CURRENT robot body. All accumulated experience below
was measured on this exact body, so it transfers to your proposals.

TASK AND ENVIRONMENT
{_context_lookup(task_key, "task")}

CONTROL GOAL
{_context_lookup(task_key, "goal")}

CURRENT BODY (this proposal round changes the controller only, never the body)
{incumbent.spec.describe()}

ACCUMULATED MORPHOLOGY-CONDITIONAL EXPERIENCE (measured on this exact body)
{belief.summary(("morph_to_law",), context_match={"morphology": morph_context})}

AVAILABLE SIGNALS
{_context_lookup(task_key, "terms")}

The controller is a weighted sum of the selected signals. Do NOT propose numeric gains:
every gain K is tuned afterward by an equal-budget Cross-Entropy Method; you select only
the signal terms, 1-8 unique terms from the allowed list. Vector-valued terms combine
componentwise (one scalar gain per term), so the same structure applies to any actuator
count. Add a term only when the task, the body geometry, or the experience above suggests
the signal helps on this specific body.

Allowed term names: {json.dumps(list(adapter.allowed_terms))}

{EFFICIENCY_GUIDANCE}

ELITE PAIRS (body, structure, CEM-tuned metrics) ranked by score:
{json.dumps(elite_payload(elites), indent=2)}

Propose structurally diverse mutations and crossovers rather than cosmetic edits. Return
ONLY a JSON array of exactly {count} objects with keys "name" and "terms".
"""


def morphology_mutation_prompt(
    task_key: str,
    template: MorphologyTemplate,
    incumbent: PairRecord,
    belief: BeliefSpace,
    elites: Sequence[PairRecord],
    count: int,
    generation: int,
) -> str:
    law_context = json.dumps(list(incumbent.structure.terms))
    fields = template.field_descriptions()
    field_names = [field["name"] for field in fields]
    if template.has_counts():
        topology_note = (
            "Topology fields (kind 'count') change the number of joints, actuators, and "
            "therefore the observation/action dimensions. The controller law is "
            "dimension-agnostic (one scalar gain per term), so the same law structure "
            "applies to every proposed body; CEM re-tunes the gains per pair. Count "
            "values must be integers inside the declared bounds."
        )
    else:
        topology_note = (
            "The joint count and the observation/action sizes must stay fixed — never "
            "propose topology changes."
        )
    return f"""Generation {generation}: evolve {count} robot body variants for the task
described below, evaluated under the CURRENT controller law. All accumulated experience
below used this exact law with its tuned gains, so it transfers to your proposals.

TASK AND ENVIRONMENT
{_context_lookup(task_key, "task")}

CONTROL GOAL
{_context_lookup(task_key, "goal")}

CURRENT LAW (fixed for this proposal round; gains already tuned by CEM)
{incumbent.structure.name}: terms = {json.dumps(list(incumbent.structure.terms))}

ACCUMULATED LAW-CONDITIONAL EXPERIENCE (hypotheses measured under this exact law)
{belief.summary(("law_to_morph",), context_match={"structure": law_context})}

MORPHOLOGY FIELD PHYSICS
{_context_lookup(task_key, "morph")}

Morphology fields with allowed ranges (default = the original body):
{json.dumps(fields, indent=2)}

Rules: every value must stay inside its bounds. Prefer small, physically motivated changes,
varying one or two fields per proposal unless experience suggests otherwise. Use the
law-conditional experience to decide which change plausibly helps (e.g., raise gear when
the energy cost is the bottleneck, lengthen or shorten limbs when the law oscillates or
cannot keep pace). Coupled geometry is handled automatically (child bodies and joints move
with the changed segment). {topology_note}

ELITE PAIRS (body, structure, CEM-tuned metrics) ranked by score:
{json.dumps(elite_payload(elites), indent=2)}

Return ONLY a JSON array of exactly {count} objects, each with ALL of these keys:
{json.dumps(field_names)}
"""


def extract_structures(response: str, allowed: tuple[str, ...]) -> list[GymStructure]:
    """Parse LLM output into validated GymStructures; mirrors the benchmark extractor."""
    text = re.sub(r"```(?:json)?|```", "", response, flags=re.IGNORECASE).strip()
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("structures", [payload])
    output: list[GymStructure] = []
    if not isinstance(payload, list):
        return output
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        try:
            structure = GymStructure(
                str(item.get("name", f"proposal_{index}"))[:50], tuple(item["terms"])
            )
        except (KeyError, TypeError, ValueError):
            continue
        if set(structure.terms) <= set(allowed) and structure.key() not in {
            existing.key() for existing in output
        }:
            output.append(structure)
    return output


def extract_morphologies(response: str, template: MorphologyTemplate) -> list[MorphologySpec]:
    """Parse LLM output into validated MorphologySpecs inside template bounds."""
    text = re.sub(r"```(?:json)?|```", "", response, flags=re.IGNORECASE).strip()
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("morphologies", [payload])
    output: list[MorphologySpec] = []
    if not isinstance(payload, list):
        return output
    field_names = [field.name for field in template.fields]
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            spec = MorphologySpec.of({name: float(item[name]) for name in field_names})
        except (KeyError, TypeError, ValueError, MorphologyError):
            continue
        if not template.validate(spec) and spec.key() not in {
            existing.key() for existing in output
        }:
            output.append(spec)
    return output
