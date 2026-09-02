from __future__ import annotations

import json
import re
from collections.abc import Sequence

from lawevo.morplaw.engine import PairRecord
from lawevo.morplaw.knowledge import (
    DirectedKnowledgeBase,
    KnowledgeHypothesis,
    KnowledgeItem,
)
from lawevo.morplaw.morphology import MorphologyError, MorphologyGenome, MorphologyTemplate
from lawevo.morplaw.navigator import SearchDirective
from lawevo.morplaw.proposals import LawProposal, MorphologyProposal
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
    "reacher_payload": "reacher",
    "reacher_gravity": "reacher",
    "reacher_precision": "reacher",
    "pusher": "pusher",
    "walker2d": "locomotion",
    "hopper": "locomotion",
    "half_cheetah": "locomotion",
    "swimmer": "swimmer",
    "ant": "ant",
    "swimmer_topology": "swimmer",
    "ant_topology": "ant",
    "robomorph_flat": "robomorph",
    "robomorph_ridged": "robomorph",
    "robomorph_frozen_lake": "robomorph",
    "robomorph_beams": "robomorph",
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
    knowledge: DirectedKnowledgeBase,
    retrieved: Sequence[KnowledgeItem],
    directive: SearchDirective,
    elites: Sequence[PairRecord],
    count: int,
    generation: int,
    *,
    responsive: bool = False,
) -> str:
    phase = "RESPONSIVE" if responsive else "PRIMARY"
    return f"""Generation {generation}, {phase} phase: evolve {count} compact controller
structures for the
task described below, evaluated on the CURRENT robot body. All accumulated experience below
comes from similar, compatible contexts and has downstream utility credit.

NAVIGATOR DIRECTIVE ({directive.mode})
Reason: {directive.reason}
{directive.law_guidance}

TASK AND ENVIRONMENT
{_context_lookup(task_key, "task")}

CONTROL GOAL
{_context_lookup(task_key, "goal")}

CURRENT BODY (this proposal round changes the controller only, never the body)
{incumbent.spec.describe()}
{json.dumps(incumbent.spec.to_dict(), indent=2)}

CURRENT PAIR DIAGNOSTICS
law terms = {json.dumps(list(incumbent.structure.terms))}
CEM gains = {json.dumps(incumbent.gains.tolist())}
metrics = {json.dumps(incumbent.metrics.to_dict(), sort_keys=True)}

RETRIEVED MORPHOLOGY-TO-LAW KNOWLEDGE
{knowledge.summary(retrieved)}

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

First form a falsifiable morphology-to-law hypothesis, then ground it as a controller
structure. Propose structurally diverse mutations and crossovers rather than cosmetic edits.
Return ONLY a JSON array of exactly {count} objects with this schema:
{{"name": "...", "terms": ["..."], "knowledge": {{"summary": "mechanistic rationale",
"recommendation": "actionable law design rule", "condition": "when it applies",
"prediction": {{"score": "increase", "success_rate": "non_decrease",
"energy_norm": "decrease|non_increase|unknown", "jerk": "decrease|unknown"}}}}}}
"""


def morphology_mutation_prompt(
    task_key: str,
    template: MorphologyTemplate,
    incumbent: PairRecord,
    knowledge: DirectedKnowledgeBase,
    retrieved: Sequence[KnowledgeItem],
    directive: SearchDirective,
    elites: Sequence[PairRecord],
    count: int,
    generation: int,
    *,
    responsive: bool = False,
) -> str:
    representation = template.field_descriptions()
    phase = "RESPONSIVE" if responsive else "PRIMARY"
    response_schema = {
        **template.proposal_schema(),
        "knowledge": {
            "summary": "mechanistic rationale",
            "recommendation": "actionable body design rule",
            "condition": "when it applies",
            "prediction": {
                "score": "increase",
                "success_rate": "non_decrease",
                "energy_norm": "decrease|non_increase|unknown",
                "jerk": "decrease|unknown",
            },
        },
    }
    return f"""Generation {generation}, {phase} phase: evolve {count} robot body variants for the task
described below, evaluated under the CURRENT controller law. All accumulated experience
below comes from similar, compatible laws and has downstream utility credit.

NAVIGATOR DIRECTIVE ({directive.mode})
Reason: {directive.reason}
{directive.morph_guidance}

TASK AND ENVIRONMENT
{_context_lookup(task_key, "task")}

CONTROL GOAL
{_context_lookup(task_key, "goal")}

CURRENT LAW (fixed for this proposal round; gains already tuned by CEM)
{incumbent.structure.name}: terms = {json.dumps(list(incumbent.structure.terms))}
CEM gains = {json.dumps(incumbent.gains.tolist())}
CURRENT PAIR METRICS = {json.dumps(incumbent.metrics.to_dict(), sort_keys=True)}

CURRENT BODY (mutate this graph/parameterization or cross it with an elite)
{json.dumps(incumbent.spec.to_dict(), indent=2)}

RETRIEVED LAW-TO-MORPHOLOGY KNOWLEDGE
{knowledge.summary(retrieved)}

MORPHOLOGY / TOPOLOGY PHYSICS
{_context_lookup(task_key, "morph")}

EXECUTABLE DESIGN GRAMMAR OR FIELDS
{json.dumps(representation, indent=2)}

DESIGN RULES
{template.proposal_guidance()}
Use law-conditional experience to decide which physical or structural change plausibly
helps. Every proposal must be a complete executable artifact, even when it is described as
a mutation or crossover of the incumbent and elites.

ELITE PAIRS (body, structure, CEM-tuned metrics) ranked by score:
{json.dumps(elite_payload(elites), indent=2)}

First form a falsifiable law-to-morphology hypothesis, then ground it as an executable body.
Return ONLY a JSON array of exactly {count} objects with this schema:
{json.dumps(response_schema, indent=2)}
"""


def law_knowledge_query(task_key: str, incumbent: PairRecord) -> dict[str, object]:
    return {
        "task": task_key,
        "morphology": incumbent.spec.to_dict(),
        "law_terms": list(incumbent.structure.terms),
        "metrics": incumbent.metrics.to_dict(),
    }


def morphology_knowledge_query(task_key: str, incumbent: PairRecord) -> dict[str, object]:
    return {
        "task": task_key,
        "law_terms": list(incumbent.structure.terms),
        "morphology": incumbent.spec.to_dict(),
        "metrics": incumbent.metrics.to_dict(),
    }


def _decode_payload(response: str, collection_key: str) -> list[object]:
    text = re.sub(r"```(?:json)?|```", "", response, flags=re.IGNORECASE).strip()
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get(collection_key, [payload])
    return payload if isinstance(payload, list) else []


def extract_law_proposals(
    response: str,
    allowed: tuple[str, ...],
    *,
    retrieved_ids: Sequence[str] = (),
    operator: str = "law_mutation",
) -> list[LawProposal]:
    """Parse executable law structures together with their knowledge-first hypotheses."""
    payload = _decode_payload(response, "structures")
    output: list[LawProposal] = []
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
            existing.structure.key() for existing in output
        }:
            modification = f"select law terms {', '.join(structure.terms)}"
            output.append(
                LawProposal(
                    structure,
                    _extract_hypothesis(item, "morph_to_law", modification),
                    tuple(retrieved_ids),
                    operator,
                )
            )
    return output


def extract_morphology_proposals(
    response: str,
    template: MorphologyTemplate,
    *,
    retrieved_ids: Sequence[str] = (),
    operator: str = "morph_mutation",
) -> list[MorphologyProposal]:
    """Parse morphology values together with their knowledge-first hypotheses."""
    payload = _decode_payload(response, "morphologies")
    output: list[MorphologyProposal] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            spec = template.parse_proposal(item)
        except (KeyError, TypeError, ValueError, MorphologyError):
            continue
        if spec.key() not in {existing.spec.key() for existing in output}:
            modification = "set morphology " + spec.describe()
            output.append(
                MorphologyProposal(
                    spec,
                    _extract_hypothesis(item, "law_to_morph", modification),
                    tuple(retrieved_ids),
                    operator,
                )
            )
    return output


def extract_structures(response: str, allowed: tuple[str, ...]) -> list[GymStructure]:
    """Backward-compatible artifact-only law parser."""
    return [proposal.structure for proposal in extract_law_proposals(response, allowed)]


def extract_morphologies(response: str, template: MorphologyTemplate) -> list[MorphologyGenome]:
    """Backward-compatible artifact-only morphology parser."""
    return [proposal.spec for proposal in extract_morphology_proposals(response, template)]


def _extract_hypothesis(
    item: dict[str, object], direction: str, modification: str
) -> KnowledgeHypothesis:
    raw = item.get("knowledge", {})
    knowledge = raw if isinstance(raw, dict) else {}
    prediction_raw = knowledge.get("prediction", {})
    prediction = (
        {str(key): str(value) for key, value in prediction_raw.items()}
        if isinstance(prediction_raw, dict)
        else {"score": "increase"}
    )
    if not prediction:
        prediction = {"score": "increase"}
    direction_name = "morph_to_law" if direction == "morph_to_law" else "law_to_morph"
    return KnowledgeHypothesis(
        direction_name,
        str(knowledge.get("summary", f"Test whether {modification} improves the pair.")),
        str(knowledge.get("recommendation", modification)),
        str(knowledge.get("condition", "under a similar task and parent context")),
        prediction,
        modification,
    )
