from lawevo.morplaw import topology
from lawevo.morplaw.engine import (
    MorpLawConfig,
    MorpLawGenerationReport,
    MorpLawRunner,
    PairRecord,
)
from lawevo.morplaw.evaluate import (
    PairMetrics,
    evaluate_pair,
    make_morph_env,
    morph_cost,
    pair_formula,
    tune_pair_cem,
)
from lawevo.morplaw.morphology import (
    ASSET_DIR,
    KIND_COUNT,
    KIND_GEAR,
    KIND_LENGTH,
    KIND_MASS,
    KIND_RADIUS,
    AntTemplate,
    HalfCheetahTemplate,
    HopperTemplate,
    MorphologyError,
    MorphologyField,
    MorphologySpec,
    MorphologyTemplate,
    ReacherTemplate,
    SwimmerTemplate,
    Walker2dTemplate,
)
from lawevo.morplaw.prompts import (
    extract_morphologies,
    extract_structures,
    law_mutation_prompt,
    morphology_mutation_prompt,
)
from lawevo.morplaw.topology import AntTopologyTemplate, SwimmerTopologyTemplate

TEMPLATES: dict[str, MorphologyTemplate] = {
    "walker2d": Walker2dTemplate(),
    "reacher": ReacherTemplate(),
    "hopper": HopperTemplate(),
    "half_cheetah": HalfCheetahTemplate(),
    "swimmer": SwimmerTemplate(),
    "ant": AntTemplate(),
    "swimmer_topology": topology.SwimmerTopologyTemplate(),
    "ant_topology": topology.AntTopologyTemplate(),
}

TEMPLATE_ADAPTERS: dict[str, str] = {
    "walker2d": "walker2d",
    "reacher": "reacher",
    "hopper": "hopper",
    "half_cheetah": "half_cheetah",
    "swimmer": "swimmer",
    "ant": "ant",
    "swimmer_topology": "swimmer",
    "ant_topology": "ant",
}

__all__ = [
    "ASSET_DIR",
    "KIND_COUNT",
    "KIND_GEAR",
    "KIND_LENGTH",
    "KIND_MASS",
    "KIND_RADIUS",
    "TEMPLATES",
    "TEMPLATE_ADAPTERS",
    "AntTemplate",
    "AntTopologyTemplate",
    "HalfCheetahTemplate",
    "HopperTemplate",
    "MorpLawConfig",
    "MorpLawGenerationReport",
    "MorpLawRunner",
    "MorphologyError",
    "MorphologyField",
    "MorphologySpec",
    "MorphologyTemplate",
    "PairMetrics",
    "PairRecord",
    "ReacherTemplate",
    "SwimmerTemplate",
    "SwimmerTopologyTemplate",
    "Walker2dTemplate",
    "evaluate_pair",
    "extract_morphologies",
    "extract_structures",
    "law_mutation_prompt",
    "make_morph_env",
    "morph_cost",
    "morphology_mutation_prompt",
    "pair_formula",
    "tune_pair_cem",
]
