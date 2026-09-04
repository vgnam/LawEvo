"""PyBullet-native MorpLaw templates for the five harder Panda-Gym variants.

Unlike the MJCF templates, these bodies are not rendered XML: the morphology
fields are physical environment parameters (goal speed, table friction, gate
width, cube mass, stack tolerance, settle speed) that the registered variant
environments accept as ``gym.make`` keyword arguments. ``make_morph_env`` in
``lawevo.morplaw.evaluate`` detects non-MJCF templates via the
``mjcf_template`` flag and forwards the spec values instead of an XML path, so
the standard MorpLaw engine, CEM tuning, knowledge channels, and factorial
counterfactuals run unchanged.
"""

from __future__ import annotations

from lawevo.morplaw.morphology import (
    KIND_FRICTION,
    KIND_LENGTH,
    KIND_MASS,
    KIND_SPEED,
    MorphologyField,
    MorphologyGenome,
    MorphologySpec,
    MorphologyTemplate,
)
from lawevo.pid.panda_gym_variants import (
    PICK_DISTRACTOR_ENV_ID,
    PUSH_ICE_ENV_ID,
    REACH_MOVING_ENV_ID,
    SLIDE_GATE_ENV_ID,
    STACK_NARROW_ENV_ID,
)


class _PandaVariantTemplate(MorphologyTemplate):
    """Base for PyBullet variant bodies: fields are gym.make kwargs."""

    asset_path = None  # no MJCF source; the env class owns the scene
    mjcf_template = False

    def render(self, spec: MorphologyGenome) -> str:
        self.check(spec)
        assert isinstance(spec, MorphologySpec)
        return f"{self.env_id} parameters: {spec.describe()}"

    def compile(self, spec: MorphologyGenome) -> dict[str, float]:
        """Validity gate: bounded, finite fields (the env compiles itself)."""
        self.check(spec)
        assert isinstance(spec, MorphologySpec)
        return spec.to_dict()

    def total_mass(self, spec: MorphologyGenome) -> float:
        """Nominal Panda-plus-payload mass for energy normalization."""
        del spec
        return 5.0


class PandaReachMovingTemplate(_PandaVariantTemplate):
    """Reach-MovingGoal: the goal orbit speed is the design variable."""

    env_id = REACH_MOVING_ENV_ID
    fields = (
        MorphologyField("goal_speed", KIND_SPEED, 0.05, (0.02, 0.15), "m/s"),
    )


class PandaPushIceTemplate(_PandaVariantTemplate):
    """Push-IceObstacle: table friction is the design variable."""

    env_id = PUSH_ICE_ENV_ID
    fields = (
        MorphologyField("table_friction", KIND_FRICTION, 0.1, (0.02, 0.5), "coefficient"),
    )


class PandaSlideGateTemplate(_PandaVariantTemplate):
    """Slide-Gate: the gate opening is the design variable."""

    env_id = SLIDE_GATE_ENV_ID
    fields = (
        MorphologyField("gate_width", KIND_LENGTH, 0.09, (0.06, 0.20), "m"),
    )


class PandaPickDistractorTemplate(_PandaVariantTemplate):
    """Pick-HeavyDistractor: the cube mass is the design variable."""

    env_id = PICK_DISTRACTOR_ENV_ID
    fields = (
        MorphologyField("cube_mass", KIND_MASS, 1.5, (1.0, 3.0), "kg"),
    )


class PandaStackNarrowTemplate(_PandaVariantTemplate):
    """Stack-NarrowSettle: tolerance and settle speed are design variables."""

    env_id = STACK_NARROW_ENV_ID
    fields = (
        MorphologyField("distance_threshold", KIND_LENGTH, 0.025, (0.015, 0.05), "m"),
        MorphologyField("settle_speed", KIND_SPEED, 0.08, (0.03, 0.2), "m/s"),
    )
