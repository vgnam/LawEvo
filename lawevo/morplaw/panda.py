"""PyBullet-native MorpLaw templates for the five harder Panda-Gym variants.

Unlike the MJCF templates, these bodies are not rendered MJCF: each individual
renders a **parametric URDF** of the Franka Panda arm (arm link lengths, link
masses, base height, wrist length, motor strength) plus — for the variant
tasks — physical environment parameters (goal speed, table friction, gate
width, cube mass, stack tolerance, settle speed). ``make_morph_env`` in
``lawevo.morplaw.evaluate`` detects URDF templates via the ``mjcf_template``
flag and forwards ``urdf_path`` plus the remaining parameters as ``gym.make``
keyword arguments, so the standard MorpLaw engine, CEM tuning, knowledge
channels, and factorial counterfactuals run unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lawevo.morplaw.morphology import (
    ASSET_DIR,
    KIND_FRICTION,
    KIND_FORCE,
    KIND_LENGTH,
    KIND_MASS,
    KIND_SPEED,
    MorphologyError,
    MorphologyField,
    MorphologyGenome,
    MorphologySpec,
    MorphologyTemplate,
)

PANDA_ASSET_DIR = ASSET_DIR / "assets_panda"
URDF_TEMPLATE_PATH = PANDA_ASSET_DIR / "panda_parametric.urdf"

# Defaults equal the stock panda_gym URDF so the default spec reproduces the
# standard arm exactly.
PANDA_DEFAULTS: dict[str, float] = {
    "base_height": 0.333,
    "upper_arm_len": 0.316,
    "forearm_len": 0.384,
    "wrist_len": 0.088,
    "shoulder_offset": -0.0825,
    "mass_link1": 2.7,
    "mass_link2": 2.73,
    "mass_link3": 2.04,
    "mass_link4": 2.08,
    "mass_link5": 3.0,
    "mass_link6": 1.3,
    "mass_link7": 0.2,
    "motor_force": 1.0,
}

PANDA_ARM_FIELDS: tuple[MorphologyField, ...] = (
    MorphologyField("base_height", KIND_LENGTH, 0.333, (0.28, 0.40), "m"),
    MorphologyField("upper_arm_len", KIND_LENGTH, 0.316, (0.25, 0.42), "m"),
    MorphologyField("forearm_len", KIND_LENGTH, 0.384, (0.30, 0.48), "m"),
    MorphologyField("wrist_len", KIND_LENGTH, 0.088, (0.06, 0.13), "m"),
    MorphologyField("shoulder_offset", KIND_LENGTH, -0.0825, (-0.12, -0.05), "m"),
    MorphologyField("mass_link1", KIND_MASS, 2.7, (1.5, 4.5), "kg"),
    MorphologyField("mass_link2", KIND_MASS, 2.73, (1.5, 4.5), "kg"),
    MorphologyField("mass_link3", KIND_MASS, 2.04, (1.0, 3.5), "kg"),
    MorphologyField("mass_link4", KIND_MASS, 2.08, (1.0, 3.5), "kg"),
    MorphologyField("mass_link5", KIND_MASS, 3.0, (1.5, 5.0), "kg"),
    MorphologyField("mass_link6", KIND_MASS, 1.3, (0.6, 2.2), "kg"),
    MorphologyField("mass_link7", KIND_MASS, 0.2, (0.1, 0.6), "kg"),
    MorphologyField("motor_force", KIND_FORCE, 1.0, (0.6, 1.6), "joint-force multiplier"),
)


def render_panda_urdf(spec: MorphologyGenome) -> str:
    """Render the parametric Panda URDF with the spec's arm parameters."""
    assert isinstance(spec, MorphologySpec)
    values = {**PANDA_DEFAULTS, **spec.to_dict()}
    source = URDF_TEMPLATE_PATH.read_text(encoding="utf-8")
    xml = source
    for name, value in values.items():
        xml = xml.replace("{" + name + "}", repr(float(value)) if name != "shoulder_offset" else str(value))
    unsubstituted = [token for token in xml.split("{")[1:]]
    if any("}" in token for token in unsubstituted):
        raise MorphologyError(f"unsubstituted placeholder in rendered Panda URDF: {unsubstituted[:2]}")
    return xml


class PandaUrdfTemplate(MorphologyTemplate):
    """Base for morphology templates that render a parametric Panda URDF.

    Subclasses append task-specific environment-parameter fields (goal speed,
    friction, ...) to ``PANDA_ARM_FIELDS``; ``urdf_parameters`` separates the
    arm fields (which go into the URDF) from the environment fields (which go
    to ``gym.make``).
    """

    env_id: str = ""
    asset_path = None  # no MJCF; the URDF template lives in assets_panda
    mjcf_template = False
    urdf_template = True
    fields: tuple[MorphologyField, ...] = PANDA_ARM_FIELDS

    def urdf_parameters(self, spec: MorphologyGenome) -> dict[str, float]:
        """Arm fields that substitute into the URDF template."""
        assert isinstance(spec, MorphologySpec)
        return {name: value for name, value in spec.to_dict().items() if name in PANDA_DEFAULTS}

    def environment_parameters(self, spec: MorphologyGenome) -> dict[str, float]:
        """Remaining fields forwarded to ``gym.make`` as environment kwargs."""
        assert isinstance(spec, MorphologySpec)
        return {name: value for name, value in spec.to_dict().items() if name not in PANDA_DEFAULTS}

    def render(self, spec: MorphologyGenome) -> str:
        self.check(spec)
        return render_panda_urdf(spec)

    def compile(self, spec: MorphologyGenome):
        """Validity gate: the rendered URDF must load in a headless PyBullet."""
        import os

        import pybullet as p

        self.check(spec)
        urdf_path = self.urdf_path(spec)
        client = p.connect(p.DIRECT)
        previous_dir = os.getcwd()
        try:
            # Mesh references are relative to the URDF; resolve from its folder.
            os.chdir(urdf_path.parent)
            body = p.loadURDF(str(urdf_path), useFixedBase=True, physicsClientId=client)
            if p.getNumJoints(body, physicsClientId=client) != 12:
                raise MorphologyError(f"{self.env_id}: rendered URDF has unexpected joint count")
            positions = [
                p.getDynamicsInfo(body, i, physicsClientId=client)[0]
                for i in range(-1, p.getNumJoints(body, physicsClientId=client))
            ]
            if not all(np.isfinite(positions)):
                raise MorphologyError(f"{self.env_id}: rendered URDF has non-finite masses")
        except MorphologyError:
            raise
        except Exception as exc:
            raise MorphologyError(f"{self.env_id}: PyBullet rejected rendered URDF: {exc}") from exc
        finally:
            os.chdir(previous_dir)
            p.disconnect(physicsClientId=client)
        return urdf_path

    def urdf_path(self, spec: MorphologyGenome) -> Path:
        """Render, cache, and return the per-spec URDF file path.

        The URDF is written **next to the template** (not into the system temp
        dir) because PyBullet resolves the ``meshes/...`` references relative
        to the URDF's own folder, and the vendored ``meshes`` tree lives there.
        """
        import hashlib

        self.check(spec)
        digest = hashlib.sha256(
            json.dumps([self.cache_namespace(), spec.key()], sort_keys=True).encode()
        ).hexdigest()[:16]
        directory = URDF_TEMPLATE_PATH.parent
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"lawevo_{self.env_id}_{digest}.urdf"
        if not path.exists():
            path.write_text(self.render(spec), encoding="utf-8")
        return path

    def total_mass(self, spec: MorphologyGenome) -> float:
        """Nominal total mass for energy normalization, from the rendered URDF."""
        import re

        xml = self.render(spec)
        masses = re.findall(r'<mass value="([0-9.eE+-]+)"/>', xml)
        return float(sum(float(m) for m in masses))

    def cost(self, spec: MorphologyGenome, weight: float = 0.05) -> float:
        """Penalty for arm-shape distance from the stock Panda."""
        if not isinstance(spec, MorphologySpec):
            raise MorphologyError("URDF template requires MorphologySpec")
        total = 0.0
        for name, value in spec.to_dict().items():
            if name in PANDA_DEFAULTS:
                base = PANDA_DEFAULTS[name]
                total += abs((value - base) / abs(base))
        return weight * total


class PandaReachMovingUrdfTemplate(PandaUrdfTemplate):
    """Reach-MovingGoal: evolvable arm plus the orbit speed parameter."""

    env_id = "LawevoPandaReachMoving-v0"
    fields = PANDA_ARM_FIELDS + (
        MorphologyField("goal_speed", KIND_SPEED, 0.05, (0.02, 0.15), "m/s"),
    )


class PandaPushIceUrdfTemplate(PandaUrdfTemplate):
    """Push-IceObstacle: evolvable arm plus the table friction parameter."""

    env_id = "LawevoPandaPushIce-v0"
    fields = PANDA_ARM_FIELDS + (
        MorphologyField("table_friction", KIND_FRICTION, 0.1, (0.02, 0.5), "coefficient"),
    )


class PandaSlideGateUrdfTemplate(PandaUrdfTemplate):
    """Slide-Gate: evolvable arm plus the gate width parameter."""

    env_id = "LawevoPandaSlideGate-v0"
    fields = PANDA_ARM_FIELDS + (
        MorphologyField("gate_width", KIND_LENGTH, 0.09, (0.06, 0.20), "m"),
    )


class PandaPickDistractorUrdfTemplate(PandaUrdfTemplate):
    """Pick-HeavyDistractor: evolvable arm plus the cube mass parameter."""

    env_id = "LawevoPandaPickDistractor-v0"
    fields = PANDA_ARM_FIELDS + (
        MorphologyField("cube_mass", KIND_MASS, 1.5, (1.0, 3.0), "kg"),
    )


class PandaStackNarrowUrdfTemplate(PandaUrdfTemplate):
    """Stack-NarrowSettle: evolvable arm plus tolerance and settle parameters."""

    env_id = "LawevoPandaStackNarrow-v0"
    fields = PANDA_ARM_FIELDS + (
        MorphologyField("distance_threshold", KIND_LENGTH, 0.025, (0.015, 0.05), "m"),
        MorphologyField("settle_speed", KIND_SPEED, 0.08, (0.03, 0.2), "m/s"),
    )


class PandaReachUrdfTemplate(PandaUrdfTemplate):
    """Stock Reach with the evolvable Panda arm (no extra environment field)."""

    env_id = "LawevoPandaReach-v1"


class PandaPushUrdfTemplate(PandaUrdfTemplate):
    """Stock Push with the evolvable Panda arm."""

    env_id = "LawevoPandaPush-v1"


class PandaSlideUrdfTemplate(PandaUrdfTemplate):
    """Stock Slide with the evolvable Panda arm."""

    env_id = "LawevoPandaSlide-v1"


class PandaPickAndPlaceUrdfTemplate(PandaUrdfTemplate):
    """Stock PickAndPlace with the evolvable Panda arm."""

    env_id = "LawevoPandaPickAndPlace-v1"


class PandaStackUrdfTemplate(PandaUrdfTemplate):
    """Stock Stack with the evolvable Panda arm."""

    env_id = "LawevoPandaStack-v1"
