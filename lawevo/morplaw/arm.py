from __future__ import annotations

from lawevo.morplaw.morphology import (
    ASSET_DIR,
    KIND_GEAR,
    KIND_LENGTH,
    KIND_MASS,
    KIND_RADIUS,
    MorphologyError,
    MorphologyField,
    MorphologyTemplate,
    ReacherTemplate,
    _format_number,
)
from lawevo.pid.gym_benchmark import ReacherAdapter


def _replace_required(xml: str, old: str, new: str, variant: str) -> str:
    if old not in xml:
        raise MorphologyError(f"{variant}: expected Reacher MJCF marker is missing")
    return xml.replace(old, new, 1)


class ReacherPayloadTemplate(ReacherTemplate):
    """Horizontal two-link reacher with an evolvable endpoint payload."""

    fields = (
        *ReacherTemplate.fields,
        MorphologyField("payload_radius", KIND_RADIUS, 0.018, (0.01, 0.03), "m"),
        MorphologyField("payload_density", KIND_MASS, 1500.0, (500.0, 4000.0), "kg/m^3"),
    )

    def _xml(self, values: dict[str, float]) -> str:
        xml = super()._xml(values)
        fingertip = (
            '<geom contype="0" name="fingertip" pos="0 0 0" '
            'rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>'
        )
        payload = (
            fingertip
            + "\n\t\t\t\t\t"
            + f'<geom conaffinity="0" contype="0" density="{_format_number(values["payload_density"])}" '
            + 'name="payload" pos="0 0 0" rgba="0.25 0.25 0.25 1" '
            + f'size="{_format_number(values["payload_radius"])}" type="sphere"/>'
        )
        return _replace_required(xml, fingertip, payload, "reacher_payload")


class ReacherGravityTemplate(ReacherTemplate):
    """Two-link reacher operating in a vertical x-y plane under gravity."""

    def _xml(self, values: dict[str, float]) -> str:
        xml = super()._xml(values)
        return _replace_required(
            xml,
            'gravity="0 0 -9.81"',
            'gravity="0 -9.81 0"',
            "reacher_gravity",
        )


class ReacherPrecisionTemplate(ReacherTemplate):
    """Two-link reacher with a small visual target and tighter success tolerance."""

    def _xml(self, values: dict[str, float]) -> str:
        xml = super()._xml(values)
        target = (
            '<geom conaffinity="0" contype="0" name="target" pos="0 0 0" '
            'rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>'
        )
        smaller_target = target.replace('size=".009"', 'size=".003"')
        return _replace_required(xml, target, smaller_target, "reacher_precision")


class PrecisionReacherAdapter(ReacherAdapter):
    horizon = 100
    success_tolerance = 0.02


PRECISION_REACHER_ADAPTER = PrecisionReacherAdapter()


class PusherTemplate(MorphologyTemplate):
    """Morphable 7-DoF arm for Gymnasium's planar object-pushing task."""

    env_id = "Pusher-v5"
    asset_path = ASSET_DIR / "pusher.xml"
    fields = (
        MorphologyField("upper_len", KIND_LENGTH, 0.4, (0.25, 0.55), "m"),
        MorphologyField("forearm_len", KIND_LENGTH, 0.291, (0.18, 0.4), "m"),
        MorphologyField("upper_radius", KIND_RADIUS, 0.06, (0.04, 0.08), "m"),
        MorphologyField("forearm_radius", KIND_RADIUS, 0.05, (0.03, 0.07), "m"),
        MorphologyField("arm_density", KIND_MASS, 300.0, (150.0, 600.0), "kg/m^3"),
        MorphologyField("gear", KIND_GEAR, 1.0, (0.5, 2.0), "actuator gear"),
    )

    def derived(self, values: dict[str, float]) -> dict[str, float]:
        return {"wrist_x": values["forearm_len"] + 0.03}


ARM_ADAPTERS = {"reacher_precision": PRECISION_REACHER_ADAPTER}
