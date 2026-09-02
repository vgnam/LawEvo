from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"


def _format_number(value: float) -> str:
    """Format a float for MJCF; MuJoCo rejects integer-looking float attributes."""
    text = f"{value:.8g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


KIND_LENGTH = "length"
KIND_MASS = "mass"
KIND_RADIUS = "radius"
KIND_GEAR = "gear"
KIND_COUNT = "count"
VALID_KINDS = (KIND_LENGTH, KIND_MASS, KIND_RADIUS, KIND_GEAR, KIND_COUNT)


class MorphologyError(ValueError):
    """Raised when a morphology spec is invalid or its MJCF model fails to compile."""


class MorphologyGenome(Protocol):
    """Common contract for parametric and grammar-native robot bodies."""

    spec_type: ClassVar[str]

    def to_dict(self) -> dict[str, object]: ...

    def key(self) -> Hashable: ...

    def describe(self) -> str: ...


@dataclass(frozen=True)
class MorphologyField:
    name: str
    kind: str
    default: float
    bounds: tuple[float, float]
    unit: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"unknown morphology field kind {self.kind!r}")
        low, high = self.bounds
        if not (low < high):
            raise ValueError(f"field {self.name!r} bounds must satisfy low < high")
        if not np.isfinite(self.default) or not (low <= self.default <= high):
            raise ValueError(f"field {self.name!r} default {self.default} outside {self.bounds}")
        if self.kind == KIND_COUNT and not self.default.is_integer():
            raise ValueError(f"count field {self.name!r} default must be integral")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "bounds": list(self.bounds),
            "unit": self.unit,
        }


@dataclass(frozen=True)
class MorphologySpec:
    """Immutable, canonical (name-sorted) set of morphology field values."""

    values: tuple[tuple[str, float], ...]
    spec_type: ClassVar[str] = "parameters"

    def __post_init__(self) -> None:
        canonical = tuple(sorted((str(name), float(value)) for name, value in self.values))
        if any(not np.isfinite(value) for _, value in canonical):
            raise MorphologyError("morphology values must be finite")
        object.__setattr__(self, "values", canonical)

    @classmethod
    def of(cls, values: Mapping[str, float]) -> MorphologySpec:
        return cls(tuple((name, value) for name, value in values.items()))

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)

    def get(self, name: str) -> float:
        for field_name, value in self.values:
            if field_name == name:
                return value
        raise KeyError(name)

    def key(self) -> tuple[tuple[str, float], ...]:
        return self.values

    def describe(self) -> str:
        return (
            "{"
            + ", ".join(
                f"{name}={value:.0f}" if value.is_integer() else f"{name}={value:.6g}"
                for name, value in self.values
            )
            + "}"
        )


class MorphologyTemplate:
    """Base class mapping a MorphologySpec onto a parameterized MJCF asset.

    Subclasses declare `fields` and optionally `derived` placeholders whose values
    depend on multiple primitive fields (coupled geometry). `render` substitutes both
    primitive and derived values; `compile` is the MuJoCo validity gate.
    """

    env_id: str = ""
    asset_path: Path | None = ASSET_DIR
    fields: tuple[MorphologyField, ...] = ()

    def __init__(self) -> None:
        if not self.env_id:
            raise ValueError("template must declare an env_id")
        self._field_map: dict[str, MorphologyField] = {field.name: field for field in self.fields}
        self._source = (
            self.asset_path.read_text(encoding="utf-8") if self.asset_path is not None else ""
        )
        self._mass_cache: dict[Hashable, float] = {}

    def cache_namespace(self) -> str:
        """Identity of the compiler/environment contract used by disk/temp caches."""
        return f"{type(self).__module__}.{type(self).__qualname__}:{self.env_id}"

    def knowledge_key(self) -> str:
        """Task context stored alongside directed morphology/law evidence."""
        return self.env_id

    def defaults(self) -> dict[str, float]:
        return {field.name: field.default for field in self.fields}

    def default_spec(self) -> MorphologySpec:
        return MorphologySpec.of(self.defaults())

    def seed_specs(self, count: int = 1, seed: int = 0) -> tuple[MorphologyGenome, ...]:
        """Initial best-shot bodies; parametric templates keep their original body."""
        del count, seed
        return (self.default_spec(),)

    def field_descriptions(self) -> list[dict[str, object]]:
        return [field.to_dict() for field in self.fields]

    def field_kinds(self) -> dict[str, str]:
        return {field.name: field.kind for field in self.fields}

    def proposal_schema(self) -> dict[str, object]:
        """Executable artifact schema requested from the morphology generator."""
        return {"values": {field.name: "number" for field in self.fields}}

    def proposal_guidance(self) -> str:
        if self.has_counts():
            topology_note = (
                "Topology fields (kind 'count') change the number of joints, actuators, and "
                "therefore the observation/action dimensions. The controller law is "
                "dimension-agnostic (one scalar gain per term), so the same law structure "
                "applies to every proposed body; CEM re-tunes the gains per pair. Count "
                "values must be integers inside the declared bounds."
            )
        else:
            topology_note = (
                "The joint count and the observation/action sizes must stay fixed; never "
                "propose topology changes."
            )
        return (
            "Every value must stay inside its bounds. Prefer small, physically motivated "
            "changes, varying one or two fields per proposal unless experience suggests "
            "otherwise. Coupled geometry is handled automatically. " + topology_note
        )

    def parse_proposal(self, payload: Mapping[str, object]) -> MorphologySpec:
        """Turn one LLM artifact into a validated parametric morphology."""
        raw = payload.get("values", payload)
        if not isinstance(raw, Mapping):
            raise MorphologyError("morphology proposal must contain a values mapping")
        try:
            spec = MorphologySpec.of({field.name: float(raw[field.name]) for field in self.fields})
        except (KeyError, TypeError, ValueError) as exc:
            raise MorphologyError(f"invalid morphology values: {exc}") from exc
        self.check(spec)
        return spec

    def cost(self, spec: MorphologyGenome, weight: float = 0.05) -> float:
        """Penalty for parametric distance from the original body."""
        if not isinstance(spec, MorphologySpec):
            raise MorphologyError("parametric template requires MorphologySpec")
        defaults = self.defaults()
        kinds = self.field_kinds()
        total = 0.0
        for name, value in spec.values:
            base = defaults[name]
            if kinds[name] == KIND_COUNT:
                total += abs(value - base)
            else:
                total += abs((value - base) / base)
        return weight * total

    def has_counts(self) -> bool:
        return any(field.kind == KIND_COUNT for field in self.fields)

    def validate(self, spec: MorphologyGenome) -> list[str]:
        if not isinstance(spec, MorphologySpec):
            return ["parametric template requires MorphologySpec"]
        errors: list[str] = []
        known = set(self._field_map)
        seen: set[str] = set()
        for name, value in spec.values:
            if name in seen:
                errors.append(f"duplicate field {name!r}")
                continue
            seen.add(name)
            field = self._field_map.get(name)
            if field is None:
                errors.append(f"unknown field {name!r} for {self.env_id}")
                continue
            low, high = field.bounds
            if not np.isfinite(value) or not (low <= value <= high):
                errors.append(f"field {name!r} value {value} outside bounds [{low}, {high}]")
            if field.kind == KIND_COUNT and not float(value).is_integer():
                errors.append(f"count field {name!r} value {value} is not integral")
        missing = known - seen
        if missing:
            errors.append(f"missing fields for {self.env_id}: {sorted(missing)}")
        return errors

    def check(self, spec: MorphologyGenome) -> None:
        errors = self.validate(spec)
        if errors:
            raise MorphologyError("; ".join(errors))

    def derived(self, values: dict[str, float]) -> dict[str, float]:
        """Environment-specific coupled values; overridden by subclasses."""
        return {}

    def render(self, spec: MorphologyGenome) -> str:
        self.check(spec)
        assert isinstance(spec, MorphologySpec)
        xml = self._xml(spec.to_dict())
        if "{" in xml or "}" in xml:
            raise MorphologyError(f"{self.env_id}: unsubstituted placeholder in rendered XML")
        return xml

    def _xml(self, values: dict[str, float]) -> str:
        """Default source-substitution path; topology templates override this hook."""
        substitutions = {
            name: _format_number(value)
            for name, value in {**values, **self.derived(values)}.items()
        }
        try:
            return self._source.format_map(substitutions)
        except (KeyError, ValueError) as exc:
            raise MorphologyError(f"{self.env_id}: template substitution failed: {exc}") from exc

    def compile(self, spec: MorphologyGenome):
        """Compile the rendered MJCF; the primary morphology validity gate."""
        import mujoco

        self.check(spec)
        xml = self.render(spec)
        try:
            model = mujoco.MjModel.from_xml_string(xml)
        except Exception as exc:  # MuJoCo raises its own error types
            raise MorphologyError(f"{self.env_id}: MuJoCo rejected rendered XML: {exc}") from exc
        data = mujoco.MjData(model)
        try:
            mujoco.mj_forward(model, data)
        except Exception as exc:
            raise MorphologyError(f"{self.env_id}: MuJoCo forward failed: {exc}") from exc
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            raise MorphologyError(f"{self.env_id}: compiled model has non-finite state")
        return model

    def total_mass(self, spec: MorphologyGenome) -> float:
        key = spec.key()
        cached = self._mass_cache.get(key)
        if cached is not None:
            return cached
        mass = float(self.compile(spec).body_mass.sum())
        self._mass_cache[key] = mass
        return mass

    def xml_path(self, spec: MorphologyGenome) -> Path:
        """Write the rendered XML to a cached temp file for gym.make(xml_file=...)."""
        self.check(spec)
        digest = hashlib.sha256(
            json.dumps([self.cache_namespace(), spec.key()], sort_keys=True).encode()
        ).hexdigest()[:16]
        directory = Path(tempfile.gettempdir()) / "lawevo_morplaw_xml"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.env_id}_{digest}.xml"
        if not path.exists():
            path.write_text(self.render(spec), encoding="utf-8")
        return path

    def field_deltas(self, spec: MorphologyGenome, baseline: MorphologyGenome | None = None) -> str:
        """Change of each field against the actual parent (or default if omitted)."""
        if not isinstance(spec, MorphologySpec):
            raise MorphologyError("parametric template requires MorphologySpec")
        if baseline is not None and not isinstance(baseline, MorphologySpec):
            raise MorphologyError("parametric baseline requires MorphologySpec")
        defaults = baseline.to_dict() if baseline is not None else self.defaults()
        kinds = self.field_kinds()
        parts = []
        for name, value in spec.values:
            base = defaults[name]
            if kinds[name] == KIND_COUNT:
                parts.append(f"{name} {base:.0f}->{value:.0f}")
            else:
                relative = (value - base) / base
                parts.append(f"{name} {base:.6g}->{value:.6g} ({relative:+.1%})")
        return ", ".join(parts)


class Walker2dTemplate(MorphologyTemplate):
    env_id = "Walker2d-v5"
    asset_path = ASSET_DIR / "walker2d_v5.xml"
    fields = (
        MorphologyField("thigh_len", KIND_LENGTH, 0.225, (0.135, 0.315), "m"),
        MorphologyField("leg_len", KIND_LENGTH, 0.25, (0.15, 0.35), "m"),
        MorphologyField("foot_len", KIND_LENGTH, 0.1, (0.06, 0.14), "m"),
        MorphologyField("torso_density", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("thigh_density", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("leg_density", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("foot_density", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("gear", KIND_GEAR, 100.0, (50.0, 200.0), "actuator gear"),
    )

    def derived(self, values: dict[str, float]) -> dict[str, float]:
        thigh_len = values["thigh_len"]
        leg_len = values["leg_len"]
        foot_len = values["foot_len"]
        return {
            # leg body sits 0.25 below the bottom of the thigh geom (leg joint offset).
            "leg_z": -(2.0 * thigh_len + 0.25),
            # foot joint sits 0.1 above the foot body origin, at the bottom of the leg geom.
            "foot_z": -(leg_len + 0.1),
            # foot geom center is half a foot length ahead of the foot joint.
            "foot_x": -0.2 + foot_len,
        }


class ReacherTemplate(MorphologyTemplate):
    env_id = "Reacher-v5"
    asset_path = ASSET_DIR / "reacher.xml"
    fields = (
        MorphologyField("l0", KIND_LENGTH, 0.1, (0.06, 0.14), "m"),
        MorphologyField("l1", KIND_LENGTH, 0.1, (0.06, 0.14), "m"),
        MorphologyField("r0", KIND_RADIUS, 0.01, (0.005, 0.02), "m"),
        MorphologyField("r1", KIND_RADIUS, 0.01, (0.005, 0.02), "m"),
        MorphologyField("density0", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("density1", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("gear", KIND_GEAR, 200.0, (100.0, 400.0), "actuator gear"),
    )

    def derived(self, values: dict[str, float]) -> dict[str, float]:
        return {"fingertip_x": values["l1"] + 0.01}


class HopperTemplate(MorphologyTemplate):
    env_id = "Hopper-v5"
    asset_path = ASSET_DIR / "hopper.xml"
    fields = (
        MorphologyField("thigh_len", KIND_LENGTH, 0.225, (0.135, 0.315), "m"),
        MorphologyField("leg_len", KIND_LENGTH, 0.25, (0.15, 0.35), "m"),
        MorphologyField("foot_len", KIND_LENGTH, 0.195, (0.117, 0.273), "m"),
        MorphologyField("torso_density", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("thigh_density", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("leg_density", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("foot_density", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("gear", KIND_GEAR, 200.0, (100.0, 400.0), "actuator gear"),
    )

    def derived(self, values: dict[str, float]) -> dict[str, float]:
        thigh_len, leg_len, foot_len = (
            values["thigh_len"],
            values["leg_len"],
            values["foot_len"],
        )
        return {
            # leg body sits 0.25 below the bottom of the thigh geom (leg joint offset).
            "leg_z": -(2.0 * thigh_len + 0.25),
            # foot joint sits 0.1 above the foot body origin, at the bottom of the leg geom.
            "foot_z": -(leg_len + 0.1),
            # foot geom center trails the joint by one third of its halflength (upstream layout).
            "foot_x": -0.13 + foot_len / 3.0,
        }


class HalfCheetahTemplate(MorphologyTemplate):
    env_id = "HalfCheetah-v5"
    asset_path = ASSET_DIR / "half_cheetah.xml"
    fields = (
        MorphologyField("torso_len", KIND_LENGTH, 0.5, (0.3, 0.7), "m"),
        MorphologyField("bthigh_len", KIND_LENGTH, 0.145, (0.087, 0.203), "m"),
        MorphologyField("bshin_len", KIND_LENGTH, 0.15, (0.09, 0.21), "m"),
        MorphologyField("bfoot_len", KIND_LENGTH, 0.094, (0.0564, 0.1316), "m"),
        MorphologyField("fthigh_len", KIND_LENGTH, 0.133, (0.0798, 0.1862), "m"),
        MorphologyField("fshin_len", KIND_LENGTH, 0.106, (0.0636, 0.1484), "m"),
        MorphologyField("ffoot_len", KIND_LENGTH, 0.07, (0.042, 0.098), "m"),
        MorphologyField("gear_scale", KIND_GEAR, 1.0, (0.5, 2.0), "gear multiplier"),
    )

    def derived(self, values: dict[str, float]) -> dict[str, float]:
        torso_len = values["torso_len"]
        scale = values["gear_scale"]
        return {
            "bthigh_x": -torso_len,
            "fthigh_x": torso_len,
            "head_x": torso_len + 0.1,
            "bthigh_gear": 120.0 * scale,
            "bshin_gear": 90.0 * scale,
            "bfoot_gear": 60.0 * scale,
            "fthigh_gear": 120.0 * scale,
            "fshin_gear": 60.0 * scale,
            "ffoot_gear": 30.0 * scale,
        }


class SwimmerTemplate(MorphologyTemplate):
    """Parametric Swimmer: topology fixed at 3 links (torso, mid, back)."""

    env_id = "Swimmer-v5"
    asset_path = ASSET_DIR / "swimmer.xml"
    fields = (
        MorphologyField("torso_len", KIND_LENGTH, 0.5, (0.3, 0.7), "m"),
        MorphologyField("mid_len", KIND_LENGTH, 0.5, (0.3, 0.7), "m"),
        MorphologyField("back_len", KIND_LENGTH, 0.5, (0.3, 0.7), "m"),
        MorphologyField("radius", KIND_RADIUS, 0.1, (0.06, 0.14), "m"),
        MorphologyField("density", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("gear", KIND_GEAR, 150.0, (75.0, 300.0), "actuator gear"),
    )

    def derived(self, values: dict[str, float]) -> dict[str, float]:
        torso_len, mid_len, back_len = (
            values["torso_len"],
            values["mid_len"],
            values["back_len"],
        )
        return {
            "torso_from_x": torso_len + 1.0,
            "mid_to_x": -2.0 * mid_len,
            "back_x": -2.0 * mid_len,
            "back_to_x": -2.0 * back_len,
        }


class AntTemplate(MorphologyTemplate):
    """Parametric Ant: topology fixed at 4 legs."""

    env_id = "Ant-v5"
    asset_path = ASSET_DIR / "ant.xml"
    _LEG_DIRS = ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))
    fields = (
        MorphologyField("hip_len", KIND_LENGTH, 0.2, (0.1, 0.3), "m"),
        MorphologyField("ankle_len", KIND_LENGTH, 0.4, (0.2, 0.6), "m"),
        MorphologyField("density", KIND_MASS, 5.0, (2.5, 10.0), "kg/m^3"),
        MorphologyField("gear", KIND_GEAR, 150.0, (75.0, 300.0), "actuator gear"),
        MorphologyField("torso_radius", KIND_RADIUS, 0.25, (0.15, 0.35), "m"),
        MorphologyField("leg_radius", KIND_RADIUS, 0.08, (0.05, 0.12), "m"),
    )

    def derived(self, values: dict[str, float]) -> dict[str, float]:
        hip_len, ankle_len = values["hip_len"], values["ankle_len"]
        output: dict[str, float] = {}
        for index, (sign_x, sign_y) in enumerate(self._LEG_DIRS, start=1):
            output[f"aux{index}_x"] = sign_x * hip_len
            output[f"aux{index}_y"] = sign_y * hip_len
            output[f"ankle{index}_x"] = sign_x * ankle_len
            output[f"ankle{index}_y"] = sign_y * ankle_len
        return output
