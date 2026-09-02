from __future__ import annotations

import json
import math
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from lawevo.morplaw.morphology import MorphologyError, MorphologyGenome, MorphologyTemplate
from lawevo.pid.gym_benchmark import AntAdapter, BenchmarkAdapter

BODY_JOINTS = ("rigid", "roll", "twist")
LIMB_JOINTS = ("rigid", "roll", "knee", "elbow")
TERMINALS = ("foot", "wheel")
ROBOMORPH_TERRAINS = ("flat", "ridged", "frozen_lake", "beams")


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise MorphologyError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MorphologyError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise MorphologyError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class LimbSegment:
    joint: str
    length: float
    angle: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> LimbSegment:
        return cls(
            str(payload.get("joint", "knee")).lower(),
            _number(payload.get("length"), "limb length"),
            _number(payload.get("angle", 0.0), "joint angle"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"joint": self.joint, "length": self.length, "angle": self.angle}


@dataclass(frozen=True)
class LimbBranch:
    segments: tuple[LimbSegment, ...]
    terminal: str = "foot"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> LimbBranch:
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
            raise MorphologyError("limb segments must be an array")
        segments = tuple(
            LimbSegment.from_dict(item) for item in raw_segments if isinstance(item, Mapping)
        )
        if len(segments) != len(raw_segments):
            raise MorphologyError("every limb segment must be an object")
        return cls(segments, str(payload.get("terminal", "foot")).lower())

    def to_dict(self) -> dict[str, object]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class BodySegment:
    length: float
    joint_to_previous: str = "rigid"
    limb: LimbBranch | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], index: int) -> BodySegment:
        raw_limb = payload.get("limb")
        if raw_limb is not None and not isinstance(raw_limb, Mapping):
            raise MorphologyError("limb must be an object or null")
        joint = "root" if index == 0 else str(payload.get("joint_to_previous", "rigid")).lower()
        return cls(
            _number(payload.get("length"), "body length"),
            joint,
            LimbBranch.from_dict(raw_limb) if isinstance(raw_limb, Mapping) else None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "joint_to_previous": self.joint_to_previous,
            "limb": self.limb.to_dict() if self.limb is not None else None,
        }


@dataclass(frozen=True)
class RobotGraphSpec:
    """Canonical robot graph generated from a RoboMorph-style module grammar.

    Each limb entry represents a bilateral pair; symmetry is enforced by the compiler
    instead of delegated to the LLM.
    """

    body: tuple[BodySegment, ...]
    name: str = "grammar_robot"
    spec_type: ClassVar[str] = "robot_grammar"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RobotGraphSpec:
        graph = payload.get("graph", payload)
        if not isinstance(graph, Mapping):
            raise MorphologyError("graph must be an object")
        raw_body = graph.get("body")
        if not isinstance(raw_body, Sequence) or isinstance(raw_body, (str, bytes)):
            raise MorphologyError("graph.body must be an array")
        body = tuple(
            BodySegment.from_dict(item, index)
            for index, item in enumerate(raw_body)
            if isinstance(item, Mapping)
        )
        if len(body) != len(raw_body):
            raise MorphologyError("every body segment must be an object")
        return cls(body, str(payload.get("name", graph.get("name", "grammar_robot")))[:60])

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "body": [segment.to_dict() for segment in self.body]}

    def key(self) -> Hashable:
        graph = {"body": [segment.to_dict() for segment in self.body]}
        return (self.spec_type, json.dumps(graph, sort_keys=True, separators=(",", ":")))

    def counts(self) -> dict[str, int]:
        limb_pairs = sum(segment.limb is not None for segment in self.body)
        limb_segments = sum(
            2 * len(segment.limb.segments) for segment in self.body if segment.limb is not None
        )
        wheels = sum(
            2
            for segment in self.body
            if segment.limb is not None and segment.limb.terminal == "wheel"
        )
        actuators = sum(
            segment.joint_to_previous not in ("root", "rigid") for segment in self.body
        ) + sum(
            2 * sum(limb_segment.joint != "rigid" for limb_segment in segment.limb.segments)
            for segment in self.body
            if segment.limb is not None
        )
        return {
            "body_segments": len(self.body),
            "limb_pairs": limb_pairs,
            "limb_segments": limb_segments,
            "wheels": wheels,
            "actuators": actuators,
        }

    def describe(self) -> str:
        counts = self.counts()
        body_joints = "/".join(segment.joint_to_previous for segment in self.body[1:]) or "none"
        return (
            f"{self.name}[body={counts['body_segments']}, limb_pairs={counts['limb_pairs']}, "
            f"limb_segments={counts['limb_segments']}, wheels={counts['wheels']}, "
            f"actuators={counts['actuators']}, body_joints={body_joints}]"
        )


class RoboMorphGrammarTemplate(MorphologyTemplate):
    """RoboMorph-style body grammar compiled into a free-root MuJoCo robot."""

    env_id = "Ant-v5"
    asset_path = None
    fields = ()

    def __init__(self, terrain: str = "flat") -> None:
        if terrain not in ROBOMORPH_TERRAINS:
            raise ValueError(f"terrain must be one of {ROBOMORPH_TERRAINS}")
        self.terrain = terrain
        super().__init__()

    def cache_namespace(self) -> str:
        return f"{super().cache_namespace()}:{self.terrain}"

    def knowledge_key(self) -> str:
        return f"robomorph_{self.terrain}"

    def default_spec(self) -> RobotGraphSpec:
        leg = LimbBranch(
            (
                LimbSegment("knee", 0.18, 20.0),
                LimbSegment("knee", 0.20, -25.0),
            ),
            "foot",
        )
        return RobotGraphSpec(
            (
                BodySegment(0.24, "root", leg),
                BodySegment(0.24, "rigid", leg),
            ),
            "seed_quadruped",
        )

    def seed_specs(self, count: int = 3, seed: int = 0) -> tuple[MorphologyGenome, ...]:
        if not 1 <= count <= 32:
            raise ValueError("grammar seed count must be in [1, 32]")
        rng = np.random.default_rng(seed)
        output = [self.default_spec()]
        while len(output) < count:
            body_count = int(rng.integers(1, 5))
            limb_sites = rng.random(body_count) < 0.65
            limb_sites[int(rng.integers(0, body_count))] = True
            body: list[BodySegment] = []
            for index in range(body_count):
                limb = None
                if limb_sites[index]:
                    link_count = int(rng.integers(1, 4))
                    links = tuple(
                        LimbSegment(
                            str(rng.choice(("roll", "knee", "elbow", "rigid"))),
                            round(float(rng.uniform(0.07, 0.24)), 3),
                            round(float(rng.uniform(-35.0, 35.0)), 1),
                        )
                        for _ in range(link_count)
                    )
                    if all(link.joint == "rigid" for link in links):
                        links = (LimbSegment("knee", links[0].length, links[0].angle), *links[1:])
                    limb = LimbBranch(links, str(rng.choice(TERMINALS)))
                body.append(
                    BodySegment(
                        round(float(rng.uniform(0.12, 0.38)), 3),
                        "root" if index == 0 else str(rng.choice(BODY_JOINTS)),
                        limb,
                    )
                )
            spec = RobotGraphSpec(tuple(body), f"random_seed_{len(output)}")
            if not self.validate(spec) and spec.key() not in {item.key() for item in output}:
                output.append(spec)
        return tuple(output)

    def field_descriptions(self) -> list[dict[str, object]]:
        return [
            {"node": "body", "count": [1, 4], "length_m": [0.1, 0.4]},
            {"node": "body_joint", "choices": list(BODY_JOINTS)},
            {"node": "bilateral_limb", "segments": [1, 3]},
            {"node": "limb_joint", "choices": list(LIMB_JOINTS)},
            {"node": "limb_link", "length_m": [0.05, 0.25]},
            {"node": "terminal", "choices": list(TERMINALS)},
        ]

    def proposal_schema(self) -> dict[str, object]:
        return {
            "name": "short design name",
            "graph": {
                "body": [
                    {
                        "length": "0.10..0.40",
                        "joint_to_previous": "root for first; then rigid|roll|twist",
                        "limb": {
                            "segments": [
                                {
                                    "joint": "rigid|roll|knee|elbow",
                                    "length": "0.05..0.25",
                                    "angle": "degrees",
                                }
                            ],
                            "terminal": "foot|wheel",
                        },
                    }
                ]
            },
        }

    def proposal_guidance(self) -> str:
        return (
            "Generate a complete robot graph rather than a fixed-template parameter vector. "
            "The body is a serial chain of 1-4 modules. The first body joint must be root; "
            "later body joints may be rigid, roll, or twist. Each body module may carry one "
            "bilateral limb definition; the compiler mirrors it left/right. A limb has 1-3 "
            "segments with rigid, roll, knee, or elbow joints and ends in a foot or passive "
            "wheel. Roll/knee angles must stay within +/-60 degrees and elbow angles within "
            "+/-180 degrees. At least one body module must have limbs and the graph must contain 2-16 "
            "actuated non-rigid joints. Use mutation or crossover of elite graphs. During "
            "exploration, non-local changes such as adding a body module, changing limb depth, "
            "changing joint type, or switching foot/wheel are allowed. During exploitation, "
            "make a smaller graph edit. Do not output raw MJCF; output only the grammar graph."
        )

    def parse_proposal(self, payload: Mapping[str, object]) -> RobotGraphSpec:
        spec = RobotGraphSpec.from_dict(payload)
        self.check(spec)
        return spec

    def validate(self, spec: MorphologyGenome) -> list[str]:
        if not isinstance(spec, RobotGraphSpec):
            return ["grammar template requires RobotGraphSpec"]
        errors: list[str] = []
        if not 1 <= len(spec.body) <= 4:
            errors.append("body must contain 1..4 segments")
        for index, body in enumerate(spec.body):
            if not 0.1 <= body.length <= 0.4:
                errors.append(f"body[{index}].length outside [0.1, 0.4]")
            expected = ("root",) if index == 0 else BODY_JOINTS
            if body.joint_to_previous not in expected:
                errors.append(f"body[{index}] has invalid joint {body.joint_to_previous!r}")
            if body.limb is None:
                continue
            if not 1 <= len(body.limb.segments) <= 3:
                errors.append(f"body[{index}] limb must contain 1..3 segments")
            if body.limb.terminal not in TERMINALS:
                errors.append(f"body[{index}] has invalid terminal {body.limb.terminal!r}")
            for link_index, link in enumerate(body.limb.segments):
                if link.joint not in LIMB_JOINTS:
                    errors.append(f"body[{index}].limb[{link_index}] invalid joint {link.joint!r}")
                if not 0.05 <= link.length <= 0.25:
                    errors.append(f"body[{index}].limb[{link_index}].length outside [0.05, 0.25]")
                angle_limit = 180.0 if link.joint == "elbow" else 60.0
                if not -angle_limit <= link.angle <= angle_limit:
                    errors.append(
                        f"body[{index}].limb[{link_index}].angle outside "
                        f"[-{angle_limit:g}, {angle_limit:g}]"
                    )
        counts = spec.counts()
        if counts["limb_pairs"] == 0:
            errors.append("at least one body segment must carry bilateral limbs")
        if not 2 <= counts["actuators"] <= 16:
            errors.append("graph must contain 2..16 actuated joints")
        return errors

    def cost(self, spec: MorphologyGenome, weight: float = 0.05) -> float:
        if not isinstance(spec, RobotGraphSpec):
            raise MorphologyError("grammar template requires RobotGraphSpec")
        counts = spec.counts()
        # Charge for physical/control complexity, not distance from one privileged topology.
        units = (
            0.5 * counts["body_segments"]
            + 0.25 * counts["limb_segments"]
            + 0.25 * counts["actuators"]
            + 0.1 * counts["wheels"]
        )
        return weight * units

    def field_deltas(self, spec: MorphologyGenome, baseline: MorphologyGenome | None = None) -> str:
        if not isinstance(spec, RobotGraphSpec):
            return "invalid grammar morphology"
        current = spec.counts()
        if not isinstance(baseline, RobotGraphSpec):
            return "set graph " + spec.describe()
        before = baseline.counts()
        changes = [
            f"{name} {before[name]}->{current[name]}"
            for name in current
            if before[name] != current[name]
        ]
        if [segment.to_dict() for segment in spec.body] != [
            segment.to_dict() for segment in baseline.body
        ]:
            changes.append("module/joint layout changed")
        return ", ".join(changes) or "graph unchanged"

    def render(self, spec: MorphologyGenome) -> str:
        self.check(spec)
        assert isinstance(spec, RobotGraphSpec)
        actuators: list[str] = []
        body_xml = self._body_chain(spec.body, 0, actuators)
        max_leg = max(
            (
                sum(link.length for link in body.limb.segments)
                for body in spec.body
                if body.limb is not None
            ),
            default=0.25,
        )
        root_height = min(0.85, max(0.35, max_leg + 0.12))
        floor_friction = "0.05 0.005 0.0001" if self.terrain == "frozen_lake" else "1.2 0.1 0.1"
        terrain_xml = self._terrain_xml()
        return f"""<mujoco model="morplaw_grammar_robot">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <default>
    <joint armature="0.08" damping="0.8" limited="true"/>
    <geom conaffinity="0" contype="1" condim="3" density="650" friction="1.2 0.1 0.1" margin="0.002" rgba="0.35 0.65 0.85 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="100"/>
  </default>
  <asset>
    <texture builtin="checker" height="100" name="ground_tex" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" type="2d" width="100"/>
    <material name="ground" reflectance="0.15" texrepeat="40 40" texture="ground_tex"/>
  </asset>
  <worldbody>
    <light diffuse="1 1 1" dir="0 0 -1" directional="true" pos="0 0 3"/>
    <geom conaffinity="1" condim="3" friction="{floor_friction}" material="ground" name="floor" pos="0 0 0" size="40 40 0.1" type="plane"/>
{terrain_xml}
    <body name="torso" pos="0 0 {root_height:.8g}">
      <camera name="track" mode="trackcom" pos="-3 -4 2" xyaxes="0.8 -0.6 0 0.25 0.35 0.9"/>
      <freejoint name="root"/>
{body_xml}
    </body>
  </worldbody>
  <actuator>
{"".join(actuators)}  </actuator>
</mujoco>
"""

    def _terrain_xml(self) -> str:
        """Terrain parameters mirror RoboMorph, with x mirrored for Gym's +x reward."""
        if self.terrain not in ("ridged", "beams"):
            return ""
        prefix = "ridge" if self.terrain == "ridged" else "beam"
        z = 0.0 if self.terrain == "ridged" else 0.5
        return "\n".join(
            (
                f'    <geom name="{prefix}_{index}" pos="{index * 2.0 + 1.0:.8g} 0 {z:.8g}" '
                'size="0.2 10.0" quat="0.7071067811865476 0.7071067811865476 0 0" '
                'type="cylinder" contype="0" conaffinity="1" rgba="0.7 0.5 0.3 1"/>'
            )
            for index in range(15)
        )

    def _body_chain(self, body: tuple[BodySegment, ...], index: int, actuators: list[str]) -> str:
        segment = body[index]
        half = 0.5 * segment.length
        indent = "      " if index == 0 else "  " * (4 + index)
        content: list[str] = []
        if index > 0:
            joint_name = f"body_joint_{index}"
            joint = self._joint_xml(joint_name, segment.joint_to_previous, 0.0, body_joint=True)
            if joint:
                content.append(f"{indent}{joint}\n")
                actuators.append(f'    <motor joint="{joint_name}"/>\n')
        content.append(
            f'{indent}<geom fromto="{-half:.8g} 0 0 {half:.8g} 0 0" '
            f'name="body_geom_{index}" size="0.055" type="capsule"/>\n'
        )
        if segment.limb is not None:
            content.append(self._limb_pair(index, segment.limb, indent, actuators))
        if index + 1 < len(body):
            next_half = 0.5 * body[index + 1].length
            content.append(
                f'{indent}<body name="body_{index + 1}" pos="{half + next_half:.8g} 0 0">\n'
            )
            content.append(self._body_chain(body, index + 1, actuators))
            content.append(f"{indent}</body>\n")
        return "".join(content)

    def _limb_pair(
        self, body_index: int, limb: LimbBranch, indent: str, actuators: list[str]
    ) -> str:
        return "".join(self._limb(body_index, side, limb, indent, actuators) for side in ("l", "r"))

    def _limb(
        self,
        body_index: int,
        side: str,
        limb: LimbBranch,
        indent: str,
        actuators: list[str],
    ) -> str:
        sign = 1.0 if side == "l" else -1.0

        def build(link_index: int, child_indent: str) -> str:
            link = limb.segments[link_index]
            name = f"limb_b{body_index}_{side}_{link_index}"
            lateral = sign * link.length * (0.35 if link_index == 0 else 0.12)
            vertical = -math.sqrt(max(link.length**2 - lateral**2, 1e-8))
            joint = self._joint_xml(name, link.joint, link.angle, body_joint=False)
            parts = [f'{child_indent}<body name="{name}" pos="0 0 0">\n']
            if joint:
                parts.append(f"{child_indent}  {joint}\n")
                actuators.append(f'    <motor joint="{name}"/>\n')
            parts.append(
                f'{child_indent}  <geom fromto="0 0 0 0 {lateral:.8g} {vertical:.8g}" '
                f'name="{name}_geom" size="0.035" type="capsule"/>\n'
            )
            endpoint_indent = child_indent + "  "
            if link_index + 1 < len(limb.segments):
                parts.append(
                    f'{endpoint_indent}<body name="{name}_end" '
                    f'pos="0 {lateral:.8g} {vertical:.8g}">\n'
                )
                parts.append(build(link_index + 1, endpoint_indent + "  "))
                parts.append(f"{endpoint_indent}</body>\n")
            else:
                parts.append(
                    self._terminal_xml(
                        f"terminal_b{body_index}_{side}",
                        limb.terminal,
                        lateral,
                        vertical,
                        endpoint_indent,
                    )
                )
            parts.append(f"{child_indent}</body>\n")
            return "".join(parts)

        return build(0, indent)

    @staticmethod
    def _joint_xml(name: str, kind: str, angle: float, *, body_joint: bool) -> str:
        if kind == "rigid":
            return ""
        if kind == "roll":
            axis, limit = "1 0 0", 45 if body_joint else 70
        elif kind == "twist":
            axis, limit = "0 0 1", 45
        elif kind == "knee":
            axis, limit = "0 1 0", 80
        elif kind == "elbow":
            axis, limit = "0 0 1", 180
        else:
            raise MorphologyError(f"unsupported joint {kind!r}")
        reference = max(-limit, min(limit, angle))
        return (
            f'<joint axis="{axis}" name="{name}" range="{-limit} {limit}" '
            f'ref="{reference:.8g}" type="hinge"/>'
        )

    @staticmethod
    def _terminal_xml(
        name: str, terminal: str, lateral: float, vertical: float, indent: str
    ) -> str:
        if terminal == "wheel":
            return (
                f'{indent}<body name="{name}" pos="0 {lateral:.8g} {vertical:.8g}">\n'
                f'{indent}  <joint axis="0 1 0" limited="false" name="{name}_spin" '
                f'type="hinge" damping="0.02"/>\n'
                f'{indent}  <geom euler="90 0 0" name="{name}_geom" size="0.065 0.025" '
                f'type="cylinder" rgba="0.15 0.15 0.15 1"/>\n'
                f"{indent}</body>\n"
            )
        return (
            f'{indent}<geom name="{name}_geom" pos="0 {lateral:.8g} {vertical:.8g}" '
            f'size="0.05" type="sphere" rgba="0.85 0.45 0.2 1"/>\n'
        )


class RoboMorphLocomotionAdapter(BenchmarkAdapter):
    """Topology-agnostic symbolic controller adapter for grammar-generated robots."""

    env_id = "Ant-v5"
    horizon = 300
    target_height = 0.55
    target_speed = 1.0
    phase_frequency = 1.4
    allowed_terms = AntAdapter.allowed_terms
    classical = AntAdapter.classical
    energy_weight, jerk_weight = 0.01, 0.000001

    @staticmethod
    def morph_env_kwargs() -> dict[str, object]:
        """Match RoboMorph FlatEnv dynamics/reward settings where Gym exposes them."""
        return {
            "ctrl_cost_weight": 0.0,
            "healthy_reward": 0.0,
            "terminate_when_unhealthy": True,
            "healthy_z_range": (0.05, 5.0),
            "reset_noise_scale": 0.1,
            "exclude_current_positions_from_observation": True,
            "include_cfrc_ext_in_observation": False,
        }

    def prepare_reset(self, env, observation, seed):
        rng = np.random.default_rng(seed + 8111)
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "_lawevo_base_body_mass"):
            unwrapped._lawevo_base_body_mass = unwrapped.model.body_mass.copy()
        unwrapped.model.body_mass[:] = unwrapped._lawevo_base_body_mass
        unwrapped.model.body_mass[1:] *= rng.uniform(
            0.9, 1.1, size=len(unwrapped.model.body_mass) - 1
        )
        unwrapped._lawevo_start_x = float(unwrapped.data.qpos[0])
        import mujoco

        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        return unwrapped._get_obs()

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory["step"] = 0
        return memory

    @staticmethod
    def _actuated_state(env) -> tuple[np.ndarray, np.ndarray, list[str]]:
        import mujoco

        model, data = env.unwrapped.model, env.unwrapped.data
        joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=int)
        qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
        dof_addresses = np.asarray(model.jnt_dofadr[joint_ids], dtype=int)
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)) or ""
            for joint_id in joint_ids
        ]
        return data.qpos[qpos_addresses].copy(), data.qvel[dof_addresses].copy(), names

    @staticmethod
    def _patterns(names: Sequence[str]) -> dict[str, np.ndarray]:
        phase: list[float] = []
        roll: list[float] = []
        pitch: list[float] = []
        speed: list[float] = []
        for index, name in enumerate(names):
            match = re.search(r"_b(\d+)_([lr])_", name)
            if match:
                body_index = int(match.group(1))
                side = match.group(2)
                parity = (body_index + (side == "r")) % 2
                side_sign = 1.0 if side == "l" else -1.0
                front_sign = 1.0 if body_index % 2 == 0 else -1.0
            else:
                parity = index % 2
                side_sign = 0.0
                front_sign = 1.0 if index % 2 == 0 else -1.0
            phase.append(0.0 if parity == 0 else math.pi)
            roll.append(side_sign)
            pitch.append(front_sign)
            speed.append(front_sign)
        size = len(names)
        return {
            "phase": np.asarray(phase),
            "roll": np.asarray(roll),
            "pitch": np.asarray(pitch),
            "height": np.ones(size),
            "speed": np.asarray(speed),
        }

    def features(self, env, observation, memory, dt):
        del observation
        qpos, qvel, names = self._actuated_state(env)
        posture_error = -qpos
        memory["integral"] = np.clip(memory["integral"] + posture_error * dt, -2.0, 2.0)
        patterns = self._patterns(names)
        phase = 2.0 * math.pi * self.phase_frequency * float(memory["step"]) * dt
        memory["step"] += 1
        data = env.unwrapped.data
        w, x, y, z = map(float, data.qpos[3:7])
        roll = float(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = float(math.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
        return {
            "phase_sin": np.sin(phase + patterns["phase"]),
            "phase_cos": np.cos(phase + patterns["phase"]),
            "posture_error": posture_error,
            "joint_velocity": qvel,
            "integral_posture": memory["integral"],
            "tanh_posture": np.tanh(2.0 * posture_error),
            "tanh_velocity": np.tanh(qvel),
            "body_angle": roll * patterns["roll"] + pitch * patterns["pitch"],
            "height_error": (self.target_height - float(data.qpos[2])) * patterns["height"],
            "forward_speed_error": (self.target_speed - float(data.qvel[0])) * patterns["speed"],
        }

    def success(self, env, observation, steps, terminated):
        del observation
        displacement = float(env.unwrapped.data.qpos[0]) - float(env.unwrapped._lawevo_start_x)
        return not terminated and steps == self.horizon and displacement > 1.0


ROBOMORPH_ADAPTER = RoboMorphLocomotionAdapter()
