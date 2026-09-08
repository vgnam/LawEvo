from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from morphlaw.control.expression import SymbolicExpression
from morphlaw.control.objective import objective_for

# Backward-compatible alias: the evolved law genome is now a free-form symbolic
# expression instead of a flat term list. Legacy callers that construct laws
# from a term tuple still work (they build the equivalent linear expression).
GymStructure = SymbolicExpression


def _wrap(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)


@dataclass(frozen=True)
class GymMetrics:
    score: float
    episode_return: float
    success_rate: float
    energy: float
    jerk: float
    complexity: int
    sg: float | None = None
    q: float | None = None

    @property
    def selection_key(self) -> tuple[float, float, float]:
        """No finite return/effort gain can compensate for lower measured SR."""
        import math

        sg = 0.0 if self.sg is None else self.sg
        if not all(math.isfinite(x) for x in (self.success_rate, sg, self.score)):
            return (-math.inf, -math.inf, -math.inf)
        return (self.success_rate, -sg, self.score)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "score": self.score,
            "episode_return": self.episode_return,
            "success_rate": self.success_rate,
            "energy": self.energy,
            "jerk": self.jerk,
            "complexity": self.complexity,
            "sg": self.sg,
            "q": self.q,
        }


@dataclass(frozen=True)
class GymEpisode:
    episode_return: float
    success: bool
    energy: float
    jerk: float
    sg: float | None = None
    q: float | None = None
    constraint_violations: dict[str, float] | None = None
    progress_predicates: dict[str, bool] | None = None
    diagnostics: dict[str, float] | None = None
    barrier_verification: dict | None = None


def clip_violation(excess: float, scale: float) -> float:
    """Normalized violation in [0, 1] of a threshold that was exceeded by ``excess``."""
    return float(np.clip(excess / max(scale, 1e-9), 0.0, 1.0))


def strict_upper_violation(value: float, threshold: float, scale: float) -> float:
    """Gap for value < threshold, preserving failure exactly at the boundary."""
    if not np.isfinite(value):
        return 1.0
    if value < threshold:
        return 0.0
    return max(1e-12, clip_violation(value - threshold, scale))


class EpisodeTracker:
    """Per-episode physical-state recorder for the SR/SG/Q metric layer.

    The generic engine calls ``update`` after every env step and the task
    adapter's ``success_constraints``/``progress_predicates`` close over the
    recorded trajectory, so all three metrics derive from raw physics instead
    of the environment reward.
    """

    def __init__(self, horizon: int) -> None:
        self.steps: list[dict[str, object]] = []
        self.diagnostics: dict[str, float] = {}

    def update(self, env, observation, terminated: bool) -> None:
        """Record one (post-step) physical snapshot of the episode."""

    def reset(self) -> None:
        self.steps.clear()
        self.diagnostics.clear()


class _ReacherTracker(EpisodeTracker):
    def update(self, env, observation, terminated: bool) -> None:
        data = env.unwrapped.data
        distance = float(np.linalg.norm(
            data.body("fingertip").xpos[:2] - data.body("target").xpos[:2]
        ))
        self.steps.append({"distance": distance, "terminated": bool(terminated)})
        self.diagnostics.update(steps=len(self.steps), final_distance=distance)


class BenchmarkAdapter:
    env_id: str
    horizon: int
    allowed_terms: tuple[str, ...]
    classical: tuple[GymStructure, ...]
    energy_weight: float
    jerk_weight: float
    complexity_weight: float = 0.02
    fallback_dt: float = 0.05

    def make_env(self):
        return gym.make(self.env_id, max_episode_steps=self.horizon)

    def reset_controller(self, action_dim: int) -> dict[str, np.ndarray | float]:
        return {"integral": np.zeros(action_dim), "previous_action": np.zeros(action_dim)}

    def prepare_reset(self, env, observation: np.ndarray, seed: int) -> np.ndarray:
        return observation

    def features(
        self, env, observation: np.ndarray, memory: dict, dt: float
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def success(self, env, observation: np.ndarray, steps: int, terminated: bool) -> bool:
        raise NotImplementedError

    def make_tracker(self) -> EpisodeTracker:
        return EpisodeTracker(self.horizon)

    def success_constraints(
        self, tracker: EpisodeTracker
    ) -> dict[str, float] | None:
        """Named normalized violations in [0, 1]; 0 means the constraint holds.

        Must describe exactly the same full-success specification as ``success``.
        ``None`` disables SG for adapters without a metric definition.
        """
        return None

    def progress_predicates(
        self, tracker: EpisodeTracker
    ) -> dict[str, bool] | None:
        """Named binary subgoal achievements; ``None`` disables Q."""
        return None

    @property
    def success_gap_spec(self):
        """Serializable definition, shared with prompts and run manifests."""
        return None

    @property
    def progress_spec(self):
        """Optional serializable Q milestones; Q never enters selection_key."""
        return None

    def score(self, episode_return: float, energy: float, jerk: float, complexity: int) -> float:
        return self.objective_config.score(episode_return, energy, jerk, complexity)

    @property
    def objective_config(self):
        return objective_for(self.env_id)

    @property
    def signal_contract(self):
        from morphlaw.control.signal_schema import contract_for

        return contract_for(self.env_id, self.allowed_terms)

    def validate_structure(self, structure):
        from morphlaw.control.signal_schema import validate_expression

        structure.validate(self.allowed_terms)
        return validate_expression(structure, self.signal_contract)


class ReacherAdapter(BenchmarkAdapter):
    env_id = "Reacher-v5"
    horizon = 50
    success_tolerance = 0.05
    distance_gap_scale = 0.20
    allowed_terms = (
        "jt_error",
        "joint_velocity",
        "integral_jt_error",
        "tanh_jt_error",
        "tanh_velocity",
        "normalized_jt_error",
        "task_damping",
    )
    classical = (
        GymStructure("Task P", ("jt_error",)),
        GymStructure("Task PI", ("jt_error", "integral_jt_error")),
        GymStructure("Task PD", ("jt_error", "joint_velocity")),
        GymStructure("Task PID", ("jt_error", "integral_jt_error", "joint_velocity")),
        GymStructure("Jacobian-transpose PD", ("jt_error", "task_damping")),
        GymStructure("Saturated PD", "K1*tanh(K2*jt_error) - K3*joint_velocity"),
    )
    energy_weight, jerk_weight = 0.01, 0.00002

    def prepare_reset(self, env, observation, seed):
        # Payload is an explicit paired variant, not hidden reset randomization.
        return observation

    def features(self, env, observation, memory, dt):
        del observation
        unwrapped = env.unwrapped
        model, data = unwrapped.model, unwrapped.data
        body_id = model.body("fingertip").id
        joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=int)
        dof_addresses = np.asarray(model.jnt_dofadr[joint_ids], dtype=int)
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacobian, None, body_id)
        jxy = jacobian[:2, dof_addresses]
        fingertip = np.asarray(data.body("fingertip").xpos[:2], dtype=float)
        target = np.asarray(data.body("target").xpos[:2], dtype=float)
        task_error = target - fingertip
        qvel = np.asarray(data.qvel[dof_addresses], dtype=float)
        jt_error = jxy.T @ task_error
        memory["integral"] = np.clip(memory["integral"] + jt_error * dt, -0.5, 0.5)
        norm = float(np.linalg.norm(jt_error))
        return {
            "jt_error": jt_error,
            "joint_velocity": qvel,
            "integral_jt_error": memory["integral"],
            "tanh_jt_error": np.tanh(10 * jt_error),
            "tanh_velocity": np.tanh(qvel),
            "normalized_jt_error": jt_error / max(norm, 1e-6),
            "task_damping": jxy.T @ (jxy @ qvel),
        }

    def success(self, env, observation, steps, terminated):
        del observation, steps, terminated
        data = env.unwrapped.data
        error = data.body("fingertip").xpos[:2] - data.body("target").xpos[:2]
        return float(np.linalg.norm(error)) < self.success_tolerance

    def make_tracker(self):
        return _ReacherTracker(self.horizon)

    def success_constraints(self, tracker):
        distance = tracker.steps[-1]["distance"] if tracker.steps else float("inf")
        return {"goal_position": strict_upper_violation(
            distance, self.success_tolerance, self.distance_gap_scale
        )}

    def progress_predicates(self, tracker):
        distances = [s["distance"] for s in tracker.steps]

        def inside(distance, tolerance):
            return bool(strict_upper_violation(distance, tolerance, self.distance_gap_scale) == 0)

        return {
            "near_goal_once": any(inside(d, 2 * self.success_tolerance) for d in distances),
            "goal_reached_once": any(inside(d, self.success_tolerance) for d in distances),
            "goal_reached_final": bool(distances and inside(distances[-1], self.success_tolerance)),
        }

    @property
    def progress_spec(self):
        return {
            "aggregation": "fraction of achieved milestones per episode, then mean over seeds",
            "selection": "diagnostic only; excluded from selection_key",
            "near_goal_once": "any post-step fingertip-target xy distance strictly below near radius",
            "goal_reached_once": "any post-step distance strictly below success tolerance",
            "goal_reached_final": "final distance strictly below success tolerance",
            "near_radius_m": 2 * self.success_tolerance,
            "success_tolerance_m": self.success_tolerance,
        }

    @property
    def success_gap_spec(self):
        return {
            "aggregation": "mean of constraint violations per episode, then mean over seeds",
            "goal_position": "strict upper-bound violation of final fingertip-target xy distance",
            "distance_tolerance_m": self.success_tolerance,
            "distance_scale_m": self.distance_gap_scale,
            "strict_boundary_floor": 1e-12,
        }


class LocomotionAdapter(BenchmarkAdapter):
    """Classical CPG plus posture-feedback features for planar MuJoCo locomotion."""

    horizon = 300
    action_dim: int
    joint_position_slice: slice
    joint_velocity_slice: slice
    forward_velocity_index: int
    height_index: int = 0
    angle_index: int = 1
    target_height: float
    target_speed: float
    phase_frequency: float
    phase_offsets: np.ndarray
    balance_pattern: np.ndarray
    height_pattern: np.ndarray
    speed_pattern: np.ndarray
    allowed_terms = (
        "phase_sin",
        "phase_cos",
        "posture_error",
        "joint_velocity",
        "integral_posture",
        "tanh_posture",
        "tanh_velocity",
        "body_angle",
        "height_error",
        "forward_speed_error",
    )
    classical = (
        GymStructure("Posture P", ("posture_error",)),
        GymStructure("Posture PD", ("posture_error", "joint_velocity")),
        GymStructure("CPG", ("phase_sin", "phase_cos")),
        GymStructure(
            "CPG + PD",
            ("phase_sin", "phase_cos", "posture_error", "joint_velocity"),
        ),
    )
    energy_weight, jerk_weight = 0.01, 0.000001

    def prepare_reset(self, env, observation, seed):
        rng = np.random.default_rng(seed + 3701)
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "_morphlaw_base_body_mass"):
            unwrapped._morphlaw_base_body_mass = unwrapped.model.body_mass.copy()
        unwrapped.model.body_mass[:] = unwrapped._morphlaw_base_body_mass
        unwrapped.model.body_mass[1:] *= rng.uniform(
            0.9, 1.1, size=len(unwrapped.model.body_mass) - 1
        )
        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        return unwrapped._get_obs()

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory["step"] = 0
        return memory

    def features(self, env, observation, memory, dt):
        del env
        joint_position = np.asarray(observation[self.joint_position_slice], dtype=float)
        joint_velocity = np.asarray(observation[self.joint_velocity_slice], dtype=float)
        posture_error = -joint_position
        memory["integral"] = np.clip(memory["integral"] + posture_error * dt, -2.0, 2.0)
        phase = 2 * np.pi * self.phase_frequency * float(memory["step"]) * dt
        memory["step"] += 1
        angle = float(observation[self.angle_index])
        height_error = self.target_height - float(observation[self.height_index])
        speed_error = self.target_speed - float(observation[self.forward_velocity_index])
        return {
            "phase_sin": np.sin(phase + self.phase_offsets),
            "phase_cos": np.cos(phase + self.phase_offsets),
            "posture_error": posture_error,
            "joint_velocity": joint_velocity,
            "integral_posture": memory["integral"],
            "tanh_posture": np.tanh(2 * posture_error),
            "tanh_velocity": np.tanh(joint_velocity),
            "body_angle": angle * self.balance_pattern,
            "height_error": height_error * self.height_pattern,
            "forward_speed_error": speed_error * self.speed_pattern,
        }

    def success(self, env, observation, steps, terminated):
        del env
        return (
            not terminated
            and steps == self.horizon
            and float(observation[self.forward_velocity_index]) > 0.5 * self.target_speed
        )


class HopperAdapter(LocomotionAdapter):
    env_id = "Hopper-v5"
    action_dim = 3
    joint_position_slice = slice(2, 5)
    joint_velocity_slice = slice(8, 11)
    forward_velocity_index = 5
    target_height = 1.25
    target_speed = 1.5
    phase_frequency = 1.8
    phase_offsets = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    balance_pattern = np.array([1.0, -0.5, -0.5])
    height_pattern = np.array([1.0, 0.7, 0.4])
    speed_pattern = np.array([1.0, -0.5, 0.5])


class Walker2dAdapter(LocomotionAdapter):
    env_id = "Walker2d-v5"
    action_dim = 6
    joint_position_slice = slice(2, 8)
    joint_velocity_slice = slice(11, 17)
    forward_velocity_index = 8
    target_height = 1.25
    target_speed = 1.5
    phase_frequency = 1.5
    phase_offsets = np.array([0.0, np.pi / 2, np.pi, np.pi, 3 * np.pi / 2, 0.0])
    balance_pattern = np.array([1.0, -0.5, -0.5, -1.0, 0.5, 0.5])
    height_pattern = np.array([1.0, 0.7, 0.4, 1.0, 0.7, 0.4])
    speed_pattern = np.array([1.0, -0.5, 0.5, 1.0, -0.5, 0.5])


class HalfCheetahAdapter(LocomotionAdapter):
    env_id = "HalfCheetah-v5"
    action_dim = 6
    joint_position_slice = slice(2, 8)
    joint_velocity_slice = slice(11, 17)
    forward_velocity_index = 8
    target_height = 0.0
    target_speed = 3.0
    phase_frequency = 1.7
    phase_offsets = np.array([0.0, np.pi / 2, np.pi, np.pi, 3 * np.pi / 2, 0.0])
    balance_pattern = np.array([1.0, -0.5, 0.5, -1.0, 0.5, -0.5])
    height_pattern = np.array([0.5, 0.3, 0.2, 0.5, 0.3, 0.2])
    speed_pattern = np.array([1.0, -0.5, 0.5, 1.0, -0.5, 0.5])


class SwimmerAdapter(BenchmarkAdapter):
    env_id = "Swimmer-v5"
    horizon = 300
    allowed_terms = (
        "phase_sin",
        "phase_cos",
        "posture_error",
        "joint_velocity",
        "integral_posture",
        "tanh_posture",
        "tanh_velocity",
        "body_angle",
        "lateral_velocity",
        "forward_speed_error",
    )
    classical = (
        GymStructure("Posture P", ("posture_error",)),
        GymStructure("Posture PD", ("posture_error", "joint_velocity")),
        GymStructure("CPG", ("phase_sin", "phase_cos")),
        GymStructure(
            "CPG + PD",
            ("phase_sin", "phase_cos", "posture_error", "joint_velocity"),
        ),
    )
    energy_weight, jerk_weight = 0.01, 0.000001
    target_speed = 0.2
    phase_frequency = 1.0

    def prepare_reset(self, env, observation, seed):
        rng = np.random.default_rng(seed + 4201)
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "_morphlaw_base_body_mass"):
            unwrapped._morphlaw_base_body_mass = unwrapped.model.body_mass.copy()
        unwrapped.model.body_mass[:] = unwrapped._morphlaw_base_body_mass
        unwrapped.model.body_mass[1:] *= rng.uniform(
            0.9, 1.1, size=len(unwrapped.model.body_mass) - 1
        )
        unwrapped._morphlaw_start_x = float(unwrapped.data.qpos[0])
        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        return unwrapped._get_obs()

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory["step"] = 0
        return memory

    def features(self, env, observation, memory, dt):
        del observation
        data = env.unwrapped.data
        nu = int(data.ctrl.size)
        joint_position = np.asarray(data.qpos[3 : 3 + nu], dtype=float)
        joint_velocity = np.asarray(data.qvel[3 : 3 + nu], dtype=float)
        posture_error = -joint_position
        memory["integral"] = np.clip(memory["integral"] + posture_error * dt, -2.0, 2.0)
        phase = 2 * np.pi * self.phase_frequency * float(memory["step"]) * dt
        memory["step"] += 1
        alternating = np.resize(np.array([1.0, -1.0]), nu)
        phase_offsets = np.resize(np.array([0.0, np.pi]), nu)
        return {
            "phase_sin": np.sin(phase + phase_offsets),
            "phase_cos": np.cos(phase + phase_offsets),
            "posture_error": posture_error,
            "joint_velocity": joint_velocity,
            "integral_posture": memory["integral"],
            "tanh_posture": np.tanh(2.0 * posture_error),
            "tanh_velocity": np.tanh(joint_velocity),
            "body_angle": float(data.qpos[2]) * alternating,
            "lateral_velocity": float(data.qvel[1]) * alternating,
            "forward_speed_error": (self.target_speed - float(data.qvel[0])) * np.ones(nu),
        }

    def success(self, env, observation, steps, terminated):
        del observation
        displacement = float(env.unwrapped.data.qpos[0]) - float(env.unwrapped._morphlaw_start_x)
        return not terminated and steps == self.horizon and displacement > 1.0


class FreeJointLocomotionAdapter(LocomotionAdapter):
    """Locomotion features read from MuJoCo state for free-root robots.

    Topology-changing morphologies vary the actuator count, so adapters may
    override `joint_order_for` and `patterns_for` to derive their layouts from
    the live action dimension instead of fixed class attributes.
    """

    joint_order: np.ndarray
    roll_pattern: np.ndarray
    pitch_pattern: np.ndarray

    def joint_order_for(self, nu: int) -> np.ndarray:
        return self.joint_order

    def patterns_for(self, nu: int) -> dict[str, np.ndarray]:
        del nu
        return {
            "phase_offsets": self.phase_offsets,
            "roll_pattern": self.roll_pattern,
            "pitch_pattern": self.pitch_pattern,
            "balance_pattern": self.balance_pattern,
            "height_pattern": self.height_pattern,
            "speed_pattern": self.speed_pattern,
        }

    def features(self, env, observation, memory, dt):
        del observation
        data = env.unwrapped.data
        nu = int(data.ctrl.size)
        joint_order = self.joint_order_for(nu)
        patterns = self.patterns_for(nu)
        joint_position = np.asarray(data.qpos[7:], dtype=float)[joint_order]
        joint_velocity = np.asarray(data.qvel[6:], dtype=float)[joint_order]
        posture_error = -joint_position
        memory["integral"] = np.clip(memory["integral"] + posture_error * dt, -2.0, 2.0)
        phase = 2 * np.pi * self.phase_frequency * float(memory["step"]) * dt
        memory["step"] += 1
        w, x, y, z = map(float, data.qpos[3:7])
        roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
        height_error = self.target_height - float(data.qpos[2])
        speed_error = self.target_speed - float(data.qvel[0])
        return {
            "phase_sin": np.sin(phase + patterns["phase_offsets"]),
            "phase_cos": np.cos(phase + patterns["phase_offsets"]),
            "posture_error": posture_error,
            "joint_velocity": joint_velocity,
            "integral_posture": memory["integral"],
            "tanh_posture": np.tanh(2 * posture_error),
            "tanh_velocity": np.tanh(joint_velocity),
            "body_angle": roll * patterns["roll_pattern"] + pitch * patterns["pitch_pattern"],
            "height_error": height_error * patterns["height_pattern"],
            "forward_speed_error": speed_error * patterns["speed_pattern"],
        }

    def success(self, env, observation, steps, terminated):
        del observation
        return (
            not terminated
            and steps == self.horizon
            and float(env.unwrapped.data.qvel[0]) > 0.5 * self.target_speed
        )


def _ant_patterns(nu: int) -> dict[str, np.ndarray]:
    """Per-actuator patterns for an Ant with nu actuators (two per leg: hip, ankle).

    For the default four-legged Ant (nu=8) this reproduces the historical
    hardcoded pattern arrays exactly; larger leg counts extend the same rules.
    """
    n_legs = nu // 2
    phase: list[float] = []
    roll: list[float] = []
    pitch: list[float] = []
    height: list[float] = []
    speed: list[float] = []
    for leg in range(n_legs):
        pair = (leg // 2) % 2
        diagonal = 1.0 if (leg % 4) in (0, 3) else -1.0
        phase_leg = 0.0 if diagonal > 0 else np.pi
        phase.extend([phase_leg, phase_leg])
        roll_sign = 1.0 if leg % 2 == 0 else -1.0
        roll.extend([1.0 * roll_sign, 0.4 * roll_sign])
        pitch_sign = 1.0 if pair == 0 else -1.0
        pitch.extend([1.0 * pitch_sign, 0.4 * pitch_sign])
        height.extend([1.0, 0.5])
        speed.extend([1.0 * diagonal, 0.5 * diagonal])
    return {
        "phase_offsets": np.array(phase),
        "roll_pattern": np.array(roll),
        "pitch_pattern": np.array(pitch),
        "balance_pattern": np.zeros(nu),
        "height_pattern": np.array(height),
        "speed_pattern": np.array(speed),
    }


class AntAdapter(FreeJointLocomotionAdapter):
    env_id = "Ant-v5"
    action_dim = 8
    target_height = 0.65
    target_speed = 1.5
    phase_frequency = 1.5

    def joint_order_for(self, nu: int) -> np.ndarray:
        return np.arange(nu)

    def patterns_for(self, nu: int) -> dict[str, np.ndarray]:
        return _ant_patterns(nu)


class PusherAdapter(BenchmarkAdapter):
    env_id = "Pusher-v5"
    horizon = 100
    allowed_terms = (
        "jt_object_error",
        "jt_goal_error",
        "jt_push_error",
        "joint_velocity",
        "integral_jt_push",
        "tanh_jt_push",
        "normalized_jt_push",
        "task_damping",
        "posture_error",
    )
    classical = (
        GymStructure("Task P", ("jt_push_error",)),
        GymStructure("Task PI", ("jt_push_error", "integral_jt_push")),
        GymStructure("Task PD", ("jt_push_error", "joint_velocity")),
        GymStructure("Task PID", ("jt_push_error", "integral_jt_push", "joint_velocity")),
    )
    energy_weight, jerk_weight = 0.01, 0.000001

    def prepare_reset(self, env, observation, seed):
        rng = np.random.default_rng(seed + 4701)
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "_morphlaw_base_body_mass"):
            unwrapped._morphlaw_base_body_mass = unwrapped.model.body_mass.copy()
        unwrapped.model.body_mass[:] = unwrapped._morphlaw_base_body_mass
        unwrapped.model.body_mass[1:] *= rng.uniform(
            0.9, 1.1, size=len(unwrapped.model.body_mass) - 1
        )
        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        return unwrapped._get_obs()

    def features(self, env, observation, memory, dt):
        del observation
        unwrapped = env.unwrapped
        tip_id = unwrapped.model.body("tips_arm").id
        jacobian = np.zeros((3, unwrapped.model.nv))
        mujoco.mj_jacBody(unwrapped.model, unwrapped.data, jacobian, None, tip_id)
        jxy = jacobian[:2, :7]
        tip = np.asarray(unwrapped.data.body("tips_arm").xpos[:2], dtype=float)
        obj = np.asarray(unwrapped.data.body("object").xpos[:2], dtype=float)
        goal = np.asarray(unwrapped.data.body("goal").xpos[:2], dtype=float)
        qvel = np.asarray(unwrapped.data.qvel[:7], dtype=float)
        object_error = obj - tip
        goal_error = goal - obj
        push_error = object_error + 0.5 * goal_error
        jt_object = jxy.T @ object_error
        jt_goal = jxy.T @ goal_error
        jt_push = jxy.T @ push_error
        memory["integral"] = np.clip(memory["integral"] + jt_push * dt, -0.5, 0.5)
        norm = float(np.linalg.norm(jt_push))
        return {
            "jt_object_error": jt_object,
            "jt_goal_error": jt_goal,
            "jt_push_error": jt_push,
            "joint_velocity": qvel,
            "integral_jt_push": memory["integral"],
            "tanh_jt_push": np.tanh(10 * jt_push),
            "normalized_jt_push": jt_push / max(norm, 1e-6),
            "task_damping": jxy.T @ (jxy @ qvel),
            "posture_error": -np.asarray(unwrapped.data.qpos[:7], dtype=float),
        }

    def success(self, env, observation, steps, terminated):
        del observation, steps, terminated
        data = env.unwrapped.data
        obj = np.asarray(data.body("object").xpos[:2], dtype=float)
        goal = np.asarray(data.body("goal").xpos[:2], dtype=float)
        return float(np.linalg.norm(goal - obj)) < 0.1


ADAPTERS = {
    "reacher": ReacherAdapter(),
    "pusher": PusherAdapter(),
}

LOCOMOTION_ADAPTERS = {
    "hopper": HopperAdapter(),
    "walker2d": Walker2dAdapter(),
    "half_cheetah": HalfCheetahAdapter(),
    "swimmer": SwimmerAdapter(),
    "ant": AntAdapter(),
}


def run_episode(
    adapter: BenchmarkAdapter,
    structure: GymStructure,
    gains: np.ndarray,
    seed: int,
    *,
    env=None,
    verify_barriers=False,
):
    adapter.validate_structure(structure)
    owns_env = env is None
    if owns_env:
        env = adapter.make_env()
    try:
        observation, _ = env.reset(seed=seed)
        observation = adapter.prepare_reset(env, observation, seed)
        action_dim = int(np.prod(env.action_space.shape))
        memory = adapter.reset_controller(action_dim)
        dt = float(env.unwrapped.dt) if hasattr(env.unwrapped, "dt") else adapter.fallback_dt
        total_return = energy = jerk = 0.0
        previous = np.zeros(action_dim)
        terminated = False
        steps = 0
        tracker = adapter.make_tracker()
        contract = adapter.signal_contract
        verifier = None
        if verify_barriers:
            from morphlaw.control.verification import ControllerBarrierVerifier

            verifier = ControllerBarrierVerifier(adapter.env_id, env, observation, dt, seed)
        for steps in range(1, adapter.horizon + 1):
            features = adapter.features(env, observation, memory, dt)
            if steps == 1 and contract.complete:
                contract.validate_values(features)
            action = structure.evaluate(features, gains)
            if verifier is not None:
                with np.errstate(all="ignore"):
                    raw = structure.evaluate_raw(features, gains)
                valid = all(np.isfinite(value).all() for value in features.values())
                if contract.complete:
                    valid = valid and all(np.asarray(features[name]).shape == spec.shape
                                          for name, spec in contract.signals.items())
                verifier.check_action(raw, env.action_space.low, env.action_space.high, valid)
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            energy += dt * float(action @ action)
            jerk += dt * float(((action - previous) / dt) @ ((action - previous) / dt))
            previous = action.copy()
            tracker.update(env, observation, terminated)
            if verifier is not None:
                verifier.update(env, observation)
            if terminated or truncated:
                break
        violations = adapter.success_constraints(tracker)
        predicates = adapter.progress_predicates(tracker)
        if violations is not None:
            # Section-10 rule: the adapter converts raw state into thresholded
            # violations, then success is exactly "all violations == 0", so SR
            # and SG share one full-success specification by construction.
            sg_episode = float(np.mean(list(violations.values()))) if violations else 0.0
            success = all(value == 0 for value in violations.values())
        else:
            sg_episode = None
            success = adapter.success(env, observation, steps, terminated)
        if predicates is not None:
            q_episode = float(np.mean([float(bool(value)) for value in predicates.values()])) if predicates else 0.0
        else:
            q_episode = None
        return GymEpisode(
            total_return,
            bool(success),
            energy,
            jerk,
            sg_episode,
            q_episode,
            violations,
            predicates,
            tracker.diagnostics,
            None if verifier is None else verifier.report(),
        )
    finally:
        if owns_env:
            env.close()


def evaluate_gym_structure(
    adapter: BenchmarkAdapter,
    structure: GymStructure,
    gains: np.ndarray,
    seeds: list[int],
    *,
    envs: list | None = None,
    verify_barriers: bool = False,
) -> tuple[GymMetrics, list[GymEpisode]]:
    adapter.validate_structure(structure)
    if not set(structure.signals) <= set(adapter.allowed_terms):
        raise ValueError("structure uses unavailable signals")
    if envs is None and hasattr(adapter, "evaluate_episodes") and not verify_barriers:
        episodes = adapter.evaluate_episodes(structure, gains, seeds)
    else:
        episodes = (
            [run_episode(adapter, structure, gains, seed, verify_barriers=verify_barriers) for seed in seeds]
            if envs is None
            else [
                run_episode(adapter, structure, gains, seed, env=env, verify_barriers=verify_barriers)
                for seed, env in zip(seeds, envs, strict=True)
            ]
        )
    episode_return = float(np.mean([item.episode_return for item in episodes]))
    success = float(np.mean([item.success for item in episodes]))
    energy = float(np.mean([item.energy for item in episodes]))
    jerk = float(np.mean([item.jerk for item in episodes]))
    sg_values = [item.sg for item in episodes if item.sg is not None]
    q_values = [item.q for item in episodes if item.q is not None]
    sg = float(np.mean(sg_values)) if sg_values else None
    q = float(np.mean(q_values)) if q_values else None
    score = adapter.score(episode_return, energy, jerk, structure.complexity)
    return GymMetrics(score, episode_return, success, energy, jerk, structure.complexity, sg, q), episodes


def tune_gym_cem(
    adapter: BenchmarkAdapter,
    structure: GymStructure,
    seeds: list[int],
    *,
    iterations: int = 5,
    population_size: int = 24,
) -> tuple[np.ndarray, GymMetrics]:
    adapter.validate_structure(structure)
    digest = hashlib.sha256(
        json.dumps({"env": adapter.env_id, "law": structure.to_expression_string()}, sort_keys=True).encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))
    dimension = structure.parameter_count
    mean, sigma = np.zeros(dimension), np.full(dimension, 3.0)
    if hasattr(adapter, "evaluate_gain_batch"):
        best_gains = mean.copy()
        best_metrics = adapter.evaluate_gain_batch(structure, mean[None, :], seeds)[0]
        elite_count = max(2, round(0.2 * population_size))
        for _ in range(iterations):
            samples = np.clip(rng.normal(mean, sigma, size=(population_size, dimension)), -20, 20)
            metrics = adapter.evaluate_gain_batch(structure, samples, seeds)
            scored = sorted(
                zip(samples, metrics, strict=True),
                key=lambda item: item[1].selection_key,
                reverse=True,
            )
            elites = np.vstack([item[0] for item in scored[:elite_count]])
            mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
            sigma = np.maximum(0.05, 0.25 * sigma + 0.75 * elites.std(axis=0))
            if scored[0][1].selection_key > best_metrics.selection_key:
                best_gains, best_metrics = scored[0][0].copy(), scored[0][1]
        return best_gains, best_metrics

    envs = [adapter.make_env() for _ in seeds]
    try:
        best_gains = mean.copy()
        best_metrics, _ = evaluate_gym_structure(adapter, structure, best_gains, seeds, envs=envs)
        elite_count = max(2, round(0.2 * population_size))
        for _ in range(iterations):
            samples = np.clip(rng.normal(mean, sigma, size=(population_size, dimension)), -20, 20)
            scored = [
                (
                    sample,
                    evaluate_gym_structure(adapter, structure, sample, seeds, envs=envs)[0],
                )
                for sample in samples
            ]
            scored.sort(key=lambda item: item[1].selection_key, reverse=True)
            elites = np.vstack([item[0] for item in scored[:elite_count]])
            mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
            sigma = np.maximum(0.05, 0.25 * sigma + 0.75 * elites.std(axis=0))
            if scored[0][1].selection_key > best_metrics.selection_key:
                best_gains, best_metrics = scored[0][0].copy(), scored[0][1]
        return best_gains, best_metrics
    finally:
        for env in envs:
            env.close()
