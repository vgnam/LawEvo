from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are


def _wrap(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)


@dataclass(frozen=True)
class GymStructure:
    name: str
    terms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.terms) <= 8 or len(set(self.terms)) != len(self.terms):
            raise ValueError("a structure requires 1-8 unique terms")

    def key(self) -> tuple[str, ...]:
        return self.terms

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "terms": list(self.terms)}


@dataclass(frozen=True)
class GymMetrics:
    score: float
    episode_return: float
    success_rate: float
    energy: float
    jerk: float
    complexity: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "score": self.score,
            "episode_return": self.episode_return,
            "success_rate": self.success_rate,
            "energy": self.energy,
            "jerk": self.jerk,
            "complexity": self.complexity,
        }


@dataclass(frozen=True)
class GymEpisode:
    episode_return: float
    success: bool
    energy: float
    jerk: float


class BenchmarkAdapter:
    env_id: str
    horizon: int
    allowed_terms: tuple[str, ...]
    classical: tuple[GymStructure, ...]
    energy_weight: float
    jerk_weight: float
    complexity_weight: float = 0.02

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

    def score(self, episode_return: float, energy: float, jerk: float, complexity: int) -> float:
        return (
            episode_return
            - self.energy_weight * energy
            - self.jerk_weight * jerk
            - self.complexity_weight * complexity
        )


class PendulumAdapter(BenchmarkAdapter):
    env_id = "Pendulum-v1"
    horizon = 200
    allowed_terms = (
        "angle",
        "angular_velocity",
        "integral_angle",
        "sin_angle",
        "tanh_angle",
        "tanh_velocity",
        "sqrt_angle",
        "cubic_angle",
    )
    classical = (
        GymStructure("P", ("angle",)),
        GymStructure("PI", ("angle", "integral_angle")),
        GymStructure("PD", ("angle", "angular_velocity")),
        GymStructure("PID", ("angle", "integral_angle", "angular_velocity")),
    )
    energy_weight, jerk_weight = 0.002, 0.00002

    def prepare_reset(self, env, observation, seed):
        rng = np.random.default_rng(seed + 701)
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "_lawevo_base_m"):
            unwrapped._lawevo_base_m = float(unwrapped.m)
            unwrapped._lawevo_base_l = float(unwrapped.l)
        unwrapped.m = unwrapped._lawevo_base_m * float(rng.uniform(0.85, 1.15))
        unwrapped.l = unwrapped._lawevo_base_l * float(rng.uniform(0.9, 1.1))
        return observation

    def features(self, env, observation, memory, dt):
        del env
        angle = _wrap(float(np.arctan2(observation[1], observation[0])))
        velocity = float(observation[2])
        memory["integral"] = np.clip(memory["integral"] + angle * dt, -4, 4)
        integral = float(memory["integral"][0])
        scalar = {
            "angle": angle,
            "angular_velocity": velocity,
            "integral_angle": integral,
            "sin_angle": float(np.sin(angle)),
            "tanh_angle": float(np.tanh(angle)),
            "tanh_velocity": float(np.tanh(velocity)),
            "sqrt_angle": float(np.sign(angle) * np.sqrt(abs(angle))),
            "cubic_angle": angle**3,
        }
        return {key: np.array([value]) for key, value in scalar.items()}

    def success(self, env, observation, steps, terminated):
        del env, steps, terminated
        angle = abs(_wrap(float(np.arctan2(observation[1], observation[0]))))
        return angle < 0.2 and abs(float(observation[2])) < 0.75


class InvertedPendulumAdapter(BenchmarkAdapter):
    env_id = "InvertedPendulum-v5"
    horizon = 500
    allowed_terms = (
        "cart_position",
        "pole_angle",
        "cart_velocity",
        "pole_velocity",
        "integral_cart",
        "integral_angle",
        "tanh_cart",
        "tanh_angle",
        "tanh_cart_velocity",
        "tanh_pole_velocity",
    )
    classical = (
        GymStructure("P", ("cart_position", "pole_angle")),
        GymStructure("PI", ("cart_position", "pole_angle", "integral_cart", "integral_angle")),
        GymStructure("PD", ("cart_position", "pole_angle", "cart_velocity", "pole_velocity")),
        GymStructure(
            "PID",
            (
                "cart_position",
                "pole_angle",
                "cart_velocity",
                "pole_velocity",
                "integral_cart",
                "integral_angle",
            ),
        ),
    )
    energy_weight, jerk_weight = 0.001, 0.000002

    def prepare_reset(self, env, observation, seed):
        rng = np.random.default_rng(seed + 1701)
        qpos = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.14, 0.14)])
        qvel = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.15, 0.15)])
        env.unwrapped.set_state(qpos, qvel)
        # Vary both moving bodies without changing the XML or observation contract.
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "_lawevo_base_body_mass"):
            unwrapped._lawevo_base_body_mass = unwrapped.model.body_mass.copy()
        unwrapped.model.body_mass[:] = unwrapped._lawevo_base_body_mass
        unwrapped.model.body_mass[1:] *= rng.uniform(0.85, 1.15, size=2)
        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        return unwrapped._get_obs()

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory["integral_cart"] = 0.0
        memory["integral_angle"] = 0.0
        return memory

    def features(self, env, observation, memory, dt):
        del env
        x, angle, xdot, angledot = map(float, observation)
        memory["integral_cart"] = float(np.clip(memory["integral_cart"] + x * dt, -2, 2))
        memory["integral_angle"] = float(np.clip(memory["integral_angle"] + angle * dt, -1, 1))
        scalar = {
            "cart_position": x,
            "pole_angle": angle,
            "cart_velocity": xdot,
            "pole_velocity": angledot,
            "integral_cart": memory["integral_cart"],
            "integral_angle": memory["integral_angle"],
            "tanh_cart": float(np.tanh(x)),
            "tanh_angle": float(np.tanh(5 * angle)),
            "tanh_cart_velocity": float(np.tanh(xdot)),
            "tanh_pole_velocity": float(np.tanh(angledot)),
        }
        return {key: np.array([value]) for key, value in scalar.items()}

    def success(self, env, observation, steps, terminated):
        del env
        return not terminated and steps == self.horizon and abs(observation[1]) < 0.1


class ReacherAdapter(BenchmarkAdapter):
    env_id = "Reacher-v5"
    horizon = 50
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
    )
    energy_weight, jerk_weight = 0.01, 0.00002

    def prepare_reset(self, env, observation, seed):
        rng = np.random.default_rng(seed + 2701)
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "_lawevo_base_body_mass"):
            unwrapped._lawevo_base_body_mass = unwrapped.model.body_mass.copy()
        unwrapped.model.body_mass[:] = unwrapped._lawevo_base_body_mass
        unwrapped.model.body_mass[1:] *= rng.uniform(
            0.9, 1.1, size=len(unwrapped.model.body_mass) - 1
        )
        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        return unwrapped._get_obs()

    def features(self, env, observation, memory, dt):
        unwrapped = env.unwrapped
        body_id = unwrapped.model.body("fingertip").id
        jacobian = np.zeros((3, unwrapped.model.nv))
        mujoco.mj_jacBody(unwrapped.model, unwrapped.data, jacobian, None, body_id)
        jxy = jacobian[:2, :2]
        # Reacher observation stores fingertip - target in the final two elements.
        task_error = -np.asarray(observation[-2:], dtype=float)
        qvel = np.asarray(observation[6:8], dtype=float)
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
        del env, steps, terminated
        return float(np.linalg.norm(observation[-2:])) < 0.05


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
        if not hasattr(unwrapped, "_lawevo_base_body_mass"):
            unwrapped._lawevo_base_body_mass = unwrapped.model.body_mass.copy()
        unwrapped.model.body_mass[:] = unwrapped._lawevo_base_body_mass
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
        memory["integral"] = np.clip(
            memory["integral"] + posture_error * dt, -2.0, 2.0
        )
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


ADAPTERS = {
    "pendulum": PendulumAdapter(),
    "inverted_pendulum": InvertedPendulumAdapter(),
    "reacher": ReacherAdapter(),
}

LOCOMOTION_ADAPTERS = {
    "hopper": HopperAdapter(),
    "walker2d": Walker2dAdapter(),
    "half_cheetah": HalfCheetahAdapter(),
}


def inverted_pendulum_lqr() -> tuple[GymStructure, np.ndarray]:
    """Linearize MuJoCo at the upright equilibrium and solve the discrete Riccati equation."""
    env = gym.make("InvertedPendulum-v5")
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        unwrapped.set_state(np.zeros(2), np.zeros(2))
        unwrapped.data.ctrl[:] = 0.0
        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        a = np.zeros((4, 4))
        b = np.zeros((4, 1))
        mujoco.mjd_transitionFD(unwrapped.model, unwrapped.data, 1e-6, 1, a, b, None, None)
        q = np.diag([1.0, 20.0, 0.1, 0.5])
        r = np.array([[0.1]])
        p = solve_discrete_are(a, b, q, r)
        k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        structure = GymStructure(
            "LQR", ("cart_position", "pole_angle", "cart_velocity", "pole_velocity")
        )
        # The controller applies u=-Kx; DSL terms are summed as gain_i*signal_i.
        return structure, -k.ravel()
    finally:
        env.close()


def run_episode(
    adapter: BenchmarkAdapter,
    structure: GymStructure,
    gains: np.ndarray,
    seed: int,
    *,
    env=None,
):
    owns_env = env is None
    if owns_env:
        env = adapter.make_env()
    try:
        observation, _ = env.reset(seed=seed)
        observation = adapter.prepare_reset(env, observation, seed)
        action_dim = int(np.prod(env.action_space.shape))
        memory = adapter.reset_controller(action_dim)
        dt = float(env.unwrapped.dt) if hasattr(env.unwrapped, "dt") else 0.05
        total_return = energy = jerk = 0.0
        previous = np.zeros(action_dim)
        terminated = False
        steps = 0
        for steps in range(1, adapter.horizon + 1):
            features = adapter.features(env, observation, memory, dt)
            action = sum(gain * features[term] for gain, term in zip(gains, structure.terms))
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            energy += dt * float(action @ action)
            jerk += dt * float(((action - previous) / dt) @ ((action - previous) / dt))
            previous = action.copy()
            if terminated or truncated:
                break
        return GymEpisode(
            total_return,
            adapter.success(env, observation, steps, terminated),
            energy,
            jerk,
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
) -> tuple[GymMetrics, list[GymEpisode]]:
    if not set(structure.terms) <= set(adapter.allowed_terms):
        raise ValueError("structure uses unavailable terms")
    episodes = (
        [run_episode(adapter, structure, gains, seed) for seed in seeds]
        if envs is None
        else [
            run_episode(adapter, structure, gains, seed, env=env)
            for seed, env in zip(seeds, envs, strict=True)
        ]
    )
    episode_return = float(np.mean([item.episode_return for item in episodes]))
    success = float(np.mean([item.success for item in episodes]))
    energy = float(np.mean([item.energy for item in episodes]))
    jerk = float(np.mean([item.jerk for item in episodes]))
    score = adapter.score(episode_return, energy, jerk, len(structure.terms))
    return GymMetrics(score, episode_return, success, energy, jerk, len(structure.terms)), episodes


def tune_gym_cem(
    adapter: BenchmarkAdapter,
    structure: GymStructure,
    seeds: list[int],
    *,
    iterations: int = 5,
    population_size: int = 24,
) -> tuple[np.ndarray, GymMetrics]:
    digest = hashlib.sha256(
        json.dumps({"env": adapter.env_id, "terms": structure.terms}, sort_keys=True).encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))
    dimension = len(structure.terms)
    mean, sigma = np.zeros(dimension), np.full(dimension, 3.0)
    envs = [adapter.make_env() for _ in seeds]
    try:
        best_gains = mean.copy()
        best_metrics, _ = evaluate_gym_structure(
            adapter, structure, best_gains, seeds, envs=envs
        )
        elite_count = max(2, round(0.2 * population_size))
        for _ in range(iterations):
            samples = np.clip(
                rng.normal(mean, sigma, size=(population_size, dimension)), -20, 20
            )
            scored = [
                (
                    sample,
                    evaluate_gym_structure(
                        adapter, structure, sample, seeds, envs=envs
                    )[0],
                )
                for sample in samples
            ]
            scored.sort(key=lambda item: item[1].score, reverse=True)
            elites = np.vstack([item[0] for item in scored[:elite_count]])
            mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
            sigma = np.maximum(0.05, 0.25 * sigma + 0.75 * elites.std(axis=0))
            if scored[0][1].score > best_metrics.score:
                best_gains, best_metrics = scored[0][0].copy(), scored[0][1]
        return best_gains, best_metrics
    finally:
        for env in envs:
            env.close()
