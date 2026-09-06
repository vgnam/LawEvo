"""Single-factor variants for the five-task classical-control benchmark.

Factories are registered lazily. They wrap stock simulators, preserving their
reset distributions, observation/action contracts and rewards. Public adapters
share the same signals and classical structures as their corresponding base.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

from lawevo.pid.gym_benchmark import InvertedPendulumAdapter, ReacherAdapter
from lawevo.pid.panda_gym_benchmark import (
    PandaPushAdapter,
    PandaReachAdapter,
    PandaSlideAdapter,
)
from lawevo.pid.panda_gym_variants import (
    PandaReachMovingAdapter,
    _install_quiet_bullet_client,
)


class PulseWrapper(gym.Wrapper):
    """Apply one horizontal force pulse for exactly one control interval."""

    def __init__(self, env, pulse_step=250, pulse_force=5.0):
        super().__init__(env)
        self.pulse_step = int(pulse_step)
        self.pulse_force = float(pulse_force)
        self.elapsed_steps = 0
        self.cart_id = self.unwrapped.model.body("cart").id

    def reset(self, **kwargs):
        self.elapsed_steps = 0
        observation, info = self.env.reset(**kwargs)
        self.unwrapped.data.xfrc_applied[:] = 0.0
        return observation, info

    def step(self, action):
        self.elapsed_steps += 1
        force = self.pulse_force if self.elapsed_steps == self.pulse_step else 0.0
        data = self.unwrapped.data
        data.xfrc_applied[self.cart_id, 0] = force
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
        finally:
            data.xfrc_applied[self.cart_id, 0] = 0.0
        return observation, reward, terminated, truncated, {**info, "pulse_force": force}


class MovingSlowWrapper(gym.Wrapper):
    """Translate the goal along a bounded orbit, without terminating on contact."""

    def __init__(self, env, amplitude=0.025, period=5.0):
        super().__init__(env)
        self.amplitude = float(amplitude)
        self.period = float(period)
        self.elapsed = 0.0
        self.initial_goal = np.zeros(3)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.elapsed = 0.0
        self.initial_goal = self.unwrapped.task.goal.copy()
        return observation, info

    def step(self, action):
        self.elapsed += float(self.unwrapped.sim.dt)
        phase = 2 * np.pi * self.elapsed / self.period
        # Offset is zero at reset, preserving the paired initial observation.
        offset = self.amplitude * np.array([np.sin(phase), 0.5 * np.sin(2 * phase), 0.0])
        task = self.unwrapped.task
        task.goal = self.initial_goal + offset
        task.sim.set_base_pose("target", task.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        observation, reward, _, truncated, info = self.env.step(action)
        return observation, reward, False, truncated, info


def make_pulse_env(pulse_step=250, pulse_force=5.0, **kwargs):
    return PulseWrapper(
        gym.make("InvertedPendulum-v5", max_episode_steps=500, **kwargs),
        pulse_step, pulse_force,
    )


def make_payload_env(payload_fraction=0.15, **kwargs):
    if payload_fraction < 0:
        raise ValueError("payload_fraction must be nonnegative")
    env = gym.make("Reacher-v5", max_episode_steps=50, **kwargs)
    model, data = env.unwrapped.model, env.unwrapped.data
    tip = model.body("fingertip").id
    # Stock fingertip is a centered 1 cm sphere. Add a concentric payload;
    # no center-of-mass shift is needed and all three principal inertias grow.
    payload_mass = float(payload_fraction) * float(model.body_mass[model.body("body1").id])
    model.body_mass[tip] += payload_mass
    model.body_inertia[tip] += (2.0 / 5.0) * payload_mass * 0.01**2
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    return env


def _panda_env(env_id, horizon, **kwargs):
    import panda_gym  # noqa: F401 -- registers stock environments

    _install_quiet_bullet_client()
    kwargs.setdefault("render_mode", "rgb_array")
    kwargs.setdefault("renderer", "Tiny")
    return gym.make(env_id, max_episode_steps=horizon, **kwargs)


def make_moving_slow_env(amplitude=0.025, period=5.0, **kwargs):
    return MovingSlowWrapper(
        _panda_env("PandaReachDense-v3", PandaReachAdapter.horizon, **kwargs),
        amplitude, period,
    )


def make_moving_slow_v1_env(**kwargs):
    env = make_moving_slow_env(**kwargs)
    env.unwrapped.task.distance_threshold = 0.01
    return env


def _friction_env(env_id, factor, **kwargs):
    if factor <= 0:
        raise ValueError("friction factor must be positive")
    env = _panda_env(env_id, 50, **kwargs)
    sim = env.unwrapped.sim
    table = sim._bodies_idx["table"]
    nominal = sim.physics_client.getDynamicsInfo(table, -1)[1]
    sim.physics_client.changeDynamics(table, -1, lateralFriction=nominal * factor)
    return env


def make_low_friction_env(friction_factor=0.75, **kwargs):
    return _friction_env("PandaPushDense-v3", friction_factor, **kwargs)


def make_friction_shift_env(friction_factor=1.25, **kwargs):
    return _friction_env("PandaSlideDense-v3", friction_factor, **kwargs)


VARIANT_FACTORIES = {
    "LawevoInvertedPendulumPulse-v0": make_pulse_env,
    "LawevoReacherPayload-v0": make_payload_env,
    "LawevoPandaReachMovingSlow-v0": make_moving_slow_env,
    "LawevoPandaReachMovingSlow-v1": make_moving_slow_v1_env,
    "LawevoPandaPushLowFriction-v0": make_low_friction_env,
    "LawevoPandaSlideFrictionShift-v0": make_friction_shift_env,
}


class _RegisteredVariant:
    def make_env(self):
        for env_id, factory in VARIANT_FACTORIES.items():
            if env_id not in gym.registry:
                gym.register(env_id, entry_point=factory)
        return gym.make(self.env_id, max_episode_steps=self.horizon)


class InvertedPendulumPulseAdapter(_RegisteredVariant, InvertedPendulumAdapter):
    env_id = "LawevoInvertedPendulumPulse-v0"


class ReacherPayloadAdapter(_RegisteredVariant, ReacherAdapter):
    env_id = "LawevoReacherPayload-v0"


class PandaReachMovingSlowAdapter(_RegisteredVariant, PandaReachMovingAdapter):
    env_id = "LawevoPandaReachMovingSlow-v0"
    horizon = PandaReachAdapter.horizon
    classical = PandaReachAdapter.classical + (
        PandaReachMovingAdapter.classical[2],  # Feedforward P
        PandaReachMovingAdapter.classical[3],  # Tracking PD
    )


class PandaReachMovingSlowV1Adapter(PandaReachMovingSlowAdapter):
    """Same orbit and physics; one-centimetre tracking/final-position radius."""

    env_id = "LawevoPandaReachMovingSlow-v1"
    tracking_radius = 0.01

    @property
    def progress_spec(self):
        return {"version": 1, "radius_m": self.tracking_radius, "warmup_steps": 10,
                "milestones": ["inside radius once after warmup", "tracking rate >= 0.3",
                               "tracking rate >= 0.6", "final position inside radius"],
                "aggregation": "equal mean over milestones then episodes; diagnostic only"}


class PandaPushLowFrictionAdapter(_RegisteredVariant, PandaPushAdapter):
    env_id = "LawevoPandaPushLowFriction-v0"


class PandaSlideFrictionShiftAdapter(_RegisteredVariant, PandaSlideAdapter):
    env_id = "LawevoPandaSlideFrictionShift-v0"


CONTROLLED_VARIANT_ADAPTERS = {
    "inverted_pendulum_pulse": InvertedPendulumPulseAdapter(),
    "reacher_payload": ReacherPayloadAdapter(),
    "panda_reach_moving_slow": PandaReachMovingSlowAdapter(),
    "panda_reach_moving_slow_v1": PandaReachMovingSlowV1Adapter(),
    "panda_push_low_friction": PandaPushLowFrictionAdapter(),
    "panda_slide_friction_shift": PandaSlideFrictionShiftAdapter(),
}
