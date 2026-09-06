"""Sampled discrete-time barrier audit of executed controllers, not a certificate.

No control-affine model or global Lipschitz bound is assumed for contact tasks.
The existing BarrierVerifier remains the separate model-based feasibility tool.
"""
from dataclasses import asdict, dataclass

import numpy as np

SUPPORTED_ENVIRONMENTS = frozenset({
    "InvertedPendulum-v5", "LawevoInvertedPendulumPulse-v0",
    "Reacher-v5", "LawevoReacherPayload-v0",
    "PandaReachDense-v3", "LawevoPandaReachMovingSlow-v0", "LawevoPandaReachMovingSlow-v1",
    "PandaPushDense-v3", "LawevoPandaPushLowFriction-v0",
    "PandaSlideDense-v3", "LawevoPandaSlideFrictionShift-v0",
})


@dataclass(frozen=True)
class ControllerBarrierConfig:
    decay_rate: float = 5.0  # s^-1; declared audit choice, not fitted to episodes
    tolerance: float = 1e-6
    reacher_speed_limit: float = 20.0  # rad/s; engineering envelope, not native limit

    def __post_init__(self):
        if (not np.isfinite([self.decay_rate, self.tolerance, self.reacher_speed_limit]).all()
                or self.decay_rate <= 0 or self.tolerance < 0 or self.reacher_speed_limit <= 0):
            raise ValueError("invalid controller barrier configuration")


def barrier_spec(env_id, config=None):
    if env_id not in SUPPORTED_ENVIRONMENTS:
        return None
    config = config or ControllerBarrierConfig()
    if "InvertedPendulum" in env_id:
        constraints = ["h=1-(pole_angle/0.2 rad)^2; native upright episode boundary"]
    elif "Reacher" in env_id:
        constraints = ["h_i=1-(qdot_i/reacher_speed_limit)^2, two arm joints; declared engineering envelope"]
    else:
        constraints = ["h_i=1-((q_i-mid_i)/half_range_i)^2; native Panda arm joint position limits"]
        if "Reach" not in env_id:
            constraints += ["object AABB inside table AABB in world x/y; each edge clearance divided by table width",
                            "object top >= table top; vertical margin divided by 0.1 m (fall diagnostic)"]
    return {
        "version": 1, "method": "sampled executed-controller discrete-time barrier audit",
        "constraints": constraints, "config": asdict(config),
        "residual": "h(x_next)-exp(-decay_rate*dt)*h(x); only checked when h(x)>=-tolerance",
        "certified_between_samples": False,
        "coverage": "initial state and control-step endpoints of reported test seeds only; no substep or unseen-state guarantee",
        "selection": "report only; no filtering or reranking; success metrics unchanged",
        "not_checked": "global stability, collision freedom, contact forces, robustness beyond tested trajectories",
    }


class ControllerBarrierVerifier:
    """Collect margins and counterexamples without changing controller actions."""

    def __init__(self, env_id, env, observation, dt, seed, config=None):
        self.config = config or ControllerBarrierConfig()
        self.spec = barrier_spec(env_id, self.config)
        if self.spec is None:
            raise ValueError(f"no controller barrier profile for {env_id}")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.env_id, self.dt, self.seed = env_id, dt, int(seed)
        self.steps = self.numeric_failures = self.saturated_steps = 0
        self.unsafe_samples = self.residual_failures = 0
        self.first_failure = None
        self.minimum_margin = self.minimum_residual = None
        self.previous = self.margins(env, observation)
        self._check_margins(self.previous, 0)

    def margins(self, env, observation):
        base = env.unwrapped
        if "InvertedPendulum" in self.env_id:
            return {"pole_angle": 1 - (float(base.data.qpos[1]) / 0.2)**2}
        if "Reacher" in self.env_id:
            return {f"joint_speed_{i}": 1-(float(v)/self.config.reacher_speed_limit)**2
                    for i, v in enumerate(base.data.qvel[:2])}
        sim, robot = base.sim, base.robot
        client = sim.physics_client
        robot_id = sim._bodies_idx[robot.body_name]
        margins = {}
        for index in robot.joint_indices[:7]:
            info = client.getJointInfo(robot_id, int(index))
            low, high = float(info[8]), float(info[9])
            if high <= low:
                continue
            q = float(client.getJointState(robot_id, int(index))[0])
            margins[f"joint_position_{index}"] = 1-((q-(low+high)/2)/((high-low)/2))**2
        if "Reach" not in self.env_id:
            table_low, table_high = map(np.asarray, client.getAABB(sim._bodies_idx["table"]))
            obj_low, obj_high = map(np.asarray, client.getAABB(sim._bodies_idx["object"]))
            for i, axis in enumerate("xy"):
                width = float(table_high[i]-table_low[i])
                margins[f"object_{axis}_low"] = float((obj_low[i]-table_low[i])/width)
                margins[f"object_{axis}_high"] = float((table_high[i]-obj_high[i])/width)
            margins["object_not_fallen"] = float((obj_high[2]-table_high[2])/0.1)
        if not margins:
            raise ValueError("barrier profile produced no constraints")
        return margins

    def _failure(self, kind, step, barrier=None, value=None):
        if self.first_failure is None:
            self.first_failure = {"kind": kind, "seed": self.seed, "step": step,
                                  "barrier": barrier, "value": value if value is None or np.isfinite(value) else None}

    def _check_margins(self, margins, step):
        for name, value in margins.items():
            if not np.isfinite(value):
                self.numeric_failures += 1
                self._failure("nonfinite_state", step, name)
            else:
                self.minimum_margin = value if self.minimum_margin is None else min(self.minimum_margin, value)
                if value < -self.config.tolerance:
                    self.unsafe_samples += 1
                    self._failure("outside_envelope", step, name, value)

    def check_action(self, raw, low, high, features_valid=True):
        raw = np.asarray(raw)
        if raw.shape == ():
            raw = np.full(low.shape, raw.item())  # DSL explicitly permits scalar broadcast.
        if not features_valid or raw.shape != low.shape or not np.isfinite(raw).all():
            self.numeric_failures += 1
            self._failure("invalid_raw_action_or_signal", self.steps + 1)
        elif np.any((raw < low) | (raw > high)):
            self.saturated_steps += 1

    def update(self, env, observation):
        self.steps += 1
        current = self.margins(env, observation)
        self._check_margins(current, self.steps)
        for name, value in current.items():
            previous = self.previous[name]
            if not np.isfinite([value, previous]).all() or previous < -self.config.tolerance:
                continue
            residual = float(value - np.exp(-self.config.decay_rate*self.dt)*previous)
            self.minimum_residual = residual if self.minimum_residual is None else min(self.minimum_residual, residual)
            if residual < -self.config.tolerance:
                self.residual_failures += 1
                self._failure("negative_barrier_residual", self.steps, name, residual)
        self.previous = current

    def report(self):
        passed = self.steps > 0 and not (self.numeric_failures or self.unsafe_samples or self.residual_failures)
        return {"status": "passed_sampled_checks" if passed else "failed_sampled_checks",
                "certified_between_samples": False, "seed": self.seed, "steps": self.steps,
                "envelope_satisfied": self.steps > 0 and not (self.numeric_failures or self.unsafe_samples),
                "numeric_failures": self.numeric_failures, "outside_envelope_count": self.unsafe_samples,
                "negative_residual_count": self.residual_failures, "saturated_steps": self.saturated_steps,
                "minimum_margin": self.minimum_margin, "minimum_residual": self.minimum_residual,
                "first_failure": self.first_failure, "spec": self.spec}
