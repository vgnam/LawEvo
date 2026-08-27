from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from lawevo.filter import CBFSafetyFilter, InfeasibleSafetyFilter
from lawevo.robot.base import Array, RobotInterface

NominalPolicy = Callable[[dict[str, object], Array, Sequence[object]], Array]


@dataclass(frozen=True)
class RolloutConfig:
    dt: float = 0.05
    steps: int = 250
    goal_tolerance: float = 0.15
    collision_tolerance: float = 1e-6


@dataclass(frozen=True)
class Trajectory:
    states: Array
    controls: Array
    nominal_controls: Array
    cbf_residuals: Array
    reached_goal: bool
    safety_violation: bool
    filter_infeasible: bool
    energy: float
    jerk: float
    intervention: float
    task_reward: float

    def fitness(self, energy_weight: float = 0.02, jerk_weight: float = 0.01) -> float:
        if self.safety_violation or self.filter_infeasible:
            return -1e6
        return self.task_reward - energy_weight * self.energy - jerk_weight * self.jerk


def rollout(
    robot: RobotInterface,
    policy: NominalPolicy,
    safety_filter: CBFSafetyFilter,
    initial_state: Array,
    goal: Array,
    config: RolloutConfig | None = None,
) -> Trajectory:
    cfg = config or RolloutConfig()
    x = robot.validate_state(np.asarray(initial_state, dtype=float)).copy()
    goal = np.asarray(goal, dtype=float)
    states = [x.copy()]
    controls: list[Array] = []
    nominal_controls: list[Array] = []
    residuals: list[float] = []
    violation = False
    infeasible = False
    reached = False
    intervention = 0.0
    obstacles = tuple(robot.primitive_params().get("obstacles", ()))

    for step_index in range(cfg.steps):
        state_view: dict[str, object] = {
            "x": x.copy(),
            "pos": x[:2].copy(),
            "theta": float(x[2]),
            "step": step_index,
        }
        nominal = np.asarray(policy(state_view, goal, obstacles), dtype=float)
        try:
            result = safety_filter.filter(x, nominal)
        except InfeasibleSafetyFilter:
            infeasible = True
            break
        x = robot.step(x, result.control, cfg.dt)
        states.append(x.copy())
        controls.append(result.control)
        nominal_controls.append(nominal)
        residuals.append(result.cbf_residual)
        intervention += result.intervention_norm
        barrier_value, _ = safety_filter.barrier.value_gradient(x, robot)
        if barrier_value < -cfg.collision_tolerance:
            violation = True
            break
        if float(np.linalg.norm(x[:2] - goal[:2])) <= cfg.goal_tolerance:
            reached = True
            break

    control_array = np.asarray(controls, dtype=float).reshape((-1, robot.control_dim()))
    nominal_array = np.asarray(nominal_controls, dtype=float).reshape((-1, robot.control_dim()))
    energy = cfg.dt * float(np.sum(control_array * control_array))
    if len(control_array) >= 2:
        rates = np.diff(control_array, axis=0) / cfg.dt
        jerk = cfg.dt * float(np.sum(rates * rates))
    else:
        jerk = 0.0
    final_distance = float(np.linalg.norm(states[-1][:2] - goal[:2]))
    reward = (100.0 if reached else 0.0) - final_distance - 0.1 * len(controls)
    return Trajectory(
        np.asarray(states),
        control_array,
        nominal_array,
        np.asarray(residuals),
        reached,
        violation,
        infeasible,
        energy,
        jerk,
        intervention,
        reward,
    )
