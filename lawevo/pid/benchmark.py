from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from lawevo.robot import UnicycleRobot
from lawevo.robot.base import Array


def wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)


@dataclass(frozen=True)
class PIDGains:
    kp_distance: float
    ki_distance: float
    kd_distance: float
    kp_heading: float
    ki_heading: float
    kd_heading: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PIDBounds:
    kp_distance: tuple[float, float] = (0.1, 3.0)
    ki_distance: tuple[float, float] = (0.0, 0.5)
    kd_distance: tuple[float, float] = (0.0, 1.0)
    kp_heading: tuple[float, float] = (0.2, 6.0)
    ki_heading: tuple[float, float] = (0.0, 0.5)
    kd_heading: tuple[float, float] = (0.0, 1.5)

    def contains(self, gains: PIDGains) -> bool:
        return all(
            np.isfinite(value) and lower <= value <= upper
            for value, (lower, upper) in zip(gains.to_dict().values(), asdict(self).values())
        )


@dataclass(frozen=True)
class PIDScenario:
    initial_state: Array
    goal: Array


@dataclass(frozen=True)
class PIDMetrics:
    score: float
    success_rate: float
    mean_final_distance: float
    mean_settling_time: float
    mean_energy: float
    mean_jerk: float
    mean_heading_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PIDTrajectory:
    states: Array
    controls: Array
    distances: Array
    success: bool
    final_distance: float
    settling_time: float
    energy: float
    jerk: float
    mean_heading_error: float


def generate_scenarios(count: int, seed: int) -> list[PIDScenario]:
    rng = np.random.default_rng(seed)
    scenarios: list[PIDScenario] = []
    while len(scenarios) < count:
        start = rng.uniform(-3.5, 3.5, size=2)
        goal = rng.uniform(-3.5, 3.5, size=2)
        if np.linalg.norm(goal - start) < 2.0:
            continue
        heading = rng.uniform(-np.pi, np.pi)
        scenarios.append(PIDScenario(np.array([*start, heading]), goal))
    return scenarios


def simulate_pid(
    gains: PIDGains,
    scenario: PIDScenario,
    *,
    dt: float = 0.05,
    steps: int = 240,
    tolerance: float = 0.12,
) -> PIDTrajectory:
    robot = UnicycleRobot(workspace=((-5.0, 5.0), (-5.0, 5.0)))
    x = scenario.initial_state.astype(float).copy()
    states = [x.copy()]
    controls: list[Array] = []
    distances: list[float] = []
    distance_integral = 0.0
    heading_integral = 0.0
    previous_distance: float | None = None
    previous_heading: float | None = None
    heading_errors: list[float] = []
    success = False
    settling_time = steps * dt

    for index in range(steps):
        delta = scenario.goal - x[:2]
        distance = float(np.linalg.norm(delta))
        target_heading = float(np.arctan2(delta[1], delta[0]))
        heading_error = wrap_angle(target_heading - x[2])
        distance_integral = float(np.clip(distance_integral + distance * dt, -5.0, 5.0))
        heading_integral = float(np.clip(heading_integral + heading_error * dt, -2.5, 2.5))
        distance_derivative = (
            0.0 if previous_distance is None else (distance - previous_distance) / dt
        )
        heading_derivative = (
            0.0 if previous_heading is None else wrap_angle(heading_error - previous_heading) / dt
        )
        v = (
            gains.kp_distance * distance
            + gains.ki_distance * distance_integral
            + gains.kd_distance * distance_derivative
        )
        # Suppress forward motion when the robot points away from the goal.
        v *= max(0.0, float(np.cos(heading_error)))
        omega = (
            gains.kp_heading * heading_error
            + gains.ki_heading * heading_integral
            + gains.kd_heading * heading_derivative
        )
        control = np.clip(np.array([v, omega]), *robot.control_bounds())
        x = robot.step(x, control, dt)
        states.append(x.copy())
        controls.append(control)
        distances.append(distance)
        heading_errors.append(abs(heading_error))
        previous_distance = distance
        previous_heading = heading_error
        if distance <= tolerance:
            success = True
            settling_time = (index + 1) * dt
            break

    controls_array = np.asarray(controls, dtype=float).reshape((-1, 2))
    final_distance = float(np.linalg.norm(scenario.goal - states[-1][:2]))
    energy = dt * float(np.sum(controls_array**2))
    jerk = (
        dt * float(np.sum((np.diff(controls_array, axis=0) / dt) ** 2))
        if len(controls_array) > 1
        else 0.0
    )
    return PIDTrajectory(
        states=np.asarray(states),
        controls=controls_array,
        distances=np.asarray(distances),
        success=success,
        final_distance=final_distance,
        settling_time=settling_time,
        energy=energy,
        jerk=jerk,
        mean_heading_error=float(np.mean(heading_errors)),
    )


def evaluate_gains(gains: PIDGains, scenarios: list[PIDScenario]) -> PIDMetrics:
    trajectories = [simulate_pid(gains, scenario) for scenario in scenarios]
    success_rate = float(np.mean([trajectory.success for trajectory in trajectories]))
    final_distance = float(np.mean([trajectory.final_distance for trajectory in trajectories]))
    settling_time = float(np.mean([trajectory.settling_time for trajectory in trajectories]))
    energy = float(np.mean([trajectory.energy for trajectory in trajectories]))
    jerk = float(np.mean([trajectory.jerk for trajectory in trajectories]))
    heading_error = float(np.mean([trajectory.mean_heading_error for trajectory in trajectories]))
    score = (
        120.0 * success_rate
        - 10.0 * final_distance
        - 1.2 * settling_time
        - 0.08 * energy
        - 0.0003 * jerk
        - 2.0 * heading_error
    )
    return PIDMetrics(
        score, success_rate, final_distance, settling_time, energy, jerk, heading_error
    )


def grid_tune(scenarios: list[PIDScenario]) -> tuple[PIDGains, PIDMetrics]:
    axes = (
        (0.8, 1.2, 1.6),
        (0.0, 0.05, 0.1),
        (0.0, 0.1, 0.2),
        (1.8, 2.6, 3.4),
        (0.0, 0.04),
        (0.0, 0.15),
    )
    best_gains: PIDGains | None = None
    best_metrics: PIDMetrics | None = None
    for values in product(*axes):
        gains = PIDGains(*values)
        metrics = evaluate_gains(gains, scenarios)
        if best_metrics is None or metrics.score > best_metrics.score:
            best_gains, best_metrics = gains, metrics
    assert best_gains is not None and best_metrics is not None
    return best_gains, best_metrics
