from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from lawevo.pid.benchmark import PIDMetrics, PIDScenario, wrap_angle
from lawevo.robot import UnicycleRobot
from lawevo.robot.base import Array

DISTANCE_TERMS = (
    "error",
    "integral",
    "derivative",
    "tanh_error",
    "sqrt_error",
    "quadratic_error",
    "tanh_derivative",
    "heading_coupling",
)
HEADING_TERMS = (
    "error",
    "integral",
    "derivative",
    "tanh_error",
    "sqrt_error",
    "quadratic_error",
    "tanh_derivative",
    "distance_coupling",
)
VELOCITY_GATES = ("positive_cosine", "cosine", "none")


@dataclass(frozen=True)
class ControllerStructure:
    name: str
    distance_terms: tuple[str, ...]
    heading_terms: tuple[str, ...]
    velocity_gate: str = "positive_cosine"

    def __post_init__(self) -> None:
        if not 1 <= len(self.distance_terms) <= 6 or not 1 <= len(self.heading_terms) <= 6:
            raise ValueError("each channel must contain 1-6 terms")
        if len(set(self.distance_terms)) != len(self.distance_terms):
            raise ValueError("distance terms must be unique")
        if len(set(self.heading_terms)) != len(self.heading_terms):
            raise ValueError("heading terms must be unique")
        if not set(self.distance_terms) <= set(DISTANCE_TERMS):
            raise ValueError("unknown distance term")
        if not set(self.heading_terms) <= set(HEADING_TERMS):
            raise ValueError("unknown heading term")
        if self.velocity_gate not in VELOCITY_GATES:
            raise ValueError("unknown velocity gate")

    @property
    def parameter_count(self) -> int:
        return len(self.distance_terms) + len(self.heading_terms)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["distance_terms"] = list(self.distance_terms)
        data["heading_terms"] = list(self.heading_terms)
        return data

    def formula(self, gains: Array) -> str:
        distance = " + ".join(
            f"({gain:.4g})·{term}" for gain, term in zip(gains, self.distance_terms)
        )
        heading = " + ".join(
            f"({gain:.4g})·{term}"
            for gain, term in zip(gains[len(self.distance_terms) :], self.heading_terms)
        )
        return f"v = gate[{self.velocity_gate}]({distance}); omega = {heading}"


@dataclass(frozen=True)
class StructureTrajectory:
    success: bool
    final_distance: float
    settling_time: float
    energy: float
    jerk: float
    mean_heading_error: float


def _feature_values(
    error: float, integral: float, derivative: float, other: float
) -> dict[str, float]:
    return {
        "error": error,
        "integral": integral,
        "derivative": derivative,
        "tanh_error": float(np.tanh(error)),
        "sqrt_error": float(np.sign(error) * np.sqrt(abs(error))),
        "quadratic_error": error * abs(error),
        "tanh_derivative": float(np.tanh(derivative)),
        "heading_coupling": error * max(0.0, float(np.cos(other))),
        "distance_coupling": error * min(abs(other), 3.0),
    }


def simulate_structure(
    structure: ControllerStructure,
    gains: Array,
    scenario: PIDScenario,
    *,
    dt: float = 0.05,
    steps: int = 240,
    tolerance: float = 0.12,
) -> StructureTrajectory:
    gains = np.asarray(gains, dtype=float)
    if gains.shape != (structure.parameter_count,):
        raise ValueError("gain vector does not match structure")
    split = len(structure.distance_terms)
    distance_gains, heading_gains = gains[:split], gains[split:]
    robot = UnicycleRobot(workspace=((-5.0, 5.0), (-5.0, 5.0)))
    x = scenario.initial_state.astype(float).copy()
    controls: list[Array] = []
    distance_integral = heading_integral = 0.0
    previous_distance: float | None = None
    previous_heading: float | None = None
    heading_errors: list[float] = []
    success = False
    settling_time = steps * dt

    for index in range(steps):
        delta = scenario.goal - x[:2]
        distance = float(np.linalg.norm(delta))
        heading_error = wrap_angle(float(np.arctan2(delta[1], delta[0])) - x[2])
        distance_integral = float(np.clip(distance_integral + distance * dt, -5.0, 5.0))
        heading_integral = float(np.clip(heading_integral + heading_error * dt, -2.5, 2.5))
        distance_derivative = (
            0.0 if previous_distance is None else (distance - previous_distance) / dt
        )
        heading_derivative = (
            0.0 if previous_heading is None else wrap_angle(heading_error - previous_heading) / dt
        )
        distance_features = _feature_values(
            distance, distance_integral, distance_derivative, heading_error
        )
        heading_features = _feature_values(
            heading_error, heading_integral, heading_derivative, distance
        )
        v = float(
            distance_gains @ np.asarray([distance_features[t] for t in structure.distance_terms])
        )
        omega = float(
            heading_gains @ np.asarray([heading_features[t] for t in structure.heading_terms])
        )
        if structure.velocity_gate == "positive_cosine":
            v *= max(0.0, float(np.cos(heading_error)))
        elif structure.velocity_gate == "cosine":
            v *= float(np.cos(heading_error))
        control = np.clip(np.array([v, omega]), *robot.control_bounds())
        x = robot.step(x, control, dt)
        controls.append(control)
        heading_errors.append(abs(heading_error))
        previous_distance, previous_heading = distance, heading_error
        if distance <= tolerance:
            success = True
            settling_time = (index + 1) * dt
            break

    controls_array = np.asarray(controls, dtype=float).reshape((-1, 2))
    final_distance = float(np.linalg.norm(scenario.goal - x[:2]))
    energy = dt * float(np.sum(controls_array**2))
    jerk = (
        dt * float(np.sum((np.diff(controls_array, axis=0) / dt) ** 2))
        if len(controls_array) > 1
        else 0.0
    )
    return StructureTrajectory(
        success,
        final_distance,
        settling_time,
        energy,
        jerk,
        float(np.mean(heading_errors)),
    )


def evaluate_structure(
    structure: ControllerStructure, gains: Array, scenarios: list[PIDScenario]
) -> PIDMetrics:
    runs = [simulate_structure(structure, gains, scenario) for scenario in scenarios]
    success = float(np.mean([run.success for run in runs]))
    final_distance = float(np.mean([run.final_distance for run in runs]))
    settling = float(np.mean([run.settling_time for run in runs]))
    energy = float(np.mean([run.energy for run in runs]))
    jerk = float(np.mean([run.jerk for run in runs]))
    heading = float(np.mean([run.mean_heading_error for run in runs]))
    complexity = structure.parameter_count
    score = (
        120.0 * success
        - 10.0 * final_distance
        - 1.2 * settling
        - 0.10 * energy
        - 0.003 * jerk
        - 2.0 * heading
        - 0.08 * complexity
    )
    return PIDMetrics(score, success, final_distance, settling, energy, jerk, heading)


def tune_cem(
    structure: ControllerStructure,
    scenarios: list[PIDScenario],
    *,
    iterations: int = 7,
    population_size: int = 32,
    elite_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[Array, PIDMetrics, list[float]]:
    if seed is None:
        seed_material = {
            "distance_terms": structure.distance_terms,
            "heading_terms": structure.heading_terms,
            "velocity_gate": structure.velocity_gate,
        }
        digest = hashlib.sha256(json.dumps(seed_material, sort_keys=True).encode()).digest()
        seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed)
    dimension = structure.parameter_count
    lower = np.full(dimension, -6.0)
    upper = np.full(dimension, 6.0)
    # Positive initialization favors the conventional stabilizing feedback direction.
    mean = np.full(dimension, 1.0)
    sigma = np.full(dimension, 2.0)
    elite_count = max(2, round(population_size * elite_fraction))
    best_gains = mean.copy()
    best_metrics = evaluate_structure(structure, best_gains, scenarios)
    history = [best_metrics.score]
    for _ in range(iterations):
        samples = np.clip(rng.normal(mean, sigma, size=(population_size, dimension)), lower, upper)
        scored = [(sample, evaluate_structure(structure, sample, scenarios)) for sample in samples]
        scored.sort(key=lambda item: item[1].score, reverse=True)
        elites = np.vstack([item[0] for item in scored[:elite_count]])
        mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
        sigma = np.maximum(0.05, 0.25 * sigma + 0.75 * elites.std(axis=0))
        if scored[0][1].score > best_metrics.score:
            best_gains, best_metrics = scored[0][0].copy(), scored[0][1]
        history.append(best_metrics.score)
    return best_gains, best_metrics, history
