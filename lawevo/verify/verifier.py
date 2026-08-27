from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from lawevo.dsl.ast import Barrier
from lawevo.robot.base import Array, RobotInterface


@dataclass(frozen=True)
class VerificationConfig:
    safety_margin: float = 0.25
    max_grid_points: int = 100_000
    k_max: float = 50.0
    bisection_iterations: int = 32
    feasibility_tolerance: float = 1e-9
    safe_set_tolerance: float = 1e-9
    # If supplied, this must bound the Lipschitz constant of the complete maximized
    # CBF residual, not merely h. It turns sampled verification into a grid certificate.
    residual_lipschitz: float | None = None

    def __post_init__(self) -> None:
        if self.safety_margin <= 0 or self.max_grid_points <= 0 or self.k_max < 0:
            raise ValueError("invalid verification configuration")


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    alpha: float | None
    reason: str
    sampled_points: int
    safe_points: int
    worst_residual: float | None
    worst_state: Array | None
    barrier_lipschitz: float
    grid_radius: float
    certified_between_samples: bool


class BarrierVerifier:
    def __init__(self, robot: RobotInterface, config: VerificationConfig | None = None) -> None:
        self.robot = robot
        self.config = config or VerificationConfig()

    def _grid(self, barrier_lipschitz: float) -> tuple[Array, float]:
        lower, upper = self.robot.state_bounds()
        widths = upper - lower
        if np.any(widths <= 0):
            raise ValueError("state bounds must have positive width")
        spacing = self.config.safety_margin / max(barrier_lipschitz, 1e-12)
        counts = np.maximum(2, np.ceil(widths / spacing).astype(int) + 1)
        total = int(np.prod(counts, dtype=object))
        if total > self.config.max_grid_points:
            scale = (self.config.max_grid_points / total) ** (1.0 / len(counts))
            counts = np.maximum(2, np.floor(counts * scale).astype(int))
            while int(np.prod(counts, dtype=object)) > self.config.max_grid_points:
                counts[int(np.argmax(counts))] -= 1
        axes = [np.linspace(lo, hi, count) for lo, hi, count in zip(lower, upper, counts)]
        points = np.asarray(list(product(*axes)), dtype=float)
        actual_spacing = widths / np.maximum(counts - 1, 1)
        radius = 0.5 * float(np.linalg.norm(actual_spacing))
        return points, radius

    def _residuals(
        self, barrier: Barrier, points: Array, k: float, coverage_margin: float
    ) -> tuple[Array, Array]:
        u_min, u_max = self.robot.control_bounds()
        residuals: list[float] = []
        safe_states: list[Array] = []
        for x in points:
            value, gradient = barrier.value_gradient(x, self.robot)
            if value < -self.config.safe_set_tolerance:
                continue
            lf = float(gradient @ self.robot.drift(x))
            lg = gradient @ self.robot.control_matrix(x)
            maximizing_u = np.where(lg >= 0.0, u_max, u_min)
            residuals.append(lf + float(lg @ maximizing_u) + k * value - coverage_margin)
            safe_states.append(x)
        return np.asarray(residuals), np.asarray(safe_states)

    def _feasible(
        self, barrier: Barrier, points: Array, k: float, coverage_margin: float
    ) -> tuple[bool, float | None, Array | None, int]:
        residuals, safe_states = self._residuals(barrier, points, k, coverage_margin)
        if not len(residuals):
            return False, None, None, 0
        index = int(np.argmin(residuals))
        worst = float(residuals[index])
        return (
            worst >= -self.config.feasibility_tolerance,
            worst,
            safe_states[index].copy(),
            len(residuals),
        )

    def verify(self, barrier: Barrier) -> VerificationResult:
        barrier.validate(self.robot)
        lipschitz = float(barrier.lipschitz(self.robot))
        points, radius = self._grid(lipschitz)
        certified = self.config.residual_lipschitz is not None
        coverage_margin = (self.config.residual_lipschitz or 0.0) * radius

        feasible_zero, worst, state, safe_count = self._feasible(
            barrier, points, 0.0, coverage_margin
        )
        if feasible_zero:
            return VerificationResult(
                True,
                0.0,
                "feasible",
                len(points),
                safe_count,
                worst,
                state,
                lipschitz,
                radius,
                certified,
            )

        feasible_high, worst, state, safe_count = self._feasible(
            barrier, points, self.config.k_max, coverage_margin
        )
        if not feasible_high:
            reason = "safe set has no sampled states" if safe_count == 0 else "infeasible at k_max"
            return VerificationResult(
                False,
                None,
                reason,
                len(points),
                safe_count,
                worst,
                state,
                lipschitz,
                radius,
                certified,
            )

        low, high = 0.0, self.config.k_max
        for _ in range(self.config.bisection_iterations):
            mid = (low + high) / 2.0
            feasible, _, _, _ = self._feasible(barrier, points, mid, coverage_margin)
            if feasible:
                high = mid
            else:
                low = mid
        feasible, worst, state, safe_count = self._feasible(barrier, points, high, coverage_margin)
        return VerificationResult(
            feasible,
            high if feasible else None,
            "feasible" if feasible else "infeasible",
            len(points),
            safe_count,
            worst,
            state,
            lipschitz,
            radius,
            certified,
        )
