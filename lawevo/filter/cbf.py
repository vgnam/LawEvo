from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from lawevo.dsl.ast import Barrier
from lawevo.robot.base import Array, RobotInterface


class InfeasibleSafetyFilter(RuntimeError):
    pass


@dataclass(frozen=True)
class CBFFilterResult:
    control: Array
    nominal_control: Array
    intervened: bool
    intervention_norm: float
    cbf_residual: float


class CBFSafetyFilter:
    """Exact Euclidean projection onto a box intersected with one CBF half-space.

    The active-set enumeration is dependency-free and intended for low-dimensional
    robot controls. It replaces a general QP dependency for the MVP.
    """

    def __init__(self, robot: RobotInterface, barrier: Barrier, alpha: float) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        barrier.validate(robot)
        self.robot = robot
        self.barrier = barrier
        self.alpha = float(alpha)

    @staticmethod
    def _project_box_halfspace(
        u_nom: Array, lower: Array, upper: Array, a: Array, rhs: float
    ) -> Array:
        clipped = np.clip(u_nom, lower, upper)
        if float(a @ clipped) >= rhs - 1e-10:
            return clipped
        candidates: list[Array] = []
        dimension = len(u_nom)
        # -1=lower, 0=free, +1=upper. The half-space boundary is active.
        for status in product((-1, 0, 1), repeat=dimension):
            fixed = np.array([item != 0 for item in status])
            free = ~fixed
            candidate = np.empty(dimension)
            candidate[fixed] = np.where(np.asarray(status)[fixed] < 0, lower[fixed], upper[fixed])
            remaining_rhs = rhs - float(a[fixed] @ candidate[fixed])
            if np.any(free):
                free_a = a[free]
                norm_sq = float(free_a @ free_a)
                if norm_sq <= 1e-16:
                    continue
                free_nom = u_nom[free]
                candidate[free] = free_nom + free_a * (
                    (remaining_rhs - float(free_a @ free_nom)) / norm_sq
                )
                if np.any(candidate[free] < lower[free] - 1e-10) or np.any(
                    candidate[free] > upper[free] + 1e-10
                ):
                    continue
            elif abs(float(a @ candidate) - rhs) > 1e-9:
                continue
            if float(a @ candidate) >= rhs - 1e-9:
                candidates.append(np.clip(candidate, lower, upper))

        # Include feasible vertices; relevant for degenerate coefficients.
        for sides in product((0, 1), repeat=dimension):
            vertex = np.where(np.asarray(sides, dtype=bool), upper, lower)
            if float(a @ vertex) >= rhs - 1e-9:
                candidates.append(vertex)
        if not candidates:
            raise InfeasibleSafetyFilter("CBF half-space does not intersect control bounds")
        return min(candidates, key=lambda item: float((item - u_nom) @ (item - u_nom))).copy()

    def filter(self, x: Array, nominal_control: Array) -> CBFFilterResult:
        x = self.robot.validate_state(x)
        nominal = np.asarray(nominal_control, dtype=float)
        if nominal.shape != (self.robot.control_dim(),) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal control has invalid shape or values")
        value, gradient = self.barrier.value_gradient(x, self.robot)
        lf = float(gradient @ self.robot.drift(x))
        lg = gradient @ self.robot.control_matrix(x)
        rhs = -lf - self.alpha * value
        lower, upper = self.robot.control_bounds()
        control = self._project_box_halfspace(nominal, lower, upper, lg, rhs)
        residual = lf + float(lg @ control) + self.alpha * value
        distance = float(np.linalg.norm(control - nominal))
        return CBFFilterResult(control, nominal.copy(), distance > 1e-9, distance, residual)
