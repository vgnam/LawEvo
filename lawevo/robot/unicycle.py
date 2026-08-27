from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lawevo.robot.base import Array, RobotInterface


@dataclass(frozen=True)
class CircleObstacle:
    center: tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError("obstacle radius must be non-negative")


class UnicycleRobot(RobotInterface):
    """Kinematic unicycle x=(px, py, theta), u=(v, omega)."""

    def __init__(
        self,
        obstacles: Sequence[CircleObstacle] = (),
        workspace: tuple[tuple[float, float], tuple[float, float]] = ((-5.0, 5.0), (-5.0, 5.0)),
        velocity_bounds: tuple[float, float] = (-1.0, 1.5),
        omega_bounds: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        self.obstacles = tuple(obstacles)
        self.workspace = workspace
        self._u_min = np.array([velocity_bounds[0], omega_bounds[0]], dtype=float)
        self._u_max = np.array([velocity_bounds[1], omega_bounds[1]], dtype=float)
        if np.any(self._u_min > self._u_max):
            raise ValueError("invalid control bounds")

    def drift(self, x: Array) -> Array:
        self.validate_state(x)
        return np.zeros(3)

    def control_matrix(self, x: Array) -> Array:
        x = self.validate_state(x)
        theta = x[2]
        return np.array([[np.cos(theta), 0.0], [np.sin(theta), 0.0], [0.0, 1.0]])

    def control_bounds(self) -> tuple[Array, Array]:
        return self._u_min.copy(), self._u_max.copy()

    def state_bounds(self) -> tuple[Array, Array]:
        (xmin, xmax), (ymin, ymax) = self.workspace
        return np.array([xmin, ymin, -np.pi]), np.array([xmax, ymax, np.pi])

    def state_dim(self) -> int:
        return 3

    def control_dim(self) -> int:
        return 2

    def primitive_params(self) -> dict[str, object]:
        return {
            "obstacles": self.obstacles,
            "workspace": self.workspace,
            "v_max": float(self._u_max[0]),
        }

    def available_primitives(self) -> Sequence[str]:
        # speed_margin is not valid for a kinematic state without velocity.
        return ("dist_to_obstacle", "boundary_margin")

    def primitive_value_gradient(
        self, name: str, args: tuple[object, ...], x: Array
    ) -> tuple[float, Array]:
        x = self.validate_state(x)
        grad = np.zeros(3)
        if name == "dist_to_obstacle":
            obstacle_id, margin = int(args[0]), float(args[1])
            try:
                obstacle = self.obstacles[obstacle_id]
            except IndexError as exc:
                raise ValueError(f"unknown obstacle id {obstacle_id}") from exc
            delta = x[:2] - np.asarray(obstacle.center)
            distance = float(np.linalg.norm(delta))
            if distance > 1e-12:
                grad[:2] = delta / distance
            # At the center any unit subgradient is valid; zero is stable and conservative for control.
            return distance - obstacle.radius - margin, grad
        if name == "boundary_margin":
            axis, bound, margin = str(args[0]), float(args[1]), float(args[2])
            index = {"x": 0, "y": 1, "theta": 2}.get(axis)
            if index is None:
                raise ValueError(f"unknown axis {axis!r}")
            # A signed bound encodes the side: x <= b for b>=0, x >= b for b<0.
            sign = 1.0 if bound >= 0 else -1.0
            grad[index] = -sign
            return sign * (bound - x[index]) - margin, grad
        raise ValueError(f"primitive {name!r} is unavailable for UnicycleRobot")

    def primitive_lipschitz(self, name: str, args: tuple[object, ...]) -> float:
        if name not in self.available_primitives():
            raise ValueError(f"primitive {name!r} is unavailable for UnicycleRobot")
        return 1.0

    def step(self, x: Array, u: Array, dt: float) -> Array:
        next_x = super().step(x, u, dt)
        next_x[2] = (next_x[2] + np.pi) % (2 * np.pi) - np.pi
        return next_x
