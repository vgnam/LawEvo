from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class RobotInterface(ABC):
    """The only robot-specific dependency used by verification and rollout.

    ``state_bounds`` is deliberately part of the executable interface: a finite
    verification domain cannot be inferred from dynamics alone.
    """

    @abstractmethod
    def drift(self, x: Array) -> Array:
        """Return f(x), shape ``(state_dim,)``."""

    @abstractmethod
    def control_matrix(self, x: Array) -> Array:
        """Return g(x), shape ``(state_dim, control_dim)``."""

    @abstractmethod
    def control_bounds(self) -> tuple[Array, Array]:
        """Return element-wise ``(u_min, u_max)``."""

    @abstractmethod
    def state_bounds(self) -> tuple[Array, Array]:
        """Return the compact state domain used by the verifier."""

    @abstractmethod
    def state_dim(self) -> int:
        pass

    @abstractmethod
    def control_dim(self) -> int:
        pass

    @abstractmethod
    def primitive_params(self) -> dict[str, object]:
        pass

    @abstractmethod
    def available_primitives(self) -> Sequence[str]:
        pass

    @abstractmethod
    def primitive_value_gradient(
        self, name: str, args: tuple[object, ...], x: Array
    ) -> tuple[float, Array]:
        """Evaluate a DSL terminal and its state gradient."""

    @abstractmethod
    def primitive_lipschitz(self, name: str, args: tuple[object, ...]) -> float:
        """Return a global Lipschitz bound over ``state_bounds``."""

    def validate_state(self, x: Array) -> Array:
        value = np.asarray(x, dtype=float)
        if value.shape != (self.state_dim(),):
            raise ValueError(f"expected state shape {(self.state_dim(),)}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("state must contain only finite values")
        return value

    def step(self, x: Array, u: Array, dt: float) -> Array:
        """Forward-Euler default; subclasses may override with exact integration."""
        x = self.validate_state(x)
        u = np.asarray(u, dtype=float)
        if u.shape != (self.control_dim(),):
            raise ValueError(f"expected control shape {(self.control_dim(),)}, got {u.shape}")
        lo, hi = self.control_bounds()
        u = np.clip(u, lo, hi)
        return x + dt * (self.drift(x) + self.control_matrix(x) @ u)
