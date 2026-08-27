from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np

from lawevo.robot.base import Array, RobotInterface


class Barrier(Protocol):
    def value_gradient(self, x: Array, robot: RobotInterface) -> tuple[float, Array]: ...

    def lipschitz(self, robot: RobotInterface) -> float: ...

    def validate(self, robot: RobotInterface) -> None: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class Primitive:
    name: str
    args: tuple[object, ...]

    def value_gradient(self, x: Array, robot: RobotInterface) -> tuple[float, Array]:
        return robot.primitive_value_gradient(self.name, self.args, x)

    def lipschitz(self, robot: RobotInterface) -> float:
        return robot.primitive_lipschitz(self.name, self.args)

    def validate(self, robot: RobotInterface) -> None:
        if self.name not in robot.available_primitives():
            raise ValueError(
                f"primitive {self.name!r} is not available for {type(robot).__name__}; "
                f"allowed: {', '.join(robot.available_primitives())}"
            )
        expected = {
            "dist_to_obstacle": 2,
            "speed_margin": 1,
            "joint_margin": 3,
            "boundary_margin": 3,
        }
        if self.name in expected and len(self.args) != expected[self.name]:
            raise ValueError(f"{self.name} expects {expected[self.name]} arguments")
        robot.primitive_lipschitz(self.name, self.args)

    def to_dict(self) -> dict[str, object]:
        return {"primitive": self.name, "args": list(self.args)}


@dataclass(frozen=True)
class WeightedTerm:
    weight: float
    term: Primitive

    def __post_init__(self) -> None:
        if not (0.0 < self.weight <= 1.0) or not np.isfinite(self.weight):
            raise ValueError("weights must be finite and in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {"weight": self.weight, "term": self.term.to_dict()}


Term: TypeAlias = Primitive


@dataclass(frozen=True)
class Combine:
    operator: str
    terms: tuple[Primitive | WeightedTerm, ...]

    def __post_init__(self) -> None:
        if self.operator not in {"min", "wsum"}:
            raise ValueError("operator must be 'min' or 'wsum'")
        if not self.terms:
            raise ValueError("a barrier must contain at least one term")
        if self.operator == "min" and not all(isinstance(t, Primitive) for t in self.terms):
            raise ValueError("min accepts unweighted primitives")
        if self.operator == "wsum" and not all(isinstance(t, WeightedTerm) for t in self.terms):
            raise ValueError("wsum accepts weighted primitives")

    def value_gradient(self, x: Array, robot: RobotInterface) -> tuple[float, Array]:
        if self.operator == "min":
            results = [term.value_gradient(x, robot) for term in self.terms]
            # Deterministic active branch at non-smooth ties.
            return results[int(np.argmin([item[0] for item in results]))]
        value = 0.0
        gradient = np.zeros(robot.state_dim())
        for weighted in self.terms:
            assert isinstance(weighted, WeightedTerm)
            item_value, item_gradient = weighted.term.value_gradient(x, robot)
            value += weighted.weight * item_value
            gradient += weighted.weight * item_gradient
        return value, gradient

    def lipschitz(self, robot: RobotInterface) -> float:
        if self.operator == "min":
            return max(term.lipschitz(robot) for term in self.terms)
        return sum(
            weighted.weight * weighted.term.lipschitz(robot)
            for weighted in self.terms
            if isinstance(weighted, WeightedTerm)
        )

    def validate(self, robot: RobotInterface) -> None:
        for item in self.terms:
            (item.term if isinstance(item, WeightedTerm) else item).validate(robot)

    def to_dict(self) -> dict[str, object]:
        return {"op": self.operator, "terms": [item.to_dict() for item in self.terms]}
