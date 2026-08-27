import numpy as np
import pytest

from lawevo.dsl import BarrierSyntaxError, parse_barrier
from lawevo.robot import CircleObstacle, UnicycleRobot


def test_parse_evaluate_and_lipschitz() -> None:
    robot = UnicycleRobot([CircleObstacle((0.0, 0.0), 0.5)])
    barrier = parse_barrier("min(dist_to_obstacle(0, 0.3), boundary_margin(x, 5, 0.2))")
    barrier.validate(robot)
    value, gradient = barrier.value_gradient(np.array([1.0, 0.0, 0.0]), robot)
    assert value == pytest.approx(0.2)
    assert gradient == pytest.approx([1.0, 0.0, 0.0])
    assert barrier.lipschitz(robot) == 1.0


def test_json_wsum_roundtrip() -> None:
    source = {
        "op": "wsum",
        "terms": [
            {"weight": 0.75, "term": {"primitive": "dist_to_obstacle", "args": [0, 0.2]}},
            {"weight": 0.5, "term": {"primitive": "boundary_margin", "args": ["x", 5, 0.1]}},
        ],
    }
    robot = UnicycleRobot([CircleObstacle((0, 0), 0.5)])
    barrier = parse_barrier(source)
    barrier.validate(robot)
    assert barrier.lipschitz(robot) == pytest.approx(1.25)
    assert barrier.to_dict() == source


def test_invalid_syntax_and_robot_primitive_are_rejected() -> None:
    with pytest.raises(BarrierSyntaxError):
        parse_barrier("max(speed_margin(1.0))")
    barrier = parse_barrier("min(speed_margin(1.0))")
    with pytest.raises(ValueError, match="not available"):
        barrier.validate(UnicycleRobot())
