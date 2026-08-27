import numpy as np
import pytest

from lawevo.dsl import parse_barrier
from lawevo.filter import CBFSafetyFilter
from lawevo.robot import CircleObstacle, UnicycleRobot


def test_unicycle_dynamics() -> None:
    robot = UnicycleRobot()
    x = np.array([1.0, 2.0, np.pi / 2])
    assert robot.drift(x) == pytest.approx([0, 0, 0])
    assert robot.control_matrix(x) == pytest.approx(np.array([[0, 0], [1, 0], [0, 1]]), abs=1e-12)
    next_x = robot.step(x, np.array([1.0, 0.5]), 0.1)
    assert next_x == pytest.approx([1.0, 2.1, np.pi / 2 + 0.05])


def test_filter_projects_unsafe_nominal_control() -> None:
    robot = UnicycleRobot([CircleObstacle((0, 0), 0.5)], velocity_bounds=(-1.0, 1.5))
    barrier = parse_barrier("min(dist_to_obstacle(0, 0.3))")
    safety_filter = CBFSafetyFilter(robot, barrier, alpha=1.0)
    # At x=1 facing the obstacle, positive v points inward. h=0.2 requires v <= 0.2.
    result = safety_filter.filter(np.array([1.0, 0.0, np.pi]), np.array([1.0, 0.0]))
    assert result.intervened
    assert result.control[0] == pytest.approx(0.2)
    assert result.cbf_residual >= -1e-9


def test_filter_leaves_safe_control_unchanged() -> None:
    robot = UnicycleRobot([CircleObstacle((0, 0), 0.5)])
    barrier = parse_barrier("min(dist_to_obstacle(0, 0.3))")
    result = CBFSafetyFilter(robot, barrier, 1.0).filter(
        np.array([2.0, 0.0, 0.0]), np.array([0.5, 0.1])
    )
    assert not result.intervened
    assert result.control == pytest.approx([0.5, 0.1])
