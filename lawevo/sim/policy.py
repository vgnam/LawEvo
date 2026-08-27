from __future__ import annotations

import numpy as np

from lawevo.robot.base import Array


def proportional_unicycle_policy(
    state: dict[str, object], goal: Array, obstacles: list[object] | tuple[object, ...]
) -> Array:
    """Simple hand-designed nominal controller used by the smoke test and example."""
    del obstacles
    x = np.asarray(state["x"], dtype=float)
    delta = np.asarray(goal, dtype=float)[:2] - x[:2]
    target_heading = np.arctan2(delta[1], delta[0])
    heading_error = (target_heading - x[2] + np.pi) % (2 * np.pi) - np.pi
    distance = float(np.linalg.norm(delta))
    v = min(1.2 * distance, 1.5) * max(0.0, np.cos(heading_error))
    omega = np.clip(2.5 * heading_error, -2.0, 2.0)
    return np.array([v, omega])
