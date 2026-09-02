from __future__ import annotations

import numpy as np
import pytest

from lawevo.pid import MANISKILL_ADAPTERS


def _observation(*, grasped: bool = False, tcp: tuple[float, float, float] = (0, 0, 0.2)) -> dict:
    return {
        "agent": {"qpos": np.zeros(9), "qvel": np.zeros(9)},
        "extra": {
            "tcp_pose": np.array([*tcp, 1.0, 0.0, 0.0, 0.0]),
            "obj_pose": np.array([0.1, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]),
            "goal_pos": np.array([0.25, 0.0, 0.15]),
            "is_grasped": grasped,
        },
    }


@pytest.mark.parametrize("adapter_key", tuple(MANISKILL_ADAPTERS))
def test_maniskill_adapter_features_match_delta_pose_action(adapter_key: str) -> None:
    adapter = MANISKILL_ADAPTERS[adapter_key]
    memory = adapter.reset_controller(action_dim=7)

    features = adapter.features(None, _observation(), memory, adapter.fallback_dt)

    assert set(features) == set(adapter.allowed_terms)
    assert all(np.asarray(value).shape == (7,) for value in features.values())
    assert all(np.isfinite(value).all() for value in features.values())
    for baseline in adapter.classical:
        assert set(baseline.terms) <= set(adapter.allowed_terms)


def test_maniskill_adapters_have_task_specific_prompt_context() -> None:
    from experiments.gymnasium_classical_benchmarks import (
        CONTROL_GOALS,
        ENVIRONMENT_DESCRIPTIONS,
    )

    for adapter_key in MANISKILL_ADAPTERS:
        assert adapter_key in ENVIRONMENT_DESCRIPTIONS
        assert adapter_key in CONTROL_GOALS
        assert len(ENVIRONMENT_DESCRIPTIONS[adapter_key]) > 200
        assert len(CONTROL_GOALS[adapter_key]) > 200


def test_maniskill_pick_sequence_closes_only_after_reaching() -> None:
    adapter = MANISKILL_ADAPTERS["maniskill_pick_cube"]
    far = adapter.features(
        None, _observation(grasped=False), adapter.reset_controller(7), adapter.fallback_dt
    )
    near = adapter.features(
        None,
        _observation(grasped=False, tcp=(0.1, 0.0, 0.02)),
        adapter.reset_controller(7),
        adapter.fallback_dt,
    )

    assert far["pick_place_sequence"][-1] == 1.0
    assert near["pick_place_sequence"][-1] == -1.0


def test_maniskill_env_smoke() -> None:
    pytest.importorskip("mani_skill")
    adapter = MANISKILL_ADAPTERS["maniskill_push_cube"]
    env = adapter.make_env()
    try:
        observation, _ = env.reset(seed=123)
        assert "extra" in observation
        assert env.action_space.shape == (7,)
        _, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
        assert np.isfinite(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert "success" in info
    finally:
        env.close()
