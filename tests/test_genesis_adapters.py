from __future__ import annotations

import numpy as np
import pytest

from lawevo.pid import GENESIS_ADAPTERS


@pytest.mark.parametrize("adapter_key", tuple(GENESIS_ADAPTERS))
def test_genesis_features_match_cartesian_gripper_action(adapter_key: str) -> None:
    adapter = GENESIS_ADAPTERS[adapter_key]
    memory = adapter.reset_controller(4)
    observation = {
        "tcp": np.array([0.50, 0.0, 0.20]),
        "object": np.array([0.45, 0.0, 0.02]),
        "goal": np.array([0.60, 0.02, 0.20]),
    }

    features = adapter.features(None, observation, memory, adapter.fallback_dt)

    assert set(features) == set(adapter.allowed_terms)
    assert all(np.asarray(value).shape == (4,) for value in features.values())
    assert all(np.isfinite(value).all() for value in features.values())
    for baseline in adapter.classical:
        assert set(baseline.terms) <= set(adapter.allowed_terms)


def test_genesis_adapters_have_task_specific_prompt_context() -> None:
    from experiments.gymnasium_classical_benchmarks import (
        CONTROL_GOALS,
        ENVIRONMENT_DESCRIPTIONS,
    )

    for adapter_key in GENESIS_ADAPTERS:
        assert adapter_key in ENVIRONMENT_DESCRIPTIONS
        assert adapter_key in CONTROL_GOALS
        assert len(ENVIRONMENT_DESCRIPTIONS[adapter_key]) > 200
        assert len(CONTROL_GOALS[adapter_key]) > 200


def test_genesis_pick_sequence_opens_far_and_closes_near() -> None:
    adapter = GENESIS_ADAPTERS["genesis_pick_cube"]
    memory = adapter.reset_controller(4)
    far = adapter.features(
        None,
        {
            "tcp": np.array([0.2, 0.0, 0.2]),
            "object": np.array([0.45, 0.0, 0.02]),
            "goal": np.array([0.5, 0.0, 0.2]),
        },
        memory,
        adapter.fallback_dt,
    )
    near = adapter.features(
        None,
        {
            "tcp": np.array([0.45, 0.0, 0.02]),
            "object": np.array([0.45, 0.0, 0.02]),
            "goal": np.array([0.5, 0.0, 0.2]),
        },
        adapter.reset_controller(4),
        adapter.fallback_dt,
    )

    assert far["pick_place_sequence"][-1] == 1.0
    assert near["pick_place_sequence"][-1] == -1.0
