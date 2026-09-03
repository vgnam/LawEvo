import numpy as np
import pytest

from lawevo.pid import ROBOSUITE_ADAPTERS

pytest.importorskip("robosuite")


@pytest.mark.parametrize("adapter_key", tuple(ROBOSUITE_ADAPTERS))
def test_robosuite_adapter_features_match_osc_action(adapter_key: str) -> None:
    adapter = ROBOSUITE_ADAPTERS[adapter_key]
    env = adapter.make_env()
    try:
        observation, _ = env.reset(seed=123)
        observation = adapter.prepare_reset(env, observation, seed=123)
        action_dim = int(np.prod(env.action_space.shape))
        memory = adapter.reset_controller(action_dim)

        features = adapter.features(env, observation, memory, env.dt)

        assert action_dim == 7
        assert set(features) == set(adapter.allowed_terms)
        assert all(np.asarray(value).shape == (action_dim,) for value in features.values())
        assert all(np.isfinite(value).all() for value in features.values())
        for baseline in adapter.classical:
            assert set(baseline.signals) <= set(adapter.allowed_terms)

        _, reward, terminated, truncated, _ = env.step(
            np.zeros(action_dim, dtype=np.float32)
        )
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_robosuite_adapters_have_task_specific_prompt_context() -> None:
    from experiments.gymnasium_classical_benchmarks import (
        CONTROL_GOALS,
        ENVIRONMENT_DESCRIPTIONS,
    )

    for adapter_key in ROBOSUITE_ADAPTERS:
        assert adapter_key in ENVIRONMENT_DESCRIPTIONS
        assert adapter_key in CONTROL_GOALS
        assert len(ENVIRONMENT_DESCRIPTIONS[adapter_key]) > 200
        assert len(CONTROL_GOALS[adapter_key]) > 200
