import numpy as np
import pytest

from lawevo.pid import ADAPTERS, LOCOMOTION_ADAPTERS, run_episode

NEW_ADAPTER_KEYS = (
    "swimmer",
    "inverted_double_pendulum",
    "humanoid_standup",
    "bipedal_walker",
)


@pytest.mark.parametrize("adapter_key", NEW_ADAPTER_KEYS)
def test_new_adapter_features_match_action_space(adapter_key: str) -> None:
    adapters = {**ADAPTERS, **LOCOMOTION_ADAPTERS}
    adapter = adapters[adapter_key]
    env = adapter.make_env()
    try:
        observation, _ = env.reset(seed=123)
        observation = adapter.prepare_reset(env, observation, seed=123)
        action_dim = int(np.prod(env.action_space.shape))
        memory = adapter.reset_controller(action_dim)
        dt = (
            float(env.unwrapped.dt)
            if hasattr(env.unwrapped, "dt")
            else adapter.fallback_dt
        )

        features = adapter.features(env, observation, memory, dt)

        assert set(features) == set(adapter.allowed_terms)
        assert all(np.asarray(value).shape == (action_dim,) for value in features.values())
        assert all(np.isfinite(value).all() for value in features.values())
        for baseline in adapter.classical:
            assert set(baseline.signals) <= set(adapter.allowed_terms)

        next_observation, reward, *_ = env.step(np.zeros(action_dim, dtype=np.float32))
        assert np.isfinite(next_observation).all()
        assert np.isfinite(reward)
    finally:
        env.close()


def test_new_adapters_have_task_specific_prompt_context() -> None:
    from experiments.gymnasium_classical_benchmarks import (
        CONTROL_GOALS,
        ENVIRONMENT_DESCRIPTIONS,
    )

    for adapter_key in NEW_ADAPTER_KEYS:
        assert adapter_key in ENVIRONMENT_DESCRIPTIONS
        assert adapter_key in CONTROL_GOALS
        assert len(ENVIRONMENT_DESCRIPTIONS[adapter_key]) > 200
        assert len(CONTROL_GOALS[adapter_key]) > 200


def test_swimmer_reset_drift_is_not_success() -> None:
    adapter = LOCOMOTION_ADAPTERS["swimmer"]
    structure = adapter.classical[0]

    episode = run_episode(adapter, structure, np.zeros(structure.parameter_count), seed=7)

    assert not episode.success
