import numpy as np
import pytest

from lawevo.pid import PANDA_GYM_ADAPTERS, run_episode

pytest.importorskip("panda_gym")


@pytest.mark.parametrize("adapter_key", tuple(PANDA_GYM_ADAPTERS))
def test_panda_gym_adapter_features_match_action_space(adapter_key: str) -> None:
    adapter = PANDA_GYM_ADAPTERS[adapter_key]
    env = adapter.make_env()
    try:
        observation, _ = env.reset(seed=123)
        action_dim = int(np.prod(env.action_space.shape))
        memory = adapter.reset_controller(action_dim)

        features = adapter.features(env, observation, memory, adapter.fallback_dt)

        assert set(features) == set(adapter.allowed_terms)
        assert all(np.asarray(value).shape == (action_dim,) for value in features.values())
        assert all(np.isfinite(value).all() for value in features.values())
        for baseline in adapter.classical:
            assert set(baseline.terms) <= set(adapter.allowed_terms)

        next_observation, reward, terminated, truncated, info = env.step(
            np.zeros(action_dim, dtype=np.float32)
        )
        assert set(next_observation) == {"observation", "achieved_goal", "desired_goal"}
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert not truncated
        assert "is_success" in info
    finally:
        env.close()


def test_panda_gym_adapters_have_task_specific_prompt_context() -> None:
    from experiments.gymnasium_classical_benchmarks import (
        CONTROL_GOALS,
        ENVIRONMENT_DESCRIPTIONS,
    )

    for adapter_key in PANDA_GYM_ADAPTERS:
        assert adapter_key in ENVIRONMENT_DESCRIPTIONS
        assert adapter_key in CONTROL_GOALS
        assert len(ENVIRONMENT_DESCRIPTIONS[adapter_key]) > 200
        assert len(CONTROL_GOALS[adapter_key]) > 200


def test_panda_reach_runs_through_common_episode_loop() -> None:
    adapter = PANDA_GYM_ADAPTERS["panda_reach"]
    structure = adapter.classical[0]

    episode = run_episode(adapter, structure, np.zeros(len(structure.terms)), seed=7)

    assert np.isfinite(episode.episode_return)
    assert isinstance(episode.success, bool)
