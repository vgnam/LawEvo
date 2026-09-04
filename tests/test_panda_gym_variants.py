import numpy as np
import pytest

from lawevo.pid import PANDA_GYM_ADAPTERS, PANDA_VARIANT_ADAPTERS, run_episode

pytest.importorskip("panda_gym")


@pytest.mark.parametrize("adapter_key", tuple(PANDA_VARIANT_ADAPTERS))
def test_panda_variant_features_match_action_space(adapter_key: str) -> None:
    adapter = PANDA_VARIANT_ADAPTERS[adapter_key]
    env = adapter.make_env()
    try:
        observation, _ = env.reset(seed=123)
        action_dim = int(np.prod(env.action_space.shape))
        memory = adapter.reset_controller(action_dim)
        dt = float(env.unwrapped.sim.dt)

        features = adapter.features(env, observation, memory, dt)

        assert set(features) == set(adapter.allowed_terms)
        assert all(np.asarray(value).shape == (action_dim,) for value in features.values())
        assert all(np.isfinite(value).all() for value in features.values())
        for baseline in adapter.classical:
            assert set(baseline.signals) <= set(adapter.allowed_terms)

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


def test_panda_variants_have_task_specific_prompt_context() -> None:
    from experiments.gymnasium_classical_benchmarks import (
        CONTROL_GOALS,
        ENVIRONMENT_DESCRIPTIONS,
    )

    for adapter_key in PANDA_VARIANT_ADAPTERS:
        assert adapter_key in ENVIRONMENT_DESCRIPTIONS
        assert adapter_key in CONTROL_GOALS
        assert len(ENVIRONMENT_DESCRIPTIONS[adapter_key]) > 200
        assert len(CONTROL_GOALS[adapter_key]) > 200


@pytest.mark.parametrize("adapter_key", tuple(PANDA_VARIANT_ADAPTERS))
def test_panda_variant_runs_through_common_episode_loop(adapter_key: str) -> None:
    adapter = PANDA_VARIANT_ADAPTERS[adapter_key]
    structure = adapter.classical[0]

    episode = run_episode(adapter, structure, np.zeros(structure.parameter_count), seed=7)

    assert np.isfinite(episode.episode_return)
    assert isinstance(episode.success, bool)


def test_panda_reach_moving_goal_actually_moves() -> None:
    adapter = PANDA_VARIANT_ADAPTERS["panda_reach_moving"]
    env = adapter.make_env()
    try:
        env.reset(seed=123)
        task = env.unwrapped.task
        first = np.asarray(env.unwrapped.sim.get_base_position("target"), dtype=float)
        for _ in range(10):
            env.step(np.zeros(3, dtype=np.float32))
        later = np.asarray(env.unwrapped.sim.get_base_position("target"), dtype=float)
        assert np.linalg.norm(later[:2] - first[:2]) > 1e-3
        del task
    finally:
        env.close()


def test_panda_push_ice_low_friction_is_applied() -> None:
    adapter = PANDA_VARIANT_ADAPTERS["panda_push_ice"]
    env = adapter.make_env()
    try:
        env.reset(seed=123)
        task = env.unwrapped.task
        import pybullet as p

        friction = p.getDynamicsInfo(task.sim._bodies_idx["object"], -1)[1]
        assert friction == pytest.approx(0.1)
        assert hasattr(task, "obstacle_position")
    finally:
        env.close()


def test_panda_slide_gate_walls_block_the_straight_path() -> None:
    adapter = PANDA_VARIANT_ADAPTERS["panda_slide_gate"]
    env = adapter.make_env()
    try:
        env.reset(seed=123)
        sim = env.unwrapped.sim
        left = np.asarray(sim.get_base_position("gate_wall_left"), dtype=float)
        right = np.asarray(sim.get_base_position("gate_wall_right"), dtype=float)
        gap = abs(left[1] - right[1]) - 2 * 0.05  # wall half-width is 0.05 m
        assert gap == pytest.approx(env.unwrapped.task.gate_width, abs=1e-6)
        assert env.unwrapped.task.gate_x > 0.15
    finally:
        env.close()


def test_panda_pick_distractor_heavy_cube_mass() -> None:
    adapter = PANDA_VARIANT_ADAPTERS["panda_pick_distractor"]
    env = adapter.make_env()
    try:
        env.reset(seed=123)
        task = env.unwrapped.task
        import pybullet as p

        mass = p.getDynamicsInfo(task.sim._bodies_idx["object"], -1)[0]
        assert mass == pytest.approx(1.5)
        assert hasattr(task, "set_cube_mass")
    finally:
        env.close()


def test_panda_stack_narrow_tight_tolerance_and_settle() -> None:
    adapter = PANDA_VARIANT_ADAPTERS["panda_stack_narrow"]
    env = adapter.make_env()
    try:
        env.reset(seed=123)
        task = env.unwrapped.task
        assert task.distance_threshold == pytest.approx(0.025)
        assert task.settle_speed == pytest.approx(0.08)
        # Tolerance must be tighter than the standard 0.1 m stack threshold.
        assert task.distance_threshold < 0.1
    finally:
        env.close()
