import numpy as np
import pytest

from lawevo.pid import ADAPTERS, CONTROLLED_VARIANT_ADAPTERS, PANDA_GYM_ADAPTERS, run_episode
from lawevo.pid.controlled_variants import make_pulse_env


@pytest.mark.parametrize("key", tuple(CONTROLLED_VARIANT_ADAPTERS))
def test_variant_contract_and_baselines(key):
    adapter = CONTROLLED_VARIANT_ADAPTERS[key]
    env = adapter.make_env()
    try:
        obs, _ = env.reset(seed=71)
        obs = adapter.prepare_reset(env, obs, 71)
        dim = env.action_space.shape[0]
        features = adapter.features(
            env, obs, adapter.reset_controller(dim),
            getattr(env.unwrapped, "dt", adapter.fallback_dt),
        )
        assert set(features) == set(adapter.allowed_terms)
        for law in adapter.classical:
            action = law.evaluate(features, np.ones(law.parameter_count))
            assert action.shape == (dim,)
            assert np.isfinite(action).all()
        env.step(np.zeros(dim, dtype=np.float32))
    finally:
        env.close()


def test_pulse_is_one_interval_and_restarts_after_reset():
    env = make_pulse_env(pulse_step=3, pulse_force=5.0)
    try:
        for _ in range(2):
            env.reset(seed=1)
            env.unwrapped.set_state(np.zeros(2), np.zeros(2))
            forces = []
            for _ in range(5):
                _, _, _, _, info = env.step(np.zeros(1))
                forces.append(info["pulse_force"])
                assert not np.any(env.unwrapped.data.xfrc_applied)
            assert forces == [0.0, 0.0, 5.0, 0.0, 0.0]
            assert env.unwrapped.data.qvel[0] != 0.0
    finally:
        env.close()


def test_payload_changes_only_tip_mass_and_inertia_without_accumulating():
    base = ADAPTERS["reacher"]
    variant = CONTROLLED_VARIANT_ADAPTERS["reacher_payload"]
    env, loaded = base.make_env(), variant.make_env()
    try:
        m0, m1 = env.unwrapped.model, loaded.unwrapped.model
        tip = m1.body("fingertip").id
        extra_mass = 0.15 * m0.body_mass[m0.body("body1").id]
        expected_mass = m0.body_mass.copy()
        expected_mass[tip] += extra_mass
        expected_inertia = m0.body_inertia.copy()
        expected_inertia[tip] += 0.4 * extra_mass * 0.01**2
        for seed in (7, 8, 7):
            o0, _ = env.reset(seed=seed)
            o1, _ = loaded.reset(seed=seed)
            base.prepare_reset(env, o0, seed)
            variant.prepare_reset(loaded, o1, seed)
            np.testing.assert_allclose(m1.body_mass, expected_mass)
            np.testing.assert_allclose(m1.body_inertia, expected_inertia)
            np.testing.assert_allclose(o0, o1)
        np.testing.assert_allclose(m0.dof_damping, m1.dof_damping)
    finally:
        env.close()
        loaded.close()


@pytest.mark.parametrize("base_key,variant_key,factor", [
    ("panda_push", "panda_push_low_friction", 0.75),
    ("panda_slide", "panda_slide_friction_shift", 1.25),
])
def test_friction_pair_preserves_scene_reset_and_object_dynamics(base_key, variant_key, factor):
    env = PANDA_GYM_ADAPTERS[base_key].make_env()
    variant = CONTROLLED_VARIANT_ADAPTERS[variant_key].make_env()
    try:
        for seed in (7, 8, 7):
            obs0, _ = env.reset(seed=seed)
            obs1, _ = variant.reset(seed=seed)
            for key in obs0:
                np.testing.assert_allclose(obs0[key], obs1[key])
            s0, s1 = env.unwrapped.sim, variant.unwrapped.sim
            assert set(s0._bodies_idx) == set(s1._bodies_idx)
            for name in ("table", "object"):
                d0 = s0.physics_client.getDynamicsInfo(s0._bodies_idx[name], -1)
                d1 = s1.physics_client.getDynamicsInfo(s1._bodies_idx[name], -1)
                assert d0[0] == d1[0]
                assert d1[1] == pytest.approx(d0[1] * (factor if name == "table" else 1))
    finally:
        env.close()
        variant.close()


def test_moving_slow_runs_full_horizon_after_native_success():
    adapter = CONTROLLED_VARIANT_ADAPTERS["panda_reach_moving_slow"]
    env = adapter.make_env()
    try:
        obs, _ = env.reset(seed=7)
        from lawevo.pid.controlled_variants import MovingSlowWrapper

        wrapper = env
        while not isinstance(wrapper, MovingSlowWrapper):
            wrapper = wrapper.env
        wrapper.initial_goal = obs["achieved_goal"].astype(float).copy()
        for step in range(adapter.horizon):
            _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            assert not terminated
            assert truncated == (step == adapter.horizon - 1)
            if step == 0:
                assert info["is_success"]
        assert wrapper.elapsed == pytest.approx(adapter.horizon * 0.04)
    finally:
        env.close()


def test_tracking_episode_has_post_warmup_samples():
    adapter = CONTROLLED_VARIANT_ADAPTERS["panda_reach_moving_slow"]
    law = next(s for s in adapter.classical if s.name == "Task P")
    episode = run_episode(adapter, law, np.array([10.0]), seed=90000)
    assert episode.progress_predicates["goal_acquired_once"]
    assert episode.constraint_violations["tracking_rate"] < 1.0


def test_slide_fsm_latches_strike_then_retracts_and_resets():
    adapter = PANDA_GYM_ADAPTERS["panda_slide"]
    memory = adapter.reset_controller(3)
    obj = np.array([0.0, 0.0, 0.02])
    goal = np.array([0.3, 0.0, 0.02])
    obs = {"observation": np.array([-0.06, 0.0, 0.03, 0.0, 0.0, 0.0]),
           "achieved_goal": obj, "desired_goal": goal}
    features = adapter.features(None, obs, memory, 0.04)
    assert memory["slide_phase"] == "strike"
    assert features["slide_strike"][0] > 0
    obs["desired_goal"] = np.array([0.0, 0.3, 0.02])
    features = adapter.features(None, obs, memory, 0.04)
    assert features["slide_strike"][1] == 0.0  # direction remains latched
    for _ in range(6):
        features = adapter.features(None, obs, memory, 0.04)
    assert memory["slide_phase"] == "retract"
    assert features["slide_retract"][2] > 0
    assert not np.any(features["slide_strike"])
    assert "slide_phase" not in adapter.reset_controller(3)


def test_controlled_variants_are_exposed_in_experiment_context():
    from experiments.gymnasium_classical_benchmarks import (
        CONTROL_GOALS,
        ENVIRONMENT_DESCRIPTIONS,
    )
    for key in CONTROLLED_VARIANT_ADAPTERS:
        assert len(CONTROL_GOALS[key]) > 200
        assert len(ENVIRONMENT_DESCRIPTIONS[key]) > 200


@pytest.mark.parametrize("key", tuple(CONTROLLED_VARIANT_ADAPTERS))
def test_new_variant_cli_exports_tuned_baselines_and_evolved_law(key, tmp_path, monkeypatch):
    import json
    import re
    import sys

    import experiments.gymnasium_classical_benchmarks as benchmark

    def fake_complete(self, system, prompt, **kwargs):
        signals = json.loads(re.search(r"Allowed signals: (\[[^\n]+\])", prompt).group(1))
        return json.dumps([{"name": "smoke law", "expression": f"K1*{signals[-1]}"}])

    monkeypatch.setattr(benchmark.NVIDIAChatClient, "complete", fake_complete)
    monkeypatch.setattr(benchmark, "plot_environment", lambda *args: None)
    monkeypatch.setenv("OPENAI_API_KEY", "smoke-only")
    monkeypatch.setenv("NVIDIA_API_KEY", "smoke-only")
    monkeypatch.chdir(tmp_path)
    env_id = CONTROLLED_VARIANT_ADAPTERS[key].env_id
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--environment", env_id, "--generations", "1", "--proposals", "1",
        "--proposal-attempts", "1", "--cem-iterations", "1", "--cem-population", "2",
        "--train-episodes", "1", "--test-episodes", "1",
    ])
    benchmark.main()
    summaries = list(tmp_path.glob(f"results/*/{env_id}/summary/results.json"))
    assert len(summaries) == 1
    result = json.loads(summaries[0].read_text())["result"]
    assert result["environment"] == env_id
    assert "Evolved Structure" in result["test"]
    for record in result["all_structures"]:
        assert record["train_metrics"]["sg"] is not None
        assert record["train_metrics"]["q"] is not None
    for record in result["test"].values():
        assert record["metrics"]["sg"] is not None
        assert record["metrics"]["q"] is not None
    protocol = json.loads(summaries[0].read_text())["protocol"]
    assert protocol["success_gap"] == CONTROLLED_VARIANT_ADAPTERS[key].success_gap_spec
    assert protocol["progress_spec"] == CONTROLLED_VARIANT_ADAPTERS[key].progress_spec
    if key == "inverted_pendulum_pulse":
        assert "LQR" in result["test"]
    else:
        for law in CONTROLLED_VARIANT_ADAPTERS[key].classical:
            assert law.name in result["test"]
