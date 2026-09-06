import numpy as np
import pytest

from lawevo.pid import ROBOSUITE_CLASSICAL_ADAPTERS
from lawevo.pid.gym_benchmark import run_episode


@pytest.mark.parametrize("key", tuple(ROBOSUITE_CLASSICAL_ADAPTERS))
def test_contract_and_empty_metrics(key):
    adapter = ROBOSUITE_CLASSICAL_ADAPTERS[key]
    for law in adapter.classical:
        adapter.validate_structure(law)
    assert adapter.signal_contract.complete
    assert set(adapter.signal_contract.signals) == set(adapter.allowed_terms)
    tracker = adapter.make_tracker()
    assert adapter.success_constraints(tracker) == {"task_completion": 1.0}
    assert not any(adapter.progress_predicates(tracker).values())


@pytest.mark.parametrize("key", tuple(ROBOSUITE_CLASSICAL_ADAPTERS))
def test_seeded_physics_and_rollout(key):
    pytest.importorskip("robosuite")
    adapter = ROBOSUITE_CLASSICAL_ADAPTERS[key]
    env = adapter.make_env()
    try:
        first, _ = env.reset(seed=31)
        state = env.suite_env.sim.data.qpos.copy()
        masses = env.suite_env.sim.model.body_mass.copy()
        env.reset(seed=92)
        again, _ = env.reset(seed=31)
        np.testing.assert_allclose(env.suite_env.sim.data.qpos, state, atol=1e-12)
        for name in first:
            np.testing.assert_allclose(first[name], again[name], atol=1e-12, err_msg=name)
        np.testing.assert_array_equal(env.suite_env.sim.model.body_mass, masses)
        law = adapter.classical[0]
        episode = run_episode(adapter, law, np.ones(law.parameter_count), 31, env=env)
        assert np.isfinite([episode.episode_return, episode.energy, episode.jerk]).all()
        assert 0 <= episode.sg <= 1
        assert 0 <= episode.q <= 1
        assert episode.success == (episode.sg == 0)
        assert episode.success == bool(env.suite_env._check_success())
    finally:
        env.close()


@pytest.mark.parametrize("key", tuple(ROBOSUITE_CLASSICAL_ADAPTERS))
def test_native_success_boundaries(key):
    from types import SimpleNamespace

    a = ROBOSUITE_CLASSICAL_ADAPTERS[key]
    for value in (0.0, 0.5, 1.0, 1.1):
        if a.task_name == "Lift":
            height = 0.04 * value
            suite = SimpleNamespace(
                _check_success=lambda h=height: h > 0.04,
                _check_grasp=lambda *args: False,
                robots=[SimpleNamespace(gripper=None)], cube=None,
                model=SimpleNamespace(mujoco_arena=SimpleNamespace(table_offset=[0, 0, 0])),
            )
            obs = {"cube_pos": np.array([0, 0, height]), "robot0_eef_pos": np.zeros(3)}
        elif a.task_name == "Door":
            angle = 0.3 * value
            suite = SimpleNamespace(_check_success=lambda h=angle: h > 0.3)
            obs = {"hinge_qpos": np.array([angle]), "handle_pos": np.zeros(3), "robot0_eef_pos": np.zeros(3)}
        else:
            count = min(10, round(value*10))
            suite = SimpleNamespace(_check_success=lambda n=count: n == 10, wiped_markers=list(range(count)), num_markers=10)
            obs = {}
        snapshot = a.snapshot(suite, obs)
        assert (snapshot["gap"] == 0) == suite._check_success()
        assert 0 <= snapshot["gap"] <= 1


@pytest.mark.parametrize("key", tuple(ROBOSUITE_CLASSICAL_ADAPTERS))
def test_cli_exports_new_tasks(key, tmp_path, monkeypatch):
    import json
    import sys

    pytest.importorskip("robosuite")
    import experiments.gymnasium_classical_benchmarks as benchmark

    adapter = ROBOSUITE_CLASSICAL_ADAPTERS[key]
    monkeypatch.setattr(adapter, "horizon", 4)
    monkeypatch.setattr(benchmark.NVIDIAChatClient, "complete", lambda *args, **kwargs:
                        json.dumps([{"name": "smoke", "expression": "K1*waypoint"}]))
    monkeypatch.setattr(benchmark, "plot_environment", lambda *args: None)
    monkeypatch.setenv("NVIDIA_API_KEY", "smoke-only")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--environment", adapter.env_id, "--generations", "1",
        "--proposals", "1", "--proposal-attempts", "1", "--cem-iterations", "1",
        "--cem-population", "2", "--train-episodes", "1", "--test-episodes", "1",
    ])
    benchmark.main()
    path, = tmp_path.glob(f"results/*/{adapter.env_id}/summary/results.json")
    payload = json.loads(path.read_text())
    assert payload["protocol"]["success_gap"] == adapter.success_gap_spec
    assert payload["protocol"]["progress_spec"] == adapter.progress_spec
    for row in payload["result"]["test"].values():
        assert row["metrics"]["sg"] is not None
        assert row["metrics"]["q"] is not None
