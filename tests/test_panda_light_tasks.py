import json
import sys

import numpy as np
import pytest

from lawevo.pid import PANDA_LIGHT_ADAPTERS
from lawevo.pid.gym_benchmark import run_episode

pytest.importorskip("panda_gym")


@pytest.mark.parametrize("key", tuple(PANDA_LIGHT_ADAPTERS))
def test_physical_task_seed_contract_and_success(key):
    a = PANDA_LIGHT_ADAPTERS[key]
    e = a.make_env()
    try:
        first, _ = e.reset(seed=31)
        e.step(np.ones(a.action_dim))
        e.reset(seed=78)
        repeated, _ = e.reset(seed=31)
        for name in first:
            np.testing.assert_allclose(first[name], repeated[name], atol=1e-10)
        for law in a.classical:
            a.validate_structure(law)
        assert a.signal_contract.complete
        memory = a.reset_controller(a.action_dim)
        features = a.features(e, repeated, memory, e.unwrapped.dt)
        a.signal_contract.validate_values(features)
        episode = run_episode(a, a.classical[0], np.ones(a.classical[0].parameter_count), 31, env=e)
        assert episode.success  # bounded regression for a feasible classical reference
        assert episode.sg == 0
        assert episode.q == 1
        assert np.isfinite([episode.energy, episode.jerk, episode.episode_return]).all()
        assert e.unwrapped.state_metrics()[1][-1]
    finally:
        e.close()


@pytest.mark.parametrize("key", tuple(PANDA_LIGHT_ADAPTERS))
def test_control_snapshot_matches_live_state_and_isolated_observation(key):
    adapter = PANDA_LIGHT_ADAPTERS[key]
    env = adapter.make_env()
    try:
        for seed in (31, 78):
            observation, _ = env.reset(seed=seed)
            for _ in range(3):
                scene = env.unwrapped
                assert scene._metrics == scene.state_metrics()
                live = scene._observation()
                for name in live:
                    np.testing.assert_array_equal(observation[name], live[name])
                    observation[name][...] = 999
                    np.testing.assert_array_equal(scene._observation(cached=True)[name], live[name])
                observation, *_ = env.step(np.zeros(adapter.action_dim))
    finally:
        env.close()


@pytest.mark.parametrize("task,latched,expected", [
    ("drawer", False, 20), ("drawer", True, 1),
    ("wipe", False, 20), ("wipe", True, 1), ("peg", False, 1),
])
def test_contact_query_schedule_preserves_history(task, latched, expected, monkeypatch):
    key = {"drawer": "panda_drawer", "wipe": "panda_planar_wipe", "peg": "panda_peg_insertion_easy"}[task]
    adapter = PANDA_LIGHT_ADAPTERS[key]
    env = adapter.make_env()
    try:
        env.reset(seed=31)
        scene = env.unwrapped
        scene.contact_seen = latched
        scene._wipe_complete = latched
        calls = []
        monkeypatch.setattr(scene, "_contact_update", lambda: calls.append(True))
        env.step(np.zeros(adapter.action_dim))
        assert len(calls) == expected
    finally:
        env.close()


def test_wipe_requires_physical_contact_not_just_xy_overlap():
    e = PANDA_LIGHT_ADAPTERS["panda_planar_wipe"].make_env()
    try:
        e.reset(seed=31)
        scene = e.unwrapped
        marker = scene.markers[0]
        scene.client.resetBasePositionAndOrientation(scene.tool, marker+[0, 0, 0.1], [1, 0, 0, 0])
        scene.client.performCollisionDetection()
        scene._contact_update()
        assert not scene.cleaned.any()
        scene.client.resetBasePositionAndOrientation(scene.tool, marker+[0, 0, 0.007], [1, 0, 0, 0])
        # Collision detection alone has no solved normal force: cannot earn wipe credit.
        scene.client.performCollisionDetection()
        scene._contact_update()
        assert not scene.cleaned.any()
    finally:
        e.close()


def test_peg_geometry_has_real_walls_and_clear_opening():
    e = PANDA_LIGHT_ADAPTERS["panda_peg_insertion_easy"].make_env()
    try:
        e.reset(seed=31)
        scene = e.unwrapped
        scene.client.resetBasePositionAndOrientation(scene.tool, scene.target+[0, 0, 0.03], [1, 0, 0, 0])
        scene.client.performCollisionDetection()
        assert not any(scene.client.getContactPoints(scene.tool, wall) for wall, _ in scene.walls)
        scene.client.resetBasePositionAndOrientation(scene.tool, scene.target+[0.025, 0, 0.03], [1, 0, 0, 0])
        scene.client.performCollisionDetection()
        assert any(scene.client.getContactPoints(scene.tool, wall) for wall, _ in scene.walls)
        assert scene.state_metrics()[0] > 0
    finally:
        e.close()


@pytest.mark.parametrize("key", tuple(PANDA_LIGHT_ADAPTERS))
def test_cli_exports_new_light_tasks(key, tmp_path, monkeypatch):
    import experiments.gymnasium_classical_benchmarks as benchmark

    adapter = PANDA_LIGHT_ADAPTERS[key]
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
