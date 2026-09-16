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
            "vertices": np.array([0, 0, 0.02, 0.12, 0, 0.02, 0.06, 0.1, 0.02]),
        },
    }


@pytest.mark.parametrize("adapter_key", tuple(MANISKILL_ADAPTERS))
def test_maniskill_adapter_features_match_delta_pose_action(adapter_key: str) -> None:
    adapter = MANISKILL_ADAPTERS[adapter_key]
    action_dim = 6 if "draw" in adapter_key else 7
    memory = adapter.reset_controller(action_dim=action_dim)

    features = adapter.features(None, _observation(), memory, adapter.fallback_dt)

    assert set(features) == set(adapter.allowed_terms)
    assert all(np.asarray(value).shape == (action_dim,) for value in features.values())
    assert all(np.isfinite(value).all() for value in features.values())
    for baseline in adapter.classical:
        assert set(baseline.signals) <= set(adapter.allowed_terms)


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


def test_drawing_scheduler_approaches_lowers_and_closes_triangle() -> None:
    adapter = MANISKILL_ADAPTERS["maniskill_draw_triangle"]
    memory = adapter.reset_controller(6)
    first = adapter.features(None, _observation(), memory, 0.05)
    np.testing.assert_allclose(first["path_error"], [0, 0, -0.14, 0, 0, 0])
    assert memory["waypoint"] == 0
    lower = adapter.features(None, _observation(tcp=(0, 0, 0.06)), memory, 0.05)
    assert memory["phase"] == "lower"
    assert lower["path_error"][2] == pytest.approx(-0.035)
    trace = adapter.features(None, _observation(tcp=(0, 0, 0.025)), memory, 0.05)
    assert memory["waypoint"] == 1
    assert trace["path_error"][0] > 0
    np.testing.assert_allclose(memory["path"][0], memory["path"][-1])
    assert np.max(np.linalg.norm(np.diff(memory["path"], axis=0), axis=1)) <= 0.008001
    assert adapter.reset_controller(6)["path"] is None


def test_drawing_rejects_discontinuous_outline() -> None:
    adapter = MANISKILL_ADAPTERS["maniskill_draw_svg"]
    observation = _observation()
    observation["extra"]["continuous"] = False
    with pytest.raises(ValueError, match="continuous SVG"):
        adapter.features(None, observation, adapter.reset_controller(6), 0.05)


@pytest.mark.parametrize("adapter_key", ["maniskill_push_cube", "maniskill_draw_triangle", "maniskill_draw_svg"])
def test_maniskill_env_smoke(adapter_key) -> None:
    pytest.importorskip("mani_skill")
    adapter = MANISKILL_ADAPTERS[adapter_key]
    env = adapter.make_env()
    try:
        observation, _ = env.reset(seed=123)
        assert "extra" in observation
        action_dim = 6 if "draw" in adapter_key else 7
        assert env.action_space.shape == (action_dim,)
        features = adapter.features(env, observation, adapter.reset_controller(action_dim), 0.05)
        assert all(value.shape == (action_dim,) for value in features.values())
        _, reward, terminated, truncated, info = env.step(np.zeros(action_dim, dtype=np.float32))
        assert np.isfinite(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert "success" in info
        if "draw" in adapter_key:
            import torch

            from lawevo.pid.cpu_kinematics import SingleRobotJacobian

            kinematics = env.unwrapped.agent.controller.controllers["arm"].kinematics
            chain = getattr(kinematics, "pk_chain", None)
            if chain is None:
                return  # Native Pinocchio installations do not use the CPU fallback.
            fast = chain.jacobian
            assert isinstance(fast, SingleRobotJacobian)
            rng = np.random.default_rng(12)
            for _ in range(25):
                joints = torch.tensor(rng.uniform(-2, 2, (1, 7)), dtype=torch.float32)
                torch.testing.assert_close(fast(joints), fast.original(joints), atol=1e-6, rtol=1e-5)
            batch = joints.repeat(2, 1)
            torch.testing.assert_close(fast(batch), fast.original(batch))
            joints.requires_grad_(True)
            assert fast(joints).requires_grad

            trajectories = []
            for implementation in (fast.original, fast):
                chain.jacobian = implementation
                obs, _ = env.reset(seed=123)
                memory = adapter.reset_controller(6)
                positions = []
                for _ in range(adapter.horizon):
                    signals = adapter.features(env, obs, memory, 0.05)
                    action = np.clip(10 * signals["path_error"], -1, 1).astype(np.float32)
                    obs, _, terminated, truncated, info = env.step(action)
                    positions.append(obs["extra"]["tcp_pose"][:3])
                    if terminated or truncated:
                        break
                assert info["success"]
                trajectories.append(positions)
            np.testing.assert_allclose(trajectories[0], trajectories[1], atol=2e-5, rtol=1e-4)
    finally:
        env.close()
