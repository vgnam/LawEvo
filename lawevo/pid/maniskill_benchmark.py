from __future__ import annotations

import warnings
from typing import Any

import gymnasium as gym
import numpy as np

from lawevo.pid.gym_benchmark import BenchmarkAdapter, GymStructure


def _action(
    action_dim: int,
    translation: np.ndarray | None = None,
    gripper: float = 0.0,
) -> np.ndarray:
    """Build a normalized ManiSkill delta-pose action without commanding rotation."""
    action = np.zeros(action_dim, dtype=float)
    if translation is not None:
        action[:3] = np.asarray(translation, dtype=float)
    action[-1] = gripper
    return action


def _normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def _extra(observation: dict[str, Any]) -> dict[str, Any]:
    """Accept both CPUGymWrapper's state_dict and a bare task-extra dictionary."""
    return observation.get("extra", observation)


def _position(extra: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(extra[key], dtype=float)
    return value[:3]


def _as_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


class ManiSkillAdapter(BenchmarkAdapter):
    """Common CPU adapter for ManiSkill 3 Panda tabletop tasks."""

    horizon = 50
    fallback_dt = 0.05
    energy_weight, jerk_weight = 0.01, 0.00001

    def make_env(self):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="pinnochio package is not installed.*",
                    module="sapien.wrapper.pinocchio_model",
                )
                import mani_skill.envs  # noqa: F401 -- registers ManiSkill environments
                from mani_skill.agents.controllers.utils import kinematics
                from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
        except ImportError as exc:
            raise RuntimeError(
                "ManiSkill benchmark support requires: pip install -e '.[benchmarks]'"
            ) from exc

        # SAPIEN's Windows wheel currently has no Pinocchio bindings. ManiSkill's
        # PyTorch-kinematics implementation works on CPU too, so select that existing
        # backend when Pinocchio is unavailable instead of failing during env creation.
        if kinematics.PinocchioModel is None:
            kinematics.Kinematics._setup_cpu = kinematics.Kinematics._setup_gpu

        env = gym.make(
            self.env_id,
            num_envs=1,
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            reward_mode="dense",
            max_episode_steps=self.horizon,
        )
        return CPUGymWrapper(env)

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory["action_dim"] = action_dim
        memory["previous_tcp"] = None
        memory["integral_xyz"] = np.zeros(3)
        return memory

    def _common(self, observation, memory, dt):
        extra = _extra(observation)
        tcp = _position(extra, "tcp_pose")
        previous = memory["previous_tcp"]
        velocity = np.zeros(3) if previous is None else (tcp - previous) / dt
        memory["previous_tcp"] = tcp.copy()
        return extra, tcp, velocity, int(memory["action_dim"])

    def success(self, env, observation, steps, terminated):
        del observation, steps, terminated
        result = env.unwrapped.evaluate()
        return _as_bool(result["success"])


class ManiSkillPushCubeAdapter(ManiSkillAdapter):
    env_id = "PushCube-v1"
    allowed_terms = (
        "reach_push_pose",
        "normalized_reach_push_pose",
        "object_goal_error",
        "normalized_object_goal_error",
        "eef_damping",
        "contact_then_goal",
    )
    classical = (
        GymStructure("Reach P", ("reach_push_pose",)),
        GymStructure("Reach PD", ("reach_push_pose", "eef_damping")),
        GymStructure("Object Goal P", ("reach_push_pose", "object_goal_error")),
        GymStructure("Contact + Goal PD", ("contact_then_goal", "eef_damping")),
    )

    def features(self, env, observation, memory, dt):
        del env
        extra, tcp, velocity, action_dim = self._common(observation, memory, dt)
        obj = _position(extra, "obj_pose")
        goal = _position(extra, "goal_pos")
        push_pose = obj + np.array([-0.025, 0.0, 0.0])
        reach = push_pose - tcp
        goal_error = goal - obj
        sequential = goal_error if np.linalg.norm(reach) < 0.04 else reach
        return {
            "reach_push_pose": _action(action_dim, reach),
            "normalized_reach_push_pose": _action(action_dim, _normalized(reach)),
            "object_goal_error": _action(action_dim, goal_error),
            "normalized_object_goal_error": _action(action_dim, _normalized(goal_error)),
            "eef_damping": _action(action_dim, -velocity),
            "contact_then_goal": _action(action_dim, sequential),
        }


class ManiSkillPickCubeAdapter(ManiSkillAdapter):
    env_id = "PickCube-v1"
    allowed_terms = (
        "reach_cube",
        "normalized_reach_cube",
        "integral_reach",
        "eef_damping",
        "grasp_close",
        "lift_then_transport",
        "release_on_target",
        "pick_place_sequence",
    )
    classical = (
        GymStructure("Reach P", ("reach_cube",)),
        GymStructure("Reach PD", ("reach_cube", "eef_damping")),
        GymStructure("Pick + Place", ("reach_cube", "grasp_close", "lift_then_transport")),
        GymStructure(
            "Pick + Place PD",
            ("pick_place_sequence", "eef_damping", "grasp_close", "release_on_target"),
        ),
    )

    def features(self, env, observation, memory, dt):
        del env
        extra, tcp, velocity, action_dim = self._common(observation, memory, dt)
        obj = _position(extra, "obj_pose")
        goal = _position(extra, "goal_pos")
        reach = obj - tcp
        goal_error = goal - obj
        memory["integral_xyz"] = np.clip(
            memory["integral_xyz"] + reach * dt, -0.25, 0.25
        )
        grasped = _as_bool(extra.get("is_grasped", np.linalg.norm(reach) < 0.04))
        goal_distance = float(np.linalg.norm(goal_error))
        lift_height = max(obj[2], goal[2]) + 0.05
        transport = (
            goal_error
            if grasped and obj[2] >= lift_height - 0.06
            else np.array([0.0, 0.0, max(0.0, lift_height - obj[2])])
        )
        if not grasped and np.linalg.norm(reach) >= 0.04:
            sequence = _action(action_dim, reach, gripper=1.0)
        elif not grasped:
            sequence = _action(action_dim, reach, gripper=-1.0)
        elif goal_distance < 0.03:
            sequence = _action(action_dim, goal_error, gripper=1.0)
        else:
            sequence = _action(action_dim, transport, gripper=-1.0)
        return {
            "reach_cube": _action(action_dim, reach),
            "normalized_reach_cube": _action(action_dim, _normalized(reach)),
            "integral_reach": _action(action_dim, memory["integral_xyz"]),
            "eef_damping": _action(action_dim, -velocity),
            "grasp_close": _action(action_dim, gripper=-1.0),
            "lift_then_transport": _action(action_dim, transport),
            "release_on_target": _action(
                action_dim, gripper=1.0 if goal_distance < 0.03 else 0.0
            ),
            "pick_place_sequence": sequence,
        }


MANISKILL_ADAPTERS = {
    "maniskill_push_cube": ManiSkillPushCubeAdapter(),
    "maniskill_pick_cube": ManiSkillPickCubeAdapter(),
}
