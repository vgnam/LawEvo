from __future__ import annotations

import numpy as np

from lawevo.pid.gym_benchmark import BenchmarkAdapter, GymStructure


def _action(action_dim: int, xyz: np.ndarray | None = None, gripper: float = 0.0) -> np.ndarray:
    """Build a Panda-Gym end-effector displacement command."""
    action = np.zeros(action_dim, dtype=float)
    if xyz is not None:
        action[:3] = np.asarray(xyz, dtype=float)
    if action_dim == 4:
        # Panda-Gym uses negative finger displacement to close the gripper.
        action[3] = gripper
    return action


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=float) / max(norm, 1e-6)


class PandaGymAdapter(BenchmarkAdapter):
    """Base adapter for Panda-Gym's goal-conditioned PyBullet tasks."""

    fallback_dt = 0.04  # 20 PyBullet substeps at 0.002 s each.
    energy_weight, jerk_weight = 0.005, 0.00001

    def make_env(self):
        try:
            import panda_gym  # noqa: F401 -- import registers the Gymnasium environments
        except ImportError as exc:
            raise RuntimeError(
                "Panda-Gym benchmark support requires: pip install -e '.[benchmarks]'"
            ) from exc
        return super().make_env()

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory["integral_xyz"] = np.zeros(3)
        return memory

    @staticmethod
    def _state(observation: dict[str, np.ndarray]):
        state = np.asarray(observation["observation"], dtype=float)
        achieved = np.asarray(observation["achieved_goal"], dtype=float)
        desired = np.asarray(observation["desired_goal"], dtype=float)
        return state[:3], state[3:6], achieved, desired

    def success(self, env, observation, steps, terminated):
        del steps, terminated
        achieved = np.asarray(observation["achieved_goal"], dtype=float)
        desired = np.asarray(observation["desired_goal"], dtype=float)
        return bool(env.unwrapped.task.is_success(achieved, desired))


class PandaReachAdapter(PandaGymAdapter):
    env_id = "PandaReachDense-v3"
    horizon = 50
    allowed_terms = (
        "goal_error",
        "normalized_goal_error",
        "integral_goal_error",
        "eef_damping",
        "tanh_goal_error",
    )
    classical = (
        GymStructure("Task P", ("goal_error",)),
        GymStructure("Task PI", ("goal_error", "integral_goal_error")),
        GymStructure("Task PD", ("goal_error", "eef_damping")),
        GymStructure("Task PID", ("goal_error", "integral_goal_error", "eef_damping")),
    )

    def features(self, env, observation, memory, dt):
        del env
        eef, velocity, _, goal = self._state(observation)
        error = goal[:3] - eef
        memory["integral_xyz"] = np.clip(
            memory["integral_xyz"] + error * dt, -0.25, 0.25
        )
        return {
            "goal_error": error,
            "normalized_goal_error": _normalized(error),
            "integral_goal_error": memory["integral_xyz"].copy(),
            "eef_damping": -velocity,
            "tanh_goal_error": np.tanh(10.0 * error),
        }


class PandaObjectMotionAdapter(PandaGymAdapter):
    """Shared features for pushing or sliding one object to a Cartesian goal."""

    horizon = 50
    allowed_terms = (
        "reach_object",
        "normalized_reach_object",
        "object_goal_error",
        "normalized_object_goal_error",
        "eef_damping",
        "contact_then_goal",
    )
    classical = (
        GymStructure("Reach P", ("reach_object",)),
        GymStructure("Reach PD", ("reach_object", "eef_damping")),
        GymStructure("Object Goal P", ("reach_object", "object_goal_error")),
        GymStructure("Contact + Goal PD", ("contact_then_goal", "eef_damping")),
    )

    def features(self, env, observation, memory, dt):
        del env, memory, dt
        eef, velocity, achieved, desired = self._state(observation)
        obj = achieved[:3]
        goal_error = desired[:3] - obj
        reach = obj - eef
        near_object = float(np.linalg.norm(reach)) < 0.08
        sequential = goal_error if near_object else reach
        return {
            "reach_object": reach,
            "normalized_reach_object": _normalized(reach),
            "object_goal_error": goal_error,
            "normalized_object_goal_error": _normalized(goal_error),
            "eef_damping": -velocity,
            "contact_then_goal": sequential,
        }


class PandaPushAdapter(PandaObjectMotionAdapter):
    env_id = "PandaPushDense-v3"


class PandaSlideAdapter(PandaObjectMotionAdapter):
    env_id = "PandaSlideDense-v3"


class PandaPickAndPlaceAdapter(PandaGymAdapter):
    env_id = "PandaPickAndPlaceDense-v3"
    horizon = 50
    allowed_terms = (
        "reach_object",
        "normalized_reach_object",
        "object_goal_error",
        "eef_damping",
        "grasp_close",
        "lift_then_transport",
        "release_on_target",
        "pick_place_sequence",
    )
    classical = (
        GymStructure("Reach P", ("reach_object",)),
        GymStructure("Reach PD", ("reach_object", "eef_damping")),
        GymStructure(
            "Pick + Place", ("reach_object", "grasp_close", "lift_then_transport")
        ),
        GymStructure(
            "Pick + Place PD",
            ("pick_place_sequence", "eef_damping", "grasp_close", "release_on_target"),
        ),
    )

    def features(self, env, observation, memory, dt):
        del env, memory, dt
        eef, velocity, achieved, desired = self._state(observation)
        obj, goal = achieved[:3], desired[:3]
        reach = obj - eef
        goal_error = goal - obj
        reach_distance = float(np.linalg.norm(reach))
        goal_distance = float(np.linalg.norm(goal_error))
        grasped = reach_distance < 0.07
        lift_target = np.array([0.0, 0.0, max(0.0, 0.12 - obj[2])])
        transport = goal_error if grasped and obj[2] >= 0.08 else lift_target
        if not grasped:
            sequence = _action(4, reach)
        elif goal_distance < 0.05:
            sequence = _action(4, goal_error, gripper=1.0)
        else:
            sequence = _action(4, transport, gripper=-1.0)
        return {
            "reach_object": _action(4, reach),
            "normalized_reach_object": _action(4, _normalized(reach)),
            "object_goal_error": _action(4, goal_error),
            "eef_damping": _action(4, -velocity),
            "grasp_close": _action(4, gripper=-1.0),
            "lift_then_transport": _action(4, transport),
            "release_on_target": _action(4, gripper=1.0 if goal_distance < 0.05 else 0.0),
            "pick_place_sequence": sequence,
        }


class PandaStackAdapter(PandaGymAdapter):
    env_id = "PandaStackDense-v3"
    horizon = 100
    allowed_terms = (
        "reach_cube_one",
        "cube_one_goal_error",
        "eef_damping",
        "grasp_close",
        "lift_then_stack",
        "release_on_stack",
        "stack_sequence",
    )
    classical = (
        GymStructure("Reach P", ("reach_cube_one",)),
        GymStructure("Reach PD", ("reach_cube_one", "eef_damping")),
        GymStructure("Pick + Stack", ("reach_cube_one", "grasp_close", "lift_then_stack")),
        GymStructure(
            "Pick + Stack PD",
            ("stack_sequence", "eef_damping", "grasp_close", "release_on_stack"),
        ),
    )

    def features(self, env, observation, memory, dt):
        del env, memory, dt
        eef, velocity, achieved, desired = self._state(observation)
        cube_one, cube_one_goal = achieved[:3], desired[:3]
        reach = cube_one - eef
        goal_error = cube_one_goal - cube_one
        reach_distance = float(np.linalg.norm(reach))
        goal_distance = float(np.linalg.norm(goal_error))
        grasped = reach_distance < 0.07
        lift_target = np.array([0.0, 0.0, max(0.0, cube_one_goal[2] + 0.06 - cube_one[2])])
        transport = goal_error if grasped and cube_one[2] >= cube_one_goal[2] + 0.03 else lift_target
        if not grasped:
            sequence = _action(4, reach)
        elif goal_distance < 0.05:
            sequence = _action(4, goal_error, gripper=1.0)
        else:
            sequence = _action(4, transport, gripper=-1.0)
        return {
            "reach_cube_one": _action(4, reach),
            "cube_one_goal_error": _action(4, goal_error),
            "eef_damping": _action(4, -velocity),
            "grasp_close": _action(4, gripper=-1.0),
            "lift_then_stack": _action(4, transport),
            "release_on_stack": _action(4, gripper=1.0 if goal_distance < 0.05 else 0.0),
            "stack_sequence": sequence,
        }


PANDA_GYM_ADAPTERS = {
    "panda_reach": PandaReachAdapter(),
    "panda_push": PandaPushAdapter(),
    "panda_slide": PandaSlideAdapter(),
    "panda_pick_and_place": PandaPickAndPlaceAdapter(),
    "panda_stack": PandaStackAdapter(),
}
