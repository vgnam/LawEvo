from __future__ import annotations

import numpy as np

from lawevo.pid.gym_benchmark import (
    BenchmarkAdapter,
    EpisodeTracker,
    GymStructure,
    clip_violation,
)


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


# SG normalization scales: fixed before any controller comparison, sized from the
# tasks' physical reach (goal zones span roughly +/-0.15 m around the arm).
REACH_DISTANCE_SCALE = 0.15
PUSH_DISTANCE_SCALE = 0.20
SLIDE_DISTANCE_SCALE = 0.35
PICK_DISTANCE_SCALE = 0.25
STACK_POSITION_SCALE = 0.15
STACK_VELOCITY_SCALE = 0.30


class _PandaStateTracker(EpisodeTracker):
    """Record per-step end-effector and object state from the dict observation.

    ``ee`` is the end-effector position, ``goal`` the desired goal, and ``obj``
    the achieved goal (the controlled object; equal to ``ee`` for Reach). The
    task's success tolerance is recorded once so metric thresholds always match
    the environment specification.
    """

    distance_threshold: float = 0.05

    def update(self, env, observation, terminated: bool) -> None:
        state = np.asarray(observation["observation"], dtype=float)
        achieved = np.asarray(observation["achieved_goal"], dtype=float).reshape(-1)
        desired = np.asarray(observation["desired_goal"], dtype=float).reshape(-1)
        task = getattr(env.unwrapped, "task", None)
        if task is not None and hasattr(task, "distance_threshold"):
            self.distance_threshold = float(task.distance_threshold)
        self.steps.append(
            {
                "ee": state[:3].copy(),
                "obj": achieved.copy(),
                "goal": desired.copy(),
                "terminated": bool(terminated),
            }
        )


class _PandaStackTracker(EpisodeTracker):
    """Track the top cube (object1) toward its desired stacking pose.

    ``object1`` is the cube the robot must grasp and stack; ``object2`` stays
    under it. ``desired_goal[:3]`` is object1's target pose, so the position
    error below matches the task's own success distance.
    """

    settle_speed_limit: float | None = None
    settle_steps_required: int = 5

    def __init__(self, horizon: int) -> None:
        super().__init__(horizon)
        self.position_errors: list[float] = []
        self.speeds: list[float] = []
        self.grasp_distances: list[float] = []
        self.heights: list[float] = []

    def update(self, env, observation, terminated: bool) -> None:
        state = np.asarray(observation["observation"], dtype=float)
        achieved = np.asarray(observation["achieved_goal"], dtype=float).reshape(-1)
        desired = np.asarray(observation["desired_goal"], dtype=float).reshape(-1)
        cube_one = achieved[:3]
        goal = desired[:3]
        # is_success compares the full 6-D goal (both cubes), so the recorded
        # position error uses the same 6-D distance.
        self.position_errors.append(float(np.linalg.norm(desired - achieved)))
        self.heights.append(float(cube_one[2]))
        sim = env.unwrapped.sim
        velocity = np.asarray(sim.get_base_velocity("object1"), dtype=float)
        self.speeds.append(float(np.linalg.norm(velocity)))
        task = getattr(env.unwrapped, "task", None)
        if task is not None and hasattr(task, "settle_speed"):
            self.settle_speed_limit = float(task.settle_speed)
        if task is not None and hasattr(task, "distance_threshold"):
            self.distance_threshold = float(task.distance_threshold)
        self.grasp_distances.append(float(np.linalg.norm(cube_one - state[:3])))
        self.steps.append({"terminated": bool(terminated)})

    distance_threshold: float = 0.1
    settle_speed_limit: float | None = None
    settle_steps_required: int = 5

    @property
    def position_error_final(self) -> float:
        return self.position_errors[-1] if self.position_errors else float("inf")

    @property
    def final_speed(self) -> float:
        return self.speeds[-1] if self.speeds else float("inf")

    @property
    def max_lift_height(self) -> float:
        if not self.heights:
            return 0.0
        return max(self.heights) - self.heights[0]

    @property
    def min_grasp_distance(self) -> float:
        return min(self.grasp_distances) if self.grasp_distances else float("inf")

    @property
    def max_consecutive_stable_steps(self) -> int:
        if self.settle_speed_limit is None:
            stable = [e <= self.distance_threshold for e in self.position_errors]
        else:
            stable = [
                e <= self.distance_threshold and s <= self.settle_speed_limit
                for e, s in zip(self.position_errors, self.speeds)
            ]
        longest = run = 0
        for flag in stable:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        return longest


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

    def make_tracker(self) -> EpisodeTracker:
        return _PandaStateTracker(self.horizon)


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

    def success_constraints(self, tracker):
        # C1: final end-effector-to-goal distance within the task threshold.
        final = np.asarray(tracker.steps[-1]["ee"], dtype=float)
        goal = np.asarray(tracker.steps[-1]["goal"], dtype=float)
        final_distance = float(np.linalg.norm(goal - final))
        tolerance = float(tracker.distance_threshold)
        return {
            "goal_position": clip_violation(final_distance - tolerance, REACH_DISTANCE_SCALE),
        }

    def progress_predicates(self, tracker):
        goal = np.asarray(tracker.steps[0]["goal"], dtype=float)
        tolerance = float(tracker.distance_threshold)
        distances = [
            float(np.linalg.norm(goal - np.asarray(step["ee"], dtype=float)))
            for step in tracker.steps
        ]
        return {
            "goal_reached": any(d <= tolerance for d in distances),
            "goal_reached_final": distances[-1] <= tolerance,
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

    def success_constraints(self, tracker):
        # C1: the object finishes inside the goal tolerance.
        final_obj = np.asarray(tracker.steps[-1]["obj"], dtype=float)
        goal = np.asarray(tracker.steps[-1]["goal"], dtype=float)
        final_distance = float(np.linalg.norm(goal - final_obj))
        tolerance = float(tracker.distance_threshold)
        return {
            "goal_position": clip_violation(final_distance - tolerance, PUSH_DISTANCE_SCALE),
        }

    def progress_predicates(self, tracker):
        start_obj = np.asarray(tracker.steps[0]["obj"], dtype=float)
        goal = np.asarray(tracker.steps[0]["goal"], dtype=float)
        tolerance = float(tracker.distance_threshold)
        total = float(np.linalg.norm(goal - start_obj))
        distances = [
            float(np.linalg.norm(goal - np.asarray(step["obj"], dtype=float)))
            for step in tracker.steps
        ]
        # P2 uses half the initial object-goal distance, capped to a physical
        # minimum so a nearly-solved reset cannot grant the predicate for free.
        progress_threshold = max(0.5 * total, tolerance + 0.02)
        return {
            "object_contacted": any(
                float(np.linalg.norm(np.asarray(step["obj"], dtype=float) - np.asarray(step["ee"], dtype=float)))
                <= 0.06
                for step in tracker.steps
            ),
            "object_displaced_toward_goal": (total - min(distances)) >= progress_threshold,
            "object_in_goal_region": any(d <= tolerance for d in distances),
            "object_final_in_goal": distances[-1] <= tolerance,
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

    def success_constraints(self, tracker):
        # C1: the target object finishes inside the goal tolerance.
        final_obj = np.asarray(tracker.steps[-1]["obj"], dtype=float)
        goal = np.asarray(tracker.steps[-1]["goal"], dtype=float)
        final_distance = float(np.linalg.norm(goal - final_obj))
        tolerance = float(tracker.distance_threshold)
        return {
            "goal_position": clip_violation(final_distance - tolerance, PICK_DISTANCE_SCALE),
        }

    def progress_predicates(self, tracker):
        tolerance = float(tracker.distance_threshold)
        obj_positions = [np.asarray(step["obj"], dtype=float) for step in tracker.steps]
        ee_positions = [np.asarray(step["ee"], dtype=float) for step in tracker.steps]
        goal = np.asarray(tracker.steps[0]["goal"], dtype=float)
        start_obj = obj_positions[0]
        grasp_distances = [
            float(np.linalg.norm(obj - ee)) for obj, ee in zip(obj_positions, ee_positions)
        ]
        goal_distances = [float(np.linalg.norm(goal - obj)) for obj in obj_positions]
        start_height, max_height = float(start_obj[2]), max(float(o[2]) for o in obj_positions)
        return {
            "object_contacted": min(grasp_distances) <= 0.07,
            "object_grasped": min(grasp_distances) <= 0.07 and max_height - start_height > 0.02,
            "object_lifted": max_height - start_height > 0.04,
            "object_in_goal_neighborhood": any(d <= 0.10 for d in goal_distances),
            "object_final_in_goal": goal_distances[-1] <= tolerance,
        }


class PandaStackAdapter(PandaGymAdapter):
    env_id = "PandaStackDense-v3"
    horizon = 100

    def make_tracker(self) -> EpisodeTracker:
        return _PandaStackTracker(self.horizon)

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

    def success_constraints(self, tracker):
        # C1: the stacked cube finishes inside the tightened position tolerance.
        # The narrow-settle variant appends its velocity (C2) and sustained
        # settle (C3) constraints.
        violations = {
            "position_error": clip_violation(
                tracker.position_error_final - tracker.distance_threshold,
                STACK_POSITION_SCALE,
            ),
        }
        if tracker.settle_speed_limit is not None:
            violations["low_velocity"] = clip_violation(
                tracker.final_speed - tracker.settle_speed_limit, STACK_VELOCITY_SCALE
            )
            violations["settle_duration"] = clip_violation(
                tracker.settle_steps_required - tracker.max_consecutive_stable_steps,
                float(tracker.settle_steps_required),
            )
        return violations

    def progress_predicates(self, tracker):
        tolerance = float(tracker.distance_threshold)
        max_run = tracker.max_consecutive_stable_steps
        return {
            "cube_contacted": tracker.min_grasp_distance <= 0.07,
            "cube_lifted": tracker.max_lift_height > 0.02,
            "cube_above_base": min(tracker.position_errors) < 3.0 * tolerance,
            "cube_within_tolerance": min(tracker.position_errors) <= tolerance,
            "cube_position_and_velocity_ok": (
                tracker.position_error_final <= tolerance
                and (
                    tracker.settle_speed_limit is None
                    or tracker.final_speed <= tracker.settle_speed_limit
                )
            ),
            "cube_settled_n_steps": max_run >= tracker.settle_steps_required,
        }


PANDA_GYM_ADAPTERS = {
    "panda_reach": PandaReachAdapter(),
    "panda_push": PandaPushAdapter(),
    "panda_slide": PandaSlideAdapter(),
    "panda_pick_and_place": PandaPickAndPlaceAdapter(),
    "panda_stack": PandaStackAdapter(),
}
