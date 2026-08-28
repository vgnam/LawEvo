from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from lawevo.pid.gym_benchmark import BenchmarkAdapter, GymStructure


def _osc_action(
    translation: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    gripper: float = 0.0,
) -> np.ndarray:
    """Construct Panda OSC pose + gripper command in robosuite action order."""
    action = np.zeros(7, dtype=float)
    if translation is not None:
        action[:3] = np.asarray(translation, dtype=float)
    if rotation is not None:
        action[3:6] = np.asarray(rotation, dtype=float)
    action[6] = gripper
    return action


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-6)


class RobosuiteGymWrapper:
    """Expose the robosuite four-return API through the interface used by LawEvo."""

    def __init__(self, suite_env: Any):
        self.suite_env = suite_env
        low, high = suite_env.action_spec
        self.action_space = gym.spaces.Box(
            np.asarray(low, dtype=np.float32),
            np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )
        self.dt = 1.0 / float(suite_env.control_freq)

    @property
    def unwrapped(self) -> RobosuiteGymWrapper:
        return self

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.suite_env.seed = seed
            self.suite_env.rng = np.random.default_rng(seed)
        return self.suite_env.reset(), {}

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.suite_env.step(action)
        return observation, reward, False, done, info

    def close(self) -> None:
        self.suite_env.close()


class RobosuiteAdapter(BenchmarkAdapter):
    """Panda OSC adapter shared by task-specific robosuite benchmarks."""

    task_name: str
    horizon = 200
    energy_weight, jerk_weight = 0.01, 0.00001
    object_body_attributes: tuple[str, ...] = ()
    object_body_names: tuple[str, ...] = ()

    def make_env(self):
        try:
            import robosuite
            from robosuite.controllers import load_composite_controller_config
        except ImportError as exc:
            raise RuntimeError(
                "robosuite benchmark support requires: pip install -e '.[benchmarks]'"
            ) from exc

        controller = load_composite_controller_config(controller="BASIC")
        suite_env = robosuite.make(
            env_name=self.task_name,
            robots="Panda",
            controller_configs=controller,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            reward_scale=1.0,
            horizon=self.horizon,
            control_freq=20,
            hard_reset=False,
        )
        return RobosuiteGymWrapper(suite_env)

    def prepare_reset(self, env, observation, seed):
        rng = np.random.default_rng(seed + 6201)
        suite_env = env.suite_env
        model = suite_env.sim.model
        if not hasattr(suite_env, "_lawevo_base_body_mass"):
            suite_env._lawevo_base_body_mass = model.body_mass.copy()
        model.body_mass[:] = suite_env._lawevo_base_body_mass
        for attribute in self.object_body_attributes:
            body_id = int(getattr(suite_env, attribute))
            model.body_mass[body_id] *= float(rng.uniform(0.9, 1.1))
        for body_name in self.object_body_names:
            body_id = int(model.body_name2id(body_name))
            model.body_mass[body_id] *= float(rng.uniform(0.9, 1.1))
        suite_env.sim.forward()
        return observation

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory["previous_eef"] = None
        memory["integral_xyz"] = np.zeros(3)
        return memory

    def _common(self, observation: dict[str, np.ndarray], memory: dict, dt: float):
        eef = np.asarray(observation["robot0_eef_pos"], dtype=float)
        previous = memory["previous_eef"]
        velocity = np.zeros(3) if previous is None else (eef - previous) / dt
        memory["previous_eef"] = eef.copy()
        return eef, velocity

    def success(self, env, observation, steps, terminated):
        del observation, steps, terminated
        return bool(env.suite_env._check_success())


class RobosuiteLiftAdapter(RobosuiteAdapter):
    env_id = "RobosuiteLift-v0"
    task_name = "Lift"
    object_body_attributes = ("cube_body_id",)
    allowed_terms = (
        "reach_cube",
        "normalized_reach",
        "integral_reach",
        "eef_damping",
        "grasp_close",
        "lift_height_error",
        "grasp_gated_lift",
        "table_centering",
    )
    classical = (
        GymStructure("Reach P", ("reach_cube",)),
        GymStructure("Reach PD", ("reach_cube", "eef_damping")),
        GymStructure("Pick + Lift", ("reach_cube", "grasp_close", "lift_height_error")),
        GymStructure(
            "Pick + Lift PD",
            ("reach_cube", "eef_damping", "grasp_close", "grasp_gated_lift"),
        ),
    )

    def features(self, env, observation, memory, dt):
        del env
        eef, velocity = self._common(observation, memory, dt)
        cube = np.asarray(observation["cube_pos"], dtype=float)
        reach = cube - eef
        memory["integral_xyz"] = np.clip(
            memory["integral_xyz"] + reach * dt, -0.25, 0.25
        )
        distance = float(np.linalg.norm(reach))
        lift_error = max(0.0, 0.95 - float(cube[2]))
        return {
            "reach_cube": _osc_action(reach),
            "normalized_reach": _osc_action(_normalized(reach)),
            "integral_reach": _osc_action(memory["integral_xyz"]),
            "eef_damping": _osc_action(-velocity),
            "grasp_close": _osc_action(gripper=1.0),
            "lift_height_error": _osc_action(np.array([0.0, 0.0, lift_error])),
            "grasp_gated_lift": _osc_action(
                np.array([0.0, 0.0, lift_error if distance < 0.06 else 0.0]),
                gripper=1.0 if distance < 0.08 else 0.0,
            ),
            "table_centering": _osc_action(np.array([-cube[0], -cube[1], 0.0])),
        }


class RobosuiteStackAdapter(RobosuiteAdapter):
    env_id = "RobosuiteStack-v0"
    task_name = "Stack"
    object_body_attributes = ("cubeA_body_id", "cubeB_body_id")
    allowed_terms = (
        "reach_cube_a",
        "normalized_reach_a",
        "integral_reach",
        "eef_damping",
        "grasp_close",
        "lift_cube_a",
        "align_above_cube_b",
        "release_on_target",
    )
    classical = (
        GymStructure("Reach P", ("reach_cube_a",)),
        GymStructure("Reach PD", ("reach_cube_a", "eef_damping")),
        GymStructure(
            "Pick + Stack",
            ("reach_cube_a", "grasp_close", "lift_cube_a", "align_above_cube_b"),
        ),
        GymStructure(
            "Pick + Stack PD",
            (
                "reach_cube_a",
                "eef_damping",
                "grasp_close",
                "lift_cube_a",
                "align_above_cube_b",
                "release_on_target",
            ),
        ),
    )

    def features(self, env, observation, memory, dt):
        del env
        eef, velocity = self._common(observation, memory, dt)
        cube_a = np.asarray(observation["cubeA_pos"], dtype=float)
        cube_b = np.asarray(observation["cubeB_pos"], dtype=float)
        reach = cube_a - eef
        memory["integral_xyz"] = np.clip(
            memory["integral_xyz"] + reach * dt, -0.25, 0.25
        )
        reach_distance = float(np.linalg.norm(reach))
        stack_goal = cube_b + np.array([0.0, 0.0, 0.05])
        stack_error = stack_goal - cube_a
        near_target = float(np.linalg.norm(stack_error)) < 0.04
        grasped = reach_distance < 0.07
        return {
            "reach_cube_a": _osc_action(reach),
            "normalized_reach_a": _osc_action(_normalized(reach)),
            "integral_reach": _osc_action(memory["integral_xyz"]),
            "eef_damping": _osc_action(-velocity),
            "grasp_close": _osc_action(gripper=1.0),
            "lift_cube_a": _osc_action(
                np.array([0.0, 0.0, max(0.0, 0.98 - cube_a[2]) if grasped else 0.0])
            ),
            "align_above_cube_b": _osc_action(stack_error if grasped else np.zeros(3)),
            "release_on_target": _osc_action(gripper=-1.0 if near_target else 0.0),
        }


class RobosuiteNutAssemblyAdapter(RobosuiteAdapter):
    env_id = "RobosuiteNutAssemblySquare-v0"
    task_name = "NutAssemblySquare"
    object_body_names = ("SquareNut_main",)
    allowed_terms = (
        "reach_nut",
        "normalized_reach_nut",
        "integral_reach",
        "eef_damping",
        "grasp_close",
        "lift_nut",
        "align_above_peg",
        "insert_on_peg",
    )
    classical = (
        GymStructure("Reach P", ("reach_nut",)),
        GymStructure("Reach PD", ("reach_nut", "eef_damping")),
        GymStructure(
            "Pick + Insert",
            ("reach_nut", "grasp_close", "lift_nut", "align_above_peg", "insert_on_peg"),
        ),
        GymStructure(
            "Pick + Insert PD",
            (
                "reach_nut",
                "eef_damping",
                "grasp_close",
                "lift_nut",
                "align_above_peg",
                "insert_on_peg",
            ),
        ),
    )

    def features(self, env, observation, memory, dt):
        eef, velocity = self._common(observation, memory, dt)
        nut = np.asarray(observation["SquareNut_pos"], dtype=float)
        peg = np.asarray(env.suite_env.sim.data.body_xpos[env.suite_env.peg1_body_id], dtype=float)
        reach = nut - eef
        memory["integral_xyz"] = np.clip(
            memory["integral_xyz"] + reach * dt, -0.25, 0.25
        )
        reach_distance = float(np.linalg.norm(reach))
        grasped = reach_distance < 0.08
        above_peg = peg + np.array([0.0, 0.0, 0.10])
        horizontal_error = above_peg - nut
        horizontal_error[2] = 0.0
        aligned = float(np.linalg.norm(horizontal_error[:2])) < 0.025
        return {
            "reach_nut": _osc_action(reach),
            "normalized_reach_nut": _osc_action(_normalized(reach)),
            "integral_reach": _osc_action(memory["integral_xyz"]),
            "eef_damping": _osc_action(-velocity),
            "grasp_close": _osc_action(gripper=1.0),
            "lift_nut": _osc_action(
                np.array([0.0, 0.0, max(0.0, above_peg[2] - nut[2]) if grasped else 0.0])
            ),
            "align_above_peg": _osc_action(horizontal_error if grasped else np.zeros(3)),
            "insert_on_peg": _osc_action(
                np.array([0.0, 0.0, peg[2] - nut[2]]) if aligned else np.zeros(3),
                gripper=1.0 if aligned else 0.0,
            ),
        }


class RobosuiteDoorAdapter(RobosuiteAdapter):
    env_id = "RobosuiteDoor-v0"
    task_name = "Door"
    object_body_names = ("Door_door", "Door_latch")
    allowed_terms = (
        "reach_handle",
        "normalized_reach_handle",
        "integral_reach",
        "eef_damping",
        "grasp_close",
        "turn_handle",
        "door_tangent",
        "hinge_open_error",
    )
    classical = (
        GymStructure("Reach P", ("reach_handle",)),
        GymStructure("Reach PD", ("reach_handle", "eef_damping")),
        GymStructure(
            "Door P", ("reach_handle", "grasp_close", "turn_handle", "door_tangent")
        ),
        GymStructure(
            "Door PD",
            ("reach_handle", "eef_damping", "grasp_close", "turn_handle", "door_tangent"),
        ),
    )

    def features(self, env, observation, memory, dt):
        del env
        eef, velocity = self._common(observation, memory, dt)
        handle = np.asarray(observation["handle_pos"], dtype=float)
        door = np.asarray(observation["door_pos"], dtype=float)
        reach = handle - eef
        memory["integral_xyz"] = np.clip(
            memory["integral_xyz"] + reach * dt, -0.25, 0.25
        )
        near_handle = float(np.linalg.norm(reach)) < 0.07
        handle_angle = float(observation["handle_qpos"])
        hinge_angle = float(observation["hinge_qpos"])
        radial = handle[:2] - door[:2]
        tangent = np.array([-radial[1], radial[0], 0.0])
        tangent = _normalized(tangent) * max(0.0, 0.4 - abs(hinge_angle))
        return {
            "reach_handle": _osc_action(reach),
            "normalized_reach_handle": _osc_action(_normalized(reach)),
            "integral_reach": _osc_action(memory["integral_xyz"]),
            "eef_damping": _osc_action(-velocity),
            "grasp_close": _osc_action(gripper=1.0),
            "turn_handle": _osc_action(
                rotation=np.array([0.5 - handle_angle, 0.0, 0.0])
                if near_handle
                else np.zeros(3)
            ),
            "door_tangent": _osc_action(tangent if near_handle else np.zeros(3)),
            "hinge_open_error": _osc_action(
                tangent * max(0.0, 0.4 - abs(hinge_angle)) if near_handle else np.zeros(3)
            ),
        }


ROBOSUITE_ADAPTERS = {
    "robosuite_lift": RobosuiteLiftAdapter(),
    "robosuite_stack": RobosuiteStackAdapter(),
    "robosuite_nut_assembly_square": RobosuiteNutAssemblyAdapter(),
    "robosuite_door": RobosuiteDoorAdapter(),
}
