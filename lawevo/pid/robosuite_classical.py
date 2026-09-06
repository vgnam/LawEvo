"""Moderate Panda tasks with explicit OSC support and shared staged features.

These are LawEvo adapter IDs, not registrations in Gymnasium. Legacy robosuite
adapters remain available without changes to their task protocols.
"""
from __future__ import annotations

import numpy as np

from lawevo.pid.gym_benchmark import EpisodeTracker, GymStructure
from lawevo.pid.objective import ObjectiveConfig
from lawevo.pid.robosuite_benchmark import RobosuiteAdapter, RobosuiteGymWrapper
from lawevo.pid.signal_schema import SignalContract, SignalSpec


class _SeededWrapper(RobosuiteGymWrapper):
    def reset(self, seed=None):
        # Samplers / arena retain references to this generator: reseed in place.
        if seed is not None:
            self.suite_env.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        return self.suite_env.reset(), {}

    def step(self, action):
        obs, reward, done, info = self.suite_env.step(action)
        timeout = self.suite_env.timestep >= self.suite_env.horizon
        # Wipe has genuine early terminal events, including arm collisions.
        return obs, reward, bool(done and not timeout), bool(done and timeout), info


class _TaskTracker(EpisodeTracker):
    def __init__(self, adapter):
        super().__init__(adapter.horizon)
        self.adapter = adapter

    def update(self, env, observation, terminated):
        state = self.adapter.snapshot(env.suite_env, observation)
        self.steps.append(state)
        self.diagnostics.update(state)
        self.diagnostics["steps"] = len(self.steps)


class ClassicalRobosuiteAdapter(RobosuiteAdapter):
    horizon = 300
    action_dim = 7
    allowed_terms = ("waypoint", "damping", "integral", "orientation", "gripper")
    classical = (
        GymStructure("Staged OSC P", ("waypoint", "orientation", "gripper")),
        GymStructure("Staged OSC PD", ("waypoint", "damping", "orientation", "gripper")),
        GymStructure("Staged OSC PID", ("waypoint", "damping", "integral", "orientation", "gripper")),
    )

    def make_env(self):
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        config = load_composite_controller_config(controller="BASIC")
        # Explicit world frame: feature xyz and rotation errors use world axes.
        arm = config["body_parts"]["right"]
        config["body_parts"] = {"right": arm}
        arm.update(input_ref_frame="world", input_type="delta", kp=150,
                   damping_ratio=1, impedance_mode="fixed")
        kwargs = {"use_latch": False} if self.task_name == "Door" else {}
        env = robosuite.make(
            self.task_name, robots="Panda", controller_configs=config,
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
            use_object_obs=True, reward_shaping=True, reward_scale=1.0,
            horizon=self.horizon, control_freq=20, hard_reset=False, seed=0, **kwargs,
        )
        wrapper = _SeededWrapper(env)
        if wrapper.action_space.shape != (self.action_dim,):
            wrapper.close()
            raise ValueError(f"{self.env_id}: unexpected OSC action shape {wrapper.action_space.shape}")
        return wrapper

    def prepare_reset(self, env, observation, seed):
        # Stock masses, inertias, friction: no hidden mass randomization.
        return observation

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory.update(phase="approach", phase_time=0.0, target_index=None,
                      rotation=None, previous_point=None)
        return memory

    def make_tracker(self):
        return _TaskTracker(self)

    @property
    def objective_config(self):
        return ObjectiveConfig(float(self.horizon), 10.0, 100.0)

    def success_constraints(self, tracker):
        return {"task_completion": tracker.steps[-1]["gap"] if tracker.steps else 1.0}

    def progress_predicates(self, tracker):
        return {
            "first_stage_once": any(s["first"] for s in tracker.steps),
            "second_stage_once": any(s["second"] for s in tracker.steps),
            "success_final": bool(tracker.steps and tracker.steps[-1]["success"]),
        }

    @property
    def success_gap_spec(self):
        return {"version": 1, "formula": self.gap_description,
                "success": "native robosuite _check_success at final state",
                "boundary": "strict thresholds: minimum failure gap 1e-12",
                "aggregation": "mean over episodes", "physics": "stock; no added mass jitter"}

    @property
    def progress_spec(self):
        return {"version": 1, "milestones": list(self.milestones),
                "aggregation": "equal mean of two ever-achieved milestones and final success; diagnostic only"}

    @property
    def signal_contract(self):
        mapping = ("world xyz delta / 0.05 m; world rotation delta / 0.5 rad; "
                   + ("last channel +1 close, -1 open" if self.action_dim == 7 else "no gripper channel"))
        rows = {
            "waypoint": ("position_feedback", "(stage_target - controlled_point)/0.05 m in xyz; other channels zero", "target equals controlled point", self.stage_description),
            "damping": ("velocity_feedback", "-finite_difference(controlled_point)/(1 m/s) in xyz", "first call or stationary point", "previous controlled point; reset each episode"),
            "integral": ("integral_feedback", "clip(integral((target-point) dt), -0.02, 0.02)/(0.05 m*s) in xyz", "integral zero", "cleared at reset and phase changes"),
            "orientation": ("orientation_feedback", "orientation_error(reference_rotation,current_eef_rotation)/0.5 in rotation channels; Lift holds initial rotation; Wipe holds initial yaw with z down; Door follows handle long axis with z down", "reference orientation matched; xyz and gripper always zero", "Lift/Wipe reference stored on first call; Door reference from current latch body rotation"),
            "gripper": ("gripper_command", "-1 approach, +1 after reaching grasp waypoint; xyz and rotation zero", "not used by Wipe", self.stage_description),
        }
        specs = {name: SignalSpec(kind, "1", (0, 0), (self.action_dim,), formula,
                                 "positive gain follows target or opposes velocity", zero,
                                 "no clipping except integral; summed action clipped to [-1,1]",
                                 memory, "world", mapping)
                 for name, (kind, formula, zero, memory) in rows.items() if name in self.allowed_terms}
        return SignalContract(specs, (self.action_dim,), mapping + "; fixed OSC kp=150, damping ratio=1")

    def features(self, env, observation, memory, dt):
        from robosuite.utils.control_utils import orientation_error
        from robosuite.utils.transform_utils import quat2mat

        rotation = quat2mat(np.asarray(observation["robot0_eef_quat"]))
        if memory["rotation"] is None:
            memory["rotation"] = rotation.copy()
            if self.task_name == "Wipe":
                # Remove reset roll/pitch so the entire tool face meets the table.
                x = np.array([rotation[0, 0], rotation[1, 0], 0.0])
                x /= max(np.linalg.norm(x), 1e-9)
                z = np.array([0.0, 0.0, -1.0])
                memory["rotation"] = np.column_stack((x, np.cross(z, x), z))
        if self.task_name == "Door":
            suite = env.suite_env
            body = suite.object_body_ids["latch"]
            x = suite.sim.data.body_xmat[body].reshape(3, 3)[:, 0].copy()
            x[2] = 0
            x /= max(np.linalg.norm(x), 1e-9)
            z = np.array([0.0, 0.0, -1.0])
            memory["rotation"] = np.column_stack((x, np.cross(z, x), z))
        old_phase = memory["phase"]
        point, target, grip = self.waypoint(env.suite_env, observation, memory, dt)
        if old_phase != memory["phase"]:
            memory["integral_xyz"][:] = 0
        error = target - point
        previous = memory["previous_point"]
        velocity = np.zeros(3) if previous is None else (point - previous) / dt
        memory["previous_point"] = point.copy()
        memory["integral_xyz"] = np.clip(memory["integral_xyz"] + error * dt, -0.02, 0.02)
        def xyz(v):
            a = np.zeros(self.action_dim)
            a[:3] = v
            return a
        orient = xyz(np.zeros(3))
        orient[3:6] = orientation_error(memory["rotation"], rotation) / 0.5
        values = {"waypoint": xyz(error / 0.05), "damping": xyz(-velocity),
                  "integral": xyz(memory["integral_xyz"] / 0.05), "orientation": orient}
        if "gripper" in self.allowed_terms:
            values["gripper"] = xyz(np.zeros(3))
            values["gripper"][-1] = grip
        return values

    @staticmethod
    def result(success, gap, first, second, **diagnostics):
        gap = 0.0 if success else max(1e-12, float(np.clip(gap, 0, 1)))
        return dict(success=bool(success), gap=gap, first=bool(first), second=bool(second), **diagnostics)


class RobosuiteLiftNominalAdapter(ClassicalRobosuiteAdapter):
    env_id = "RobosuiteLiftNominal-v0"
    task_name = "Lift"
    gap_description = "0 on native success; otherwise clip((table_z+0.04-cube_z)/0.04,0,1), floor 1e-12"
    milestones = ("eef within 0.04 m of cube once", "native two-finger grasp once", "cube_z > table_z+0.04 at final step")
    stage_description = ("Hand-written FSM: approach cube+0.08m z, descend when within 0.015m; "
                         "close when within 0.012m of cube; wait >=0.3s and native grasp then lift; "
                         "retry approach after 1s without grasp or loss of grasp; lift target eef+"
                         "max(0,table_z+0.12-cube_z) in z. Reset every episode.")

    def waypoint(self, suite, obs, memory, dt):
        eef, cube = np.asarray(obs["robot0_eef_pos"]), np.asarray(obs["cube_pos"])
        grasp = suite._check_grasp(suite.robots[0].gripper, suite.cube)
        phase = memory["phase"]
        if phase == "approach" and np.linalg.norm(cube + [0, 0, 0.08] - eef) < 0.015:
            phase = "descend"
        elif phase == "descend" and np.linalg.norm(cube - eef) < 0.012:
            phase = "close"
            memory["phase_time"] = 0.0
        elif phase == "close":
            memory["phase_time"] += dt
            if grasp and memory["phase_time"] >= 0.3:
                phase = "lift"
            elif memory["phase_time"] > 1.0:
                phase = "approach"
        elif phase == "lift" and not grasp:
            phase = "approach"
        memory["phase"] = phase
        target = cube.copy()
        if phase == "approach":
            target[2] += 0.08
        elif phase == "lift":
            target = eef.copy()
            target[2] += max(0.0, suite.model.mujoco_arena.table_offset[2] + 0.12 - cube[2])
        return eef, target, 1.0 if phase in ("close", "lift") else -1.0

    def snapshot(self, suite, obs):
        cube = np.asarray(obs["cube_pos"])
        height = float(cube[2] - suite.model.mujoco_arena.table_offset[2])
        distance = float(np.linalg.norm(cube - obs["robot0_eef_pos"]))
        return self.result(suite._check_success(), (0.04-height)/0.04, distance < 0.04,
                           suite._check_grasp(suite.robots[0].gripper, suite.cube), height=height)


class RobosuiteDoorUnlockedAdapter(ClassicalRobosuiteAdapter):
    env_id = "RobosuiteDoorUnlocked-v0"
    task_name = "Door"
    gap_description = "0 on native success; otherwise clip((0.3-hinge_angle)/0.3,0,1), floor 1e-12"
    milestones = ("eef within 0.05 m of handle once", "hinge angle > 0.15 rad once", "hinge angle > 0.3 rad at final step")
    stage_description = ("Hand-written proximity FSM: approach 0.08m above handle with gripper open; "
                         "within 0.04m descend; within 0.05m of handle close for 0.3s, latch eef-handle "
                         "offset then follow positive hinge tangent; realign if distance>0.12m. "
                         "Tangent uses MuJoCo world joint axis and anchor, length min(0.04m, "
                         "0.1m/rad * max(0,0.4rad-hinge_angle)). Reset each episode.")

    def waypoint(self, suite, obs, memory, dt):
        eef, handle = np.asarray(obs["robot0_eef_pos"]), np.asarray(obs["handle_pos"])
        distance = np.linalg.norm(handle - eef)
        if memory["phase"] == "approach" and np.linalg.norm(handle + [0, 0, 0.08] - eef) < 0.04:
            memory["phase"] = "descend"
        elif memory["phase"] == "descend" and distance < 0.05:
            memory.update(phase="close", phase_time=0.0, handle_offset=(eef-handle).copy())
        elif memory["phase"] == "close":
            memory["phase_time"] += dt
            if memory["phase_time"] >= 0.3:
                memory["phase"] = "open"
        if memory["phase"] in ("close", "open") and distance > 0.12:
            memory["phase"] = "approach"
        target = handle.copy()
        if memory["phase"] == "approach":
            target[2] += 0.08
        elif memory["phase"] in ("close", "open"):
            target += memory["handle_offset"]
        if memory["phase"] == "open":
            joint = suite.sim.model.joint_name2id(suite.door.joints[0])
            tangent = np.cross(suite.sim.data.xaxis[joint], handle - suite.sim.data.xanchor[joint])
            tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
            angle = float(np.asarray(obs["hinge_qpos"]).item())
            target += tangent * min(0.04, 0.1 * max(0, 0.4-angle))
        return eef, target, 1.0 if memory["phase"] in ("close", "open") else -1.0

    def snapshot(self, suite, obs):
        angle = float(np.asarray(obs["hinge_qpos"]).item())
        distance = float(np.linalg.norm(obs["handle_pos"] - obs["robot0_eef_pos"]))
        return self.result(suite._check_success(), (0.3-angle)/0.3, distance < 0.05,
                           angle > 0.15, hinge_angle=angle)


class RobosuiteWipeAdapter(ClassicalRobosuiteAdapter):
    env_id = "RobosuiteWipe-v0"
    task_name = "Wipe"
    horizon = 500
    action_dim = 6
    allowed_terms = ("waypoint", "damping", "integral", "orientation")
    classical = (
        GymStructure("Waypoint Wipe P", ("waypoint", "orientation")),
        GymStructure("Waypoint Wipe PD", ("waypoint", "damping", "orientation")),
        GymStructure("Waypoint Wipe PID", ("waypoint", "damping", "integral", "orientation")),
    )
    gap_description = "1 - number_of_wiped_markers / num_markers; native success requires all markers"
    milestones = ("at least one marker wiped", "at least half the markers wiped", "all markers wiped at final step")
    stage_description = ("Hand-written waypoint scheduler over privileged marker positions and wiped flags: "
                         "latch nearest unwiped marker until wiped, then choose next; approach 0.04m above "
                         "marker until within 0.015m, then sweep at marker_z-0.003m. Controlled point is "
                         "mean of four wiping-tool corner geom positions; hold position when all clean. "
                         "This is position-based pressing with fixed OSC impedance, not force feedback.")

    def waypoint(self, suite, obs, memory, dt):
        corners = suite.robots[0].gripper["right"].important_geoms["corners"]
        point = np.mean([suite.sim.data.geom_xpos[suite.sim.model.geom_name2id(n)] for n in corners], axis=0)
        markers = suite.model.mujoco_arena.markers
        remaining = [i for i, marker in enumerate(markers) if marker not in suite.wiped_markers]
        if not remaining:
            return point, point.copy(), 0.0
        positions = {i: suite.sim.data.body_xpos[suite.sim.model.body_name2id(markers[i].root_body)].copy()
                     for i in remaining}
        if memory["target_index"] not in remaining:
            memory["target_index"] = min(remaining, key=lambda i: np.linalg.norm(positions[i]-point))
        target = positions[memory["target_index"]].copy()
        if memory["phase"] == "approach":
            target[2] += 0.04
            if np.linalg.norm(target-point) < 0.015:
                memory["phase"] = "sweep"
        else:
            target[2] -= 0.003
        return point, target, 0.0

    def snapshot(self, suite, obs):
        fraction = len(suite.wiped_markers) / suite.num_markers
        return self.result(suite._check_success(), 1-fraction, fraction > 0, fraction >= 0.5,
                           wiped_fraction=fraction)


ROBOSUITE_CLASSICAL_ADAPTERS = {
    "robosuite_lift_nominal": RobosuiteLiftNominalAdapter(),
    "robosuite_door_unlocked": RobosuiteDoorUnlockedAdapter(),
    "robosuite_wipe": RobosuiteWipeAdapter(),
}
