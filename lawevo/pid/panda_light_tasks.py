"""Three small physical-contact Panda scenes, with state-only observations."""
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np

from lawevo.pid.gym_benchmark import BenchmarkAdapter, EpisodeTracker, GymStructure
from lawevo.pid.objective import ObjectiveConfig
from lawevo.pid.signal_schema import SignalContract, SignalSpec


class PandaLightEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, task_name):
        from panda_gym.envs.robots.panda import Panda
        from panda_gym.pybullet import PyBullet

        self.task_name = task_name
        self.sim = PyBullet(render_mode="rgb_array", renderer="Tiny")  # DIRECT; no images read
        self.robot = Panda(self.sim, block_gripper=task_name != "drawer",
                           base_position=np.array([-0.6, 0, 0]), control_type="ee")
        self.client = self.sim.physics_client
        self.dt = self.sim.dt
        self.action_space = self.robot.action_space
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.robot_id = self.sim._bodies_idx["panda"]
        self.table_id = self.sim._bodies_idx["table"]
        self.tool = None
        self.walls = []
        if task_name == "drawer":
            self.drawer = self.client.loadURDF(str(Path(__file__).parent / "assets" / "simple_drawer.urdf"),
                                               basePosition=[0.08, 0, 0.05], useFixedBase=True)
            self.client.setJointMotorControl2(self.drawer, 0, self.client.VELOCITY_CONTROL, force=0)
            self.client.changeDynamics(self.drawer, 0, lateralFriction=1.0)
        else:
            self.tool_half_height = 0.008 if task_name == "wipe" else 0.03
            if task_name == "wipe":
                self.sim.create_box("tool", half_extents=np.array([0.025, 0.025, 0.008]),
                                    mass=0.05, position=np.array([0, 0, 0.3]),
                                    rgba_color=np.array([0.2, 0.5, 0.9, 1]))
            else:
                self.sim.create_cylinder("tool", radius=0.012, height=0.06, mass=0.05,
                                         position=np.array([0, 0, 0.3]),
                                         rgba_color=np.array([0.2, 0.5, 0.9, 1]))
                # Four real collision walls, 36 mm square opening around 24 mm peg.
                for index, (half, offset) in enumerate((
                    ([0.012, 0.042, 0.035], [0.030, 0, 0.045]),
                    ([0.012, 0.042, 0.035], [-0.030, 0, 0.045]),
                    ([0.018, 0.012, 0.035], [0, 0.030, 0.045]),
                    ([0.018, 0.012, 0.035], [0, -0.030, 0.045]),
                )):
                    name = f"wall{index}"
                    self.sim.create_box(name, half_extents=np.array(half), mass=0,
                                        position=np.array(offset), rgba_color=np.array([0.6, 0.6, 0.6, 1]))
                    self.walls.append((self.sim._bodies_idx[name], np.array(offset)))
            self.tool = self.sim._bodies_idx["tool"]
            self.client.createConstraint(self.robot_id, self.robot.ee_link, self.tool, -1,
                                         self.client.JOINT_FIXED, [0, 0, 0], [0, 0, 0.04], [0, 0, 0])
            # A mounted tool should not collide with its own mounting hand.
            for link in range(-1, self.client.getNumJoints(self.robot_id)):
                self.client.setCollisionFilterPair(self.robot_id, self.tool, link, -1, 0)
        self.robot.reset()
        self.initial_state = self.client.saveState()
        self.cleaned = np.zeros(9, dtype=bool)
        self.markers = np.zeros((9, 3))
        self.target = np.zeros(3)
        self.contact_seen = False
        self.contact_force = 0.0
        self.reset(seed=0)
        example = self._observation()
        self.observation_space = gym.spaces.Dict({
            name: gym.spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=np.float64)
            for name, value in example.items()
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.client.restoreState(self.initial_state)
        self.robot.reset()
        self.cleaned[:] = False
        self._wipe_complete = False
        self.contact_seen = False
        self.contact_force = 0.0
        self.target = np.array([self.np_random.uniform(-0.03, 0.05), self.np_random.uniform(-0.06, 0.06), 0.0])
        if self.task_name == "drawer":
            self.client.resetBasePositionAndOrientation(self.drawer, [0.08, self.target[1], 0.05], [0, 0, 0, 1])
            self.client.resetJointState(self.drawer, 0, 0, 0)
            self.client.setJointMotorControl2(self.drawer, 0, self.client.VELOCITY_CONTROL, force=0)
        else:
            state = self.client.getLinkState(self.robot_id, self.robot.ee_link, computeForwardKinematics=True)
            pos, quat = self.client.multiplyTransforms(state[4], state[5], [0, 0, 0.04], [0, 0, 0, 1])
            self.client.resetBasePositionAndOrientation(self.tool, pos, quat)
            self.client.resetBaseVelocity(self.tool, [0, 0, 0], [0, 0, 0])
            if self.task_name == "wipe":
                self.markers = np.array([self.target + [x, y, 0] for x in (-0.035, 0, 0.035)
                                         for y in (-0.035, 0, 0.035)])
            else:
                for wall, offset in self.walls:
                    self.client.resetBasePositionAndOrientation(wall, self.target + offset, [0, 0, 0, 1])
                self.target[2] = 0.035
        self.client.performCollisionDetection()
        self._refresh_control_state()
        return self._observation(cached=True), {"is_success": False}

    def _refresh_control_state(self):
        """One physics snapshot per reset/control step, shared by all consumers.

        Direct physics accessors remain uncached for inspection/teleport tests.
        Internal snapshots are replaced, never updated in place, at each step.
        """
        self._robot_observation = self.robot.get_obs()
        self._eef = self._robot_observation[:3].copy()
        self._point = self._eef if self.task_name == "drawer" else self.tool_tip()
        self._handle = self.handle() if self.task_name == "drawer" else None
        self._opening = (float(self.client.getJointState(self.drawer, 0)[0])
                         if self.task_name == "drawer" else None)
        self._metrics = self.state_metrics(cached_inputs=True)

    def tool_tip(self):
        pos, quat = self.client.getBasePositionAndOrientation(self.tool)
        tip, _ = self.client.multiplyTransforms(pos, quat, [0, 0, self.tool_half_height], [0, 0, 0, 1])
        return np.array(tip)

    def handle(self):
        state = self.client.getLinkState(self.drawer, 0, computeForwardKinematics=True)
        pos, _ = self.client.multiplyTransforms(state[4], state[5], [-0.13, 0, 0.04], [0, 0, 0, 1])
        return np.array(pos)

    def _contact_update(self):
        if self.task_name == "drawer":
            contacts = self.client.getContactPoints(self.robot_id, self.drawer)
        else:
            contacts = self.client.getContactPoints(self.tool)
        self.contact_force = sum(max(0.0, float(c[9])) for c in contacts)
        self.contact_seen |= self.contact_force > 0.1
        if self.task_name == "wipe" and not self._wipe_complete:
            table_force = sum(max(0.0, float(c[9])) for c in contacts if c[2] == self.table_id)
            if table_force > 0.1:
                # Transform marker coordinates into the actual tool frame.
                pos, quat = self.client.getBasePositionAndOrientation(self.tool)
                rotation = np.array(self.client.getMatrixFromQuaternion(quat)).reshape(3, 3)
                local = (self.markers - pos) @ rotation
                touched = ((np.abs(local[:, 0]) <= 0.025) & (np.abs(local[:, 1]) <= 0.025)
                           & (np.abs(local[:, 2] - self.tool_half_height) <= 0.004))
                self.cleaned |= touched
                self._wipe_complete = bool(self.cleaned.all())

    def state_metrics(self, *, cached_inputs=False):
        if self.task_name == "drawer":
            opening = self._opening if cached_inputs else float(self.client.getJointState(self.drawer, 0)[0])
            gap = float(np.clip((0.12-opening)/0.12, 0, 1))
            return gap, (self.contact_seen, opening >= 0.06, opening >= 0.12), opening
        if self.task_name == "wipe":
            fraction = float(self.cleaned.mean())
            return 1-fraction, (self.contact_seen, fraction >= 0.5, fraction == 1), fraction
        tip = self._point if cached_inputs else self.tool_tip()
        lateral = float(np.linalg.norm(tip[:2]-self.target[:2]))
        # Match physical opening: 6 mm per-side clearance, 4 mm radial success tolerance.
        lateral_gap = np.clip((lateral-0.004)/0.05, 0, 1)
        depth_gap = np.clip((tip[2]-0.04)/0.08, 0, 1)
        floor_gap = np.clip((0.005-tip[2])/0.03, 0, 1)
        gap = float((lateral_gap+depth_gap+floor_gap)/3)
        return gap, (lateral <= 0.01, lateral <= 0.004 and tip[2] <= 0.07,
                     lateral <= 0.004 and 0.005 <= tip[2] <= 0.04), float(tip[2])

    def _observation(self, *, cached=False):
        state = self._robot_observation if cached else self.robot.get_obs()
        if self.task_name == "drawer":
            achieved = np.array([self._opening if cached else self.client.getJointState(self.drawer, 0)[0]])
            goal = np.array([0.12])
            extra = self._handle if cached else self.handle()
        elif self.task_name == "wipe":
            achieved, goal = self.cleaned.astype(float), np.ones(9)
            extra = np.concatenate((self._point if cached else self.tool_tip(), self.markers.ravel(), self.cleaned.astype(float)))
        else:
            achieved, goal = self._point if cached else self.tool_tip(), self.target.copy()
            extra = np.concatenate((achieved, goal))
        return {"observation": np.concatenate((state, extra, [self.contact_force])).astype(float),
                "achieved_goal": achieved.astype(float), "desired_goal": goal.astype(float)}

    def step(self, action):
        self.robot.set_action(np.asarray(action, dtype=float))
        for substep in range(self.sim.n_substeps):
            self.client.stepSimulation()
            # Preserve transient contact milestones and wiping credit. Once
            # latched, Drawer needs only the final force; a clean pad needs no
            # more footprint checks. Peg uses only final force in observations,
            # never contact history in SR/SG/Q/reward.
            final = substep == self.sim.n_substeps - 1
            need_history = ((self.task_name == "drawer" and not self.contact_seen)
                            or (self.task_name == "wipe" and not self._wipe_complete))
            if final or need_history:
                self._contact_update()
        self._refresh_control_state()
        gap, milestones, _ = self._metrics
        if self.task_name == "drawer":
            distance = np.linalg.norm(self._eef-self._handle)
        elif self.task_name == "wipe":
            distance = min((np.linalg.norm(self._point-p) for p in self.markers[~self.cleaned]), default=0.0)
        else:
            distance = np.linalg.norm(self._point-self.target)
        reward = -gap - 0.2*float(distance) + float(milestones[-1])
        return self._observation(cached=True), reward, False, False, {"is_success": bool(milestones[-1])}

    def close(self):
        self.sim.close()


class _LightTracker(EpisodeTracker):
    def update(self, env, observation, terminated):
        gap, milestones, progress = env.unwrapped._metrics
        self.steps.append({"gap": gap, "milestones": milestones})
        self.diagnostics.update(progress=progress, contact_force=env.unwrapped.contact_force)


class PandaLightAdapter(BenchmarkAdapter):
    energy_weight, jerk_weight = 0.005, 0.00001
    horizon = 150
    fallback_dt = 0.04
    allowed_terms = ("waypoint", "damping", "integral", "gripper")

    def __init__(self, task_name, env_id):
        self.task_name, self.env_id = task_name, env_id
        self.action_dim = 4 if task_name == "drawer" else 3
        if task_name != "drawer":
            self.allowed_terms = ("waypoint", "damping", "integral")
        grip = ("gripper",) if task_name == "drawer" else ()
        self.classical = tuple(GymStructure(f"{task_name.title()} {label}", terms + grip)
                               for label, terms in (("P", ("waypoint",)), ("PD", ("waypoint", "damping")),
                                                    ("PID", ("waypoint", "damping", "integral"))))

    def make_env(self):
        return gym.wrappers.TimeLimit(PandaLightEnv(self.task_name), self.horizon)

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory.update(phase="approach", timer=0.0, integral_xyz=np.zeros(3), previous_point=None, marker=None)
        return memory

    def features(self, env, observation, memory, dt):
        scene = env.unwrapped
        point = scene._point
        old_phase = memory["phase"]
        grip = 0.0
        if self.task_name == "drawer":
            handle = scene._handle
            if memory["phase"] == "approach" and np.linalg.norm(handle + [0, 0, 0.06] - point) < 0.015:
                memory["phase"] = "descend"
            elif memory["phase"] == "descend" and np.linalg.norm(handle-point) < 0.012:
                memory.update(phase="close", timer=0.0)
            elif memory["phase"] == "close":
                memory["timer"] += dt
                if memory["timer"] >= 0.4:
                    memory["phase"] = "pull"
            target = handle.copy()
            if memory["phase"] == "approach":
                target[2] += 0.06
            elif memory["phase"] == "pull":
                opening = scene._opening
                target[0] -= min(0.04, max(0, 0.14-opening))
            grip = -1.0 if memory["phase"] in ("close", "pull") else 1.0
        elif self.task_name == "wipe":
            remaining = np.flatnonzero(~scene.cleaned)
            if not len(remaining):
                target = point.copy()
            else:
                if memory["marker"] not in remaining:
                    memory["marker"] = min(remaining, key=lambda i: np.linalg.norm(scene.markers[i]-point))
                target = scene.markers[memory["marker"]].copy()
                if memory["phase"] == "approach":
                    target[2] += 0.04
                    if np.linalg.norm(target-point) < 0.015:
                        memory["phase"] = "wipe"
                else:
                    target[2] -= 0.002
        else:
            target = scene.target.copy()
            if memory["phase"] == "approach":
                target[2] = 0.11
                if np.linalg.norm(target-point) < 0.003:
                    memory["phase"] = "insert"
            elif np.linalg.norm(point[:2]-target[:2]) > 0.005:
                memory["phase"] = "approach"
                target[2] = 0.11
            else:
                target[2] = max(0.035, float(point[2])-0.005)
        if old_phase != memory["phase"]:
            memory["integral_xyz"][:] = 0
        error = target-point
        velocity = np.zeros(3) if memory["previous_point"] is None else (point-memory["previous_point"])/dt
        memory["previous_point"] = point.copy()
        memory["integral_xyz"] = np.clip(memory["integral_xyz"]+error*dt, -0.02, 0.02)
        def action(xyz):
            a = np.zeros(self.action_dim)
            a[:3] = xyz
            return a
        result = {"waypoint": action(error/0.05), "damping": action(-velocity),
                  "integral": action(memory["integral_xyz"]/0.05)}
        if self.action_dim == 4:
            result["gripper"] = action(np.zeros(3))
            result["gripper"][3] = grip
        return result

    def make_tracker(self):
        return _LightTracker(self.horizon)

    def success(self, env, observation, steps, terminated):
        return bool(env.unwrapped.state_metrics()[1][-1])

    def success_constraints(self, tracker):
        return {"task_completion": tracker.steps[-1]["gap"] if tracker.steps else 1.0}

    def progress_predicates(self, tracker):
        return {"first_stage_once": any(s["milestones"][0] for s in tracker.steps),
                "second_stage_once": any(s["milestones"][1] for s in tracker.steps),
                "final_success": bool(tracker.steps and tracker.steps[-1]["milestones"][2])}

    @property
    def objective_config(self):
        return ObjectiveConfig(float(self.horizon), 5.0, 100.0)

    @property
    def success_gap_spec(self):
        return {"version": 1, "task": self.task_name, "aggregation": "mean per episode then seeds",
                "definition": {
                    "drawer": "clip((0.12m-opening)/0.12m,0,1); success opening>=0.12m",
                    "wipe": "fraction of 9 uncleared markers; contact with table >0.1N and marker in physical tool footprint required",
                    "peg": "mean clipped deficits: radial error<=0.004m (scale0.05m), tip_z<=0.04m (scale0.08m), tip_z>=0.005m (scale0.03m)",
                }[self.task_name]}

    @property
    def progress_spec(self):
        return {"version": 1, "milestones": {
            "drawer": ["robot/drawer contact once", "opening>=0.06m once", "opening>=0.12m final"],
            "wipe": ["tool contact once", "at least half markers cleared once", "all markers cleared final"],
            "peg": ["radial error<=0.01m once", "radial<=0.004m and tip_z<=0.07m once", "insertion success final"],
        }[self.task_name], "aggregation": "equal mean; Q never used in selection"}

    @property
    def helper_description(self):
        return {
            "drawer": "FSM: approach handle+0.06m z; within0.015m descend; within0.012m close for0.4s then pull -x by min(0.04m,max(0,0.14m-opening)). No artificial grasp attachment.",
            "wipe": "Latch nearest uncleared marker until cleared; approach marker+0.04m z within0.015m then wipe at marker_z-0.002m. Hold when all clean. Tool mounted by physical fixed joint; privileged marker state available to every law.",
            "peg": "Peg pre-mounted by physical fixed joint, fixed downward IK orientation. Approach hole xy at tip_z0.11m within0.003m then insert toward max(0.035m,tip_z-0.005m); return to approach if radial error>0.005m. No grasping phase.",
        }[self.task_name] + " Helper memory reset each episode; helper complexity excluded from AST size."

    @property
    def signal_contract(self):
        rows = {
            "waypoint": ("position_feedback", "(target-controlled_point)/0.05m in xyz", "point at selected waypoint", self.helper_description),
            "damping": ("velocity_feedback", "-finite_difference(point)/(1m/s) in xyz", "first call or stationary point", "previous point, reset per episode"),
            "integral": ("integral_feedback", "clip(integral(error dt),-0.02,0.02)/(0.05m*s) in xyz", "integral zero", "reset per episode and phase change"),
            "gripper": ("gripper_command", "+1 before close; -1 in close/pull; xyz zero", "xyz always zero", self.helper_description),
        }
        mapping = "xyz normalized world displacement, Panda IK scale0.05m, fixed down orientation; Drawer fourth channel +open/-close"
        specs = {name: SignalSpec(kind, "1", (0, 0), (self.action_dim,), formula,
                                 "positive gain follows target; damping opposes velocity", zero,
                                 "unclipped except integral; final action clipped [-1,1]", memory,
                                 "world", mapping) for name, (kind, formula, zero, memory) in rows.items()
                 if name in self.allowed_terms}
        return SignalContract(specs, (self.action_dim,), mapping)


PANDA_LIGHT_ADAPTERS = {
    "panda_drawer": PandaLightAdapter("drawer", "LawevoPandaDrawer-v0"),
    "panda_planar_wipe": PandaLightAdapter("wipe", "LawevoPandaPlanarWipe-v0"),
    "panda_peg_insertion_easy": PandaLightAdapter("peg", "LawevoPandaPegInsertionEasy-v0"),
}
