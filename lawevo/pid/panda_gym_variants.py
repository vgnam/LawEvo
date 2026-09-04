"""Harder Panda-Gym arm variants with exposed physical parameters.

Five difficulty-graded variants of the standard Panda-Gym tasks. Each variant
subclasses a ``panda_gym`` task/environment with one exposed physical parameter
so classical baselines stay honest and evolved laws have structure to exploit:

- Reach-MovingGoal: the Cartesian goal orbits a reset-sampled center on a
  Lissajous path (``goal_speed``); success tracks the moving goal.
- Push-IceObstacle: low-friction table plus a static obstacle between the
  object start zone and the goal zone (``table_friction``).
- Slide-Gate: two walls form a gate the puck must pass through before the
  goal zone (``gate_width``).
- Pick-HeavyDistractor: a heavier cube plus a movable clutter box resampled
  near the goal every episode (``cube_mass``).
- Stack-NarrowSettle: tight success tolerance plus a settle requirement that
  the stacked cube be nearly at rest (``distance_threshold``, ``settle_speed``).

The module imports panda_gym lazily so the package still works on a NumPy-only
install; environments register with Gymnasium on first ``make_env``.
"""

from __future__ import annotations

import numpy as np

from lawevo.pid.expression import SymbolicExpression
from lawevo.pid.gym_benchmark import GymStructure  # noqa: F401 -- re-exported alias
from lawevo.pid.panda_gym_benchmark import (
    PandaGymAdapter,
    PandaObjectMotionAdapter,
    PandaPickAndPlaceAdapter,
    PandaStackAdapter,
    _action,
    _normalized,
)

REACH_MOVING_ENV_ID = "LawevoPandaReachMoving-v0"
PUSH_ICE_ENV_ID = "LawevoPandaPushIce-v0"
SLIDE_GATE_ENV_ID = "LawevoPandaSlideGate-v0"
PICK_DISTRACTOR_ENV_ID = "LawevoPandaPickDistractor-v0"
STACK_NARROW_ENV_ID = "LawevoPandaStackNarrow-v0"

_ORBIT_AMPLITUDE = 0.05
_REGISTERED = False


def _morphable_panda_factory():
    """Build a Panda robot class that loads a custom (morphed) URDF.

    The stock ``Panda`` hard-codes ``franka_panda/panda.urdf``; this factory
    subclasses it so ``make_env(urdf_path=...)`` can pass a rendered parametric
    URDF. Joint indices, forces, and the EE link stay identical because the
    template only substitutes link origins/masses inside the same structure.
    """

    import panda_gym  # noqa: F401 -- ensures the stock assets are importable

    from panda_gym.envs.robots.panda import Panda

    class MorphablePanda(Panda):
        def __init__(self, sim, urdf_path, block_gripper=False, base_position=None, control_type="ee"):
            self._lawevo_urdf_path = str(urdf_path)
            super().__init__(
                sim,
                block_gripper=block_gripper,
                base_position=base_position,
                control_type=control_type,
            )

        def _load_robot(self, file_name, base_position):
            import os

            # Mesh references inside the URDF are relative to its folder.
            previous_dir = os.getcwd()
            os.chdir(os.path.dirname(os.path.abspath(self._lawevo_urdf_path)))
            try:
                self.sim.loadURDF(
                    body_name=self.body_name,
                    fileName=self._lawevo_urdf_path,
                    basePosition=base_position,
                    useFixedBase=True,
                )
            finally:
                os.chdir(previous_dir)

    return MorphablePanda


def _build_panda(sim, *, urdf_path, block_gripper, base_position, control_type, motor_force=1.0):
    """Create a Panda robot, optionally from a morphed URDF with scaled motors."""
    from panda_gym.envs.robots.panda import Panda

    if urdf_path is None:
        robot = Panda(
            sim,
            block_gripper=block_gripper,
            base_position=base_position,
            control_type=control_type,
        )
    else:
        robot = _morphable_panda_factory()(
            sim,
            urdf_path=urdf_path,
            block_gripper=block_gripper,
            base_position=base_position,
            control_type=control_type,
        )
    if motor_force != 1.0:
        # Scale the position-control motor forces: the "gear" of PyBullet.
        robot.joint_forces = robot.joint_forces * float(motor_force)
    return robot


def _register_variant_environments() -> None:
    """Define and register the five variant environments (idempotent, lazy)."""
    global _REGISTERED
    if _REGISTERED:
        return
    import panda_gym  # noqa: F401 -- registers the stock Panda environments

    from gymnasium.envs.registration import register, registry
    from panda_gym.envs.core import RobotTaskEnv
    from panda_gym.envs.panda_tasks import (
        PandaPickAndPlaceEnv,
        PandaPushEnv,
        PandaReachEnv,
        PandaSlideEnv,
        PandaStackEnv,
    )
    from panda_gym.envs.tasks.pick_and_place import PickAndPlace
    from panda_gym.envs.tasks.push import Push
    from panda_gym.envs.tasks.reach import Reach
    from panda_gym.envs.tasks.slide import Slide
    from panda_gym.envs.tasks.stack import Stack

    class ReachMovingTask(Reach):
        """Reach whose goal orbits a reset-sampled center on a Lissajous path.

        ``self.goal`` tracks the moving target, so ``is_success`` and the dense
        reward measure the instantaneous distance to the orbiting goal.
        """

        def __init__(self, sim, get_ee_position, goal_speed: float = 0.05, **kwargs) -> None:
            self.goal_speed = float(goal_speed)
            self.t = 0.0
            self.omega = self.goal_speed / _ORBIT_AMPLITUDE
            self.phase_x = 0.0
            self.phase_y = np.pi / 2
            self.center = np.zeros(3)
            super().__init__(sim, get_ee_position=get_ee_position, **kwargs)

        def reset(self) -> None:
            self.t = 0.0
            self.omega = self.goal_speed / _ORBIT_AMPLITUDE
            self.phase_x = float(self.np_random.uniform(0.0, 2 * np.pi))
            self.phase_y = float(self.np_random.uniform(0.0, 2 * np.pi))
            super().reset()
            # Keep the orbit center comfortably inside the reachable box.
            self.center = np.array(self.goal, dtype=float)
            self.center[2] = float(np.clip(self.center[2], 0.06, 0.22))
            self._place_target()

        def advance(self, dt: float) -> None:
            self.t += float(dt)
            self._place_target()

        def _place_target(self) -> None:
            x = self.center[0] + _ORBIT_AMPLITUDE * np.sin(self.omega * self.t + self.phase_x)
            y = self.center[1] + _ORBIT_AMPLITUDE * np.sin(self.omega * self.t + self.phase_y)
            self.goal = np.array([x, y, self.center[2]])
            self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    class PandaReachMovingEnv(PandaReachEnv):
        """PandaReach with an orbiting goal; the target advances every step.

        The stock env classes hard-code their task, so variant envs rewire
        sim/robot/task themselves (mirroring the stock ``__init__`` bodies).
        """

        def __init__(
            self,
            render_mode: str = "rgb_array",
            reward_type: str = "dense",
            control_type: str = "ee",
            goal_speed: float = 0.05,
            urdf_path: str | None = None,
            motor_force: float = 1.0,
            **kwargs,
        ) -> None:
            from panda_gym.pybullet import PyBullet

            del kwargs
            sim = PyBullet(render_mode=render_mode, renderer="Tiny")
            robot = _build_panda(
                sim,
                urdf_path=urdf_path,
                block_gripper=True,
                base_position=np.array([-0.6, 0.0, 0.0]),
                control_type=control_type,
                motor_force=motor_force,
            )
            task = ReachMovingTask(
                sim,
                reward_type=reward_type,
                get_ee_position=robot.get_ee_position,
                goal_speed=goal_speed,
            )
            RobotTaskEnv.__init__(self, robot, task)

        def step(self, action):
            self.task.advance(self.sim.dt)
            return RobotTaskEnv.step(self, action)

    class PushIceTask(Push):
        """Push across a low-friction table past a static offset obstacle."""

        def __init__(self, sim, table_friction: float = 0.1, **kwargs) -> None:
            self.table_friction = float(table_friction)
            self.obstacle_position = np.array([0.01, 0.0, 0.02])
            super().__init__(sim, **kwargs)

        def _create_scene(self) -> None:
            super()._create_scene()
            with self.sim.no_rendering():
                self.sim.create_cylinder(
                    body_name="obstacle",
                    mass=0.0,
                    radius=0.03,
                    height=0.04,
                    position=self.obstacle_position,
                    rgba_color=np.array([0.9, 0.5, 0.1, 1.0]),
                )
            self.set_friction(self.table_friction)

        def set_friction(self, value: float) -> None:
            self.table_friction = float(value)
            for body in ("table", "object", "obstacle"):
                self.sim.set_lateral_friction(
                    body=body, link=-1, lateral_friction=self.table_friction
                )

        def reset(self) -> None:
            super().reset()
            # Obstacle between the left-hand start zone and right-hand goal
            # zone; the lateral offset is resampled every episode.
            y_obstacle = float(self.np_random.uniform(-0.08, 0.08))
            self.obstacle_position = np.array([0.01, y_obstacle, 0.02])
            self.sim.set_base_pose(
                "obstacle", self.obstacle_position, np.array([0.0, 0.0, 0.0, 1.0])
            )

        def _sample_object(self) -> np.ndarray:
            position = np.array([0.0, 0.0, self.object_size / 2])
            position[0] = float(self.np_random.uniform(-0.15, -0.03))
            position[1] = float(self.np_random.uniform(-0.12, 0.12))
            return position

        def _sample_goal(self) -> np.ndarray:
            goal = np.array([0.0, 0.0, self.object_size / 2])
            goal[0] = float(self.np_random.uniform(0.05, 0.15))
            goal[1] = float(self.np_random.uniform(-0.12, 0.12))
            return goal

    class PandaPushIceEnv(PandaPushEnv):
        """PandaPush on a low-friction table with an obstacle to route around."""

        def __init__(
            self,
            render_mode: str = "rgb_array",
            reward_type: str = "dense",
            control_type: str = "ee",
            table_friction: float = 0.1,
            urdf_path: str | None = None,
            motor_force: float = 1.0,
            **kwargs,
        ) -> None:
            from panda_gym.pybullet import PyBullet

            del kwargs
            sim = PyBullet(render_mode=render_mode, renderer="Tiny")
            robot = _build_panda(
                sim,
                urdf_path=urdf_path,
                block_gripper=True,
                base_position=np.array([-0.6, 0.0, 0.0]),
                control_type=control_type,
                motor_force=motor_force,
            )
            task = PushIceTask(
                sim, reward_type=reward_type, table_friction=table_friction
            )
            RobotTaskEnv.__init__(self, robot, task)

    class GateSlideTask(Slide):
        """Slide the puck through a narrow gate placed before the goal zone."""

        gate_x = 0.20

        def __init__(self, sim, gate_width: float = 0.09, **kwargs) -> None:
            self.gate_width = float(gate_width)
            super().__init__(sim, **kwargs)

        def _create_scene(self) -> None:
            super()._create_scene()
            with self.sim.no_rendering():
                self.sim.create_box(
                    body_name="gate_wall_left",
                    half_extents=np.array([0.0125, 0.05, 0.03]),
                    mass=0.0,
                    position=self._wall_position(+1.0),
                    rgba_color=np.array([0.9, 0.2, 0.2, 1.0]),
                )
                self.sim.create_box(
                    body_name="gate_wall_right",
                    half_extents=np.array([0.0125, 0.05, 0.03]),
                    mass=0.0,
                    position=self._wall_position(-1.0),
                    rgba_color=np.array([0.9, 0.2, 0.2, 1.0]),
                )

        def _wall_position(self, sign: float) -> np.ndarray:
            # Inner faces sit at y = +/- gate_width / 2.
            return np.array([self.gate_x, sign * (self.gate_width / 2 + 0.05), 0.03])

        def set_gate_width(self, width: float) -> None:
            self.gate_width = float(width)
            for name, sign in (("gate_wall_left", 1.0), ("gate_wall_right", -1.0)):
                self.sim.set_base_pose(
                    name, self._wall_position(sign), np.array([0.0, 0.0, 0.0, 1.0])
                )

        def _sample_goal(self) -> np.ndarray:
            goal = np.array([0.0, 0.0, self.object_size / 2])
            goal[0] = float(self.np_random.uniform(0.30, 0.55))
            margin = max(self.gate_width / 2 - 0.015, 0.005)
            goal[1] = float(self.np_random.uniform(-margin, margin))
            return goal

    class PandaSlideGateEnv(PandaSlideEnv):
        """PandaSlide where the puck must pass through a gated wall pair."""

        def __init__(
            self,
            render_mode: str = "rgb_array",
            reward_type: str = "dense",
            control_type: str = "ee",
            gate_width: float = 0.09,
            urdf_path: str | None = None,
            motor_force: float = 1.0,
            **kwargs,
        ) -> None:
            from panda_gym.pybullet import PyBullet

            del kwargs
            sim = PyBullet(render_mode=render_mode, renderer="Tiny")
            robot = _build_panda(
                sim,
                urdf_path=urdf_path,
                block_gripper=True,
                base_position=np.array([-0.6, 0.0, 0.0]),
                control_type=control_type,
                motor_force=motor_force,
            )
            task = GateSlideTask(sim, reward_type=reward_type, gate_width=gate_width)
            RobotTaskEnv.__init__(self, robot, task)

    class DistractorPickAndPlaceTask(PickAndPlace):
        """Pick and place with a heavier cube and a movable clutter box."""

        def __init__(self, sim, cube_mass: float = 1.5, **kwargs) -> None:
            self.cube_mass = float(cube_mass)
            super().__init__(sim, **kwargs)

        def _create_scene(self) -> None:
            super()._create_scene()
            with self.sim.no_rendering():
                self.sim.create_box(
                    body_name="distractor",
                    half_extents=np.ones(3) * 0.02,
                    mass=1.0,
                    position=np.array([0.1, 0.1, 0.02]),
                    rgba_color=np.array([0.8, 0.7, 0.1, 1.0]),
                )
            self.set_cube_mass(self.cube_mass)

        def set_cube_mass(self, mass: float) -> None:
            self.cube_mass = float(mass)
            self.sim.physics_client.changeDynamics(
                bodyUniqueId=self.sim._bodies_idx["object"],
                linkIndex=-1,
                mass=self.cube_mass,
            )

        def reset(self) -> None:
            super().reset()
            object_position = np.asarray(self.sim.get_base_position("object"), dtype=float)
            position = np.array([0.12, -0.12, 0.02])
            for _ in range(20):
                candidate = np.array(
                    [
                        float(self.np_random.uniform(0.02, 0.15)),
                        float(self.np_random.uniform(-0.15, 0.15)),
                        0.02,
                    ]
                )
                if (
                    np.linalg.norm(candidate[:2] - object_position[:2]) > 0.09
                    and np.linalg.norm(candidate[:2] - np.asarray(self.goal)[:2]) > 0.07
                ):
                    position = candidate
                    break
            self.sim.set_base_pose("distractor", position, np.array([0.0, 0.0, 0.0, 1.0]))

    class PandaPickDistractorEnv(PandaPickAndPlaceEnv):
        """PandaPickAndPlace with a heavier cube and resampled clutter."""

        def __init__(
            self,
            render_mode: str = "rgb_array",
            reward_type: str = "dense",
            control_type: str = "ee",
            cube_mass: float = 1.5,
            urdf_path: str | None = None,
            motor_force: float = 1.0,
            **kwargs,
        ) -> None:
            from panda_gym.pybullet import PyBullet

            del kwargs
            sim = PyBullet(render_mode=render_mode, renderer="Tiny")
            robot = _build_panda(
                sim,
                urdf_path=urdf_path,
                block_gripper=False,
                base_position=np.array([-0.6, 0.0, 0.0]),
                control_type=control_type,
                motor_force=motor_force,
            )
            task = DistractorPickAndPlaceTask(
                sim, reward_type=reward_type, cube_mass=cube_mass
            )
            RobotTaskEnv.__init__(self, robot, task)

    class NarrowStackTask(Stack):
        """Stack with a tight tolerance and an at-rest settle requirement."""

        def __init__(
            self,
            sim,
            distance_threshold: float = 0.025,
            settle_speed: float = 0.08,
            **kwargs,
        ) -> None:
            self.settle_speed = float(settle_speed)
            super().__init__(sim, distance_threshold=float(distance_threshold), **kwargs)

        def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
            from panda_gym.utils import distance

            d = distance(achieved_goal, desired_goal)
            positioned = d < self.distance_threshold
            speed = float(
                np.linalg.norm(np.asarray(self.sim.get_base_velocity("object1"), dtype=float))
            )
            return np.array(positioned and speed < self.settle_speed, dtype=bool)

    class PandaStackNarrowEnv(PandaStackEnv):
        """PandaStack with a tight tolerance and a settle-speed success gate."""

        def __init__(
            self,
            render_mode: str = "rgb_array",
            reward_type: str = "dense",
            control_type: str = "ee",
            distance_threshold: float = 0.025,
            settle_speed: float = 0.08,
            urdf_path: str | None = None,
            motor_force: float = 1.0,
            **kwargs,
        ) -> None:
            from panda_gym.pybullet import PyBullet

            del kwargs
            sim = PyBullet(render_mode=render_mode, renderer="Tiny")
            robot = _build_panda(
                sim,
                urdf_path=urdf_path,
                block_gripper=False,
                base_position=np.array([-0.6, 0.0, 0.0]),
                control_type=control_type,
                motor_force=motor_force,
            )
            task = NarrowStackTask(
                sim,
                reward_type=reward_type,
                distance_threshold=distance_threshold,
                settle_speed=settle_speed,
            )
            RobotTaskEnv.__init__(self, robot, task)

    for env_id, env_class, max_steps in (
        (REACH_MOVING_ENV_ID, PandaReachMovingEnv, 50),
        (PUSH_ICE_ENV_ID, PandaPushIceEnv, 50),
        (SLIDE_GATE_ENV_ID, PandaSlideGateEnv, 50),
        (PICK_DISTRACTOR_ENV_ID, PandaPickDistractorEnv, 50),
        (STACK_NARROW_ENV_ID, PandaStackNarrowEnv, 100),
    ):
        if env_id not in registry:
            register(
                id=env_id,
                entry_point=env_class,
                kwargs={"reward_type": "dense", "control_type": "ee"},
                max_episode_steps=max_steps,
            )
    _REGISTERED = True


class _VariantAdapter(PandaGymAdapter):
    """Shared lazy-registration hook for the five variant adapters."""

    def make_env(self):
        _register_variant_environments()
        return super().make_env()


class PandaReachMovingAdapter(_VariantAdapter):
    """Track an orbiting goal; feedforward terms can beat reactive PD."""

    env_id = REACH_MOVING_ENV_ID
    horizon = 50
    allowed_terms = (
        "goal_error",
        "normalized_goal_error",
        "integral_goal_error",
        "goal_velocity",
        "phase_sin",
        "phase_cos",
        "eef_damping",
        "tanh_goal_error",
    )
    classical = (
        SymbolicExpression("Task P", ("goal_error",)),
        SymbolicExpression("Task PD", ("goal_error", "eef_damping")),
        SymbolicExpression(
            "Feedforward P", ("goal_error", "goal_velocity")
        ),
        SymbolicExpression(
            "Tracking PD",
            ("goal_error", "goal_velocity", "eef_damping", "phase_sin", "phase_cos"),
        ),
    )

    def features(self, env, observation, memory, dt):
        del env
        eef, velocity, _, goal = self._state(observation)
        error = goal[:3] - eef
        memory["integral_xyz"] = np.clip(memory["integral_xyz"] + error * dt, -0.25, 0.25)
        now = float(memory.get("t", 0.0)) + dt
        memory["t"] = now
        previous_goal = memory.get("previous_goal")
        goal_velocity = np.zeros(3) if previous_goal is None else (goal[:3] - previous_goal) / dt
        memory["previous_goal"] = goal[:3].copy()
        return {
            "goal_error": error,
            "normalized_goal_error": _normalized(error),
            "integral_goal_error": memory["integral_xyz"].copy(),
            "goal_velocity": goal_velocity,
            "phase_sin": np.array([np.sin(now), 0.0, 0.0]),
            "phase_cos": np.array([np.cos(now), 0.0, 0.0]),
            "eef_damping": -velocity,
            "tanh_goal_error": np.tanh(10.0 * error),
        }

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory["t"] = 0.0
        memory["previous_goal"] = None
        return memory


class PandaPushIceAdapter(_VariantAdapter, PandaObjectMotionAdapter):
    """Push on a near-frictionless table around a static obstacle."""

    env_id = PUSH_ICE_ENV_ID
    horizon = 50
    allowed_terms = PandaObjectMotionAdapter.allowed_terms + ("obstacle_repel",)
    classical = (
        SymbolicExpression("Reach PD", ("reach_object", "eef_damping")),
        SymbolicExpression(
            "Contact + Goal PD", ("contact_then_goal", "eef_damping")
        ),
        SymbolicExpression(
            "Obstacle-aware PD",
            ("contact_then_goal", "obstacle_repel", "eef_damping"),
        ),
    )

    def features(self, env, observation, memory, dt):
        base = PandaObjectMotionAdapter.features(self, env, observation, memory, dt)
        obj = np.asarray(env.unwrapped.sim.get_base_position("object"), dtype=float)
        obstacle = np.asarray(env.unwrapped.task.obstacle_position, dtype=float)
        away = obj - obstacle
        away[2] = 0.0
        base["obstacle_repel"] = _normalized(away)
        return base


class PandaSlideGateAdapter(_VariantAdapter, PandaObjectMotionAdapter):
    """Strike the puck through a gated wall pair before the goal."""

    env_id = SLIDE_GATE_ENV_ID
    horizon = 50
    allowed_terms = PandaObjectMotionAdapter.allowed_terms + ("through_gate",)
    classical = (
        SymbolicExpression("Reach PD", ("reach_object", "eef_damping")),
        SymbolicExpression("Object Goal P", ("reach_object", "object_goal_error")),
        SymbolicExpression("Through-gate P", ("through_gate", "reach_object")),
        SymbolicExpression("Through-gate PD", ("through_gate", "eef_damping")),
    )

    def features(self, env, observation, memory, dt):
        base = PandaObjectMotionAdapter.features(self, env, observation, memory, dt)
        obj = np.asarray(env.unwrapped.sim.get_base_position("object"), dtype=float)
        gate_x = float(env.unwrapped.task.gate_x)
        if obj[0] < gate_x:
            target = np.array([gate_x, 0.0, obj[2]])
        else:
            target = np.asarray(env.unwrapped.task.goal, dtype=float)[:3]
        base["through_gate"] = target - obj
        return base


class PandaPickDistractorAdapter(_VariantAdapter, PandaPickAndPlaceAdapter):
    """Pick a heavier cube while a clutter box sits near the goal."""

    env_id = PICK_DISTRACTOR_ENV_ID
    horizon = 50
    allowed_terms = PandaPickAndPlaceAdapter.allowed_terms + ("distractor_error",)
    classical = PandaPickAndPlaceAdapter.classical + (
        SymbolicExpression(
            "Selective Pick PD",
            (
                "pick_place_sequence",
                "eef_damping",
                "grasp_close",
                "release_on_target",
                "distractor_error",
            ),
        ),
    )

    def features(self, env, observation, memory, dt):
        base = PandaPickAndPlaceAdapter.features(self, env, observation, memory, dt)
        eef, _, _, _ = self._state(observation)
        distractor = np.asarray(
            env.unwrapped.sim.get_base_position("distractor"), dtype=float
        )
        base["distractor_error"] = _action(4, distractor - eef)
        return base


class PandaStackNarrowAdapter(_VariantAdapter, PandaStackAdapter):
    """Stack into a tight tolerance with an at-rest settle requirement."""

    env_id = STACK_NARROW_ENV_ID
    horizon = 100
    allowed_terms = PandaStackAdapter.allowed_terms + ("settle_velocity",)
    classical = PandaStackAdapter.classical + (
        SymbolicExpression(
            "Settle PD",
            (
                "stack_sequence",
                "eef_damping",
                "grasp_close",
                "release_on_stack",
                "settle_velocity",
            ),
        ),
    )

    def features(self, env, observation, memory, dt):
        base = PandaStackAdapter.features(self, env, observation, memory, dt)
        velocity = np.asarray(
            env.unwrapped.sim.get_base_velocity("object1"), dtype=float
        )
        base["settle_velocity"] = np.concatenate([-velocity, [0.0]])
        return base


PANDA_VARIANT_ADAPTERS = {
    "panda_reach_moving": PandaReachMovingAdapter(),
    "panda_push_ice": PandaPushIceAdapter(),
    "panda_slide_gate": PandaSlideGateAdapter(),
    "panda_pick_distractor": PandaPickDistractorAdapter(),
    "panda_stack_narrow": PandaStackNarrowAdapter(),
}
