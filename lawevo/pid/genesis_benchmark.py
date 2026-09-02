from __future__ import annotations

import os
from typing import Any

import numpy as np

from lawevo.pid.gym_benchmark import (
    BenchmarkAdapter,
    GymEpisode,
    GymMetrics,
    GymStructure,
)


def _normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-6)


def _action(vector: np.ndarray | None = None, gripper: float = 0.0) -> np.ndarray:
    action = np.zeros(4, dtype=float)
    if vector is not None:
        action[:3] = np.asarray(vector, dtype=float)
    action[3] = gripper
    return action


_SCENE: _GenesisFrankaScene | None = None


class _GenesisFrankaScene:
    """One reusable GPU scene shared by all Genesis tasks and CEM batches."""

    physics_dt = 0.02
    action_repeat = 4
    dt = physics_dt * action_repeat
    cube_size = 0.04

    def __init__(self, batch_size: int):
        try:
            import genesis as gs
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Genesis support requires: pip install -e '.[benchmarks]'"
            ) from exc

        self.gs = gs
        self.torch = torch
        self.batch_size = batch_size
        if not gs._initialized:
            requested = os.environ.get("LAWEVO_GENESIS_DEVICE", "gpu").lower()
            use_gpu = requested != "cpu" and torch.cuda.is_available()
            gs.init(
                backend=gs.gpu if use_gpu else gs.cpu,
                precision="32",
                logging_level="error",
                seed=1,
            )
        self.device = gs.device
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.physics_dt),
            rigid_options=gs.options.RigidOptions(
                dt=self.physics_dt / 2,
                enable_collision=True,
                enable_joint_limit=True,
                constraint_solver=gs.constraint_solver.Newton,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(self.cube_size,) * 3,
                pos=(0.45, 0.0, self.cube_size / 2),
                fixed=False,
            ),
            surface=gs.surfaces.Rough(),
        )
        self.scene.build(n_envs=batch_size)
        self.hand = self.robot.get_link("hand")
        self.left_finger = self.robot.get_link("left_finger")
        self.right_finger = self.robot.get_link("right_finger")
        self.arm_dofs = torch.arange(7, device=self.device)
        self.initial_qpos = torch.tensor(
            [0.0, -0.2, 0.0, -2.0, 0.0, 2.0, 0.8, 0.04, 0.04],
            dtype=torch.float32,
            device=self.device,
        ).repeat(batch_size, 1)
        self.robot.set_dofs_kp(
            torch.tensor(
                [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
                device=self.device,
            )
        )
        self.robot.set_dofs_kv(
            torch.tensor(
                [450, 450, 350, 350, 200, 200, 200, 10, 10],
                device=self.device,
            )
        )
        self.robot.set_dofs_force_range(
            torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100], device=self.device),
            torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100], device=self.device),
        )
        self.damping = (0.01**2) * torch.eye(6, device=self.device)
        self.goal = torch.zeros((batch_size, 3), device=self.device)

    @property
    def tcp(self):
        return (self.left_finger.get_pos() + self.right_finger.get_pos()) / 2

    def reset(self, task: str, seeds: list[int]) -> None:
        positions = []
        goals = []
        for seed in seeds:
            rng = np.random.default_rng(seed + 17001)
            obj = np.array(
                [rng.uniform(0.40, 0.50), rng.uniform(-0.08, 0.08), self.cube_size / 2]
            )
            if task == "push":
                goal = obj + np.array([rng.uniform(0.14, 0.20), rng.uniform(-0.04, 0.04), 0.0])
            else:
                goal = np.array(
                    [rng.uniform(0.38, 0.55), rng.uniform(-0.12, 0.12), rng.uniform(0.16, 0.28)]
                )
            positions.append(obj)
            goals.append(goal)
        pos = self.torch.tensor(np.asarray(positions), dtype=self.torch.float32, device=self.device)
        self.goal.copy_(
            self.torch.tensor(np.asarray(goals), dtype=self.torch.float32, device=self.device)
        )
        self.robot.set_qpos(self.initial_qpos, zero_velocity=True, skip_forward=True)
        self.cube.set_pos(pos, zero_velocity=True, skip_forward=True)
        self.cube.set_quat(
            self.torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                self.batch_size, 1
            ),
            zero_velocity=True,
        )
        self.robot.control_dofs_position(self.initial_qpos)

    def apply_action(self, action) -> None:
        torch = self.torch
        delta_pose = torch.zeros((self.batch_size, 6), device=self.device)
        delta_pose[:, :3] = action[:, :3] * 0.035
        jacobian = self.robot.get_jacobian(link=self.hand)
        system = torch.baddbmm(self.damping, jacobian, jacobian.mT)
        solved = torch.linalg.solve(system, delta_pose)
        delta_q = (jacobian.mT @ solved.unsqueeze(-1)).squeeze(-1)
        qpos = self.robot.get_qpos() + delta_q
        finger = 0.02 * (torch.clamp(action[:, 3], -1.0, 1.0) + 1.0)
        qpos[:, 7] = finger
        qpos[:, 8] = finger
        self.robot.control_dofs_position(qpos)
        for _ in range(self.action_repeat):
            self.scene.step()


def _scene() -> _GenesisFrankaScene:
    global _SCENE
    batch_size = int(os.environ.get("LAWEVO_GENESIS_BATCH_SIZE", "32"))
    if batch_size < 1:
        raise ValueError("LAWEVO_GENESIS_BATCH_SIZE must be positive")
    if _SCENE is None:
        _SCENE = _GenesisFrankaScene(batch_size)
    elif _SCENE.batch_size != batch_size:
        raise RuntimeError("Genesis batch size cannot change after the GPU scene is built")
    return _SCENE


class GenesisAdapter(BenchmarkAdapter):
    """GPU-batched Genesis World adapter used directly by LawEvo's CEM loop."""

    horizon = 75
    fallback_dt = _GenesisFrankaScene.dt
    energy_weight, jerk_weight = 0.01, 0.00001
    task: str

    def make_env(self):
        raise RuntimeError("Genesis adapters use GPU-batched evaluation, not single Gym envs")

    def reset_controller(self, action_dim):
        memory = super().reset_controller(action_dim)
        memory["previous_tcp"] = None
        memory["integral_xyz"] = np.zeros(3)
        return memory

    def features(self, env, observation: dict[str, Any], memory, dt):
        del env
        tcp = np.asarray(observation["tcp"], dtype=float)
        obj = np.asarray(observation["object"], dtype=float)
        goal = np.asarray(observation["goal"], dtype=float)
        previous = memory["previous_tcp"]
        velocity = np.zeros(3) if previous is None else (tcp - previous) / dt
        memory["previous_tcp"] = tcp.copy()
        return self._numpy_features(tcp, obj, goal, velocity, memory)

    def _numpy_features(self, tcp, obj, goal, velocity, memory):
        raise NotImplementedError

    def _torch_features(self, scene, tcp, obj, velocity, integral):
        raise NotImplementedError

    def _reward_success(self, scene, tcp, obj, action):
        raise NotImplementedError

    def evaluate_episodes(self, structure, gains, seeds):
        return self._evaluate(structure, np.asarray(gains, dtype=float)[None, :], seeds)[0]

    def evaluate_gain_batch(self, structure, gains, seeds):
        episode_groups = self._evaluate(structure, np.asarray(gains, dtype=float), seeds)
        metrics = []
        for episodes in episode_groups:
            episode_return = float(np.mean([item.episode_return for item in episodes]))
            success = float(np.mean([item.success for item in episodes]))
            energy = float(np.mean([item.energy for item in episodes]))
            jerk = float(np.mean([item.jerk for item in episodes]))
            metrics.append(
                GymMetrics(
                    self.score(episode_return, energy, jerk, len(structure.terms)),
                    episode_return,
                    success,
                    energy,
                    jerk,
                    len(structure.terms),
                )
            )
        return metrics

    def _evaluate(self, structure, gains, seeds):
        scene = _scene()
        torch = scene.torch
        cases = [(gain_index, seed) for gain_index in range(len(gains)) for seed in seeds]
        grouped: list[list[GymEpisode]] = [[] for _ in gains]
        for offset in range(0, len(cases), scene.batch_size):
            valid = cases[offset : offset + scene.batch_size]
            padded = valid + [valid[-1]] * (scene.batch_size - len(valid))
            scene.reset(self.task, [seed for _, seed in padded])
            batch_gains = torch.tensor(
                np.asarray([gains[index] for index, _ in padded]),
                dtype=torch.float32,
                device=scene.device,
            )
            integral = torch.zeros((scene.batch_size, 3), device=scene.device)
            previous_tcp = scene.tcp.clone()
            previous_action = torch.zeros((scene.batch_size, 4), device=scene.device)
            returns = torch.zeros(scene.batch_size, device=scene.device)
            energy = torch.zeros_like(returns)
            jerk = torch.zeros_like(returns)
            success = torch.zeros(scene.batch_size, dtype=torch.bool, device=scene.device)
            for _ in range(self.horizon):
                tcp = scene.tcp
                obj = scene.cube.get_pos()
                velocity = (tcp - previous_tcp) / scene.dt
                features, integral = self._torch_features(
                    scene, tcp, obj, velocity, integral
                )
                action = torch.zeros((scene.batch_size, 4), device=scene.device)
                for term_index, term in enumerate(structure.terms):
                    action += batch_gains[:, term_index, None] * features[term]
                action = torch.clamp(action, -1.0, 1.0)
                scene.apply_action(action)
                tcp_after = scene.tcp
                obj_after = scene.cube.get_pos()
                reward, step_success = self._reward_success(
                    scene, tcp_after, obj_after, action
                )
                returns += reward
                energy += scene.dt * torch.sum(action * action, dim=1)
                rate = (action - previous_action) / scene.dt
                jerk += scene.dt * torch.sum(rate * rate, dim=1)
                success = step_success
                previous_tcp = tcp_after.clone()
                previous_action = action
            values = zip(
                returns.detach().cpu().numpy(),
                success.detach().cpu().numpy(),
                energy.detach().cpu().numpy(),
                jerk.detach().cpu().numpy(),
                strict=True,
            )
            for (gain_index, _), (ret, ok, effort, roughness) in zip(valid, values, strict=False):
                grouped[gain_index].append(
                    GymEpisode(float(ret), bool(ok), float(effort), float(roughness))
                )
        return grouped


class GenesisPushCubeAdapter(GenesisAdapter):
    env_id = "GenesisPushCube-v0"
    task = "push"
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

    def _numpy_features(self, tcp, obj, goal, velocity, memory):
        del memory
        push_pose = obj + np.array([-0.035, 0.0, 0.015])
        reach = push_pose - tcp
        goal_error = goal - obj
        sequential = goal_error if np.linalg.norm(reach) < 0.05 else reach
        return {
            "reach_push_pose": _action(reach),
            "normalized_reach_push_pose": _action(_normalized(reach)),
            "object_goal_error": _action(goal_error),
            "normalized_object_goal_error": _action(_normalized(goal_error)),
            "eef_damping": _action(-velocity),
            "contact_then_goal": _action(sequential),
        }

    def _torch_features(self, scene, tcp, obj, velocity, integral):
        torch = scene.torch
        push_pose = obj + torch.tensor([-0.035, 0.0, 0.015], device=scene.device)
        reach = push_pose - tcp
        goal_error = scene.goal - obj
        near = torch.linalg.norm(reach, dim=1) < 0.05
        sequential = torch.where(near[:, None], goal_error, reach)
        zero = torch.zeros((scene.batch_size, 1), device=scene.device)

        def act(x):
            return torch.cat([x, zero], dim=1)

        return {
            "reach_push_pose": act(reach),
            "normalized_reach_push_pose": act(torch.nn.functional.normalize(reach, dim=1)),
            "object_goal_error": act(goal_error),
            "normalized_object_goal_error": act(
                torch.nn.functional.normalize(goal_error, dim=1)
            ),
            "eef_damping": act(-velocity),
            "contact_then_goal": act(sequential),
        }, integral

    def _reward_success(self, scene, tcp, obj, action):
        del action
        torch = scene.torch
        push_pose = obj + torch.tensor([-0.035, 0.0, 0.015], device=scene.device)
        reach_distance = torch.linalg.norm(push_pose - tcp, dim=1)
        goal_distance = torch.linalg.norm(scene.goal[:, :2] - obj[:, :2], dim=1)
        reached = reach_distance < 0.06
        success = (goal_distance < 0.06) & (obj[:, 2] < 0.06)
        reward = 1 - torch.tanh(5 * reach_distance)
        reward += reached * (1 - torch.tanh(5 * goal_distance))
        reward += success * 2.0
        return reward, success


class GenesisPickCubeAdapter(GenesisAdapter):
    env_id = "GenesisPickCube-v0"
    task = "pick"
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

    def _numpy_features(self, tcp, obj, goal, velocity, memory):
        reach = obj - tcp
        goal_error = goal - obj
        memory["integral_xyz"] = np.clip(
            memory["integral_xyz"] + reach * self.fallback_dt, -0.25, 0.25
        )
        near = np.linalg.norm(reach) < 0.07
        transport = goal_error if obj[2] > 0.10 else np.array([0.0, 0.0, 0.16 - obj[2]])
        sequence = _action(reach, 1.0) if not near else _action(transport, -1.0)
        return {
            "reach_cube": _action(reach),
            "normalized_reach_cube": _action(_normalized(reach)),
            "integral_reach": _action(memory["integral_xyz"]),
            "eef_damping": _action(-velocity),
            "grasp_close": _action(gripper=-1.0),
            "lift_then_transport": _action(transport),
            "release_on_target": _action(gripper=1.0 if np.linalg.norm(goal_error) < 0.05 else 0.0),
            "pick_place_sequence": sequence,
        }

    def _torch_features(self, scene, tcp, obj, velocity, integral):
        torch = scene.torch
        reach = obj - tcp
        goal_error = scene.goal - obj
        integral = torch.clamp(integral + reach * scene.dt, -0.25, 0.25)
        near = torch.linalg.norm(reach, dim=1) < 0.07
        fingers_closed = torch.mean(scene.robot.get_qpos()[:, 7:9], dim=1) < 0.03
        ready_to_lift = near & fingers_closed
        lifted = obj[:, 2] > 0.10
        lift = torch.zeros_like(obj)
        lift[:, 2] = torch.clamp(0.16 - obj[:, 2], min=0.0)
        transport = torch.where(lifted[:, None], goal_error, lift)
        zero = torch.zeros((scene.batch_size, 1), device=scene.device)

        def act(x, grip=None):
            return torch.cat([x, zero if grip is None else grip[:, None]], dim=1)

        open_grip = torch.ones(scene.batch_size, device=scene.device)
        close_grip = -open_grip
        sequence_xyz = torch.where(ready_to_lift[:, None], transport, reach)
        sequence_grip = torch.where(near, close_grip, open_grip)
        release = torch.where(torch.linalg.norm(goal_error, dim=1) < 0.05, open_grip, zero[:, 0])
        return {
            "reach_cube": act(reach),
            "normalized_reach_cube": act(torch.nn.functional.normalize(reach, dim=1)),
            "integral_reach": act(integral),
            "eef_damping": act(-velocity),
            "grasp_close": act(torch.zeros_like(obj), close_grip),
            "lift_then_transport": act(transport),
            "release_on_target": act(torch.zeros_like(obj), release),
            "pick_place_sequence": act(sequence_xyz, sequence_grip),
        }, integral

    def _reward_success(self, scene, tcp, obj, action):
        torch = scene.torch
        reach_distance = torch.linalg.norm(obj - tcp, dim=1)
        goal_distance = torch.linalg.norm(scene.goal - obj, dim=1)
        closed_near = (action[:, 3] < -0.25) & (reach_distance < 0.07)
        lifted = obj[:, 2] > 0.08
        success = goal_distance < 0.055
        reward = 1 - torch.tanh(5 * reach_distance)
        reward += closed_near.to(reward.dtype)
        reward += lifted * (1 - torch.tanh(5 * goal_distance))
        reward += success * 2.0
        return reward, success


GENESIS_ADAPTERS = {
    "genesis_push_cube": GenesisPushCubeAdapter(),
    "genesis_pick_cube": GenesisPickCubeAdapter(),
}
