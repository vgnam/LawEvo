"""Small serial-chain CPU Jacobian, retaining the simulator's existing IK solver."""

import numpy as np


class SingleRobotJacobian:
    """Cache constant joint transforms; fall back for batches, tools, or autograd."""

    def __init__(self, chain):
        self.original = chain.jacobian
        self.frames = []
        for frame in chain._serial_frames:
            joint = frame.joint
            offset = (np.eye(4) if joint.offset is None else
                      joint.offset.get_matrix()[0].detach().cpu().numpy().copy())
            axis = joint.axis.detach().cpu().numpy().astype(float)
            x, y, z = axis
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            self.frames.append((joint.joint_type, offset, axis, skew, skew @ skew))

    def __call__(self, th, locations=None, **kwargs):
        import torch

        if (not isinstance(th, torch.Tensor) or th.device.type != "cpu"
                or th.requires_grad or th.ndim != 2 or th.shape[0] != 1
                or locations is not None or kwargs):
            return self.original(th, locations=locations, **kwargs)
        q = th.detach().numpy()[0]
        transform = np.eye(4)
        origins, axes, kinds = [], [], []
        index = 0
        for kind, offset, axis, skew, square in self.frames:
            transform = transform @ offset
            if kind == "fixed":
                continue
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ axis)
            kinds.append(kind)
            motion = np.eye(4)
            if kind == "revolute":
                angle = float(q[index])
                motion[:3, :3] += np.sin(angle) * skew + (1 - np.cos(angle)) * square
            elif kind == "prismatic":
                motion[:3, 3] = axis * q[index]
            else:
                return self.original(th)
            transform = transform @ motion
            index += 1
        axes = np.asarray(axes)
        jacobian = np.zeros((6, len(q)))
        jacobian[:3] = np.cross(axes, transform[:3, 3] - np.asarray(origins)).T
        jacobian[3:] = axes.T
        for index, kind in enumerate(kinds):
            if kind == "prismatic":
                jacobian[:3, index] = axes[index]
                jacobian[3:, index] = 0
        return torch.as_tensor(jacobian[None], dtype=th.dtype)


def accelerate_cpu_jacobians(env):
    """Apply only to this environment's PyTorch CPU kinematic chains."""
    controllers = env.unwrapped.agent.controller.controllers.values()
    for controller in controllers:
        kinematics = getattr(controller, "kinematics", None)
        chain = getattr(kinematics, "pk_chain", None)
        if chain is not None and chain.device.type == "cpu":
            chain.jacobian = SingleRobotJacobian(chain)
