# Copyright (c) 2026-2027 zh
"""内容：依据真实 Go2 MJCF，以同一广义状态计算髋、膝、足位置和足速。

父子树递推移植自 BlinkMimic d264658 的 src/blinkmimic/kinematics.py，
并按本项目足端 site 增加刚体偏置的位置及速度运输。许可见 LICENSE_blinkmimic。
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from torch import nn

from src.assets.robots.go2.go2_reference import GO2_XML

from .reference_rotations import axis_angle_to_mat, mm3, mv3, quat_to_mat
from .state_spec import FOOT_NAMES, JOINT_NAMES


class Go2Kinematics(nn.Module):
    """当前参考 Go2 的可微正运动学；几何常量从资产读取并注册为 buffer。"""

    def __init__(self, device="cpu"):
        super().__init__()
        model = mujoco.MjModel.from_xml_path(str(GO2_XML))
        joint_indices = {name: index for index, name in enumerate(JOINT_NAMES)}
        bodies, parents, joints, axes, offsets, rotations = [], [], [], [], [], []
        mj_to_local = {}
        for body in range(1, model.nbody):
            name = model.body(body).name
            if model.body_jntnum[body] != 1:
                raise ValueError(f"Go2 FK requires exactly one joint per body: {name}")
            joint = int(model.body_jntadr[body])
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                if name != "base" or model.body_parentid[body] != 0 or bodies:
                    raise ValueError("Go2 FK requires a single free base before the leg joints")
                parents.append(-1)
                joints.append(-1)
            else:
                joint_name = model.joint(joint).name
                if (model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE or joint_name not in joint_indices
                        or np.any(model.jnt_pos[joint] != 0) or model.qpos0[model.jnt_qposadr[joint]] != 0):
                    raise ValueError(f"Go2 FK requires the named zero-anchor, zero-reference hinge: {joint_name}")
                parent = int(model.body_parentid[body])
                if parent not in mj_to_local:
                    raise ValueError(f"Go2 FK requires a preceding moving parent: {name}")
                parents.append(mj_to_local[parent])
                joints.append(joint_indices[joint_name])
            mj_to_local[body] = len(bodies)
            bodies.append(name)
            axes.append(model.jnt_axis[joint].copy())
            offsets.append(model.body_pos[body].copy())
            rotations.append(model.body_quat[body].copy())
        if sorted(joint for joint in joints if joint >= 0) != list(range(len(JOINT_NAMES))):
            raise ValueError("Go2 FK joint set differs from the task joint order")
        self.parents, self.joints = tuple(parents), tuple(joints)
        self.num_bodies = len(bodies)
        self.limb_indices = tuple(bodies.index(f"{leg}_{link}") for link in ("thigh", "calf") for leg in FOOT_NAMES)
        sites = [model.site(name).id for name in FOOT_NAMES]
        self.site_parents = tuple(mj_to_local[int(model.site_bodyid[site])] for site in sites)
        self.register_buffer("axes", torch.tensor(np.array(axes), dtype=torch.float32, device=device))
        self.register_buffer("offsets", torch.tensor(np.array(offsets), dtype=torch.float32, device=device))
        self.register_buffer("rotations", quat_to_mat(torch.tensor(np.array(rotations), dtype=torch.float32, device=device)))
        self.register_buffer("site_offsets", torch.tensor(model.site_pos[sites].copy(), dtype=torch.float32, device=device))

    def forward(self, p, R, v, w, q, qd):
        """返回世界系髋4／膝4／足4位置和足4线速度，保留任意前置批次维度。

        p/R/v/w 是根部世界位置、旋转、线速度与角速度；q/qd 为 JOINT_NAMES
        顺序的真实关节角和速度。几何 buffer 与输入应在同一设备和精度。
        """
        positions, rotations, velocities, angular = ([None] * self.num_bodies for _ in range(4))
        for index in range(self.num_bodies):
            parent = self.parents[index]
            if parent < 0:
                positions[index], rotations[index], velocities[index], angular[index] = p, R, v, w
                continue
            joint = self.joints[index]
            displacement = mv3(rotations[parent], self.offsets[index])
            position = positions[parent] + displacement
            rotation_offset = mm3(rotations[parent], self.rotations[index])
            rotation = mm3(rotation_offset, axis_angle_to_mat(self.axes[index], q[..., joint]))
            omega = angular[parent] + mv3(rotation_offset, self.axes[index]) * qd[..., joint:joint + 1]
            velocity = velocities[parent] + torch.cross(angular[parent], displacement, dim=-1)
            positions[index], rotations[index], velocities[index], angular[index] = position, rotation, velocity, omega
        feet, foot_velocities = [], []
        for index, parent in enumerate(self.site_parents):
            displacement = mv3(rotations[parent], self.site_offsets[index])
            feet.append(positions[parent] + displacement)
            foot_velocities.append(velocities[parent] + torch.cross(angular[parent], displacement, dim=-1))
        limbs = [positions[index] for index in self.limb_indices] + feet
        return torch.stack(limbs, -2), torch.stack(foot_velocities, -2)
