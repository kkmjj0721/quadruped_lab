# Copyright (c) 2026-2027 zh
"""内容：从 Go1 MJCF 读取含固定相机刚体的父子树，计算可微足端位置和速度。

父子树速度运输来源为 BlinkMimic d264658 的 src/blinkmimic/kinematics.py，
固定刚体与 Go1 site 几何由本任务适配。许可见 LICENSE_blinkmimic。
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from torch import nn

from src.assets.robots.go1_depth.go1_depth_constants import GO1_DEPTH_XML, ROOT_BODY_NAME

from .reference_rotations import axis_angle_to_mat, mm3, mv3, quat_to_mat
from .state_spec import FOOT_NAMES, JOINT_NAMES


class Go1Kinematics(nn.Module):
    """Go1 的浮动根、12 个命名铰链和任意固定子刚体共用一棵 FK。"""

    def __init__(self, device="cpu"):
        super().__init__()
        model = mujoco.MjModel.from_xml_path(str(GO1_DEPTH_XML))
        joint_indices = {name: index for index, name in enumerate(JOINT_NAMES)}
        bodies, parents, joints, axes, offsets, rotations = [], [], [], [], [], []
        mj_to_local = {}
        for body in range(1, model.nbody):
            name = model.body(body).name
            joint_count = int(model.body_jntnum[body])
            joint = int(model.body_jntadr[body])
            if joint_count == 1 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                if name != ROOT_BODY_NAME or model.body_parentid[body] != 0 or bodies:
                    raise ValueError("Go1 FK requires one leading free base_link")
                parents.append(-1)
                joints.append(-1)
                axes.append(np.zeros(3))
            else:
                parent = int(model.body_parentid[body])
                if parent not in mj_to_local:
                    raise ValueError(f"Go1 FK requires a preceding parent body: {name}")
                parents.append(mj_to_local[parent])
                if joint_count == 0:
                    joints.append(-1)
                    axes.append(np.zeros(3))
                elif joint_count == 1:
                    joint_name = model.joint(joint).name
                    if (model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE or joint_name not in joint_indices
                            or np.any(model.jnt_pos[joint] != 0) or model.qpos0[model.jnt_qposadr[joint]] != 0):
                        raise ValueError(f"Go1 FK requires named zero-anchor, zero-reference hinges: {joint_name}")
                    joints.append(joint_indices[joint_name])
                    axes.append(model.jnt_axis[joint].copy())
                else:
                    raise ValueError(f"Go1 FK does not support multiple joints in one body: {name}")
            mj_to_local[body] = len(bodies)
            bodies.append(name)
            offsets.append(model.body_pos[body].copy())
            rotations.append(model.body_quat[body].copy())
        if sorted(joint for joint in joints if joint >= 0) != list(range(len(JOINT_NAMES))):
            raise ValueError("Go1 FK joint set differs from the task joint order")
        self.body_names = tuple(bodies)
        self.parents, self.joints = tuple(parents), tuple(joints)
        self.num_bodies = len(bodies)
        self.limb_indices = tuple(bodies.index(f"{leg}_{link}") for link in ("thigh", "calf") for leg in FOOT_NAMES)
        sites = [model.site(name).id for name in FOOT_NAMES]
        self.site_parents = tuple(mj_to_local[int(model.site_bodyid[site])] for site in sites)
        for name, values in (("axes", axes), ("offsets", offsets), ("site_offsets", model.site_pos[sites].copy())):
            self.register_buffer(name, torch.tensor(np.asarray(values), dtype=torch.float32, device=device))
        self.register_buffer("rotations", quat_to_mat(torch.tensor(np.asarray(rotations), dtype=torch.float32, device=device)))

    def body_states(self, p, R, v, w, q, qd):
        """递推全部刚体，固定子刚体继承父角速度而不误读前一个关节。"""
        positions, rotations, velocities, angular = ([None] * self.num_bodies for _ in range(4))
        for index in range(self.num_bodies):
            parent = self.parents[index]
            if parent < 0:
                positions[index], rotations[index], velocities[index], angular[index] = p, R, v, w
                continue
            displacement = mv3(rotations[parent], self.offsets[index])
            position = positions[parent] + displacement
            rotation = mm3(rotations[parent], self.rotations[index])
            omega = angular[parent]
            joint = self.joints[index]
            if joint >= 0:
                omega = omega + mv3(rotation, self.axes[index]) * qd[..., joint:joint + 1]
                rotation = mm3(rotation, axis_angle_to_mat(self.axes[index], q[..., joint]))
            velocity = velocities[parent] + torch.cross(angular[parent], displacement, dim=-1)
            positions[index], rotations[index], velocities[index], angular[index] = position, rotation, velocity, omega
        return positions, rotations, velocities, angular

    def forward(self, p, R, v, w, q, qd):
        """返回 FL/FR/RL/RR 顺序的髋4、膝4、足4位置和足4世界线速度。"""
        positions, rotations, velocities, angular = self.body_states(p, R, v, w, q, qd)
        feet, foot_velocities = [], []
        for index, parent in enumerate(self.site_parents):
            displacement = mv3(rotations[parent], self.site_offsets[index])
            feet.append(positions[parent] + displacement)
            foot_velocities.append(velocities[parent] + torch.cross(angular[parent], displacement, dim=-1))
        limbs = [positions[index] for index in self.limb_indices] + feet
        return torch.stack(limbs, -2), torch.stack(foot_velocities, -2)
