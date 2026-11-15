# Copyright (c) 2026-2027 zh
"""
内容：
    从 Go1 MJCF 读取关节树，计算可微髋、膝、足位置和足速。
    树递推依据 BlinkMimic d264658 的 MIT 实现，许可见 Go2 mdp/LICENSE_blinkmimic；
    Go1 固定相机节点、关节和足端几何由本资产重新读取。
"""

import mujoco
import numpy as np
import torch
from torch import nn

from src.assets.robots.go1_depth.go1_depth_constants import GO1_DEPTH_XML, JOINT_NAMES, LEG_NAMES, ROOT_BODY_NAME
from src.tasks.locomotion.go2_supermimic.mdp.reference_rotations import axis_angle_to_mat, mm3, mv3, quat_to_mat


class Go1Kinematics(nn.Module):
    """支持无关节固定子体的 Go1 正运动学，四足顺序为 FL、FR、RL、RR。"""

    def __init__(self, device="cpu"):
        super().__init__()
        model = mujoco.MjModel.from_xml_path(str(GO1_DEPTH_XML))
        joint_indices = {name: index for index, name in enumerate(JOINT_NAMES)}
        names, parents, joints, axes, offsets, rotations = [], [], [], [], [], []
        body_indices = {}
        for body in range(1, model.nbody):
            name = model.body(body).name
            count = int(model.body_jntnum[body])
            joint = int(model.body_jntadr[body])
            if count == 1 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                if name != ROOT_BODY_NAME or model.body_parentid[body] != 0 or names:
                    raise ValueError("Go1 FK requires one free root before all other bodies")
                parent, index, axis = -1, -1, np.zeros(3)
            else:
                parent = body_indices[int(model.body_parentid[body])]
                if count == 0:
                    index, axis = -1, np.zeros(3)
                elif count == 1 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                    joint_name = model.joint(joint).name
                    if joint_name not in joint_indices or np.any(model.jnt_pos[joint]) or model.qpos0[model.jnt_qposadr[joint]] != 0:
                        raise ValueError(f"Unsupported Go1 FK joint: {joint_name}")
                    index, axis = joint_indices[joint_name], model.jnt_axis[joint].copy()
                else:
                    raise ValueError(f"Unsupported Go1 FK body joint layout: {name}")
            body_indices[body] = len(names)
            names.append(name)
            parents.append(parent)
            joints.append(index)
            axes.append(axis)
            offsets.append(model.body_pos[body].copy())
            rotations.append(model.body_quat[body].copy())
        if sorted(index for index in joints if index >= 0) != list(range(12)):
            raise ValueError("Go1 FK joint set differs from the task specification")
        self.parents, self.joints, self.num_bodies = tuple(parents), tuple(joints), len(names)
        self.limb_indices = tuple(names.index(f"{leg}_{link}") for link in ("thigh", "calf") for leg in LEG_NAMES)
        sites = [model.site(name).id for name in LEG_NAMES]
        self.site_parents = tuple(body_indices[int(model.site_bodyid[site])] for site in sites)
        self.register_buffer("axes", torch.tensor(np.array(axes), dtype=torch.float32, device=device))
        self.register_buffer("offsets", torch.tensor(np.array(offsets), dtype=torch.float32, device=device))
        self.register_buffer("rotations", quat_to_mat(torch.tensor(np.array(rotations), dtype=torch.float32, device=device)))
        self.register_buffer("site_offsets", torch.tensor(model.site_pos[sites].copy(), dtype=torch.float32, device=device))

    def forward(self, p, R, v, w, q, qd):
        """由世界系根部状态和实际关节角返回十二个位置及四足世界速度。"""
        positions, rotations, velocities, angular = ([None] * self.num_bodies for _ in range(4))
        for index, parent in enumerate(self.parents):
            if parent < 0:
                positions[index], rotations[index], velocities[index], angular[index] = p, R, v, w
                continue
            displacement = mv3(rotations[parent], self.offsets[index])
            positions[index] = positions[parent] + displacement
            rotation = mm3(rotations[parent], self.rotations[index])
            velocities[index] = velocities[parent] + torch.cross(angular[parent], displacement, dim=-1)
            joint = self.joints[index]
            if joint >= 0:
                rotations[index] = mm3(rotation, axis_angle_to_mat(self.axes[index], q[..., joint]))
                angular[index] = angular[parent] + mv3(rotation, self.axes[index]) * qd[..., joint:joint + 1]
            else:
                rotations[index], angular[index] = rotation, angular[parent]
        feet, foot_velocities = [], []
        for index, parent in enumerate(self.site_parents):
            displacement = mv3(rotations[parent], self.site_offsets[index])
            feet.append(positions[parent] + displacement)
            foot_velocities.append(velocities[parent] + torch.cross(angular[parent], displacement, dim=-1))
        return torch.stack([positions[index] for index in self.limb_indices] + feet, -2), torch.stack(foot_velocities, -2)
