# Copyright (c) 2026-2027 zh
"""内容：定义Go1真实／想象共用的物理状态、几何观测、奖励和终止。"""

from dataclasses import asdict
import json
import math

import torch
import torch.nn.functional as functional
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul

from agents.perceptive_supermimic.interfaces import AlgorithmSpec, TaskState
from src.tasks.locomotion.go2_supermimic.mdp.reference_rotations import quat_to_rotvec


POSITION = slice(0, 3)
QUATERNION = slice(3, 7)
LINEAR_WORLD = slice(7, 10)
ANGULAR_BODY = slice(10, 13)
JOINT_POSITION = slice(13, 25)
JOINT_VELOCITY = slice(25, 37)


class Go1DepthTask:
    """37维物理状态配36维切空间残差；绝对关节角与wxyz四元数不混入动作偏移。"""

    def __init__(self, cfg, nominal, limits, surface, *, device="cpu", latent_dim=32):
        self.cfg, self.surface, self.device = cfg, surface, torch.device(device)
        self.dt = cfg.timestep * cfg.decimation
        self.nominal = torch.as_tensor(nominal, device=device, dtype=torch.float32).clone()
        self.limits = torch.as_tensor(limits, device=device, dtype=torch.float32).clone()
        x, y = torch.arange(17, device=device) * 0.1 - 0.4, torch.arange(11, device=device) * 0.1 - 0.5
        gx, gy = torch.meshgrid(x, y, indexing="ij")
        self.scan_xy = torch.stack((gx.flatten(), gy.flatten()), -1)
        self.noise_scale = self.nominal.new_tensor([0.2] * 3 + [0.05] * 3 + [0.0] * 3 + [0.01] * 12 + [1.5] * 12 + [0.0] * 12)
        semantics = asdict(cfg)
        semantics.pop("num_envs")
        semantics.update(state="pos_world_quat_wxyz_vel_world_omega_body_q_qd",
                         increment="dp_world_drot_body_dvel_world_domega_body_dq_dqd",
                         action="FL_FR_RL_RR_hip_thigh_calf;target=nominal+0.25*a;PD=40/1;limit=33.5Nm",
                         height="clamp((terrain_z-base_z+0.35)/1m,-1,1);yaw_grid17x11;x[-.4,1.2];y[-.5,.5]",
                         model_heading="sin_yaw_cos_yaw_for_world_frame_targets",
                         terrain_query="vertical_collision_rays;bilinear_grid;stairs_edges_smoothed",
                         boundary="constant_edge_extension;root_margin_failure", latent_dim=latent_dim)
        depth_semantics = {"source": "optical_axis_meters_64x64", "resize": [cfg.depth_size, cfg.depth_size],
                           "interpolation": "bilinear_align_corners_false", "near": cfg.depth_near, "far": cfg.depth_far,
                           "invalid": "far", "noise_meters": cfg.depth_noise_std, "dropout": cfg.depth_dropout,
                           "noise_enabled": cfg.depth_noise, "output": "(clamped_meters-near)/(far-near);missing=0"}
        self.spec = AlgorithmSpec(state_dim=37, delta_dim=36, action_dim=12, observation_dim=45,
                                  privileged_dim=235 + latent_dim, model_input_dim=235,
                                  memory_dim=12, command_dim=3, command_indices=(6, 7, 8), context_dim=3, contact_dim=4,
                                  semantic_id="go1_depth_v1/" + json.dumps(semantics, sort_keys=True),
                                  depth_semantic_id=json.dumps(depth_semantics, sort_keys=True),
                                  depth_shape=(1, cfg.depth_size, cfg.depth_size))

    @staticmethod
    def world_position(state):
        return state.physical[:, POSITION]

    @staticmethod
    def joint_position(state):
        return state.physical[:, JOINT_POSITION]

    @staticmethod
    def body_linear_velocity(state):
        return quat_apply_inverse(state.physical[:, QUATERNION], state.physical[:, LINEAR_WORLD])

    @staticmethod
    def gravity(state):
        direction = torch.zeros_like(state.physical[:, POSITION])
        direction[:, 2] = -1
        return quat_apply_inverse(state.physical[:, QUATERNION], direction)

    @staticmethod
    def rotation_angle_error(left, right):
        relative = quat_mul(quat_conjugate(left.physical[:, QUATERNION]), right.physical[:, QUATERNION])
        return quat_to_rotvec(relative).norm(dim=-1)

    def base_height(self, state):
        position = self.world_position(state)
        return position[:, 2] - self.surface.height(position[:, :2])

    @staticmethod
    def heading(state):
        q = state.physical[:, QUATERNION]
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2].square() + q[:, 3].square()))

    def terrain_height(self, state):
        """高度由图像对应状态查询，偏航对齐；基准为机身下方0.35m平面。"""
        yaw = self.heading(state)
        cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
        x = self.scan_xy[:, 0] * cosine - self.scan_xy[:, 1] * sine
        y = self.scan_xy[:, 0] * sine + self.scan_xy[:, 1] * cosine
        position = self.world_position(state)
        xy = torch.stack((x, y), -1) + position[:, None, :2]
        height = (self.surface.height(xy) - position[:, 2:3] + 0.35).clamp(-1, 1)
        return height.reshape(-1, 1, 17, 11).float()

    def observe(self, state, *, noisy=False):
        observation = torch.cat((state.physical[:, ANGULAR_BODY], self.gravity(state), state.command,
                                 self.joint_position(state) - self.nominal, state.physical[:, JOINT_VELOCITY], state.memory), -1)
        if noisy and self.cfg.observation_noise:
            observation = observation + (torch.rand_like(observation) * 2 - 1) * self.noise_scale
        return observation

    def model_input(self, state, action):
        heading = self.heading(state)
        return torch.cat((self.base_height(state)[:, None], self.gravity(state), self.body_linear_velocity(state),
                          state.physical[:, ANGULAR_BODY], self.joint_position(state) - self.nominal,
                          state.physical[:, JOINT_VELOCITY], action, heading.sin()[:, None], heading.cos()[:, None],
                          self.terrain_height(state).flatten(1)), -1)

    def privileged(self, state, latent):
        return torch.cat((self.observe(state, noisy=False), self.body_linear_velocity(state),
                          self.terrain_height(state).flatten(1), latent), -1)

    def model_target(self, state, target):
        rotation = quat_mul(quat_conjugate(state.physical[:, QUATERNION]), target.physical[:, QUATERNION])
        return torch.cat((target.physical[:, POSITION] - state.physical[:, POSITION], quat_to_rotvec(rotation),
                          target.physical[:, 7:] - state.physical[:, 7:]), -1)

    def model_error(self, recorded, predicted, target):
        """完整状态的残差保持与训练增量相同单位，姿态采用最短局部旋转误差。"""
        return self.model_target(target, predicted)

    def integrate(self, state, action, delta):
        """旋转残差通过指数映射右乘姿态，其余状态按对应物理量残差更新。"""
        rotvec = delta[:, 3:6]
        angle = (rotvec.square().sum(-1, keepdim=True) + 1e-12).sqrt()
        increment = torch.cat((torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * math.pi)) * rotvec), -1)
        quaternion = functional.normalize(quat_mul(state.physical[:, QUATERNION], increment), dim=-1)
        physical = torch.cat((state.physical[:, POSITION] + delta[:, :3], quaternion, state.physical[:, 7:] + delta[:, 6:]), -1)
        return TaskState(physical, action, state.command, state.context)

    def hard_termination(self, state):
        return ((self.base_height(state) < self.cfg.minimum_height)
                | (-self.gravity(state)[:, 2] < math.cos(self.cfg.maximum_tilt))
                | (self.surface.boundary_margin(self.world_position(state)[:, :2]) <= 0))

    def soft_termination(self, prediction):
        state = prediction.state
        low = torch.sigmoid((self.cfg.minimum_height - self.base_height(state)) / 0.02)
        tilt = torch.sigmoid((math.cos(self.cfg.maximum_tilt) + self.gravity(state)[:, 2]) / 0.05)
        edge = torch.sigmoid(-self.surface.boundary_margin(self.world_position(state)[:, :2]) / 0.05)
        return 1 - (1 - low) * (1 - tilt) * (1 - edge)

    def reward_terms(self, state, action, next_state, failure):
        """任务优先；PD力矩为终态解析估计，不冒充整个控制间隔的实测力矩。"""
        velocity = self.body_linear_velocity(next_state)
        omega = next_state.physical[:, ANGULAR_BODY]
        joints = self.joint_position(next_state)
        torque = (40 * (self.nominal + 0.25 * action - joints) - next_state.physical[:, JOINT_VELOCITY]).clamp(-33.5, 33.5)
        return {
            "tracking_linear": torch.exp(-(velocity[:, :2] - state.command[:, :2]).square().sum(-1) / 0.25),
            "tracking_yaw": torch.exp(-(omega[:, 2] - state.command[:, 2]).square() / 0.25),
            "orientation": self.gravity(next_state)[:, :2].square().sum(-1),
            "vertical_velocity": velocity[:, 2].square(),
            "angular_velocity": omega[:, :2].square().sum(-1),
            "action_rate": (action - state.memory).square().sum(-1),
            "torque": torque.square().sum(-1),
            "joint_limit": ((self.limits[:, 0] - joints).clamp_min(0) + (joints - self.limits[:, 1]).clamp_min(0)).sum(-1),
            "failure": failure,
        }

    def reward(self, state, action, prediction, failure):
        terms = self.reward_terms(state, action, prediction.state, failure)
        return self.dt * sum(getattr(self.cfg, f"{name}_weight") * value for name, value in terms.items())
