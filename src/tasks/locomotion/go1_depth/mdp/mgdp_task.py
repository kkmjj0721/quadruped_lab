# Copyright (c) 2026-2027 zh
"""内容：连接 MGDP 原版观测、奖励、终止与 Go1 可微状态递推。"""

from __future__ import annotations

import json

import torch
from mjlab.utils.lab_api.math import quat_apply

from agents.perceptive_supermimic.interfaces import AlgorithmSpec, TaskState
from src.assets.robots.go1_depth.go1_mgdp_constants import JOINT_VELOCITY_LIMIT, RANDOMIZED_LINK_NAMES
from . import mgdp_state as slots
from .mgdp_rewards import MIX, PARKOUR, RewardContext, RewardGradientCfg, RewardInputs, RewardMemory, compute_rewards
from .task import Go1DepthTask


class MGDPTask(Go1DepthTask):
    """使用原坐标约定的 91 维状态；真实力/扭矩及奖励缓存速度均进入模型监督。

    静态参数及上一动作进入世界模型；每步随机延迟没有伪造为可知未来量。
    继承坐标读取与切空间增量编码，不调用旧任务的配置或奖励初始化。
    """

    def __init__(self, cfg, nominal, limits, surface, *, device="cpu", latent_dim=64):
        self.cfg, self.surface, self.device = cfg, surface, torch.device(device)
        self.dt = cfg.timestep * cfg.decimation
        self.nominal = torch.as_tensor(nominal, device=device, dtype=torch.float32).clone()
        self.limits = torch.as_tensor(limits, device=device, dtype=torch.float32).clone()
        self.reward_gradients = RewardGradientCfg()
        self.edge_gradient_width = 0.05
        self.termination_contact_temperature = 0.1
        self.termination_height_temperature = 0.02
        # 官方先 np.round(..., 2) 再转 float32；避免浮点累加跨过硬查询格界。
        x = self.nominal.new_tensor([round(-0.4 + 0.1 * index, 2) for index in range(17)])
        y = self.nominal.new_tensor([round(-0.5 + 0.1 * index, 2) for index in range(11)])
        gx, gy = torch.meshgrid(x, y, indexing="ij")
        self.scan_xy = torch.stack((gx.flatten(), gy.flatten()), -1)
        self.command_scale = self.nominal.new_tensor((2.0, 2.0, 0.25))
        self.noise_scale = self.nominal.new_tensor([0.05] * 3 + [0.05] * 3 + [0.0] * 3 + [0.01] * 12 + [0.075] * 12 + [0.0] * 12)
        self.hip_x = self.nominal.new_tensor((0.1881, 0.1881, -0.1881, -0.1881))
        self.leg_side = self.nominal.new_tensor((1.0, -1.0, 1.0, -1.0))
        semantics = {
            "mujoco_integrator": cfg.physics_integrator,
            "implicit_rne": "upstream_PR1663_positive_bias_derivative",
            "collision_affine": "translated_dominant_projection_sum1_v1",
            "randomized_links": RANDOMIZED_LINK_NAMES,
            "joint_velocity_limit": {"rad_s": JOINT_VELOCITY_LIMIT, "timing": "after_PD_before_integration;root_unchanged;end_speed_unclipped"},
            "reference": "MGDP_f43139d", "stage": cfg.stage, "dt": self.dt,
            "physical": "pos3_wxyz4_world_v3_body_w3_q12_qd12_torque12_foot_force12_body_force27_reward_linear_body3",
            "reward_linear_body": "physics_end_before_push_cached_body_velocity;state_target_supervised",
            "memory": "previous_action12_air_time4_last_contact4_camera_phase1",
            "context": "origin3_family1_class1_level1_global_step1_dynamics9_target_heading1",
            "observation": "omega*.25_gravity_command*(2,2,.25)_q-nominal_qd*.05_previous_action",
            "action": "FL_FR_RL_RR_hip_thigh_calf;P30_D.8_scale.25_effort23.7;shared_latency0to4",
            "height": "clamp(base_world_z-native_min3_height,-1,1);wz_projected_quaternion_grid17x11",
            "critic": "proprio45_body_v*2_feet_world_z4_height*5;no_latent",
            "edge_gradient_width_m": self.edge_gradient_width,
            "termination_gradient_width": [self.termination_contact_temperature, self.termination_height_temperature],
            "reward_gradient_width_N": [0.1, 0.01, 1.0],
            "positive_reward_backward": "identity_STE;hard_forward_unchanged",
            "nominal": self.nominal.tolist(), "soft_limits": self.limits.tolist(),
            "observation_noise": cfg.observation_noise, "height_noise": cfg.height_noise,
            "model_latent": latent_dim,
        }
        self.spec = AlgorithmSpec(
            state_dim=slots.PHYSICAL_DIM, delta_dim=slots.DELTA_DIM, action_dim=12,
            observation_dim=45, privileged_dim=239, model_input_dim=310,
            memory_dim=slots.MEMORY_DIM, command_dim=3, command_indices=(6, 7, 8),
            context_dim=slots.CONTEXT_DIM, contact_dim=4, history_length=4,
            semantic_id="go1_mgdp_native_v1/" + json.dumps(semantics, sort_keys=True),
            depth_semantic_id=json.dumps({
                "reference": "MGDP_f43139d1d97849819a0dce609dc0b59faf577b3f",
                "raw_shape": [120, 120], "source": "optical_axis_meters", "horizontal_fov_deg": 67,
                "translation_m": [0.34, 0, 0.07], "translation_randomization_m": [-0.02, 0.02],
                "pitch_deg": 30, "pitch_randomization_deg": [-2, 2],
                "preprocessing": "clip_meters_.1_2.5;bilinear16_align_corners_false;noise_then_dropout;normalize_0_5",
                "noise_meters_std": 0.03, "dropout": 0.1, "noise_enabled": cfg.depth_noise,
                "stack": "old_new", "shape": [2, 16, 16],
                "capture_interval": 5, "clock": "global_control_step_mod5", "reset": "forced_capture_duplicate_two_frames",
                "height_label": "capture_state_clamped_base_z_minus_native_height_plus_uniform_.1_then_clamp",
            }, sort_keys=True),
            depth_shape=(2, 16, 16), image_height_shape=(1, 17, 11),
            visual_interval=5, visual_phase_index=slots.CAMERA_PHASE,
        )

    def base_height(self, state):
        position = self.world_position(state)
        return position[:, 2] - self.surface.height(position[:, :2], state.context[:, slots.FAMILY])

    def terrain_height(self, state):
        """原版监督标签的无噪、未乘观测尺度版本，与采集状态一一对应。"""
        # 上游 quat_apply_yaw 清零 x/y 后归一化，不等同于完整姿态的 Euler yaw。
        wz = state.physical[:, [3, 6]]
        wz = wz / torch.linalg.vector_norm(wz, dim=-1, keepdim=True).clamp_min(1e-9)
        cosine = (2 * wz[:, 0].square() - 1)[:, None]
        sine = (2 * wz[:, 0] * wz[:, 1])[:, None]
        x = self.scan_xy[:, 0] * cosine - self.scan_xy[:, 1] * sine
        y = self.scan_xy[:, 0] * sine + self.scan_xy[:, 1] * cosine
        position = self.world_position(state)
        xy = torch.stack((x, y), -1) + position[:, None, :2]
        height = self.surface.height(xy, state.context[:, slots.FAMILY])
        return (position[:, 2:3] - height).clamp(-1, 1).reshape(-1, 1, 17, 11)

    def foot_positions(self, state):
        """实际 Go1 XML 四足中心 FK；足高是世界 z，不是离地间隙。"""
        joints = self.joint_position(state).reshape(-1, 4, 3)
        hip, thigh, calf = joints.unbind(-1)
        x = -0.213 * (thigh.sin() + (thigh + calf).sin())
        z = -0.213 * (thigh.cos() + (thigh + calf).cos())
        lateral = 0.08 * self.leg_side
        body = torch.stack((x + self.hip_x,
                            lateral * hip.cos() - z * hip.sin() + 0.04675 * self.leg_side,
                            lateral * hip.sin() + z * hip.cos()), -1)
        quaternion = state.physical[:, slots.QUATERNION, None].transpose(1, 2).expand(-1, 4, -1)
        return quat_apply(quaternion.reshape(-1, 4), body.reshape(-1, 3)).reshape(-1, 4, 3) + self.world_position(state)[:, None]

    def foot_height(self, state):
        return self.foot_positions(state)[..., 2]

    def observe(self, state, *, noisy=False):
        observation = torch.cat((0.25 * state.physical[:, slots.ANGULAR_BODY], self.gravity(state),
                                 state.command * self.command_scale, self.joint_position(state) - self.nominal,
                                 0.05 * state.physical[:, slots.JOINT_VELOCITY], state.memory[:, slots.PREVIOUS_ACTION]), -1)
        if noisy and self.cfg.observation_noise:
            observation = observation + (2 * torch.rand_like(observation) - 1) * self.noise_scale
        return observation.clamp(-100, 100)

    def critic_observation(self, state, *, noisy=True):
        """官方 239 维特权观测；高度乘 5 后加 U[-.5,.5]，速度不加噪。"""
        height = 5 * self.terrain_height(state).flatten(1)
        if noisy and self.cfg.height_noise:
            height = height + torch.rand_like(height) - 0.5
        return torch.cat((self.observe(state, noisy=noisy), 2 * self.body_linear_velocity(state), self.foot_height(state), height), -1).clamp(-100, 100)

    def privileged(self, state, latent):
        del latent
        return self.critic_observation(state)

    def model_input(self, state, action):
        heading = self.heading(state)
        return torch.cat((self.base_height(state)[:, None], self.gravity(state), self.body_linear_velocity(state),
                          state.physical[:, slots.ANGULAR_BODY], self.joint_position(state) - self.nominal,
                          state.physical[:, slots.JOINT_VELOCITY], action, heading.sin()[:, None], heading.cos()[:, None],
                          self.terrain_height(state).flatten(1), state.memory[:, slots.PREVIOUS_ACTION],
                          state.context[:, slots.DYNAMICS], state.physical[:, slots.TORQUES.start:]), -1)

    def foot_edges(self, state, *, imagined=False):
        xy = self.foot_positions(state)[..., :2]
        family = state.context[:, slots.FAMILY]
        hard = self.surface.foot_edges(xy, family).to(xy.dtype)
        if not imagined:
            return hard
        # 四角双线性值只提供反向，硬前向仍使用官方 round/clip 查询。
        grid = xy / self.edge_gradient_width
        lower = grid.floor()
        fraction = grid - lower
        soft = torch.zeros_like(hard)
        for x_offset, y_offset in ((0, 0), (0, 1), (1, 0), (1, 1)):
            corner = (lower + xy.new_tensor((x_offset, y_offset))) * self.edge_gradient_width
            value = self.surface.foot_edges(corner, family).to(xy.dtype)
            weight_x = fraction[..., 0] if x_offset else 1 - fraction[..., 0]
            weight_y = fraction[..., 1] if y_offset else 1 - fraction[..., 1]
            soft = soft + value * weight_x * weight_y
        return hard + (soft - soft.detach())

    def reward_values(self, state, action, next_state, *, imagined=False):
        inputs = RewardInputs(
            joint_pos=self.joint_position(next_state), joint_vel=next_state.physical[:, slots.JOINT_VELOCITY],
            previous_joint_vel=state.physical[:, slots.JOINT_VELOCITY], body_linear_velocity=next_state.physical[:, slots.REWARD_LINEAR_BODY],
            body_angular_velocity=next_state.physical[:, slots.ANGULAR_BODY], projected_gravity=self.gravity(next_state),
            command=next_state.command, action=action, previous_action=state.memory[:, slots.PREVIOUS_ACTION],
            torques=next_state.physical[:, slots.TORQUES], foot_forces=next_state.physical[:, slots.FOOT_FORCES].reshape(-1, 4, 3),
            body_forces=next_state.physical[:, slots.BODY_FORCES].reshape(-1, 9, 3), foot_edge=self.foot_edges(next_state, imagined=imagined),
            nominal_joint_pos=self.nominal, joint_limits=self.limits,
        )
        context = RewardContext(*(next_state.context[:, index] for index in (slots.FAMILY, slots.TERRAIN_CLASS, slots.LEVEL, slots.GLOBAL_STEP)))
        memory = RewardMemory(state.memory[:, slots.AIR_TIME], state.memory[:, slots.LAST_CONTACT])
        return compute_rewards(inputs, context, memory, imagined=imagined, dt=self.dt, gradients=self.reward_gradients)

    def integrate(self, state, action, delta):
        physical = super().integrate(state, action, delta).physical
        context = state.context.clone()
        context[:, slots.GLOBAL_STEP] = context[:, slots.GLOBAL_STEP] + 1
        provisional = TaskState(physical, state.memory, state.command, context)
        heading_error = state.context[:, slots.TARGET_HEADING] - self.heading(provisional)
        wrapped = torch.remainder(heading_error, 2 * torch.pi)
        wrapped = torch.where(wrapped > torch.pi, wrapped - 2 * torch.pi, wrapped)
        command = torch.cat((state.command[:, :2], (0.5 * wrapped).clamp(-1, 1)[:, None]), -1)
        provisional = TaskState(physical, state.memory, command, context)
        _, _, reward_memory = self.reward_values(state, action, provisional, imagined=True)
        phase = torch.remainder(state.memory[:, slots.CAMERA_PHASE:slots.CAMERA_PHASE + 1] + 1, self.spec.visual_interval)
        memory = torch.cat((action, reward_memory.air_time, reward_memory.last_contact, phase), -1)
        return TaskState(physical, memory, command, context)

    def _termination_margins(self, state):
        family, terrain_class = state.context[:, slots.FAMILY], state.context[:, slots.TERRAIN_CLASS]
        parkour = family == PARKOUR
        base_force = torch.linalg.vector_norm(state.physical[:, slots.BODY_FORCES].reshape(-1, 9, 3)[:, 0], dim=-1)
        base_contact = base_force - 1
        contact_enabled = ~(parkour & (terrain_class == 14))
        world_z = self.world_position(state)[:, 2]
        return ((base_contact, contact_enabled, self.termination_contact_temperature),
                (-0.1 - world_z, parkour & (terrain_class > 2), self.termination_height_temperature),
                (-0.1 - self.foot_height(state).min(-1).values, parkour & (terrain_class > 2), self.termination_height_temperature),
                (-0.2 - world_z, (family == MIX) & (terrain_class == 6), self.termination_height_temperature))

    def hard_termination(self, state):
        failure = torch.zeros(state.physical.shape[0], dtype=torch.bool, device=state.physical.device)
        for margin, enabled, _ in self._termination_margins(state):
            failure = failure | ((margin > 0) & enabled)
        return failure

    def soft_termination(self, prediction):
        """硬前向保持原版；代理宽度只改变想象反向，超时由环境负责。"""
        state = prediction.state
        survival = torch.ones(state.physical.shape[0], dtype=state.physical.dtype, device=state.physical.device)
        for margin, enabled, temperature in self._termination_margins(state):
            soft = torch.sigmoid(margin / temperature)
            event = (margin > 0).to(margin.dtype) + (soft - soft.detach())
            survival = survival * (1 - event * enabled)
        return 1 - survival

    def reward(self, state, action, prediction, failure):
        del failure
        return self.reward_values(state, action, prediction.state, imagined=True)[0]
