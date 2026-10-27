# Copyright (c) 2026-2027 zh
"""内容：将指定Go2参考状态、解析速度奖励及真实随机化上下文接入混合专家历史策略训练。"""

from copy import deepcopy
import hashlib
import json
import math

import torch
from torch import Tensor

from agents.supermimic.interfaces import AlgorithmSpec, TaskState

from .reference_rewards import reference_reward_terms
from .reference_state import (
    ANGULAR_BODY, CONTEXT_COM, FEET, GRAVITY, HEIGHT, JOINT_OFFSET, JOINT_VELOCITY, POSE,
    ROBOT_STATE, VELOCITY_HEADING, apply_pose_delta, pose_delta, tracking_velocity, yaw_rotate_inverse,
)
from .state_spec import (
    COMMAND_TIMER, CONTEXT_ENCODER_BIAS, CONTEXT_RANGES, CONTEXT_ZERO_PROBABILITY,
    FOOT_NAMES, PREVIOUS_ACTION, ELAPSED_TIME,
)


class Go2ReferenceTaskModel:
    """独立实现参考Go2的状态、命令、观测、奖励与回放事件接口。"""

    model_input_dim = 187
    observation_term_dims = (3, 3, 12, 12, 12, 3)

    def __init__(self, cfg, default_joint_pos: Tensor, joint_limits: Tensor, surface, device="cpu"):
        self.cfg = deepcopy(cfg)
        self.device = torch.device(device)
        self.surface = surface
        self.dt = cfg.sim.timestep * cfg.control.decimation
        self.default_joint_pos = torch.as_tensor(default_joint_pos, device=self.device, dtype=torch.float32).clone()
        self.joint_limits = torch.as_tensor(joint_limits, device=self.device, dtype=torch.float32).clone()
        if self.default_joint_pos.shape != (12,) or self.joint_limits.shape != (12, 2):
            raise ValueError("Reference Go2 joint definitions must have shapes [12] and [12,2]")
        if not torch.isfinite(self.default_joint_pos).all() or not torch.isfinite(self.joint_limits).all():
            raise ValueError("Reference Go2 joint definitions must be finite")
        if (self.joint_limits[:, 0] >= self.joint_limits[:, 1]).any():
            raise ValueError("Reference Go2 joint limits must be strictly ordered")
        for name in ("tracking_linear_variance", "tracking_angular_variance", "fall_angle", "fall_softness_deg"):
            value = getattr(cfg.reward, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Reference reward parameter must be positive and finite: {name}")
        if cfg.reward.fall_angle >= math.pi:
            raise ValueError("Reference fall angle must be less than pi radians")
        # 指定参考提交target/terrain.py的前移11×7网格，顺序为x主序、y次序。
        xs = -.42 + .12 * torch.arange(11, device=self.device)
        ys = -.36 + .12 * torch.arange(7, device=self.device)
        self.scan_points = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        noise = cfg.noise.noise_scales
        self.noise_amplitude = self.default_joint_pos.new_tensor(
            (noise.base_ang_vel,) * 3 + (noise.projected_gravity,) * 3 + (noise.joint_pos,) * 12
            + (noise.joint_vel,) * 12 + (0.,) * 15) * cfg.noise.noise_level
        payload = {"task": self.cfg.to_dict(), "geometry": surface.geometry_hash,
                   "default_joint_pos": self.default_joint_pos.cpu().tolist(),
                   "joint_limits": self.joint_limits.cpu().tolist(),
                   "layout": "go2_reference_state70_pose3_fl_fr_rl_rr_frame_major_v1"}
        self.semantic_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.spec = AlgorithmSpec(73, 73, 12, 45, 179, 26, 14, 3,
                                  "go2_reference_velocity_v1:" + self.semantic_hash,
                                  model_id="supermimic_moe_v1")

    # ---------- 真实状态与坐标 ----------

    def surface_height(self, xy: Tensor) -> Tensor:
        """支持域外使用有限占位；该区域在硬／软失败中阻断后续回报。"""
        valid = self.surface.contains(xy)
        safe = torch.where(valid.unsqueeze(-1), xy, torch.zeros_like(xy))
        height = self.surface.height_unchecked(safe)
        return torch.where(valid, height, torch.zeros_like(height))

    def bind_env(self, env):
        """在仿真边界固定本项目FL、FR、RL、RR顺序的关节中心与足端site。"""
        robot = env.scene["robot"]
        self.hip_ids, _ = robot.find_bodies(tuple(f"{leg}_thigh" for leg in FOOT_NAMES), preserve_order=True)
        self.knee_ids, _ = robot.find_bodies(tuple(f"{leg}_calf" for leg in FOOT_NAMES), preserve_order=True)
        self.foot_ids, _ = robot.find_sites(FOOT_NAMES, preserve_order=True)

    def read_physical(self, env) -> Tensor:
        """与参考robot_state/planar_pose一致，地形高度来自本项目实际几何。"""
        data = env.scene["robot"].data
        position, yaw = data.root_link_pos_w, data.heading_w
        limbs = (data.body_link_pos_w[:, self.hip_ids], data.body_link_pos_w[:, self.knee_ids],
                 data.site_pos_w[:, self.foot_ids])
        relative = [yaw_rotate_inverse(points - position[:, None], yaw[:, None]).flatten(1) for points in limbs]
        height = position[:, 2:3] - self.surface_height(position[:, :2]).unsqueeze(-1)
        physical = torch.cat((height, data.projected_gravity_b,
                              yaw_rotate_inverse(data.root_link_lin_vel_w, yaw), data.root_link_ang_vel_b,
                              data.joint_pos - data.default_joint_pos, data.joint_vel, *relative,
                              position[:, :2], yaw.unsqueeze(-1)), -1)
        return physical

    def world_position(self, state: TaskState) -> Tensor:
        xy = state.physical[:, POSE][:, :2]
        return torch.cat((xy, self.base_height(state).unsqueeze(-1) + self.surface_height(xy).unsqueeze(-1)), -1)

    def base_height(self, state: TaskState) -> Tensor:
        return state.physical[:, 0]

    def yaw(self, state: TaskState) -> Tensor:
        return state.physical[:, 72]

    def tracking_velocity(self, state: TaskState) -> Tensor:
        return tracking_velocity(state.physical)

    def terrain_scan(self, state: TaskState) -> Tensor:
        position, yaw = self.world_position(state), self.yaw(state)
        x, y = self.scan_points.unbind(-1)
        points = torch.stack((yaw.cos()[:, None] * x - yaw.sin()[:, None] * y,
                              yaw.sin()[:, None] * x + yaw.cos()[:, None] * y), -1) + position[:, None, :2]
        return position[:, 2:3] - self.surface_height(points)

    def physical_features(self, state: TaskState) -> Tensor:
        return state.physical[:, ROBOT_STATE]

    @staticmethod
    def simulation_context(state: TaskState) -> Tensor:
        # MjLab位置动作还会减去encoder bias，因此它同时属于动力学和价值条件。
        return torch.cat((state.context[:, :1], state.context[:, CONTEXT_COM],
                          state.context[:, CONTEXT_ENCODER_BIAS]), -1)

    def model_input(self, state: TaskState, action: Tensor) -> Tensor:
        return torch.cat((self.physical_features(state), action, state.memory[:, PREVIOUS_ACTION],
                          self.simulation_context(state), self.terrain_scan(state)), -1)

    def supervised_model_input(self, state: TaskState, action: Tensor, recorded: TaskState) -> Tensor:
        """参考监督使用记录pose处的地形relief，加上自回归预测的机身高度。"""
        scan = self.terrain_scan(recorded) - recorded.physical[:, HEIGHT] + state.physical[:, HEIGHT]
        return torch.cat((self.physical_features(state), action, state.memory[:, PREVIOUS_ACTION],
                          self.simulation_context(state), scan), -1)

    def privileged(self, state: TaskState) -> Tensor:
        return torch.cat((self.physical_features(state), state.memory[:, PREVIOUS_ACTION], state.command,
                          state.memory[:, COMMAND_TIMER, None], self.terrain_scan(state),
                          self.simulation_context(state)), -1)

    def observe(self, state: TaskState, *, noisy: bool = False) -> Tensor:
        physical = state.physical
        observation = torch.cat((physical[:, ANGULAR_BODY], physical[:, GRAVITY],
                                 physical[:, JOINT_OFFSET] + state.context[:, CONTEXT_ENCODER_BIAS],
                                 physical[:, JOINT_VELOCITY], state.memory[:, PREVIOUS_ACTION], state.command), -1)
        if noisy and self.cfg.noise.add_noise:
            observation = observation + (2 * torch.rand_like(observation) - 1) * self.noise_amplitude
        return observation

    # ---------- 命令及确定性记忆 ----------

    def sample_commands(self, context: Tensor) -> tuple[Tensor, Tensor]:
        bounds = context[:, CONTEXT_RANGES].reshape(-1, 3, 2)
        command = bounds[..., 0] + torch.rand_like(bounds[..., 0]) * (bounds[..., 1] - bounds[..., 0])
        standing = torch.rand_like(context[:, CONTEXT_ZERO_PROBABILITY]) < context[:, CONTEXT_ZERO_PROBABILITY]
        command = torch.where(standing.unsqueeze(-1), torch.zeros_like(command), command)
        lower, upper = self.cfg.commands.resampling_time
        timer = lower + (upper - lower) * torch.rand_like(context[:, CONTEXT_ZERO_PROBABILITY])
        return command, timer

    def advance(self, state: TaskState, action: Tensor, next_physical: Tensor) -> TaskState:
        """命令采样属于外生任务过程；记忆保存实际下发动作和控制时间。"""
        clock = state.memory[:, ELAPSED_TIME] + self.dt
        timer = state.memory[:, COMMAND_TIMER] - self.dt
        new_command, new_timer = self.sample_commands(state.context)
        expired = timer <= 1e-7
        command = torch.where(expired.unsqueeze(-1), new_command, state.command)
        timer = torch.where(expired, new_timer, timer)
        return TaskState(next_physical, torch.cat((action, clock.unsqueeze(-1), timer.unsqueeze(-1)), -1), command, state.context)

    # ---------- 直接增量动力学 ----------

    def model_target(self, state: TaskState, next_state: TaskState) -> Tensor:
        return torch.cat((next_state.physical[:, ROBOT_STATE] - state.physical[:, ROBOT_STATE],
                          pose_delta(state.physical[:, POSE], next_state.physical[:, POSE])), -1)

    def integrate(self, state: TaskState, action: Tensor, delta: Tensor) -> Tensor:
        del action
        return torch.cat((state.physical[:, ROBOT_STATE] + delta[:, :70],
                          apply_pose_delta(state.physical[:, POSE], delta[:, 70:])), -1)

    def model_error(self, current: TaskState, predicted: TaskState, target: TaskState,
                    recorded_current: TaskState) -> Tensor:
        """按参考分别监督70维状态和每步位姿增量，不把位姿误差累积到前序步骤。"""
        # 预测yaw由apply_pose_delta原样累加，不wrap；大于pi的错误预测仍受到完整惩罚。
        predicted_delta = yaw_rotate_inverse(predicted.physical[:, POSE] - current.physical[:, POSE], self.yaw(current))
        return torch.cat((predicted.physical[:, ROBOT_STATE] - target.physical[:, ROBOT_STATE],
                          predicted_delta - pose_delta(recorded_current.physical[:, POSE], target.physical[:, POSE])), -1)

    # ---------- 奖励、失败与稀有窗口 ----------

    def reward_terms(self, state, action, next_state, failure):
        return reference_reward_terms(self, state, action, next_state, failure)

    def reward(self, state: TaskState, action: Tensor, next_state: TaskState, failure: Tensor) -> Tensor:
        return self.dt * sum(getattr(self.cfg.reward.scales, name) * value
                             for name, value in self.reward_terms(state, action, next_state, failure).items())

    def hard_termination(self, state: TaskState) -> Tensor:
        supported = self.surface.contains(state.physical[:, POSE][:, :2], margin=self.cfg.terrain.boundary_margin_m)
        return (state.physical[:, 3] > -math.cos(self.cfg.reward.fall_angle)) | ~supported

    def soft_termination(self, state: TaskState) -> Tensor:
        tilt = torch.acos((-state.physical[:, 3]).clamp(-1 + 1e-4, 1 - 1e-4))
        probability = torch.sigmoid((tilt - self.cfg.reward.fall_angle) / math.radians(self.cfg.reward.fall_softness_deg))
        supported = self.surface.contains(state.physical[:, POSE][:, :2], margin=self.cfg.terrain.boundary_margin_m)
        return torch.where(supported, probability, torch.ones_like(probability))

    def _replay_events(self, window):
        """参考用state内足端中心估计触地；粗糙地形修正为各足下方真实地表。"""
        physical = torch.cat((window.initial.physical[:, None], window.targets.physical), 1)
        batch, length = physical.shape[:2]
        flat = physical.flatten(0, 1)
        feet = flat[:, FEET].reshape(-1, 4, 3)
        pose = flat[:, POSE]
        xy = torch.stack((pose[:, 2:3].cos() * feet[..., 0] - pose[:, 2:3].sin() * feet[..., 1],
                          pose[:, 2:3].sin() * feet[..., 0] + pose[:, 2:3].cos() * feet[..., 1]), -1) + pose[:, None, :2]
        base_ground = self.surface_height(pose[:, :2])
        height = flat[:, HEIGHT] + base_ground[:, None] + feet[..., 2] - self.surface_height(xy)
        contact = (height < .035).reshape(batch, length, 4)
        valid = torch.cat((torch.ones_like(window.valid[:, :1]), window.valid), 1)
        flight = ~contact.any(-1) & valid
        changes = (contact[:, 1:] != contact[:, :-1]).sum(-1) * window.valid
        hard_motion = (physical[..., VELOCITY_HEADING][..., 2].abs() > .6) & valid
        falls = (window.terminated & window.valid).any(-1)
        return changes, flight, hard_motion, falls

    def replay_priority(self, window) -> Tensor:
        """优先拟合接触切换、飞行和真实失败；重置后的无效数据不计分。"""
        changes, flight, _, falls = self._replay_events(window)
        return changes.sum(-1).float() + flight.sum(-1).float() + 4.0 * falls.float()

    def archive_worthy(self, window) -> Tensor:
        """归档至少两帧飞行、明显垂向运动或真实跌倒。"""
        _, flight, hard_motion, falls = self._replay_events(window)
        return (flight.sum(-1) >= 2) | hard_motion.any(-1) | falls
