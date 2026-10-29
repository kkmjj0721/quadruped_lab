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
from .commands import advance_commands, heading_command, sample_commands
from .gait_rewards import gait_phase
from .reference_state import (
    ANGULAR_BODY, AUXILIARY, CONTACT_FOUND, CONTEXT_COM, FEET, GRAVITY, HEIGHT, JOINT_OFFSET,
    JOINT_VELOCITY, PHYSICAL_DIM, POSE,
    ROBOT_STATE, VELOCITY_HEADING, apply_pose_delta, pose_delta, tracking_velocity, yaw_rotate_inverse,
)
from .state_spec import (
    COMMAND_TIMER, CONTEXT_ENCODER_BIAS,
    FOOT_NAMES, PREVIOUS_ACTION, PREVIOUS_PREVIOUS_ACTION, ELAPSED_TIME, AIR_TIME, CONTACT, CONTACT_TIME, COMMAND_OVERRIDE,
    HEADING_TARGET, STANDING, FORWARD, MEMORY_DIM, JOINT_NAMES,
)


class Go2ReferenceTaskModel:
    """独立实现参考Go2的状态、命令、观测、奖励与回放事件接口。"""

    model_input_dim = 211
    observation_term_dims = (3, 3, 12, 12, 12, 3)

    def __init__(self, cfg, default_joint_pos: Tensor, joint_limits: Tensor, surface, device="cpu"):
        self.cfg = deepcopy(cfg)
        self.device = torch.device(device)
        self.surface = surface
        self.dt = cfg.sim.timestep * cfg.control.decimation
        self.observation_term_dims = type(self).observation_term_dims + ((2,) if cfg.experiment.actor_phase else ())
        self.default_joint_pos = torch.as_tensor(default_joint_pos, device=self.device, dtype=torch.float32).clone()
        self.joint_limits = torch.as_tensor(joint_limits, device=self.device, dtype=torch.float32).clone()
        self.posture_std = torch.tensor(
            [[getattr(cfg.reward.posture, name)[joint.split("_")[1]] for joint in JOINT_NAMES]
             for name in ("std_standing", "std_walking", "std_running")], device=self.device)
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
            + (noise.joint_vel,) * 12 + (0.,) * (17 if cfg.experiment.actor_phase else 15)) * cfg.noise.noise_level
        payload = {"task": self._semantic_config(), "geometry": surface.geometry_hash,
                   "default_joint_pos": self.default_joint_pos.cpu().tolist(),
                   "joint_limits": self.joint_limits.cpu().tolist(),
                   "layout": "go2_reference_state70_pose3_motion_aux24_v4"}
        self.semantic_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.spec = AlgorithmSpec(PHYSICAL_DIM, PHYSICAL_DIM, 12, sum(self.observation_term_dims), 234, 26, MEMORY_DIM, 3,
                                  "go2_reference_velocity_v4:" + self.semantic_hash,
                                  model_id="supermimic_moe_v1")

    def _semantic_config(self) -> dict:
        """语义指纹沿用历史段名，使算法段改名前的检查点仍可续训与播放。

        段名只是本项目的组织方式，不属于算法语义；改名不应让已有训练失效。
        两段并存或都不存在时无法判断意图，直接报错而非静默算出另一个指纹。
        """
        config = self.cfg.to_dict()
        # 旧45维任务关闭两项时保留既有指纹，新协议开关完整进入语义指纹。
        if config["experiment"] == {"contact_source": "height", "actor_phase": False}:
            del config["experiment"]
        # 禁用新增站立速度项时保留已有十五项v4快照的指纹，非零权重标识新任务语义。
        if config["reward"]["scales"]["stand_still_joint_vel_l2"] == 0.0:
            del config["reward"]["scales"]["stand_still_joint_vel_l2"]
        names = [name for name in ("algo", "single") if name in config]
        if names == ["single"]:
            return config
        if names != ["algo"]:
            raise ValueError(f"Task configuration must contain exactly one algorithm section, got {names or 'none'}")
        return {("single" if name == "algo" else name): value for name, value in config.items()}

    # ---------- 真实状态与坐标 ----------

    def surface_height(self, xy: Tensor) -> Tensor:
        """来源高度栅格在域外夹取边缘，不新增任务失败条件。"""
        return self.surface.height_unchecked(xy)

    def bind_env(self, env):
        """在仿真边界固定本项目FL、FR、RL、RR顺序的关节中心与足端site。"""
        robot = env.scene["robot"]
        self.hip_ids, _ = robot.find_bodies(tuple(f"{leg}_thigh" for leg in FOOT_NAMES), preserve_order=True)
        self.knee_ids, _ = robot.find_bodies(tuple(f"{leg}_calf" for leg in FOOT_NAMES), preserve_order=True)
        self.foot_ids, _ = robot.find_sites(FOOT_NAMES, preserve_order=True)

    def read_physical(self, env) -> Tensor:
        """保留原状态，追加最后forward后的qacc、世界足速和真实接触标记。"""
        data = env.scene["robot"].data
        sensor = env.scene["feet_ground_contact"].data
        position, yaw = data.root_link_pos_w, data.heading_w
        limbs = (data.body_link_pos_w[:, self.hip_ids], data.body_link_pos_w[:, self.knee_ids],
                 data.site_pos_w[:, self.foot_ids])
        relative = [yaw_rotate_inverse(points - position[:, None], yaw[:, None]).flatten(1) for points in limbs]
        height = position[:, 2:3] - self.surface_height(position[:, :2]).unsqueeze(-1)
        physical = torch.cat((height, data.projected_gravity_b,
                              yaw_rotate_inverse(data.root_link_lin_vel_w, yaw), data.root_link_ang_vel_b,
                              data.joint_pos - data.default_joint_pos, data.joint_vel, *relative,
                              position[:, :2], yaw.unsqueeze(-1), data.joint_acc,
                              data.site_lin_vel_w[:, self.foot_ids, :2].flatten(1),
                              (sensor.found > 0).to(position.dtype)), -1)
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
        relief = (self.surface_height(position[:, :2])[:, None] - self.surface_height(points)).clamp(-1.2, 1.2)
        return self.base_height(state)[:, None] + relief

    def physical_features(self, state: TaskState) -> Tensor:
        return torch.cat((state.physical[:, ROBOT_STATE], state.physical[:, AUXILIARY]), -1)

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
        recorded_relief = (self.terrain_scan(recorded) - recorded.physical[:, HEIGHT]).half().float()
        scan = recorded_relief + state.physical[:, HEIGHT]
        return torch.cat((self.physical_features(state), action, state.memory[:, PREVIOUS_ACTION],
                          self.simulation_context(state), scan), -1)

    def privileged(self, state: TaskState) -> Tensor:
        phase = 2 * torch.pi * gait_phase(state, self.dt, self.cfg.reward.gait.period)
        heading_error = state.memory[:, HEADING_TARGET] - self.yaw(state)
        task_features = torch.cat((phase.sin()[:, None], phase.cos()[:, None],
                                   heading_error.sin()[:, None], heading_error.cos()[:, None],
                                   state.memory[:, STANDING:FORWARD + 1], state.memory[:, CONTACT],
                                   state.memory[:, AIR_TIME], state.memory[:, CONTACT_TIME],
                                   state.memory[:, COMMAND_OVERRIDE, None], state.memory[:, PREVIOUS_PREVIOUS_ACTION]), -1)
        # 来源critic的relief不对位置／航向求导，但机身高度仍属于价值导数。
        scan = state.physical[:, HEIGHT] + (self.terrain_scan(state) - state.physical[:, HEIGHT]).detach()
        return torch.cat((self.physical_features(state), state.memory[:, PREVIOUS_ACTION], state.command,
                          state.memory[:, COMMAND_TIMER, None], scan,
                          self.simulation_context(state), task_features), -1)

    def observe(self, state: TaskState, *, noisy: bool = False) -> Tensor:
        physical = state.physical
        observation = torch.cat((physical[:, ANGULAR_BODY], physical[:, GRAVITY],
                                 physical[:, JOINT_OFFSET] + state.context[:, CONTEXT_ENCODER_BIAS],
                                 physical[:, JOINT_VELOCITY], state.memory[:, PREVIOUS_ACTION], state.command), -1)
        if noisy and self.cfg.noise.add_noise:
            # 相位不采样噪声，保留旧45维本体输入相同的随机流。
            observation = observation + (2 * torch.rand_like(observation) - 1) * self.noise_amplitude[:45]
        if self.cfg.experiment.actor_phase:
            phase = 2 * torch.pi * gait_phase(state, self.dt, self.cfg.reward.gait.period)
            moving = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs() > self.cfg.reward.gait.command_threshold
            phase_features = torch.stack((phase.sin(), phase.cos()), -1) * moving[:, None]
            observation = torch.cat((observation, phase_features), -1)
        return observation

    # ---------- 命令及确定性记忆 ----------

    def initialize_state(self, physical: Tensor, context: Tensor) -> TaskState:
        """重置命令及全部确定性记忆；接触计时与来源传感器一样从零开始。"""
        command, timer, heading, standing, forward = sample_commands(self.cfg.commands, context, physical[:, 72])
        memory = physical.new_zeros((physical.shape[0], MEMORY_DIM))
        memory[:, COMMAND_TIMER] = timer
        memory[:, HEADING_TARGET] = heading
        memory[:, STANDING] = standing
        memory[:, FORWARD] = forward
        command = heading_command(self.cfg.commands, command, physical[:, 72], heading, standing)
        return TaskState(physical, memory, command, context)

    def feet_height(self, physical):
        """按各脚世界位置查询表面，得到足球中心离地高度。"""
        feet = physical[:, FEET].reshape(-1, 4, 3)
        pose = physical[:, POSE]
        xy = torch.stack((pose[:, 2:3].cos() * feet[..., 0] - pose[:, 2:3].sin() * feet[..., 1],
                          pose[:, 2:3].sin() * feet[..., 0] + pose[:, 2:3].cos() * feet[..., 1]), -1) + pose[:, None, :2]
        base_ground = self.surface_height(pose[:, :2])
        return physical[:, HEIGHT] + base_ground[:, None] + feet[..., 2] - self.surface_height(xy)

    def advance(self, state: TaskState, action: Tensor, next_physical: Tensor, *, contact_times=None) -> TaskState:
        """真实端读取子步计时；想象端使用受监督接触概率，旧快照保留足高近似。"""
        clock = state.memory[:, ELAPSED_TIME] + self.dt
        command, timer, heading, standing, forward = advance_commands(self.cfg.commands, state, next_physical, self.dt)
        if contact_times is None:
            gait = self.cfg.reward.gait
            contact = (next_physical[:, CONTACT_FOUND] if self.cfg.experiment.contact_source == "model" else
                       torch.sigmoid((gait.contact_height - self.feet_height(next_physical)) / gait.contact_softness))
            previous = state.memory[:, CONTACT]
            # 最大持续耦合：概率不变时不重复假设发生离地／触地，避免计时饱和。
            contact_stay = torch.minimum(previous, contact) / previous.clamp_min(1e-6)
            air_stay = torch.minimum(1 - previous, 1 - contact) / (1 - previous).clamp_min(1e-6)
            air_time = air_stay * state.memory[:, AIR_TIME] + (1 - contact) * self.dt
            contact_time = contact_stay * state.memory[:, CONTACT_TIME] + contact * self.dt
        else:
            air_time, contact_time = contact_times
            contact = (contact_time > 0).to(next_physical.dtype)
        memory = torch.cat((action, clock[:, None], timer[:, None], heading[:, None],
                            standing.to(action.dtype)[:, None], forward.to(action.dtype)[:, None], contact,
                            air_time, contact_time, state.memory[:, COMMAND_OVERRIDE, None],
                            state.memory[:, PREVIOUS_ACTION]), -1)
        return TaskState(next_physical, memory, command, state.context)

    # ---------- 直接增量动力学 ----------

    def model_target(self, state: TaskState, next_state: TaskState) -> Tensor:
        """原73维仍为增量；辅助瞬时量直接监督下一状态的绝对值。"""
        return torch.cat((next_state.physical[:, ROBOT_STATE] - state.physical[:, ROBOT_STATE],
                          pose_delta(state.physical[:, POSE], next_state.physical[:, POSE]),
                          next_state.physical[:, AUXILIARY]), -1)

    def integrate(self, state: TaskState, action: Tensor, delta: Tensor) -> Tensor:
        del action
        return torch.cat((state.physical[:, ROBOT_STATE] + delta[:, :70],
                          apply_pose_delta(state.physical[:, POSE], delta[:, POSE]),
                          delta[:, 73:93], delta[:, CONTACT_FOUND].sigmoid()), -1)

    def model_error(self, current: TaskState, predicted: TaskState, target: TaskState,
                    recorded_current: TaskState) -> Tensor:
        """分别监督原状态增量与辅助绝对量；接触使用预测概率与真实标记之差。"""
        # 预测yaw由apply_pose_delta原样累加，不wrap；大于pi的错误预测仍受到完整惩罚。
        predicted_delta = yaw_rotate_inverse(predicted.physical[:, POSE] - current.physical[:, POSE], self.yaw(current))
        return torch.cat((predicted.physical[:, ROBOT_STATE] - target.physical[:, ROBOT_STATE],
                          predicted_delta - pose_delta(recorded_current.physical[:, POSE], target.physical[:, POSE]),
                          predicted.physical[:, AUXILIARY] - target.physical[:, AUXILIARY]), -1)

    # ---------- 奖励、失败与稀有窗口 ----------

    def reward_terms(self, state, action, next_state, failure):
        return reference_reward_terms(self, state, action, next_state, failure)

    def reward(self, state: TaskState, action: Tensor, next_state: TaskState, failure: Tensor) -> Tensor:
        return self.dt * sum(getattr(self.cfg.reward.scales, name) * value
                             for name, value in self.reward_terms(state, action, next_state, failure).items())

    def hard_termination(self, state: TaskState) -> Tensor:
        return state.physical[:, 3] > -math.cos(self.cfg.reward.fall_angle)

    def soft_termination(self, state: TaskState) -> Tensor:
        tilt = torch.acos((-state.physical[:, 3]).clamp(-1 + 1e-4, 1 - 1e-4))
        probability = torch.sigmoid((tilt - self.cfg.reward.fall_angle) / math.radians(self.cfg.reward.fall_softness_deg))
        return probability

    def _replay_events(self, window):
        """严格沿用参考按state基准高度估计的稀有事件，不添加地形修正。"""
        physical = torch.cat((window.initial.physical[:, None], window.targets.physical), 1)
        batch, length = physical.shape[:2]
        flat = physical.flatten(0, 1)
        feet = flat[:, FEET].reshape(-1, 4, 3)
        height = flat[:, HEIGHT] + feet[..., 2]
        contact = (height < .035).reshape(batch, length, 4)
        flight = ~contact.any(-1)
        changes = (contact[:, 1:] != contact[:, :-1]).sum(-1)
        hard_motion = physical[..., VELOCITY_HEADING][..., 2].abs() > .6
        falls = ~window.valid[:, -1]
        return changes, flight, hard_motion, falls

    def replay_priority(self, window) -> Tensor:
        """严格按来源计算接触切换、飞行和窗口提前结束的分数。"""
        changes, flight, _, falls = self._replay_events(window)
        return changes.sum(-1).float() + flight.sum(-1).float() + 4.0 * falls.float()

    def archive_worthy(self, window) -> Tensor:
        """归档至少两帧飞行、明显垂向运动或真实跌倒。"""
        _, flight, hard_motion, falls = self._replay_events(window)
        return (flight.sum(-1) >= 2) | hard_motion.any(-1) | falls
