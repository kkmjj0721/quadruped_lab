# Copyright (c) 2026-2027 zh
"""内容：将指定Go2参考状态、解析速度奖励及真实随机化上下文接入混合专家历史策略训练。"""

from copy import deepcopy
import hashlib
import json
import math

import torch
from torch import Tensor

from agents.supermimic.config import RUNNER_RUNTIME_FIELDS
from agents.supermimic.interfaces import AlgorithmSpec, TaskState

from .reference_rewards import reference_reward_terms
from .commands import advance_commands, heading_command, sample_commands
from .gait_rewards import gait_phase
from .reference_kinematics import Go2Kinematics
from .reference_rotations import mat_to_quat, mv3, quat_to_mat
from .structured_dynamics import ResidualDynamics
from .reference_state import (
    ANGULAR_BODY, AUXILIARY, CONTACT_FOUND, CONTEXT_COM, FEET, FORCE_PHYSICAL_DIM, GENERALIZED_DELTA_DIM,
    GRAVITY, HEIGHT, JOINT_ACCELERATION, JOINT_OFFSET, MODEL_CONTACT,
    JOINT_VELOCITY, LEGACY_PHYSICAL_DIM, LEG_PHYSICAL_DIM, PHYSICAL_DIM, POSE,
    ROBOT_STATE, VELOCITY_HEADING, body_planar_velocity, body_vertical_velocity, tracking_velocity, yaw_rotate_inverse,
)
from .state_spec import (
    COMMAND_TIMER, CONTEXT_ENCODER_BIAS,
    FOOT_NAMES, PREVIOUS_ACTION, PREVIOUS_PREVIOUS_ACTION, ELAPSED_TIME, AIR_TIME, CONTACT, CONTACT_TIME, COMMAND_OVERRIDE,
    HEADING_TARGET, STANDING, FORWARD, MEMORY_DIM, JOINT_NAMES,
)


class Go2ReferenceTaskModel:
    """独立实现参考Go2的状态、命令、观测、奖励与回放事件接口。"""

    model_input_dim = 239
    observation_term_dims = (3, 3, 12, 12, 12, 3)

    def __init__(self, cfg, default_joint_pos: Tensor, joint_limits: Tensor, surface, device="cpu"):
        self.cfg = deepcopy(cfg)
        self.device = torch.device(device)
        self.surface = surface
        self.dt = cfg.sim.timestep * cfg.control.decimation
        self.him_rewards = cfg.env.num_critic_obs == 262
        self.legacy_sideways_lean = hasattr(cfg.reward.scales, "torso_sideways_lean")
        self.body_ang_vel_frame = cfg.reward.body_ang_vel_frame
        self.leg_collision = cfg.env.num_critic_obs in (259, 262)
        self.force_rewards = cfg.env.num_critic_obs in (258, 259, 262)
        self.physical_dim = (PHYSICAL_DIM if self.him_rewards else LEG_PHYSICAL_DIM if self.leg_collision else
                             FORCE_PHYSICAL_DIM if self.force_rewards else LEGACY_PHYSICAL_DIM)
        self.auxiliary = slice(AUXILIARY.start, self.physical_dim)
        self.root_quat = slice(self.physical_dim, self.physical_dim + 4)
        self.model_input_dim = self.physical_dim + 114
        self.dynamics = ResidualDynamics(self.dt, cfg.control.decimation)
        self.kinematics = Go2Kinematics(device=self.device)
        self.observation_term_dims = type(self).observation_term_dims + ((2,) if cfg.experiment.actor_phase else ())
        self.default_joint_pos = torch.as_tensor(default_joint_pos, device=self.device, dtype=torch.float32).clone()
        self.joint_limits = torch.as_tensor(joint_limits, device=self.device, dtype=torch.float32).clone()
        self.posture_std = torch.tensor(
            [[getattr(cfg.reward.posture, name)[joint.split("_")[1]] for joint in JOINT_NAMES]
             for name in ("std_standing", "std_walking", "std_running")], device=self.device)
        self.gait_offsets = self.default_joint_pos.new_tensor([cfg.reward.gait.offsets[name] for name in FOOT_NAMES])
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
                   "layout": ("go2_reference_state70_pose3_motion_force_leg_momentum_aux52_v7" if self.him_rewards else
                              "go2_reference_state70_pose3_motion_force_leg_aux49_v6" if self.leg_collision else
                              "go2_reference_state70_pose3_motion_force_aux48_v5" if self.force_rewards else
                              "go2_reference_state70_pose3_motion_aux24_v4"),
                   "dynamics": "blinkmimic_residual_fk_quaternion_v1"}
        self.semantic_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        version = "v7" if self.him_rewards else "v6" if self.leg_collision else "v5" if self.force_rewards else "v4"
        self.spec = AlgorithmSpec(self.physical_dim + 4, self.physical_dim - 45, 12, sum(self.observation_term_dims),
                                  cfg.env.num_critic_obs, 26, MEMORY_DIM, 3,
                                  f"go2_reference_velocity_{version}_residual_fk_v1:" + self.semantic_hash,
                                  model_id="supermimic_moe_velocity_v1")

    def _semantic_config(self) -> dict:
        """以当前任务配置建立语义指纹，运行时预算不改变任务定义。"""
        config = self.cfg.to_dict()
        config["runner"] = {name: value for name, value in config["runner"].items() if name not in RUNNER_RUNTIME_FIELDS}
        # 优化器类型在算法恢复边界核对，不改变旧策略的任务／部署语义。
        for name in ("actor_optimizer", "wm_optimizer", "muon_weight_decay"):
            config["algo"].pop(name)
        return config

    # ---------- 真实状态与坐标 ----------

    def surface_height(self, xy: Tensor) -> Tensor:
        """来源高度栅格在域外夹取边缘，不新增任务失败条件。"""
        return self.surface.height_unchecked(xy)

    def bind_env(self, env):
        """在任务设备缓存FL、FR、RL、RR顺序的关节中心与足端site索引。"""
        robot = env.scene["robot"]
        hip_ids, _ = robot.find_bodies(tuple(f"{leg}_thigh" for leg in FOOT_NAMES), preserve_order=True)
        knee_ids, _ = robot.find_bodies(tuple(f"{leg}_calf" for leg in FOOT_NAMES), preserve_order=True)
        foot_ids, _ = robot.find_sites(FOOT_NAMES, preserve_order=True)
        self.hip_ids = torch.as_tensor(hip_ids, dtype=torch.long, device=self.device)
        self.knee_ids = torch.as_tensor(knee_ids, dtype=torch.long, device=self.device)
        self.foot_ids = torch.as_tensor(foot_ids, dtype=torch.long, device=self.device)

    def read_physical(self, env) -> Tensor:
        """读取最后forward后的瞬时量；腿部历史力取最大模以保留来源any语义。"""
        data = env.scene["robot"].data
        sensor = env.scene["feet_ground_contact"].data
        position, yaw = data.root_link_pos_w, data.heading_w
        limbs = (data.body_link_pos_w[:, self.hip_ids], data.body_link_pos_w[:, self.knee_ids],
                 data.site_pos_w[:, self.foot_ids])
        relative = [yaw_rotate_inverse(points - position[:, None], yaw[:, None]).flatten(1) for points in limbs]
        height = position[:, 2:3] - self.surface_height(position[:, :2]).unsqueeze(-1)
        forces = (sensor.force.flatten(1), data.actuator_force) if self.force_rewards else ()
        leg_force = ()
        if self.leg_collision:
            history = env.scene["leg_ground_contact"].data.force_history
            leg_force = (history.norm(dim=-1).amax(dim=(1, 2)).unsqueeze(-1),)
        # heading旋转保持角动量平方和，避免模型在缺少绝对yaw输入时预测世界向量。
        momentum = (yaw_rotate_inverse(env.scene["robot/root_angmom"].data, yaw),) if self.him_rewards else ()
        physical = torch.cat((height, data.projected_gravity_b,
                              yaw_rotate_inverse(data.root_link_lin_vel_w, yaw), data.root_link_ang_vel_b,
                              data.joint_pos - data.default_joint_pos, data.joint_vel, *relative,
                              position[:, :2], yaw.unsqueeze(-1), data.joint_acc,
                              data.site_lin_vel_w[:, self.foot_ids, :2].flatten(1),
                              (sensor.found > 0).to(position.dtype), *forces, *leg_force, *momentum,
                              data.root_link_quat_w), -1)
        return physical

    def world_position(self, state: TaskState) -> Tensor:
        xy = state.physical[:, POSE][:, :2]
        return torch.cat((xy, self.base_height(state).unsqueeze(-1) + self.surface_height(xy).unsqueeze(-1)), -1)

    def base_height(self, state: TaskState) -> Tensor:
        return state.physical[:, 0]

    def yaw(self, state: TaskState) -> Tensor:
        return state.physical[:, 72]

    def body_linear_velocity(self, state: TaskState) -> Tensor:
        """从同一时刻真实状态恢复机身系xyz线速度，单位为m/s，不引入观测噪声。"""
        return torch.cat((body_planar_velocity(state.physical), body_vertical_velocity(state.physical).unsqueeze(-1)), -1)

    def tracking_velocity(self, state: TaskState) -> Tensor:
        velocity = tracking_velocity(state.physical)
        if self.cfg.reward.tracking_linear_frame == "body":
            return torch.cat((body_planar_velocity(state.physical), velocity[:, 2:]), -1)
        return velocity

    def terrain_scan(self, state: TaskState) -> Tensor:
        position, yaw = self.world_position(state), self.yaw(state)
        x, y = self.scan_points.unbind(-1)
        points = torch.stack((yaw.cos()[:, None] * x - yaw.sin()[:, None] * y,
                              yaw.sin()[:, None] * x + yaw.cos()[:, None] * y), -1) + position[:, None, :2]
        relief = (self.surface_height(position[:, :2])[:, None] - self.surface_height(points)).clamp(-1.2, 1.2)
        return self.base_height(state)[:, None] + relief

    def physical_features(self, state: TaskState) -> Tensor:
        return torch.cat((state.physical[:, ROBOT_STATE], state.physical[:, self.auxiliary]), -1)

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
        """真实端读取子步计时；想象端默认使用受监督接触概率。"""
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

    # ---------- 广义状态、残差积分与正运动学 ----------

    def generalized_state(self, state: TaskState):
        """完整姿态来自四元数；局部速度在边界转换为参考实现的世界速度。"""
        physical = state.physical
        rotation = quat_to_mat(physical[:, self.root_quat])
        velocity = yaw_rotate_inverse(physical[:, VELOCITY_HEADING], -self.yaw(state))
        return (self.world_position(state), rotation, velocity, mv3(rotation, physical[:, ANGULAR_BODY]),
                physical[:, JOINT_OFFSET] + self.default_joint_pos, physical[:, JOINT_VELOCITY])

    def physical_state(self, generalized, auxiliary: Tensor) -> Tensor:
        """从单一广义状态生成任务视图；auxiliary为不含足速的独立瞬时量。"""
        position, rotation, velocity, angular, joints, joint_velocity = generalized
        quaternion = mat_to_quat(rotation)
        rotation = quat_to_mat(quaternion)
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        height = position[:, 2:3] - self.surface_height(position[:, :2]).unsqueeze(-1)
        # 以根部为原点做同一棵FK，避免绝对位置较大时相减损失相对足端精度。
        limbs, feet_velocity = self.kinematics.forward(
            torch.zeros_like(position), rotation, velocity, angular, joints, joint_velocity)
        relative = yaw_rotate_inverse(limbs, yaw[:, None]).flatten(1)
        return torch.cat((height, -rotation[:, 2, :], yaw_rotate_inverse(velocity, yaw),
                          mv3(rotation.transpose(-1, -2), angular), joints - self.default_joint_pos, joint_velocity,
                          relative, position[:, :2], yaw[:, None], auxiliary[:, :12],
                          feet_velocity[:, :, :2].flatten(1), auxiliary[:, 12:], quaternion), -1)

    def independent_auxiliary(self, state: TaskState) -> Tensor:
        """瞬时加速度、接触与力等保留原标签；足速由速度运动学生成。"""
        return torch.cat((state.physical[:, JOINT_ACCELERATION],
                          state.physical[:, CONTACT_FOUND.start:self.physical_dim]), -1)

    def model_target(self, state: TaskState, next_state: TaskState) -> Tensor:
        """沿用BlinkMimic的速度增量与位置残差目标，附加Go2独立辅助量。"""
        return torch.cat((self.dynamics.target(self.generalized_state(state), self.generalized_state(next_state)),
                          self.independent_auxiliary(next_state)), -1)

    def model_increment(self, state: TaskState, next_state: TaskState) -> Tensor:
        """普通物理增量只用于误差尺度统计，避免残差方差改变监督权重。"""
        return torch.cat((self.dynamics.increments(self.generalized_state(state), self.generalized_state(next_state)),
                          self.independent_auxiliary(next_state)), -1)

    def integrate(self, state: TaskState, action: Tensor, delta: Tensor) -> Tensor:
        del action
        generalized = self.dynamics.integrate(self.generalized_state(state), delta[:, :GENERALIZED_DELTA_DIM])
        auxiliary = torch.cat((delta[:, GENERALIZED_DELTA_DIM:MODEL_CONTACT.start],
                               delta[:, MODEL_CONTACT].sigmoid(), delta[:, MODEL_CONTACT.stop:]), -1)
        return self.physical_state(generalized, auxiliary)

    def model_error(self, current: TaskState, predicted: TaskState, target: TaskState,
                    recorded_current: TaskState) -> Tensor:
        """按目标状态坐标监督广义物理误差；接触使用预测概率残差。"""
        del current, recorded_current
        return torch.cat((self.dynamics.error(self.generalized_state(predicted), self.generalized_state(target)),
                          self.independent_auxiliary(predicted) - self.independent_auxiliary(target)), -1)

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
