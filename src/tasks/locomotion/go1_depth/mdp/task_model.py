# Copyright (c) 2026-2027 zh
"""
内容：
    以 Go1 真实几何实现 Parkour 全状态增量、视觉本体观测与 heading 速度指令。
    复用 Go2 已有二十二项奖励公式，额外瞬时标签仅服务这些奖励，不新增奖励项。
    全状态增量和不变量对应 InstinctMJ 03563ee 的 supermimic/networks.py 与 mdp.py。
"""

from copy import deepcopy
import json

import torch

from agents.depth_supermimic.interfaces import DepthSpec
from agents.supermimic.interfaces import TaskState
from src.assets.robots.go1_depth.go1_depth_constants import JOINT_NAMES, LEG_NAMES
from src.tasks.locomotion.go2_supermimic.mdp.commands import advance_commands, heading_command, sample_commands
from src.tasks.locomotion.go2_supermimic.mdp.gait_rewards import gait_phase
from src.tasks.locomotion.go2_supermimic.mdp.reference_rotations import mm3, mv3, so3_exp, so3_log, yaw_mat
from src.tasks.locomotion.go2_supermimic.mdp.reference_state import (
    ANGULAR_BODY, ANGULAR_MOMENTUM, CONTACT_FOUND, FOOT_CONTACT_FORCE, JOINT_ACCELERATION,
    JOINT_OFFSET, JOINT_TORQUE, JOINT_VELOCITY, LEG_CONTACT_FORCE,
)
from src.tasks.locomotion.go2_supermimic.mdp.reference_task import Go2ReferenceTaskModel
from src.tasks.locomotion.go2_supermimic.mdp.state_spec import (
    AIR_TIME, COMMAND_OVERRIDE, COMMAND_TIMER, CONTACT, CONTACT_TIME, ELAPSED_TIME, FORWARD,
    HEADING_TARGET, MEMORY_DIM, PREVIOUS_ACTION, PREVIOUS_PREVIOUS_ACTION, STANDING,
)

from .kinematics import Go1Kinematics


class Go1DepthTaskModel(Go2ReferenceTaskModel):
    """仅借用形态无关的真实读取和奖励，不调用 Go2 构造器及残差动力学。"""

    observation_term_dims = (3, 3, 3, 12, 12, 12)

    def __init__(self, cfg, default_joint_pos, joint_limits, surface, device="cpu"):
        self.cfg, self.device, self.surface = deepcopy(cfg), torch.device(device), surface
        self.dt = cfg.sim.timestep * cfg.control.decimation
        if not isinstance(cfg.experiment.actor_phase, bool):
            raise ValueError("Go1 actor phase must be a boolean")
        self.observation_term_dims = type(self).observation_term_dims + ((2,) if cfg.experiment.actor_phase else ())
        observation_dim = sum(self.observation_term_dims)
        if (cfg.env.num_observations, cfg.env.num_obs) != (observation_dim, observation_dim * cfg.env.num_obs_history):
            raise ValueError("Go1 observation dimensions must match the actor phase setting and history length")
        self.him_rewards = self.force_rewards = self.leg_collision = True
        self.legacy_sideways_lean = False
        self.body_ang_vel_frame = cfg.reward.body_ang_vel_frame
        self.physical_dim, self.root_quat = 125, slice(125, 129)
        self.default_joint_pos = torch.as_tensor(default_joint_pos, dtype=torch.float32, device=device).clone()
        self.joint_limits = torch.as_tensor(joint_limits, dtype=torch.float32, device=device).clone()
        if self.default_joint_pos.shape != (12,) or self.joint_limits.shape != (12, 2):
            raise ValueError("Go1 joint definitions must have shapes [12] and [12,2]")
        self.kinematics = Go1Kinematics(device)
        self.posture_std = torch.tensor(
            [[getattr(cfg.reward.posture, name)[joint.split("_")[1]] for joint in JOINT_NAMES]
             for name in ("std_standing", "std_walking", "std_running")], device=device)
        self.gait_offsets = self.default_joint_pos.new_tensor([cfg.reward.gait.offsets[name] for name in LEG_NAMES])
        scan = cfg.sensors.height_scan
        xs = torch.tensor(scan.measured_points_x, dtype=torch.float32, device=device)
        ys = torch.tensor(scan.measured_points_y, dtype=torch.float32, device=device)
        self.scan_points = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
        self.proprio_scale = self.default_joint_pos.new_tensor((0.25,) * 3 + (1.,) * 18 + (0.05,) * 12 + (1.,) * 12)
        noise = cfg.noise.noise_scales
        self.noise_amplitude = self.default_joint_pos.new_tensor(
            (noise.base_ang_vel,) * 3 + (noise.projected_gravity,) * 3 + (0.,) * 3
            + (noise.joint_pos,) * 12 + (noise.joint_vel,) * 12 + (0.,) * 12)
        self.noise_amplitude *= self.proprio_scale * cfg.noise.noise_level
        depth = cfg.sensors.depth
        self.spec = DepthSpec(observation_dim=observation_dim, observation_term_dims=self.observation_term_dims,
                              privileged_dim=286 if cfg.commands.heading_command else 280, scan_dim=self.scan_points.shape[0],
                              history_length=cfg.env.num_obs_history, depth_shape=(depth.history_length, depth.height, depth.width))
        self.model_input_dim = self.spec.model_input_dim

    def semantic_configuration(self):
        """恢复前核对任务定义；环境规模、随机种子和执行预算不改变任务接口。"""
        config = self.cfg.to_dict()
        names = ("reward", "control", "sensors", "terrain", "commands", "collision", "domain_randomization",
                 "init_state", "noise", "sim", "reset", "experiment")
        semantic = {name: config[name] for name in names}
        semantic["terrain"].pop("field_cache_dir", None)
        semantic["env"] = {name: value for name, value in config["env"].items() if name not in ("num_envs", "seed")}
        # 与 task_config.json 一致地规范化 tuple/list，避免原样恢复被容器类型差异拒绝。
        return json.loads(json.dumps(semantic))

    # ---------- 观测与特权信息 ----------

    def observe(self, state, *, noisy=False):
        """保留原本体顺序和缩放，末尾追加与 Go2 相同的无噪步态相位。"""
        physical = state.physical
        frame = torch.cat((physical[:, ANGULAR_BODY], physical[:, 1:4], state.command,
                           physical[:, JOINT_OFFSET], physical[:, JOINT_VELOCITY],
                           state.memory[:, PREVIOUS_ACTION]), -1) * self.proprio_scale
        if noisy:
            frame = self.add_proprio_noise(frame)
        if self.cfg.experiment.actor_phase:
            phase = 2 * torch.pi * gait_phase(state, self.dt, self.cfg.reward.gait.period)
            moving = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs() > self.cfg.reward.gait.command_threshold
            frame = torch.cat((frame, torch.stack((phase.sin(), phase.cos()), -1) * moving[:, None]), -1)
        return frame

    def add_proprio_noise(self, frame):
        """独立采样的噪声只覆盖实际配置的本体分量。"""
        if not self.cfg.noise.add_noise:
            return frame
        # 相位不采噪；直接生成想象噪声时同样只消耗原45维的随机数。
        noise = (2 * torch.rand_like(frame[..., :45]) - 1) * self.noise_amplitude
        return torch.cat((frame[..., :45] + noise, frame[..., 45:]), -1)

    def invariant(self, state):
        """去世界平移及偏航的不变量，旋转取前两列按行展平。"""
        _, rotation, velocity, angular, joint, joint_velocity = self.generalized_state(state)
        transpose = rotation.transpose(-1, -2)
        tilt = mm3(yaw_mat(rotation).transpose(-1, -2), rotation)[..., :, :2].reshape(-1, 6)
        return torch.cat((tilt, mv3(transpose, velocity), mv3(transpose, angular), joint, joint_velocity), -1)

    def terrain_scan(self, state):
        """训练侧使用与碰撞同源的 yaw 对齐 17×11 网格，不传给 actor。"""
        position, yaw = self.world_position(state), self.yaw(state)
        x, y = self.scan_points.unbind(-1)
        points = torch.stack((yaw.cos()[:, None] * x - yaw.sin()[:, None] * y,
                              yaw.sin()[:, None] * x + yaw.cos()[:, None] * y), -1) + position[:, None, :2]
        return (position[:, 2:3] - self.surface_height(points)).clamp_max(self.cfg.sensors.height_scan.max_height)

    def physical_features(self, state):
        """保留真实诊断使用的 Go2 兼容派生视图。"""
        return torch.cat((state.physical[:, :70], state.physical[:, 73:125]), -1)

    def privileged(self, state):
        """价值条件包含速度命令、动力学上下文及已有奖励所需的计时和动作记忆。"""
        phase = 2 * torch.pi * gait_phase(state, self.dt, self.cfg.reward.gait.period)
        features = torch.cat((self.invariant(state), state.command, self.simulation_context(state), self.terrain_scan(state),
                              state.memory[:, AIR_TIME], state.memory[:, CONTACT_TIME],
                              phase.sin()[:, None], phase.cos()[:, None], state.memory[:, PREVIOUS_ACTION],
                              state.memory[:, PREVIOUS_PREVIOUS_ACTION], state.memory[:, CONTACT]), -1)
        if self.cfg.commands.heading_command:
            error = state.memory[:, HEADING_TARGET] - self.yaw(state)
            features = torch.cat((features, error.sin()[:, None], error.cos()[:, None],
                                  state.memory[:, STANDING:FORWARD + 1], state.memory[:, COMMAND_TIMER, None],
                                  state.memory[:, COMMAND_OVERRIDE, None]), -1)
        return features

    def failure_features(self, predicted, recorded_target, command):
        """失败监督使用真实终点计时及本次转移的旧指令，其余条件保留预测状态。"""
        memory = predicted.memory.clone()
        memory[:, AIR_TIME] = recorded_target.memory[:, AIR_TIME]
        memory[:, CONTACT_TIME] = recorded_target.memory[:, CONTACT_TIME]
        if self.cfg.commands.heading_command:
            # 真实重采样是外生事件，两路标签条件共享对应记录，不使用窗口起点的旧目标。
            memory[:, HEADING_TARGET:FORWARD + 1] = recorded_target.memory[:, HEADING_TARGET:FORWARD + 1]
            memory[:, COMMAND_TIMER] = recorded_target.memory[:, COMMAND_TIMER]
            memory[:, COMMAND_OVERRIDE] = recorded_target.memory[:, COMMAND_OVERRIDE]
        return self.privileged(TaskState(predicted.physical, memory, command, predicted.context))

    def standing_mask(self, state):
        """采集站立关噪与奖励使用同一命令阈值，转向命令仍属于运动。"""
        return state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs() <= self.cfg.reward.stand_still_command_threshold

    def time_left(self, state):
        return state.memory[:, COMMAND_TIMER]

    # ---------- 速度指令与奖励记忆 ----------

    def _sample_commands(self, batch):
        bounds = self.default_joint_pos.new_tensor(self.cfg.commands.initial_ranges)
        command = bounds[:, 0] + torch.rand(batch, 3, device=self.device) * (bounds[:, 1] - bounds[:, 0])
        standing = torch.rand(batch, device=self.device) < self.cfg.commands.rel_standing_envs
        command = torch.where(standing[:, None], torch.zeros_like(command), command)
        lower, upper = self.cfg.commands.resampling_time
        timer = lower + (upper - lower) * torch.rand(batch, device=self.device)
        return command, timer

    def initialize_state(self, physical, context):
        memory = physical.new_zeros(physical.shape[0], MEMORY_DIM)
        if self.cfg.commands.heading_command:
            command, timer, heading, standing, forward = sample_commands(self.cfg.commands, context, physical[:, 72])
            memory[:, HEADING_TARGET] = heading
            memory[:, STANDING] = standing
            memory[:, FORWARD] = forward
            command = heading_command(self.cfg.commands, command, physical[:, 72], heading, standing)
        else:
            command, timer = self._sample_commands(physical.shape[0])
        memory[:, COMMAND_TIMER] = timer
        return TaskState(physical, memory, command, context)

    def advance(self, state, action, next_physical, *, contact_times=None, resample_commands=False):
        """真实端使用传感器计时，想象沿用 Go2 接触概率持续耦合公式。"""
        contact = next_physical[:, CONTACT_FOUND]
        if contact_times is None:
            previous = state.memory[:, CONTACT]
            contact_stay = torch.minimum(previous, contact) / previous.clamp_min(1e-6)
            air_stay = torch.minimum(1 - previous, 1 - contact) / (1 - previous).clamp_min(1e-6)
            air_time = air_stay * state.memory[:, AIR_TIME] + (1 - contact) * self.dt
            contact_time = contact_stay * state.memory[:, CONTACT_TIME] + contact * self.dt
        else:
            air_time, contact_time = contact_times
            contact = (contact_time > 0).to(action.dtype)
        timer = state.memory[:, COMMAND_TIMER] - self.dt
        heading, standing, forward = state.memory[:, HEADING_TARGET], state.memory[:, STANDING], state.memory[:, FORWARD]
        if self.cfg.commands.heading_command:
            if resample_commands:
                command, timer, heading, standing, forward = advance_commands(self.cfg.commands, state, next_physical, self.dt)
            else:
                # 想象起点已保证整段指令有效，只更新反馈，不采样新目标或消耗采样随机数。
                command = heading_command(self.cfg.commands, state.command, next_physical[:, 72], heading, standing)
                command = torch.where(state.memory[:, COMMAND_OVERRIDE, None].bool(), state.command, command)
        else:
            command = state.command
            if resample_commands:
                sampled, sampled_timer = self._sample_commands(action.shape[0])
                expired = timer <= 0
                command = torch.where(expired[:, None], sampled, command)
                command = torch.where(state.memory[:, COMMAND_OVERRIDE, None].bool(), state.command, command)
                timer = torch.where(expired, sampled_timer, timer)
        memory = torch.cat((action, state.memory[:, ELAPSED_TIME, None] + self.dt, timer[:, None],
                            heading[:, None], standing[:, None], forward[:, None], contact, air_time, contact_time,
                            state.memory[:, COMMAND_OVERRIDE, None], state.memory[:, PREVIOUS_ACTION]), -1)
        return TaskState(next_physical, memory, command, state.context)

    # ---------- Parkour 全状态增量与标签 ----------

    def model_input(self, state, action):
        return torch.cat((self.invariant(state), action, self.simulation_context(state),
                          state.memory[:, PREVIOUS_ACTION], self.terrain_scan(state)), -1)

    def model_target(self, state, next_state):
        """独立预测位置、旋转、线角速度和关节增量，保持 R1 free 积分。"""
        p, rotation, velocity, angular, joint, joint_velocity = self.generalized_state(state)
        p1, rotation1, velocity1, angular1, joint1, joint_velocity1 = self.generalized_state(next_state)
        transpose = rotation.transpose(-1, -2)
        return torch.cat((mv3(yaw_mat(rotation).transpose(-1, -2), p1 - p), so3_log(mm3(transpose, rotation1)),
                          mv3(transpose, velocity1 - velocity), mv3(transpose, angular1 - angular),
                          joint1 - joint, joint_velocity1 - joint_velocity), -1)

    def model_error(self, predicted, target):
        """误差在真实目标状态坐标系表达，与 R1 WorldModel.state_error 一致。"""
        return self.model_target(target, predicted)

    @staticmethod
    def auxiliary(state):
        """四十维额外标签对应原 Go2 奖励需要的加速度、接触力、力矩和角动量。"""
        physical = state.physical
        return torch.cat((physical[:, JOINT_ACCELERATION], physical[:, FOOT_CONTACT_FORCE], physical[:, JOINT_TORQUE],
                          physical[:, LEG_CONTACT_FORCE], physical[:, ANGULAR_MOMENTUM]), -1)

    @staticmethod
    def contact(state):
        return state.physical[:, CONTACT_FOUND]

    def integrate(self, state, action, delta, contact, auxiliary):
        p, rotation, velocity, angular, joint, joint_velocity = self.generalized_state(state)
        dp, dtheta, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 12, 12), -1)
        generalized = (p + mv3(yaw_mat(rotation), dp), mm3(rotation, so3_exp(dtheta)),
                       velocity + mv3(rotation, dv), angular + mv3(rotation, dw), joint + dq, joint_velocity + dqd)
        # physical_state 的独立量布局在 qacc 后插入四足接触，足速由 FK 推导。
        independent = torch.cat((auxiliary[:, :12], contact, auxiliary[:, 12:]), -1)
        return self.advance(state, action, self.physical_state(generalized, independent))
