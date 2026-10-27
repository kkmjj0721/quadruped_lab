# Copyright (c) 2026-2027 zh
"""
内容：
    从真实Go2环境提取reset前转移，统一任务记忆、奖励、课程与部分reset时序。
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from agents.supermimic.interfaces import EnvStep, TaskState

from ..config.env_cfgs import (
    resolve_go2_supermimic_cfg, unitree_go2_flat_env_cfg, unitree_go2_rough_env_cfg,
)
from ..mdp.curriculums import CommandCurriculum, TerrainCurriculum
from ..mdp.observations import physical_observation
from ..mdp.state_spec import (
    ACTION_DIM, COMMAND_TIMER, CONTEXT_DIM, CONTEXT_ENCODER_BIAS, CONTEXT_FRICTION,
    ANGULAR_VELOCITY, CONTEXT_ORIGIN, CONTEXT_RANGES, CONTEXT_ZERO_PROBABILITY, JOINT_NAMES,
    LINEAR_VELOCITY, QUATERNION, REWARD_AUXILIARY, heading, quat_apply_inverse,
)
from ..mdp.task_model import Go2TaskModel
from ..mdp.single_task import Go2SingleTaskModel
from ..mdp.reward_sensors import collect_reward_auxiliary
from ..mdp.terrains import PlaneSurface, TerrainMap, TerrainSurface


class Go2TransitionAdapter:
    """连接原生仿真与CollectionEnv，任务规则由本目录的MDP实现。

    单步顺序为物理推进、复制终态、计算任务奖励／失败、更新真实课程、重置结束行。
    final_state保留重置前状态，next_state提供下一次决策输入。
    """

    # ---------- 初始化 ----------

    def __init__(self, cfg=None, *, device="cpu", play=False, render_mode=None,
                 video_height=None, video_width=None, no_terminations=False):
        self.cfg = resolve_go2_supermimic_cfg(cfg, play=play)
        self.single_actor = getattr(self.cfg, "algorithm", "cts") == "single_moe"
        self.play = play
        self.no_terminations = no_terminations
        self.map = TerrainMap(self.cfg.terrain)
        factory = unitree_go2_flat_env_cfg if self.cfg.terrain.mode == "flat" else unitree_go2_rough_env_cfg
        env_cfg = factory(go2_cfg=self.cfg, terrain_map=self.map)
        if no_terminations:
            env_cfg.terminations = {}
        if video_height is not None:
            env_cfg.viewer.height = video_height
        if video_width is not None:
            env_cfg.viewer.width = video_width
        self.raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
        try:
            self.device = next(iter(self.raw_env.scene.entities.values())).data.joint_pos.device
            self.num_envs = self.raw_env.num_envs
            self.robot = self.raw_env.scene["robot"]
            if tuple(self.robot.joint_names) != JOINT_NAMES:
                raise ValueError("Go2 asset joint order differs from the task specification")
            action = self.raw_env.action_manager.get_term("joint_pos")
            if tuple(action.target_names) != JOINT_NAMES:
                raise ValueError("Go2 action order differs from the task specification")
            terrain = self.raw_env.scene.terrain
            self.map.bind(terrain.terrain_origins)
            self.surface = (PlaneSurface(device=self.device) if self.cfg.terrain.mode == "flat" else
                            TerrainSurface.from_model(self.raw_env.sim.mj_model, self.device, terrain_map=self.map))
            task_type = Go2SingleTaskModel if self.single_actor else Go2TaskModel
            self.task = task_type(self.cfg, self.robot.data.default_joint_pos[0].clone(),
                                  self.robot.data.soft_joint_pos_limits[0].clone(), self.surface, device=self.device)
            self.command_curriculum = CommandCurriculum(self.cfg.commands, self.map.family_names, self.device)
            self.terrain_curriculum = TerrainCurriculum(
                self.map, self.num_envs, self.cfg.terrain.max_init_level, enabled=self.cfg.terrain.curriculum,
                episode_length_s=self.cfg.env.episode_length_s, device=self.device)
            self.context_version = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._state = None
            self._play_commands = None
            self.last_metrics = {}
        except BaseException:
            self.raw_env.close()
            raise

    @property
    def step_dt(self):
        return self.raw_env.step_dt

    def get_state(self):
        """提供当前决策态的独立副本，避免评估或教师推理修改采集状态。"""
        if self._state is None:
            raise RuntimeError("Reset the Go2 adapter before reading its state")
        return self._state.detached_clone()

    @torch.no_grad()
    def set_play_commands(self, commands=None):
        """播放时固定各环境的速度命令，None恢复原命令采样；不改变训练配置。"""
        if not self.play:
            raise RuntimeError("Command overrides are available only during playback")
        if commands is None:
            self._play_commands = None
            return
        values = torch.as_tensor(commands, dtype=torch.float32, device=self.device)
        if values.shape == (3,):
            values = values.unsqueeze(0).expand(self.num_envs, -1)
        if values.shape != (self.num_envs, 3) or not torch.isfinite(values).all():
            raise ValueError("Playback commands must be finite [3] or [num_envs,3]")
        self._play_commands = values.clone()
        if self._state is not None:
            self._state.command.copy_(self._play_commands)

    # ---------- 回合重置 ----------

    def _set_origins(self, ids):
        if self.cfg.terrain.mode == "flat":
            return
        terrain = self.raw_env.scene.terrain
        terrain.env_origins[ids] = self.terrain_curriculum.origins(ids)
        # 同步原生可视化元数据，课程计算仍只有当前adapter这一处。
        terrain.terrain_levels[ids] = self.terrain_curriculum.levels[ids] + self.map.row_offset
        terrain.terrain_types[ids] = self.terrain_curriculum.family_ids[ids] + self.map.rings

    def _context(self, ids):
        context = torch.zeros((ids.numel(), CONTEXT_DIM), dtype=torch.float32, device=self.device)
        foot = self.raw_env.sim.mj_model.geom("robot/FL_foot_collision").id
        context[:, CONTEXT_FRICTION] = float(self.raw_env.sim.mj_model.geom_friction[foot, 0])
        context[:, CONTEXT_ENCODER_BIAS] = self.robot.data.encoder_bias[ids]
        context[:, CONTEXT_ORIGIN] = self.raw_env.scene.env_origins[ids]
        bounds, probability = self.command_curriculum.parameters(self.terrain_curriculum.family_ids[ids])
        context[:, CONTEXT_RANGES] = bounds
        context[:, CONTEXT_ZERO_PROBABILITY] = probability.squeeze(-1)
        return context

    def _reset_rows(self, ids):
        self._set_origins(ids)
        self.raw_env.reset(env_ids=ids)
        physical = physical_observation(self.raw_env)
        context = self._context(ids)
        command, remaining = self.task.sample_commands(context)
        if self._play_commands is not None:
            command = self._play_commands[ids].clone()
        memory = torch.zeros((ids.numel(), self.task.spec.memory_dim), dtype=torch.float32, device=self.device)
        memory[:, COMMAND_TIMER] = remaining
        reset_state = TaskState(physical[ids].clone(), memory, command, context)
        if self._state is None:
            self._state = reset_state.detached_clone()
        else:
            for name in ("physical", "memory", "command", "context"):
                getattr(self._state, name)[ids] = getattr(reset_state, name)
        world_xy = self.robot.data.root_link_pos_w[:, :2]
        self.terrain_curriculum.reset_episodes(ids, world_xy)
        self.context_version[ids] += 1

    @torch.no_grad()
    def reset(self, env_ids=None):
        """外部reset不计晋级，保留已加载的课程阶段与逻辑等级。"""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if self._state is None and env_ids is not None:
            raise RuntimeError("Initialize all environments before a partial reset")
        if ids.numel() == 0:
            return self._state.detached_clone()
        self._reset_rows(ids)
        return self._state.detached_clone()

    # ---------- 真实转移 ----------

    @torch.no_grad()
    def step(self, actions):
        if self._state is None:
            raise RuntimeError("Reset the Go2 adapter before stepping")
        if actions.shape != (self.num_envs, ACTION_DIM) or actions.dtype != torch.float32 or actions.device != self.device:
            raise ValueError("Go2 actions must be float32 [N,12] on the environment device")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Go2 actions contain non-finite values")
        before = self._state.detached_clone()
        versions = self.context_version.clone()
        _, _, _, raw_truncated, _ = self.raw_env.step(actions)
        physical = physical_observation(self.raw_env).clone()
        if not torch.isfinite(physical).all():
            raise FloatingPointError("Go2 physical state contains non-finite values")
        final = self.task.advance(before, actions, physical).detached_clone()
        if self._play_commands is not None:
            final.command.copy_(self._play_commands)
        # 在任何部分重置之前复制传感器奖励和接触计时，作为世界模型监督标签。
        if not self.single_actor:
            final.memory[:, REWARD_AUXILIARY] = collect_reward_auxiliary(self.raw_env, before, final, self.task)
        terminated = (torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) if self.no_terminations
                      else self.task.hard_termination(final))
        truncated = raw_truncated.clone()
        terms = self.task.reward_terms(before, actions, final, terminated.float())
        reward = (self.step_dt * sum(getattr(self.cfg.reward.scales, name) * value
                                     for name, value in terms.items())).detach().clone()
        velocity = quat_apply_inverse(final.physical[:, QUATERNION], final.physical[:, LINEAR_VELOCITY])
        linear_error = velocity[:, :2] - before.command[:, :2]
        angular_error = final.physical[:, ANGULAR_VELOCITY][:, 2] - before.command[:, 2]
        # 命令在机体系；累计位移使用动作前航向，不把事后转向应用到历史命令。
        yaw = heading(before.physical[:, QUATERNION])
        vx, vy = before.command[:, 0], before.command[:, 1]
        world_velocity = torch.stack((yaw.cos() * vx - yaw.sin() * vy, yaw.sin() * vx + yaw.cos() * vy), -1)
        self.terrain_curriculum.observe(self.task.world_position(final)[:, :2], world_velocity, self.step_dt)
        if not self.play:
            self.command_curriculum.control_steps += 1
        self._state = final.detached_clone()
        done_ids = (terminated | truncated).nonzero().flatten()
        if done_ids.numel():
            self.terrain_curriculum.complete(done_ids, before.command)
            self._reset_rows(done_ids)
        self.last_metrics = {f"reward/{name}": value.mean() for name, value in terms.items()}
        self.last_metrics.update(terrain_level=self.terrain_curriculum.levels.float().mean(),
                                 command_stage_iteration=float(self.command_curriculum.iteration),
                                 tracking_linear_mse=linear_error.square().sum(-1).mean(),
                                 tracking_angular_mse=angular_error.square().mean(),
                                 action_rms=actions.square().mean().sqrt(),
                                 base_height=self.task.physical_features(final)[:, 0].mean())
        if not self.single_actor:
            self.last_metrics = {name: float(value) for name, value in self.last_metrics.items()}
        return EnvStep(final, self._state.detached_clone(), reward, terminated, truncated, versions)

    # ---------- 课程持久化与资源释放 ----------

    def curriculum_state_dict(self):
        return {"command": self.command_curriculum.state_dict(), "terrain": self.terrain_curriculum.state_dict()}

    def load_curriculum_state_dict(self, state):
        self.command_curriculum.load_state_dict(state["command"])
        self.terrain_curriculum.load_state_dict(state["terrain"])
        self.raw_env.common_step_counter = self.command_curriculum.control_steps

    def close(self):
        self.raw_env.close()


def make_go2_supermimic_env(cfg=None, *, device="cpu", play=False, render_mode=None,
                           video_height=None, video_width=None, no_terminations=False):
    """创建可直接交给SuperMimicRunner的真实Go2采集环境。"""
    return Go2TransitionAdapter(cfg, device=device, play=play, render_mode=render_mode,
                                video_height=video_height, video_width=video_width, no_terminations=no_terminations)
