# Copyright (c) 2026-2027 zh
"""
内容：
    读取 Go1 终止前状态与真实深度，在复制 final 转移后执行部分重置。
    地形课程沿用 Go2 SuperMimic，视觉算法不接触 Go2 的训练入口或策略实现。
"""

from copy import deepcopy

import torch
from mjlab.envs import ManagerBasedRlEnv

from agents.depth_supermimic.interfaces import DepthStep
from src.assets.robots.go1_depth.go1_depth_constants import FOOT_GEOM_NAMES, JOINT_NAMES, ROOT_BODY_NAME
from src.config.go1_depth.go1_depth_config import Go1DepthCfg
from src.tasks.locomotion.go2_supermimic.mdp.curriculums import TerrainCurriculum
from src.tasks.locomotion.go2_supermimic.mdp.reference_state import CONTEXT_COM
from src.tasks.locomotion.go2_supermimic.mdp.state_spec import (
    COMMAND_OVERRIDE, COMMAND_TIMER, CONTEXT_DIM, CONTEXT_ENCODER_BIAS, CONTEXT_FRICTION, CONTEXT_ORIGIN,
    CONTEXT_RANGES, CONTEXT_ZERO_PROBABILITY,
)
from src.tasks.locomotion.go2_supermimic.mdp.terrains import PlaneSurface, TerrainMap, TerrainSurface

from ..env_cfg import make_go1_depth_env_cfg
from ..mdp.task_model import Go1DepthTaskModel
from ..sensors.depth_camera import DepthCameraHistory
from ..sensors.near_clip import NearClipContinuation


class Go1DepthAdapter:
    """真实相机只在控制步末和已重置行更新，严格保留终止前图像。"""

    def __init__(self, cfg=None, *, device="cpu", play=False, render_mode=None,
                 no_terminations=False, video_height=None, video_width=None):
        self.cfg = deepcopy(cfg) if cfg is not None else Go1DepthCfg()
        self.play = play
        self.no_terminations = no_terminations
        if play:
            self.cfg.noise.add_noise = False
            self.cfg.terrain.curriculum = False
        self.map = TerrainMap(self.cfg.terrain)
        env_cfg = make_go1_depth_env_cfg(self.cfg, terrain_map=self.map, play=play)
        if video_height is not None:
            env_cfg.viewer.height = video_height
        if video_width is not None:
            env_cfg.viewer.width = video_width
        self.raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
        try:
            self.robot = self.raw_env.scene["robot"]
            self.device, self.num_envs = self.robot.data.joint_pos.device, self.raw_env.num_envs
            action = self.raw_env.action_manager.get_term("joint_pos")
            if tuple(self.robot.joint_names) != JOINT_NAMES or tuple(action.target_names) != JOINT_NAMES:
                raise ValueError("Go1 asset and action order must match the task specification")
            terrain = self.raw_env.scene.terrain
            self.map.bind(terrain.terrain_origins)
            self.surface = (PlaneSurface(self.device) if self.map.plane else
                            TerrainSurface.from_model(self.raw_env.sim.mj_model, self.device, terrain_map=self.map))
            self.task = Go1DepthTaskModel(self.cfg, self.robot.data.default_joint_pos[0],
                                         self.robot.data.soft_joint_pos_limits[0], self.surface, self.device)
            self.task.bind_env(self.raw_env)
            self.terrain_curriculum = TerrainCurriculum(
                self.map, self.num_envs, self.cfg.terrain.max_init_level, enabled=self.cfg.terrain.curriculum,
                episode_length_s=self.cfg.env.episode_length_s, device=self.device)
            if not self.map.plane:
                self.terrain_curriculum.family_ids.copy_(terrain.terrain_types)
            self.camera = DepthCameraHistory(self.cfg.sensors.depth, self.num_envs, self.device,
                                             noisy=self.cfg.noise.add_noise)
            self.near_clip = NearClipContinuation(self.raw_env, self.cfg.sensors.depth)
            self._state = None
            self._current_observation = None
            self._play_commands = None
            self.control_steps = 0
            self.last_metrics = {}
            self._episode_rewards = {name: torch.zeros(self.num_envs, device=self.device)
                                     for name in self.cfg.to_dict()["reward"]["scales"]}
        except BaseException:
            self.raw_env.close()
            raise

    @property
    def step_dt(self):
        return self.raw_env.step_dt

    def current_state(self):
        if self._state is None:
            raise RuntimeError("Reset the Go1 depth adapter before reading its state")
        return self._state.detached_clone()

    def depth_stack(self):
        return self.camera.stack()

    def proprio_frame(self):
        """返回实际采集时缓存的本体测量，同一状态不重复采噪。"""
        if self._current_observation is None:
            raise RuntimeError("Reset the Go1 depth adapter before reading its observation")
        return self._current_observation.clone()

    def get_state(self):
        """为播放入口提供与真实采集相同的独立状态副本。"""
        return self.current_state()

    @torch.no_grad()
    def set_play_commands(self, command=None):
        """播放时固定速度命令；None 恢复任务自身的命令采样。"""
        if not self.play:
            raise RuntimeError("Command overrides are available only during playback")
        if command is None:
            self._play_commands = None
            if self._state is not None:
                self._state.memory[:, COMMAND_OVERRIDE] = 0
                self._state.memory[:, COMMAND_TIMER] = 0
            return
        command = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if command.shape == (3,):
            command = command.expand(self.num_envs, -1)
        if command.shape != (self.num_envs, 3) or not torch.isfinite(command).all():
            raise ValueError("Playback commands must be finite [3] or [N,3]")
        self._play_commands = command.clone()
        if self._state is not None:
            self._state.command.copy_(self._play_commands)
            self._state.memory[:, COMMAND_OVERRIDE] = 1
            if self._current_observation is not None:
                # 命令分量不加噪、scale 为一；保留已采集的其余本体测量。
                self._current_observation[:, 6:9].copy_(self._play_commands)
                if self.task.cfg.experiment.actor_phase:
                    self._current_observation[:, 45:].copy_(self.task.observe(self._state)[:, 45:])

    # ---------- 回合重置 ----------

    def _context(self, ids):
        context = torch.zeros(ids.numel(), CONTEXT_DIM, device=self.device)
        model = self.raw_env.sim.mj_model
        foot = model.geom(f"robot/{FOOT_GEOM_NAMES[0]}").id
        friction = self.raw_env.sim.model.geom_friction
        context[:, CONTEXT_FRICTION] = friction[ids, foot, 0] if friction.ndim == 3 else friction[foot, 0]
        base = model.body(f"robot/{ROOT_BODY_NAME}").id
        ipos = self.raw_env.sim.model.body_ipos
        nominal = torch.tensor(model.body_ipos[base].copy(), dtype=torch.float32, device=self.device)
        context[:, CONTEXT_COM] = (ipos[ids, base] if ipos.ndim == 3 else ipos[base]) - nominal
        context[:, CONTEXT_ENCODER_BIAS] = self.robot.data.encoder_bias[ids]
        context[:, CONTEXT_ORIGIN] = self.raw_env.scene.env_origins[ids]
        context[:, CONTEXT_RANGES] = context.new_tensor(self.cfg.commands.initial_ranges).flatten()
        context[:, CONTEXT_ZERO_PROBABILITY] = self.cfg.commands.rel_standing_envs
        return context

    def _reset_rows(self, ids):
        if not self.map.plane:
            terrain = self.raw_env.scene.terrain
            terrain.env_origins[ids] = self.terrain_curriculum.origins(ids)
            terrain.terrain_levels[ids] = self.terrain_curriculum.levels[ids]
            terrain.terrain_types[ids] = self.terrain_curriculum.family_ids[ids]
        self.raw_env.reset(env_ids=ids)
        physical = self.task.read_physical(self.raw_env)
        reset = self.task.initialize_state(physical[ids].clone(), self._context(ids))
        if self._play_commands is not None:
            reset.command.copy_(self._play_commands[ids])
            reset.memory[:, COMMAND_OVERRIDE] = 1
        if self._state is None:
            self._state = reset.detached_clone()
        else:
            for name in ("physical", "memory", "command", "context"):
                getattr(self._state, name)[ids] = getattr(reset, name)
        observation = self.task.observe(reset, noisy=self.cfg.noise.add_noise)
        if self._current_observation is None:
            self._current_observation = observation.clone()
        else:
            self._current_observation[ids] = observation
        self.camera.reset(ids)
        raw_depth = self.near_clip.correct(self.raw_env.scene["depth"].data.depth)
        self.camera.append(raw_depth[ids], ids)
        self.terrain_curriculum.reset(ids, self.robot.data.root_link_pos_w[ids, :2])
        for values in self._episode_rewards.values():
            values[ids] = 0

    @torch.no_grad()
    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if self._state is None and env_ids is not None:
            raise RuntimeError("Initialize all environments before a partial reset")
        if ids.numel():
            self._reset_rows(ids)
        return self.current_state()

    # ---------- 终止前真实转移 ----------

    @torch.no_grad()
    def step(self, actions):
        if self._state is None:
            raise RuntimeError("Reset the Go1 depth adapter before stepping")
        if actions.shape != (self.num_envs, 12) or actions.device != self.device or actions.dtype != torch.float32:
            raise ValueError("Go1 actions must be float32 [N,12] on the environment device")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Go1 actions contain non-finite values")
        before = self._state.detached_clone()
        executed = actions.clamp(-20, 20).clone()
        _, _, _, truncated, _ = self.raw_env.step(executed)
        physical = self.task.read_physical(self.raw_env).clone()
        if not torch.isfinite(physical).all():
            raise FloatingPointError("Go1 physical state contains non-finite values")
        sensor = self.raw_env.scene["feet_ground_contact"].data
        final = self.task.advance(before, executed, physical,
                                  contact_times=(sensor.current_air_time.clone(), sensor.current_contact_time.clone()),
                                  resample_commands=True).detached_clone()
        self._current_observation = self.task.observe(final, noisy=self.cfg.noise.add_noise)
        self.camera.append(self.near_clip.correct(self.raw_env.scene["depth"].data.depth))
        final_depth = self.camera.stack()
        final_clean_depth = self.camera.stack(clean=True)
        terminated = (torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) if self.no_terminations else
                      self.task.hard_termination(final))
        truncated = truncated.clone()
        terms = self.task.reward_terms(before, executed, final, terminated.float())
        weighted = {name: self.step_dt * getattr(self.cfg.reward.scales, name) * value for name, value in terms.items()}
        reward = sum(weighted.values()).clone()
        for name, value in weighted.items():
            self._episode_rewards[name].add_(value)
        done_ids = (terminated | truncated).nonzero().flatten()
        self.terrain_curriculum.update(self.raw_env, done_ids, before.command)
        self.terrain_curriculum.advance(before.command, self.robot.data.heading_w, self.step_dt)
        episode_metrics = {}
        if done_ids.numel():
            episode_metrics = {f"episode_reward/{name}": values[done_ids].mean() / self.cfg.env.episode_length_s
                               for name, values in self._episode_rewards.items()}
        self._state = final.detached_clone()
        if done_ids.numel():
            self._reset_rows(done_ids)
        velocity = self.task.tracking_velocity(final)
        self.last_metrics = {f"reward/{name}": value.mean() for name, value in weighted.items()}
        self.last_metrics.update({f"raw_reward/{name}": value.mean() for name, value in terms.items()})
        self.last_metrics.update(episode_metrics)
        self.last_metrics.update(tracking_linear_mse=(velocity[:, :2] - before.command[:, :2]).square().sum(-1).mean(),
                                 tracking_angular_mse=(velocity[:, 2] - before.command[:, 2]).square().mean(),
                                 terrain_level=self.terrain_curriculum.levels.float().mean(),
                                 action_rms=executed.square().mean().sqrt(), base_height=final.physical[:, 0].mean())
        self.control_steps += 1
        return DepthStep(final, self.current_state(), reward, terminated, truncated, final_depth,
                         self.camera.stack(), executed, final_clean_depth, self.proprio_frame())

    def curriculum_state_dict(self):
        return {"terrain": self.terrain_curriculum.state_dict(), "control_steps": self.control_steps}

    def load_curriculum_state_dict(self, state):
        self.terrain_curriculum.load_state_dict(state["terrain"])
        self.control_steps = state["control_steps"]
        self.raw_env.common_step_counter = self.control_steps

    def close(self):
        self.raw_env.close()
