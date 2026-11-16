# Copyright (c) 2026-2027 zh
"""内容：装配 Go1 rough 任务，协调异步回合重置与每5控制步的深度采集。"""

from copy import deepcopy
import math

import torch

from agents.depth_supermimic.interfaces import DepthObservation
from src.assets.robots.go1_depth.go1_depth_constants import ROOT_BODY_NAME
from src.config.go1_depth.go1_depth_config import Go1DepthCfg

from .depth_camera import ScheduledDepthCamera, depth_samples
from .env_cfg import make_go1_supermimic_env_cfg, validate_config
from .task import Go1DepthTask
from .transition_adapter import Go1TransitionAdapter


def validate_go1_depth_config(cfg):
    """构造仿真前校验固定网络协议与深度噪声的外部边界。"""
    validate_config(cfg)
    if (cfg.env.num_observations, cfg.env.num_critic_obs, cfg.experiment.actor_phase) != (47, 262, True):
        raise ValueError("Go1 depth requires 47 observations, 262 privileged features and gait phase")
    depth = cfg.sensors.depth
    if (depth.height, depth.width, depth.period) != (64, 64, 5):
        raise ValueError("Go1 depth requires 64x64 images and a five-control-step camera period")
    if not all(math.isfinite(value) for value in (depth.near, depth.far, depth.noise_std, depth.dropout_probability)):
        raise ValueError("Depth range and corruption settings must be finite")
    if not 0 <= depth.near < depth.far or depth.noise_std < 0 or not 0 <= depth.dropout_probability <= 1:
        raise ValueError("Depth range, noise standard deviation or dropout probability is invalid")
    if not math.isclose(cfg.sim.timestep * cfg.control.decimation, 0.02, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Go1 depth requires a 0.02-second control timestep")


class Go1DepthEnv(Go1TransitionAdapter):
    """本体任务与视觉采集均在 Go1 目录维护，视觉状态按环境时钟推进。"""

    root_body_name = ROOT_BODY_NAME

    def _resolve_config(self, cfg, play):
        result = deepcopy(cfg) if cfg is not None else Go1DepthCfg()
        if play:
            result.noise.add_noise = False
            result.terrain.curriculum = False
            result.sensors.depth.noise_std = 0.0
            result.sensors.depth.dropout_probability = 0.0
        validate_go1_depth_config(result)
        return result

    def _make_env_config(self):
        config = make_go1_supermimic_env_cfg(self.cfg, terrain_map=self.map)
        if self.play:
            del config.terminations["time_out"]
        return config

    def _make_task(self):
        return Go1DepthTask(self.cfg, self.robot.data.default_joint_pos[0].clone(),
                            self.robot.data.soft_joint_pos_limits[0].clone(), self.surface, device=self.device)

    def __init__(self, cfg=None, *, device="cuda:0", **kwargs):
        super().__init__(cfg, device=device, **kwargs)
        try:
            self.camera = ScheduledDepthCamera(self.raw_env, self.cfg)
            self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._episode_id = torch.full_like(self._phase, -1)
            self._frame_index = torch.full_like(self._phase, -1)
            self._control_step = torch.zeros_like(self._phase)
            self._pending_resets = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.depth = None
        except BaseException:
            self.raw_env.close()
            raise

    def _reset_rows(self, ids):
        super()._reset_rows(ids)
        self._phase[ids] = 0
        self._episode_id[ids] += 1
        self._frame_index[ids] = -1
        self._control_step[ids] = 0
        self._pending_resets[ids] = True

    @torch.no_grad()
    def _observe_depth(self, fresh):
        ids = fresh.nonzero().flatten()
        if self.depth is None:
            shape = (self.num_envs, 1, self.cfg.sensors.depth.height, self.cfg.sensors.depth.width)
            clean, noisy = (torch.zeros(shape, device=self.device) for _ in range(2))
            valid = torch.zeros(shape, device=self.device, dtype=torch.bool)
        elif ids.numel():
            clean, noisy, valid = self.depth.clean.clone(), self.depth.noisy.clone(), self.depth.valid.clone()
        else:
            clean, noisy, valid = self.depth.clean, self.depth.noisy, self.depth.valid
        if ids.numel():
            clean[ids], noisy[ids], valid[ids] = depth_samples(self.camera.capture(ids), self.cfg.sensors.depth)
            self._frame_index[ids] += 1
        self.depth = DepthObservation(clean, noisy, valid, fresh.clone(), self._phase.clone(), self._episode_id.clone(),
                                      self._frame_index.clone(), self._control_step.clone())
        self._pending_resets.zero_()

    @torch.no_grad()
    def reset(self, env_ids=None):
        state = super().reset(env_ids)
        self._observe_depth(self._pending_resets.clone())
        return state

    @torch.no_grad()
    def step(self, actions):
        result = super().step(actions)
        self._phase = torch.where(self._pending_resets, 0, (self._phase + 1) % self.cfg.sensors.depth.period)
        self._control_step = torch.where(self._pending_resets, 0, self._control_step + 1)
        self._observe_depth((self._phase == 0) | self._pending_resets)
        return result


def make_go1_depth_env(cfg=None, *, device="cuda:0", play=False, render_mode=None,
                       video_height=None, video_width=None, no_terminations=False):
    """创建同步无附加延迟的 Go1 深度任务；step 返回重置前物理转移。"""
    return Go1DepthEnv(cfg, device=device, play=play, render_mode=render_mode, video_height=video_height,
                       video_width=video_width, no_terminations=no_terminations)
