# Copyright (c) 2026-2027 zh
"""内容：装配 Go1 视觉播放并维护查看器重复取观测与部分重置时的循环记忆。"""

from pathlib import Path

import torch

from agents.depth_supermimic.config import DepthNetworkConfig
from agents.depth_supermimic.policy import DepthActor
from src.config.go1_depth.go1_depth_config import Go1DepthCfg
from .config_loading import apply_settings, read_settings
from .playback_base import SuperMimicPlayEnv, SuperMimicVideoRecorder, _restore_curriculum

from .environment import make_go1_depth_env
from .training import TASK_MODES


class DepthPlayEnv(SuperMimicPlayEnv):
    """沿用查看器命令与本体历史适配，只在真实新决策帧更新视觉记忆。"""

    def __init__(self, adapter, actor):
        super().__init__(adapter)
        self.actor = actor
        self.hidden = torch.zeros(adapter.num_envs, 128, device=adapter.device)
        self.last_episode = torch.full((adapter.num_envs,), -1, device=adapter.device, dtype=torch.long)
        self.last_control = torch.full_like(self.last_episode, -1)
        self.latent = torch.zeros(adapter.num_envs, 32, device=adapter.device)

    @torch.no_grad()
    def get_observations(self):
        values = super().get_observations()
        depth = self.adapter.depth
        reset = depth.episode_id != self.last_episode
        changed = reset | (depth.control_step != self.last_control)
        if self.actor is not None:
            self.hidden = torch.where(reset[:, None], 0., self.hidden)
            self.hidden, self.latent = self.actor.encode(depth.noisy, self.hidden, depth.new_frame & changed)
        self.last_episode.copy_(depth.episode_id)
        self.last_control.copy_(depth.control_step)
        return {**values, "latent": self.latent}


def make_playback_env(task_id, *, checkpoint_file=None, agent="trained", num_envs=1, device="cuda:0",
                      video=False, video_length=200, video_height=None, video_width=None, no_terminations=False):
    if task_id not in TASK_MODES:
        raise ValueError(f"Unknown Go1 depth task: {task_id}")
    cfg = Go1DepthCfg()
    model, checkpoint = None, None
    if agent == "trained":
        if checkpoint_file is None:
            raise ValueError("Trained depth playback requires a checkpoint")
        checkpoint_path = Path(checkpoint_file).expanduser().resolve()
        settings, complete = read_settings(checkpoint_path.parent / "task_config.json")
        apply_settings(cfg, settings, replace_mappings=complete)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("format") != "depth_cnn_gru_v1":
            raise ValueError("Playback requires a depth CNN-GRU checkpoint")
        model = DepthActor(DepthNetworkConfig(**checkpoint["network_config"])).to(device).eval()
        model.load_state_dict(checkpoint["actor"])
    elif agent not in ("zero", "random"):
        raise ValueError(f"Unsupported playback agent: {agent}")
    cfg.terrain.mode = TASK_MODES[task_id]
    cfg.env.num_envs = num_envs if num_envs is not None else 1
    adapter = make_go1_depth_env(cfg, device=device, play=True, render_mode="rgb_array" if video else None,
                                 video_height=video_height, video_width=video_width, no_terminations=no_terminations)
    try:
        if checkpoint is not None:
            # 复用现有逐行映射，播放环境数量变化时仍恢复保存的地形家族与难度。
            _restore_curriculum(adapter, checkpoint)
        env = DepthPlayEnv(adapter, model)
        env.reset()

        @torch.no_grad()
        def policy(observations):
            if model is not None:
                return model(observations["history"], observations["latent"]).detach()
            shape = (adapter.num_envs, 12)
            return torch.zeros(shape, device=device) if agent == "zero" else 2 * torch.rand(shape, device=device) - 1

        if video:
            if checkpoint_file is None:
                raise ValueError("Depth video output requires a checkpoint directory")
            env = SuperMimicVideoRecorder(env, video_folder=Path(checkpoint_file).parent / "videos" / "play",
                                          step_trigger=lambda step: step == 0, video_length=video_length,
                                          disable_logger=True)
        return env, policy
    except BaseException:
        adapter.close()
        raise
