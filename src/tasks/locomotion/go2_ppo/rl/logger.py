# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026-2027 zh
"""
内容：
    沿用 RSL-RL 的 Go2 PPO 指标口径，以固定顺序输出每轮训练日志。
"""

import datetime
from pathlib import Path
import statistics
import sys

import torch
from rsl_rl.utils.log_writer import LogWriter
from rsl_rl.utils.logger import Logger


class VelocityTrainingLogger(Logger):
  """继承采样统计及日志后端，只在任务内维护有序的逐轮输出。"""

  def log(
    self,
    it: int,
    start_it: int,
    total_it: int,
    collect_time: float,
    learn_time: float,
    loss_dict: dict,
    learning_rate: float,
    action_std: torch.Tensor,
    rnd_weight: float | None,
    print_minimal: bool = False,
    width: int = 80,
    pad: int = 40,
  ) -> None:
    """保留参考项目的字段顺序和统计方式，重定向输出使用纯文本。"""
    if self.writer is None:
      return

    collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
    iteration_time = collect_time + learn_time
    self.tot_timesteps += collection_size
    self.tot_time += iteration_time
    fps = int(collection_size / iteration_time)

    def row(label: str, value: str) -> str:
      return f"{label + ':':>{pad}} {value}"

    # 与参考项目一致，按首次出现的顺序合并本轮各步的日志键。
    extra_lines = []
    for key in dict.fromkeys(key for info in self.ep_extras for key in info):
      samples = [torch.as_tensor(info[key], device=self.device).detach().float().reshape(-1)
                 for info in self.ep_extras if key in info]
      samples = [sample for sample in samples if sample.numel()]
      if not samples:
        continue
      value = torch.cat(samples).mean().item()
      tag = key if "/" in key else "Episode/" + key
      self.writer.add_scalar(tag, value, it)
      extra_lines.append(row(key if "/" in key else f"Mean episode {key}", f"{value:.4f}"))

    for key, value in loss_dict.items():
      self.writer.add_scalar(f"Loss/{key}", value, it)
    self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
    mean_std = action_std.mean().item()
    self.writer.add_scalar("Policy/mean_std", mean_std, it)
    self.writer.add_scalar("Perf/total_fps", fps, it)
    self.writer.add_scalar("Perf/collection_time", collect_time, it)
    self.writer.add_scalar("Perf/learning_time", learn_time, it)

    reward_lines = []
    if self.rewbuffer:
      if self.cfg["algorithm"].get("rnd_cfg"):
        for label, tag, buffer in (("extrinsic", "Rnd/mean_extrinsic_reward", self.erewbuffer),
                                   ("intrinsic", "Rnd/mean_intrinsic_reward", self.irewbuffer)):
          value = statistics.mean(buffer)
          self.writer.add_scalar(tag, value, it)
          reward_lines.append(row(f"Mean {label} reward", f"{value:.2f}"))
        self.writer.add_scalar("Rnd/weight", rnd_weight, it)
      for label, tag, buffer in (("Mean reward", "Train/mean_reward", self.rewbuffer),
                                 ("Mean episode length", "Train/mean_episode_length", self.lenbuffer)):
        value = statistics.mean(buffer)
        self.writer.add_scalar(tag, value, it)
        if self.logger_type != "WandbLogWriter":
          self.writer.add_scalar(tag + "/time", value, int(self.tot_time))
        reward_lines.append(row(label, f"{value:.2f}"))

    title = f" Learning iteration {it}/{total_it} ".center(width)
    if sys.stdout.isatty():
      title = f"\033[1m{title}\033[0m"
    lines = ["#" * width, title, ""]
    if self.cfg.get("run_name"):
      lines.append(row("Run name", self.cfg["run_name"]))
    lines.extend([
      row("Total steps", str(self.tot_timesteps)),
      row("Steps per second", str(fps)),
      row("Collection time", f"{collect_time:.3f}s"),
      row("Learning time", f"{learn_time:.3f}s"),
    ])
    lines.extend(row(f"Mean {key} loss", f"{value:.4f}") for key, value in loss_dict.items())
    lines.extend(reward_lines)
    lines.append(row("Mean action std", f"{mean_std:.2f}"))
    if not print_minimal:
      lines.extend(extra_lines)
    done_it = it + 1 - start_it
    remaining_it = total_it - start_it - done_it
    eta = self.tot_time / done_it * remaining_it
    lines.extend([
      "-" * width,
      row("Iteration time", f"{iteration_time:.2f}s"),
      row("Time elapsed", str(datetime.timedelta(seconds=int(self.tot_time)))),
      row("ETA", str(datetime.timedelta(seconds=int(eta)))),
      "",
    ])
    print("\n".join(lines), flush=True)

    if isinstance(self.writer, LogWriter) and self.log_dir is not None:
      for video in Path(self.log_dir).rglob("*.mp4"):
        self.writer.save_video(video, it)
    self.ep_extras.clear()
