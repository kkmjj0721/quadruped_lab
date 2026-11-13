# Copyright (c) 2026-2027 zh
"""内容：将 Go1 真实采集指标转换为通用 SuperMimic 日志的单位与名称。"""

from agents.perceptive_supermimic import PerceptiveRunner


class Go1DepthRunner(PerceptiveRunner):
    def _collection_step_metrics(self):
        metrics = {}
        tracking_names = {"tracking_linear_rmse": "tracking_linear_mse", "tracking_yaw_rmse": "tracking_angular_mse"}
        for name, value in self.env.last_metrics.items():
            if name.startswith("reward/"):
                term = name.removeprefix("reward/")
                # 两种 Go1 任务均已乘控制周期和奖励权重；这里只还原每秒奖励率。
                metrics[f"task/reward/{term}"] = value
                metrics[f"Reward_Rate/{term}"] = value / self.task.dt
            elif name in tracking_names:
                metrics["task/" + tracking_names[name]] = value ** 2
            else:
                metrics["task/" + name] = value
        return metrics
