# Copyright (c) 2026-2027 zh
"""内容：在 MuJoCo 原生窗口和 Viser 中显示 Go1 策略的最新深度帧。"""

import mujoco
import numpy as np
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


class _PolicyDepthImage:
    """按相机采集序号刷新图像，切换环境时立即显示对应帧。"""

    def __init__(self):
        self.key = None
        self.rgb = None

    def update(self, env, env_idx):
        capture, depth = env.get_depth_frame(env_idx)
        key = (env_idx, capture)
        if key == self.key:
            return False
        gray = np.rint(np.clip(depth.cpu().numpy() + 0.5, 0, 1) * 255).astype(np.uint8)
        # 固定灰度量程，最近邻放大保留原像素。
        enlarged = gray.repeat(3, axis=0).repeat(3, axis=1)
        self.rgb = enlarged[..., None].repeat(3, axis=2)
        self.key = key
        return True


class Go1DepthNativeViewer(NativeMujocoViewer):
    """在原生窗口左下角叠加策略深度图。"""

    def setup(self):
        super().setup()
        self._depth_image = _PolicyDepthImage()

    def sync_env_to_viewer(self):
        super().sync_env_to_viewer()
        if self.viewer.is_running() and self._depth_image.update(self.env.unwrapped, self.env_idx):
            image = self._depth_image.rgb
            height, width = image.shape[:2]
            self.viewer.set_images((mujoco.MjrRect(12, 12, width, height), image))


class Go1DepthViserViewer(ViserPlayViewer):
    """在浏览器面板展示当前所选环境的策略深度。"""

    def setup(self):
        super().setup()
        self._depth_image = _PolicyDepthImage()
        env = self.env.unwrapped
        self._depth_image.update(env, self._scene.env_idx)
        far = env.adapter.cfg.sensors.depth.far
        with self._server.gui.add_folder("Go1 深度相机", order=0):
            self._depth_panel = self._server.gui.add_image(
                self._depth_image.rgb, label=f"最新深度帧（0–{far:g} m，近黑远白，无回波为白）", format="png",
            )
            self._server.gui.add_markdown("落足点颜色 · 左前红 / 右前青 / 左后黄 / 右后紫")

    def sync_env_to_viewer(self):
        super().sync_env_to_viewer()
        if self._depth_image.update(self.env.unwrapped, self._scene.env_idx):
            self._depth_panel.image = self._depth_image.rgb
