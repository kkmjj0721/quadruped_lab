# Copyright (c) 2026-2027 zh
"""内容：回补 MJWarp 3.11.0 完整 implicit 的 RNE 导数符号修复。

依据：google-deepmind/mujoco_warp PR1663，提交 d8a241ef1643295ebb85bfdfdf9171360bfb2c0e。
上游使 RNE 导数始终加到积分矩阵：M - dt*df/dv = M + dt*d(qfrc_bias)/dv。
这里包装已有实现，不改安装包；这是进程级修复，须在创建仿真／捕获 CUDA 图之前调用。
已捕获旧图不会自动修正，必须重新创建环境与图。
"""

from importlib.metadata import version

from mujoco_warp._src import derivative


_original_deriv_rne_vel = derivative.deriv_rne_vel


def _positive_rne_derivative(model, data, output, flg_subtract=False):
    # 旧forward传True；与上游修复一样始终加RNE项，其余导数计算沿用安装版本。
    return _original_deriv_rne_vel(model, data, output, flg_subtract=False)


def configure_mjwarp_implicit() -> None:
    """只针对已验证的锁定版本启用；升级依赖时必须复核并移除此回补。"""
    if version("mujoco-warp") != "3.11.0":
        raise RuntimeError("MGDP implicit backport requires mujoco-warp 3.11.0; review upstream PR1663 before upgrading.")
    derivative.deriv_rne_vel = _positive_rne_derivative
