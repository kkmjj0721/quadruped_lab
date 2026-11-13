# Copyright (c) 2026-2027 zh
"""内容：用局部边向量计算 MJWarp 3.11.0 碰撞三角形的投影仿射坐标。

依据：上游 collision_gjk.py 的 _tri_affine_coord 与 _epa_witness。
保持非退化三角形上的主轴投影语义，减少大坐标行列式相减的浮点消减。
这是进程级修复，首次启用必须早于 Warp 初始化、场景构造及 CUDA 图捕获。
"""

from importlib.metadata import version

import warp as wp
from warp._src import context as wp_context
from mujoco_warp._src import collision_gjk


@wp.func
def _relative_tri_affine_coord(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> wp.vec3:
    e0 = v2 - v1
    e1 = v3 - v1
    relative = p - v1
    normal = wp.cross(e0, e1)
    x = int(0)
    y = int(1)
    if wp.abs(normal[0]) >= wp.abs(normal[1]) and wp.abs(normal[0]) >= wp.abs(normal[2]):
        x = 1
        y = 2
    elif wp.abs(normal[1]) >= wp.abs(normal[2]):
        x = 0
        y = 2
    # xz 投影行列式与法向 y 分量异号，按所选坐标直接计算。
    determinant = e0[x] * e1[y] - e0[y] * e1[x]
    second = (relative[x] * e1[y] - relative[y] * e1[x]) / determinant
    third = (e0[x] * relative[y] - e0[y] * relative[x]) / determinant
    return wp.vec3(1.0 - second - third, second, third)


def configure_mjwarp_affine() -> None:
    """在锁定依赖的首次 Warp 初始化前启用；升级后须重新核对上游实现。"""
    if version("mujoco-warp") != "3.11.0" or version("warp-lang") != "1.18.0":
        raise RuntimeError("MJWarp affine correction requires mujoco-warp 3.11.0 and warp-lang 1.18.0; review upstream before upgrading.")
    if collision_gjk._tri_affine_coord is _relative_tri_affine_coord:
        return
    if wp_context.runtime is not None:
        raise RuntimeError("MJWarp affine correction must be configured before Warp initialization; restart in a fresh process.")
    collision_gjk._tri_affine_coord = _relative_tri_affine_coord
