"""测试用**独立**面积预言。

刻意不 import 被测包 ``dateline_unwrap`` 的任何实现：向量工具、经纬度转换、
面积公式全部在此重写，只用标准库 math。提供两个预言：

1. ``gauss_bonnet_area`` —— 结构性独立预言。Gauss–Bonnet：
   A = 2pi - sum(外角)，外角由相邻边平面法向的 atan2(det, dot) 计算。
   与核心的“扇形三角形角盈求和”是完全不同的推导路径，是主预言。
2. ``fan_signed_area`` —— Oosterom–Strackee 立体角扇形的独立重写
   （atan2 行列式），枢轴 q 由测试方固定且与核心动态选出的枢轴不同，
   并以三个互相不同的固定枢轴交叉验证。它与核心是同一公式族但实现、
   枢轴均独立，作为辅助交叉检查；不调用被测代码得到任何期望值。

（另在 test_area 用解析 pi/2 与平面 eps^2/2 极限把关绝对正确性。）
满足“独立参考不得直接调用被测核心得到期望值，往返一致仅作补充”。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Ring = Sequence[tuple[float, float]]
Vec3 = tuple[float, float, float]


def _unit(lon: float, lat: float) -> Vec3:
    cl = math.cos(lat)
    return (cl * math.cos(lon), cl * math.sin(lon), math.sin(lat))


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vec3) -> Vec3:
    n = math.sqrt(_dot(v, v))
    return (v[0] / n, v[1] / n, v[2] / n)


def _fold_mod_4pi(a: float) -> float:
    """折进 (-2pi, 2pi]，与被测契约的左开右闭一致。"""
    a = math.remainder(a, 4.0 * math.pi)
    if a <= -2.0 * math.pi:
        a = 2.0 * math.pi
    return a


def fan_signed_area(ring: Ring, pivot: Vec3) -> float:
    """Oosterom–Strackee 有符号立体角扇形和（pivot 为任一固定单位向量）。"""
    pts = [_unit(lo, la) for lo, la in ring]
    n = len(pts)
    total = 0.0
    q = pivot
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        det = (
            q[0] * (a[1] * b[2] - a[2] * b[1])
            + q[1] * (a[2] * b[0] - a[0] * b[2])
            + q[2] * (a[0] * b[1] - a[1] * b[0])
        )
        denom = 1.0 + _dot(q, a) + _dot(a, b) + _dot(b, q)
        total += 2.0 * math.atan2(det, denom)
    return _fold_mod_4pi(total)


def gauss_bonnet_area(ring: Ring) -> float:
    """Gauss–Bonnet 外角和公式：A = 2pi - sum exterior angles。"""
    pts = [_unit(lo, la) for lo, la in ring]
    n = len(pts)
    edge_n = [_norm(_cross(pts[i], pts[(i + 1) % n])) for i in range(n)]
    total_turn = 0.0
    for i in range(n):
        v = pts[i]
        n_prev = edge_n[(i - 1) % n]  # 进入 v 的边法向
        n_next = edge_n[i]            # 离开 v 的边法向
        det = _dot(v, _cross(n_prev, n_next))
        total_turn += math.atan2(det, _dot(n_prev, n_next))
    return _fold_mod_4pi(2.0 * math.pi - total_turn)


# 三个互相不同的固定通用枢轴（确定性，不依赖被测代码选择）。
PIVOTS: tuple[Vec3, ...] = (
    _norm((0.17, 0.23, 0.958)),
    _norm((-0.71, 0.37, -0.603)),
    _norm((0.02, -0.88, 0.47)),
)


def oracle_areas(ring: Ring) -> tuple[float, ...]:
    """返回 (fan(q0), fan(q1), fan(q2), gauss-bonnet) 四个独立预言值。"""
    return (
        fan_signed_area(ring, PIVOTS[0]),
        fan_signed_area(ring, PIVOTS[1]),
        fan_signed_area(ring, PIVOTS[2]),
        gauss_bonnet_area(ring),
    )
