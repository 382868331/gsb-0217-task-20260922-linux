"""单位球面三维向量层。

所有真正的几何运算（点积、叉积、大圆弧上取点、求交、有向面积）
都在单位向量 (x, y, z) 上完成。经纬度只用于：

* 输入校验；
* 精确识别“经度 ±180° 切缝”和“南北极点”这类依赖地图约定的对象；
* 把结果转回经度区间输出。

绝不允许对 (lon, lat) 做平面线性插值——那会在切缝附近制造假交点
（这正是本模块要消除的错误来源）。
"""

from __future__ import annotations

import math
from typing import Final

# 判定容差在 README“误差口径”一节说明：这些 EPS 只用于“相等/共线/
# 落在线段上”这类拓扑判定，绝不用于抹平面积误差。
EPS: Final[float] = 1e-10
# 边长（弧度）小于此值视为重合。
LEN_EPS: Final[float] = 1e-12


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(dot(v, v))


def normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = norm(v)
    return (v[0] / n, v[1] / n, v[2] / n)


def unit(lon_rad: float, lat_rad: float) -> tuple[float, float, float]:
    """经纬度（弧度）-> 单位向量。约定: x=(lon 0,lat 0), y=(lon 90E,lat 0), z=北极。

    显式再归一化一次：相同 (lon,lat) 两次求值必须给出 bit 级一致的单位向量，
    否则 cos²+sin² 的求和误差（约 2e-16）会让重合顶点的角距假性变成 ~1e-8。
    """
    cl = math.cos(lat_rad)
    return normalize((cl * math.cos(lon_rad), cl * math.sin(lon_rad), math.sin(lat_rad)))


def longitude_of(v: tuple[float, float, float]) -> float:
    """单位向量 -> (-pi, pi] 内的经度（弧度）。极点处经度无定义，返回 0.0。"""
    if abs(v[0]) < EPS and abs(v[1]) < EPS:
        return 0.0
    return math.atan2(v[1], v[0])


def latitude_of(v: tuple[float, float, float]) -> float:
    """单位向量 -> 纬度（弧度）。以 z 分量为主，极点附近数值稳定。"""
    z = min(1.0, max(-1.0, v[2]))
    return math.asin(z)


def slerp(a: tuple[float, float, float], b: tuple[float, float, float], t: float,
          length: float | None = None) -> tuple[float, float, float]:
    """较短大圆弧上的插值点（球面线性插值），t in [0, 1]。

    不做经纬插值。端点重合或长度为 0 时退化为常数点。
    """
    if length is None:
        length = math.acos(min(1.0, max(-1.0, dot(a, b))))
    if length < LEN_EPS:
        return a
    s = math.sin(length)
    u = math.sin((1.0 - t) * length) / s
    w = math.sin(t * length) / s
    p = (u * a[0] + w * b[0], u * a[1] + w * b[1], u * a[2] + w * b[2])
    return normalize(p)


def edge_plane_normal(a: tuple[float, float, float],
                      b: tuple[float, float, float]) -> tuple[float, float, float]:
    """大圆弧所在过球心平面的单位法向 n = normalize(a x b)。

    沿弧自 a 向 b 行走时，左侧半球满足 dot(n, p) > 0。
    """
    return normalize(cross(a, b))


def great_circle_intersections(n1: tuple[float, float, float],
                               n2: tuple[float, float, float]
                               ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """两个大圆（由平面法向给出）的两个交点；两平面重合（共面）返回 None。"""
    d = dot(n1, n2)
    if abs(abs(d) - 1.0) < EPS:
        return None
    # ± normalize(n1 x n2)
    v = cross(n1, n2)
    p = normalize(v)
    return (p, (-p[0], -p[1], -p[2]))


def point_on_minor_arc(p: tuple[float, float, float],
                       a: tuple[float, float, float],
                       b: tuple[float, float, float],
                       length: float) -> bool:
    """点 p 是否在 a->b 的较短大圆弧（含端点）上。

    短弧（长度 < pi）恰为平面 span(a,b) 内锥 {lambda*a + mu*b : lambda,mu>=0}
    与单位圆的交。用重心系数符号判定，避免“分别归一化的等点角距 ~1e-8”
    导致 acos 角距和判据误杀端点接触。
    """
    nrm = normalize(cross(a, b))
    if abs(dot(p, nrm)) > EPS:
        return False
    c = min(1.0, max(-1.0, dot(a, b)))
    c2 = c * c
    if c2 > 1.0 - 1e-14:
        # 极短边（含被上游拒绝前的退化情形）：退化为角距和判据。
        d1 = math.acos(min(1.0, max(-1.0, dot(p, a))))
        d2 = math.acos(min(1.0, max(-1.0, dot(p, b))))
        return d1 + d2 <= length + EPS
    pa = dot(p, a)
    pb = dot(p, b)
    lam = (pa - c * pb) / (1.0 - c2)
    mu = (pb - c * pa) / (1.0 - c2)
    return lam >= -1e-9 and mu >= -1e-9
