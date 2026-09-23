"""固定输入演示（离线、确定性、无第三方依赖）。

运行::

    python demo.py

内容：
1. 正常结果：八分之一球三角形（解析面积 pi/2）及其随机轴旋转后的面积不变；
2. 正常结果：跨日期变更线的三角形，逐条边打印分裂后的连续经度渲染片段；
3. 实际触发的拒绝边界：蝴蝶结四边形非相邻边自交，抛出可定位的
   SelfIntersectionError（另演示对跖邻边被拒）。

全部结果当场计算，不 sleep、不读预录数据、不做压力测试（亚秒级）。
"""

from __future__ import annotations

import math

from dateline_unwrap import (
    AntipodalEndpointError,
    SelfIntersectionError,
    unwrap_ring,
)
from dateline_unwrap.geometry import normalize, unit


def deg(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    return math.radians(lon_deg), math.radians(lat_deg)


def rodrigues(v, axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    return normalize((
        v[0] * (c + axis[0] ** 2 * (1 - c))
        + v[1] * (axis[0] * axis[1] * (1 - c) - axis[2] * s)
        + v[2] * (axis[0] * axis[2] * (1 - c) + axis[1] * s),
        v[0] * (axis[1] * axis[0] * (1 - c) + axis[2] * s)
        + v[1] * (c + axis[1] ** 2 * (1 - c))
        + v[2] * (axis[1] * axis[2] * (1 - c) - axis[0] * s),
        v[0] * (axis[2] * axis[0] * (1 - c) - axis[1] * s)
        + v[1] * (axis[2] * axis[1] * (1 - c) + axis[0] * s)
        + v[2] * (c + axis[2] ** 2 * (1 - c)),
    ))


def to_lonlat(v):
    return math.atan2(v[1], v[0]), math.asin(min(1.0, max(-1.0, v[2])))


def fmt_lon(x):
    return "  None " if x is None else f"{math.degrees(x):8.3f}°"


def print_segments(result, title: str) -> None:
    print(f"\n{title}")
    print(f"有向面积 = {result.area:.12f} rad²（左侧为正）")
    print("边  片段类型      t0->t1      连续经度起   连续经度止   极点标注")
    for edge in result.edges:
        for s in edge.segments:
            pole = s.pole or "-"
            print(
                f"{edge.arc.edge_index:>2}  {s.kind:<11} "
                f"{s.t0:6.4f}->{s.t1:<6.4f} "
                f"{fmt_lon(s.lon_a)}  {fmt_lon(s.lon_b)}  {pole:<5} {s.note}"
            )


def main() -> None:
    print("=" * 72)
    print("1) 解析参考：八分之一球三角形 (0°,0°)->(90°E,0°)->(0°,90°N)")
    print("=" * 72)
    octant = [deg(0, 0), deg(90, 0), deg(0, 90)]
    r = unwrap_ring(octant)
    print(f"有向面积 = {r.area:.15f} rad²")
    print(f"解析值   = {math.pi / 2:.15f} rad²")
    print(f"绝对误差 = {abs(r.area - math.pi / 2):.3e} rad²")
    print(f"反向环面积 = {unwrap_ring(list(reversed(octant))).area:.15f}（方向反转取负）")

    axis = normalize((0.3, -0.5, 0.8))
    angle = 1.23456
    rotated = [to_lonlat(rodrigues(unit(lo, la), axis, angle)) for lo, la in octant]
    r_rot = unwrap_ring(rotated)
    print(f"绕任意轴旋转 {angle:.6f} rad 后面积 = {r_rot.area:.15f}")
    print(f"旋转前后面积差 = {abs(r_rot.area - r.area):.3e}（球面旋转等距不变量）")

    print("\n" + "=" * 72)
    print("2) 跨日期变更线三角形：(170°E,10°N)->(170°W,10°N)->(180°,60°N)")
    print("=" * 72)
    cross_ring = [deg(170, 10), deg(-170, 10), (-math.pi, math.radians(60))]
    print_segments(unwrap_ring(cross_ring), "分裂片段（连续经度，禁止跨 ±180° 拉直线）：")
    print("说明：边 0 在切缝处裂为 [170°E -> 180°] 与 [-180° -> 170°W] 两段，")
    print("      两段端点经度分别为 +180° 与 -180°，渲染端不得把它们直接相连。")

    print("\n" + "=" * 72)
    print("3) 过北极的边：(170°E,70°N)->(10°W,70°N)，极点交点标 ambiguous")
    print("=" * 72)
    pole_ring = [deg(170, 70), deg(-10, 70), deg(-90, 10)]
    print_segments(unwrap_ring(pole_ring), "分裂片段：")

    print("\n" + "=" * 72)
    print("4) 实际触发的拒绝边界（当场计算，非预录）")
    print("=" * 72)
    bowtie = [deg(0, 0), deg(20, 0), deg(0, 20), deg(20, 20)]
    try:
        unwrap_ring(bowtie)
    except SelfIntersectionError as exc:
        print("蝴蝶结四边形被拒绝（非相邻边自交）：")
        print(f"  {type(exc).__name__}: {exc}")
        print(f"  相交边下标 = {exc.edge_i}, {exc.edge_j}（可定位）")
        print(f"  交点单位球坐标 = ({exc.point[0]:.6f}, {exc.point[1]:.6f}, {exc.point[2]:.6f})")

    try:
        unwrap_ring([deg(0, 0), deg(180, 0), deg(10, 10)])
    except AntipodalEndpointError as exc:
        print("对跖邻边被拒绝（大圆弧不唯一）：")
        print(f"  {type(exc).__name__}: {exc}（edge_index={exc.edge_index}）")

    print("\n演示结束。运行测试：python -m unittest discover -s tests -v")


if __name__ == "__main__":
    main()
