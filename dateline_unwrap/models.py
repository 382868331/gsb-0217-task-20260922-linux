"""公开数据结构（契约先于实现固定）。

角度约定
--------
* 对外输入输出的经纬度一律使用**弧度**。
* 经度合法域 [-pi, pi]（pi 与 -pi 是切缝同一条半子午线）；
  纬度合法域 [-pi/2, pi/2]。
* 有向面积：沿顶点顺序行走、**左侧为正**（逆时针为正），
  取值范围 (-2*pi, 2*pi]。

片段分类
--------
每个 :class:`ArcSegment` 的 ``kind`` 取以下之一：

* ``"normal"``      —— 不跨缝的连续经度片段，lon_a/lon_b 均为真实经度；
* ``"dateline"``    —— 沿经度 ±180° 切缝走向的片段（端点落在切缝上且
  弧本身与切缝共面），lon_a/lon_b 为 None，不得跨缝连线；
* ``"ambiguous"``   —— 端点恰好是南北极：该点经度无定义，单片段退化为
  极点单点标注，渲染端必须自行决定如何与相邻片段连接，本库不替它拉线；
* ``"coincident"``  —— 整条原弧与切缝重合（输入边本身就是 ±180° 子午线段）。
  保留原弧参数原样输出，不生成任何跨缝连接。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vertex:
    """经纬度顶点（弧度）。"""

    lon: float
    lat: float

    def degrees(self) -> tuple[float, float]:
        return (math.degrees(self.lon), math.degrees(self.lat))


@dataclass(frozen=True)
class ArcParams:
    """原弧（唯一较短大圆弧）的几何参数。

    属性:
        edge_index: 边在环线中的下标，边 i 从顶点 i 到顶点 i+1（末边回到顶点 0）。
        start, end: 端点经纬度（弧度）。
        length: 弧长（弧度，0 < length < pi）。
        azimuth: 起点处大圆弧初始方位角（弧度，自正北顺时针，0=北, pi/2=东），
            起点为极点时方位角依赖地图经度、无内蕴意义，保留计算值。
        plane_normal: 弧所在大圆平面的单位法向 (x, y, z)；
            沿弧行走方向的左侧半球 dot(plane_normal, p) > 0。
    """

    edge_index: int
    start: Vertex
    end: Vertex
    length: float
    azimuth: float
    plane_normal: tuple[float, float, float]


@dataclass(frozen=True)
class ArcSegment:
    """一条边被切缝分裂后的一个渲染片段。

    属性:
        edge_index: 所属原边下标。
        kind: "normal" | "dateline" | "ambiguous" | "coincident"，见模块文档。
        t0, t1: 片段在原弧参数 t in [0, 1] 上的起止位置，t0 < t1
            （ambiguous 单点片段 t0 == t1）。
        lon_a, lon_b: normal 片段两端的**连续展开经度**（弧度，可超出
            [-pi, pi]），从 lon_a 单调连续走到 lon_b 即为该片段；
            其余种类为 None。
        point_a, point_b: 片段端点在单位球上的 (x, y, z)。
        pole: ambiguous 片段的极点标注，"north" / "south" / None。
        note: 机器可读的补充标记，如 "pole_intersection"、"endpoint"。
    """

    edge_index: int
    kind: str
    t0: float
    t1: float
    lon_a: float | None
    lon_b: float | None
    point_a: tuple[float, float, float]
    point_b: tuple[float, float, float]
    pole: str | None = None
    note: str = ""


@dataclass(frozen=True)
class SplitEdge:
    """一条原边的完整描述：原弧参数 + 有序分裂片段。"""

    arc: ArcParams
    segments: tuple[ArcSegment, ...]


@dataclass(frozen=True)
class RingResult:
    """:func:`dateline_unwrap.unwrap_ring` 的完整返回。

    属性:
        vertices: 校验并归一化后的顶点（经度折叠到 [-pi, pi]）。
        edges: 每条原边的参数与分裂片段，顺序与输入边一致。
        area: 有向球面面积（弧度^2），左转为正，范围 (-2*pi, 2*pi]。
        segments: 全部片段按边顺序的扁平视图，便于渲染端直接消费。
    """

    vertices: tuple[Vertex, ...]
    edges: tuple[SplitEdge, ...]
    area: float

    @property
    def segments(self) -> tuple[ArcSegment, ...]:
        out: list[ArcSegment] = []
        for e in self.edges:
            out.extend(e.segments)
        return tuple(out)
