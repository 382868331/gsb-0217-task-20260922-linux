"""dateline_unwrap —— 球面环线的日期变更线展开。

单位球面闭环（每边为唯一较短大圆弧，顶点经纬度给出）的：

* 有向球面面积（左侧为正，范围 (-2pi, 2pi]）；
* 每条原边与经度 ±180° 半子午线（切缝）的球面交点；
* 分裂为各自连续经度范围的渲染片段；极点交点单独标 ambiguous；
  整段沿切缝的边标 coincident，保留原弧、不造跨缝假连接。

典型用法::

    from dateline_unwrap import unwrap_ring

    result = unwrap_ring([(lon0, lat0), (lon1, lat1), ...])
    print(result.area)
    for edge in result.edges:
        for seg in edge.segments:
            ...

所有角度对外均为弧度。详见 README。
"""

from __future__ import annotations

from .errors import (
    AntipodalEndpointError,
    ComputationBudgetError,
    DegenerateEdgeError,
    InvalidCoordinateError,
    RingSizeError,
    SelfIntersectionError,
    SphericalLoopError,
)
from .models import ArcParams, ArcSegment, RingResult, SplitEdge, Vertex
from .unwrap import MAX_VERTICES, MIN_VERTICES, area_of_ring, unwrap_ring

__all__ = [
    "unwrap_ring",
    "area_of_ring",
    "RingResult",
    "SplitEdge",
    "ArcSegment",
    "ArcParams",
    "Vertex",
    "SphericalLoopError",
    "InvalidCoordinateError",
    "RingSizeError",
    "DegenerateEdgeError",
    "AntipodalEndpointError",
    "SelfIntersectionError",
    "ComputationBudgetError",
    "MIN_VERTICES",
    "MAX_VERTICES",
]

__version__ = "1.0.0"
