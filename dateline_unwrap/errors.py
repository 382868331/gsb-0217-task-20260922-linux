"""可定位的错误类型。

所有由本库主动拒绝的输入都继承 :class:`SphericalLoopError`，
并携带足够定位问题的结构化字段（顶点/边索引、字段名等）。
"""

from __future__ import annotations


class SphericalLoopError(ValueError):
    """球面环线处理错误的公共基类。"""


class InvalidCoordinateError(SphericalLoopError):
    """单个经纬角非法（NaN/Infinity、越界、类型错误等）。

    属性:
        vertex_index: 顶点在输入序列中的下标；-1 表示与具体顶点无关。
        field: ``"lon"`` 或 ``"lat"``。
    """

    def __init__(self, message: str, vertex_index: int = -1, field: str = ""):
        super().__init__(message)
        self.vertex_index = vertex_index
        self.field = field


class RingSizeError(SphericalLoopError):
    """顶点数量不在契约范围 [3, 100] 内，或顶点容器类型错误。"""


class DegenerateEdgeError(SphericalLoopError):
    """相邻顶点重合，唯一较短大圆弧不存在（边长数值为 0）。"""

    def __init__(self, message: str, edge_index: int):
        super().__init__(message)
        self.edge_index = edge_index


class AntipodalEndpointError(SphericalLoopError):
    """相邻顶点对跖，大圆弧不唯一（两条半大圆），契约禁止。"""

    def __init__(self, message: str, edge_index: int):
        super().__init__(message)
        self.edge_index = edge_index


class SelfIntersectionError(SphericalLoopError):
    """非相邻边（或相邻边在公共顶点之外）相交，环线不是简单环线。

    属性:
        edge_i, edge_j: 相交的两条边下标。
        point: 交点在单位球上的 (x, y, z)。
    """

    def __init__(self, message: str, edge_i: int, edge_j: int, point: tuple[float, float, float]):
        super().__init__(message)
        self.edge_i = edge_i
        self.edge_j = edge_j
        self.point = point


class ComputationBudgetError(RuntimeError):
    """确定性构造的有限候选预算耗尽（区别于“输入被证明非法”的拒绝）。

    本库唯一的搜索环节是面积计算中辅助扇形顶点 q 的退化回避：
    候选 q 由固定确定性序列生成。全部候选都数值退化时抛此错误——
    它表示“没能算出来”，不表示输入无解或非法，调用方可改变浮点舍入环境后重试。
    在双精度下该错误实际不应发生；保留它是为了遵守“预算耗尽 ≠ 证明无解”。
    """
