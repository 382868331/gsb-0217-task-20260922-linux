"""球面环线：有向面积 + 日期变更线（±180° 半子午线）展开。

公开入口只有 :func:`unwrap_ring`（另有便捷函数 :func:`area_of_ring`）。
算法分四步：

1. 严格输入校验（见 :mod:`dateline_unwrap.errors` 的错误语义）；
2. 经纬度 -> 单位向量，全部后续几何在三维单位球面上进行；
3. O(n^2) 简单性检查（n <= 100）：非相邻边相交即拒绝；
4. 球面多边形扇形三角化 + L'Huilier 公式求有向面积（左转/左侧为正）；
   每条边在球面几何上求与切缝半子午线的交点并分裂。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .errors import (
    AntipodalEndpointError,
    ComputationBudgetError,
    DegenerateEdgeError,
    InvalidCoordinateError,
    RingSizeError,
    SelfIntersectionError,
)
from .geometry import (
    EPS,
    LEN_EPS,
    cross,
    dot,
    great_circle_intersections,
    norm,
    normalize,
    point_on_minor_arc,
    slerp,
    unit,
)
from .models import ArcParams, ArcSegment, RingResult, SplitEdge, Vertex

MAX_VERTICES: int = 100
MIN_VERTICES: int = 3
_SEAM_DOT_EPS: float = 1e-12  # 判“就在切缝半子午线上”用更紧的容差
_NEAR_DELTA: float = 1e-9      # 取交点两侧近点推算连续经度侧别
_ANTIPODAL_DOT_EPS: float = 1e-14  # 端点夹角距 pi 的数值判定阈值（见 README 误差口径）


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _is_real_number(x: Any) -> bool:
    # bool 是 int 的子类：按契约必须先排除 bool；NaN/Infinity 不是有限实数。
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float)) and math.isfinite(x)


def _validate_vertices(vertices: Any) -> list[Vertex]:
    if not isinstance(vertices, Sequence) or isinstance(vertices, (str, bytes, bytearray)):
        raise RingSizeError("vertices 必须是顶点序列（list/tuple），每个元素为 (lon, lat)")
    n = len(vertices)
    if not (MIN_VERTICES <= n <= MAX_VERTICES):
        raise RingSizeError(
            f"顶点数必须在 [{MIN_VERTICES}, {MAX_VERTICES}] 之间，收到 {n}"
        )

    out: list[Vertex] = []
    for i, item in enumerate(vertices):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)):
            raise InvalidCoordinateError(
                f"顶点 {i} 必须是 (lon, lat) 二元序列", i, ""
            )
        if len(item) != 2:
            raise InvalidCoordinateError(
                f"顶点 {i} 必须恰好含 2 个分量 (lon, lat)，收到 {len(item)} 个", i, ""
            )
        lon, lat = item
        if not _is_real_number(lon):
            raise InvalidCoordinateError(
                f"顶点 {i} 的经度必须是有限实数（拒绝 bool/NaN/Infinity）", i, "lon"
            )
        if not _is_real_number(lat):
            raise InvalidCoordinateError(
                f"顶点 {i} 的纬度必须是有限实数（拒绝 bool/NaN/Infinity）", i, "lat"
            )
        lon = float(lon)
        lat = float(lat)
        if not (-math.pi <= lon <= math.pi):
            raise InvalidCoordinateError(
                f"顶点 {i} 经度 {lon!r} 超出 [-pi, pi]", i, "lon"
            )
        if not (-math.pi / 2.0 <= lat <= math.pi / 2.0):
            raise InvalidCoordinateError(
                f"顶点 {i} 纬度 {lat!r} 超出 [-pi/2, pi/2]", i, "lat"
            )
        out.append(Vertex(lon, lat))
    return out


# --------------------------------------------------------------------------- #
# 有向面积：扇形三角化 + L'Huilier，左转为正
# --------------------------------------------------------------------------- #
def _signed_triangle_angle(p0, p1, p2) -> float:
    """(p0,p1,p2) 球面三角形的有向角盈（立体角），Oosterom–Strackee 公式。

    tan(Omega/2) = det(p0,p1,p2) / (1 + p0·p1 + p1·p2 + p2·p0)，
    用 atan2 直接求有符号角：对极小三角形无 L'Huilier 四 tan 连乘的
    灾难性抵消（北极 eps 直角三角形可达 eps^2/2 的相对机器精度）。
    """
    det = dot(p0, cross(p1, p2))
    denom = 1.0 + dot(p0, p1) + dot(p1, p2) + dot(p2, p0)
    return 2.0 * math.atan2(det, denom)


# 辅助扇形顶点 q 的确定性候选（无需在多边形内）。优先使用环自身顶点：
# 对很小的多边形这使扇形三角形就是多边形本身，避免大角度项相消吃掉小面积；
# 顶点全部退化（每个顶点都有对跖伙伴等）时退回固定通用候选序列。
def _generic_pivot_stream(points):
    s = (sum(p[0] for p in points), sum(p[1] for p in points), sum(p[2] for p in points))
    if norm(s) > 1e-12:
        yield normalize(s)
    for m in range(64):
        v = (
            math.cos(0.700 * m + 0.300) + 0.5 * math.sin(1.100 * m),
            math.sin(1.300 * m + 0.700),
            math.sin(0.900 * m + 0.100) + 0.3 * math.cos(0.500 * m + 1.700),
        )
        yield normalize(v)


def _pivot_candidates(points):
    for p in points:
        yield ("vertex", p)
    for p in _generic_pivot_stream(points):
        yield ("generic", p)


def _signed_area(points, normals, lengths) -> float:
    """有向球面面积（左转为正）。

    以辅助点 q 做有向扇形三角化：sum_i signed_angle(q, p_i, p_{i+1})，
    每个三角形用 Oosterom–Strackee atan2 行列式公式。q 不必在多边形内部
    （有向绕数恒等），只需不产生退化三角形。候选顺序：环顶点优先，其次
    65 个确定性通用点。全部退化时抛 ComputationBudgetError（“预算耗尽”），
    它与 SelfIntersectionError 等“证明输入非法”的拒绝不同。
    """
    n = len(points)
    for kind, q in _pivot_candidates(points):
        ok = True
        incident = set()
        if kind == "vertex":
            # 找到 q 对应哪个顶点（候选直接取自 points，身份比较即可）。
            qi = next(k for k in range(n) if points[k] is q or dot(points[k], q) > 1.0 - 1e-15)
            incident = {qi, (qi - 1) % n}
            # 顶点枢轴与其它顶点“很近但不重合”是正常的——那正是多边形自身的
            # 三角形；只拒绝 bit 级共线（理论上重合/对跖已在上游拒绝，这里是双保险）。
            for k, p in enumerate(points):
                if k == qi:
                    continue
                if norm(cross(q, p)) == 0.0:
                    ok = False
                    break
        else:
            for p in points:
                if abs(dot(q, p)) > 1.0 - 1e-10:
                    ok = False
                    break
        if not ok:
            continue
        for i in range(n):
            if i in incident:
                continue  # 与 q 共顶点的扇形三角形恒为 0，跳过。
            if lengths[i] >= 1e-8 and abs(dot(q, normals[i])) < 1e-12:
                ok = False  # q 落在边所在大圆上：三角形精确退化。
                break
        if not ok:
            continue

        total = 0.0
        for i in range(n):
            if i in incident:
                continue
            a, b = points[i], points[(i + 1) % n]
            total += _signed_triangle_angle(q, a, b)
        # 球面有向面积只在模 4pi 意义下与扇形和相等：取 (-2pi, 2pi] 内的
        # 唯一代表元。简单环线真实面积必在该区间，折叠不改变任何合法结果。
        two_pi = 2.0 * math.pi
        four_pi = 4.0 * math.pi
        total = math.remainder(total, four_pi)  # 结果在 [-2pi, 2pi]
        if total > two_pi:                       # 数值尾巴
            total = two_pi
        # -2pi 是开区间端点：西向绕完整大圆的真实有向面积恰为 -2pi，
        # 按模 4pi 同余归并到闭端 +2pi（见 README“误差口径/边界约定”）。
        if total <= -two_pi + 1e-12:
            total = two_pi
        return total

    raise ComputationBudgetError(
        "面积辅助点的全部确定性候选（环顶点 + 65 个通用点）都产生退化扇形；"
        "这是预算耗尽，不代表输入无解"
    )


# --------------------------------------------------------------------------- #
# 简单性检查
# --------------------------------------------------------------------------- #
def _check_simple(points, normals, lengths) -> None:
    n = len(points)
    for i in range(n):
        ai, bi = points[i], points[(i + 1) % n]
        ni, li = normals[i], lengths[i]
        for j in range(i + 1, n):
            aj, bj = points[j], points[(j + 1) % n]
            nj, lj = normals[j], lengths[j]
            adjacent = (j == i + 1) or (i == 0 and j == n - 1)
            shared = bi if j == i + 1 else (ai if (i == 0 and j == n - 1) else None)

            ips = great_circle_intersections(ni, nj)
            if ips is None:
                # 共面（同一大圆）：检查两段短弧是否重叠。
                tests = (
                    slerp(ai, bi, 0.5, li), slerp(aj, bj, 0.5, lj),
                    ai, bi, aj, bj,
                )
                for p in tests:
                    if not point_on_minor_arc(p, ai, bi, li) or not point_on_minor_arc(p, aj, bj, lj):
                        continue
                    if adjacent and shared is not None and dot(p, shared) > 1.0 - EPS:
                        continue
                    raise SelfIntersectionError(
                        f"边 {i} 与边 {j} 在同一大圆上重合/重叠", i, j, p
                    )
                continue

            for p in ips:
                if point_on_minor_arc(p, ai, bi, li) and point_on_minor_arc(p, aj, bj, lj):
                    if adjacent and shared is not None and dot(p, shared) > 1.0 - EPS:
                        continue
                    raise SelfIntersectionError(
                        f"非相邻（或退化相邻）边 {i} 与边 {j} 相交", i, j, p
                    )


# --------------------------------------------------------------------------- #
# 切缝分裂
# --------------------------------------------------------------------------- #
def _is_pole(p) -> bool:
    return abs(p[2]) > 1.0 - 1e-9


def _on_seam_half(p) -> bool:
    # 切缝 = x<=0 且 y=0 的半子午线（含两极）。
    return abs(p[1]) <= _SEAM_DOT_EPS and p[0] <= _SEAM_DOT_EPS


def _arc_t(p, a, length: float) -> float:
    t = math.acos(min(1.0, max(-1.0, dot(p, a)))) / length
    return min(1.0, max(0.0, t))


def _near_point(a, b, length: float, t: float, side: int):
    tt = t + side * _NEAR_DELTA
    tt = min(1.0 - _NEAR_DELTA, max(_NEAR_DELTA, tt))
    return slerp(a, b, tt, length)


def _seam_crossing_lon(p_near) -> float:
    """非极点切缝交点处的连续经度极限：必为 +pi 或 -pi，由进入侧决定。"""
    return math.copysign(math.pi, math.atan2(p_near[1], p_near[0]))


def _pole_approach_lon(a, b, length: float, t: float, side: int) -> float:
    """弧在极点交点处沿某一侧趋近时的经度极限。

    弧经过极点时其所在平面含 z 轴，趋近方向即该平面与赤道圆的交线方向；
    用近点向赤道平面投影得到该方向（比直接读近点经度数值稳定）。
    """
    p = _near_point(a, b, length, t, side)
    u = normalize((p[0], p[1], 0.0))
    lon = math.atan2(u[1], u[0])
    # 数值上贴着切缝的非 coplanar 弧（理论上 coplanar 已单独处理）不做吸附：
    # 真实极限是多少就报多少，避免替渲染端改经度。
    return lon


def _segment(i: int, kind: str, t0: float, t1: float, p0, p1,
             lon0: float | None, lon1: float | None, note: str = "",
             pole: str | None = None) -> ArcSegment:
    return ArcSegment(
        edge_index=i, kind=kind, t0=t0, t1=t1,
        lon_a=lon0, lon_b=lon1, point_a=p0, point_b=p1, pole=pole, note=note,
    )


def _split_generic(i: int, a, b, length: float, nrm,
                   va: Vertex, vb: Vertex) -> list[ArcSegment]:
    """边所在大圆不与切缝共面时的分裂。

    与切缝半子午线的交点分两类：
    * 非极点横穿点：弧从东/西半球穿到另一侧，两侧连续经度极限为 ±pi；
    * 极点（弧的大圆恰过极点）：单独标 ambiguous，相邻片段各自按自己
      趋近极点的经度方向收尾，不跨极点拉线。
    """
    seam_n = (0.0, 1.0, 0.0)
    ips = great_circle_intersections(nrm, seam_n)
    order: list[tuple[float, Any, str]] = []
    for p in ips:  # type: ignore[union-attr]
        if _on_seam_half(p) and point_on_minor_arc(p, a, b, length):
            t = _arc_t(p, a, length)
            # 交点与端点重合时吸附到精确的 0/1，避免数值误差切出假碎片。
            if dot(p, a) > 1.0 - EPS:
                t = 0.0
            elif dot(p, b) > 1.0 - EPS:
                t = 1.0
            kind = "pole" if _is_pole(p) else "cross"
            order.append((t, p, kind))
    order.sort(key=lambda e: e[0])

    bounds: list[tuple[float, str | None]] = [(0.0, None)]
    for t, _, kind in order:
        if abs(t - bounds[-1][0]) > 1e-11:
            bounds.append((t, kind))
    if bounds[-1][0] != 1.0:
        bounds.append((1.0, None))

    def boundary_lon(t: float, kind: str | None, into: int) -> float:
        """into=+1: 取 t 处向弧前方 (+eps) 趋近的经度；-1 向后。"""
        if t == 0.0 and not _on_seam_half(a):
            return va.lon
        if t == 1.0 and not _on_seam_half(b):
            return vb.lon
        if kind == "pole":
            return _pole_approach_lon(a, b, length, t, into)
        # 普通横穿点，或端点本身落在切缝上且非极点：按近点侧给 ±pi。
        return _seam_crossing_lon(_near_point(a, b, length, t, into))

    segs: list[ArcSegment] = []
    for k in range(len(bounds) - 1):
        (lo, klo), (hi, khi) = bounds[k], bounds[k + 1]
        p_lo = a if lo == 0.0 else slerp(a, b, lo, length)
        p_hi = b if hi == 1.0 else slerp(a, b, hi, length)
        lon_lo = boundary_lon(lo, klo, +1)
        lon_hi = boundary_lon(hi, khi, -1)
        segs.append(_segment(i, "normal", lo, hi, p_lo, p_hi, lon_lo, lon_hi,
                             note="span"))

    for t, p, kind in order:
        if kind == "pole":
            segs.append(_segment(
                i, "ambiguous", t, t, p, p, None, None,
                note="pole_intersection", pole="north" if p[2] > 0 else "south",
            ))
    segs.sort(key=lambda s: (s.t0, 0 if s.kind != "ambiguous" else 1))
    return segs


def _split_coplanar(i: int, a, b, length: float,
                    va: Vertex, vb: Vertex) -> list[ArcSegment]:
    """边所在大圆恰好就是切缝大圆（±180°/0° 子午线圈）。"""
    # 整段都在切缝半侧（短弧不可能越过极点再折返）：coincident，保留原弧。
    if _on_seam_half(a) and _on_seam_half(b):
        return [_segment(i, "coincident", 0.0, 1.0, a, b, None, None,
                         note="entire_arc_on_seam")]

    north = (0.0, 0.0, 1.0)
    south = (0.0, 0.0, -1.0)
    pole_events: list[tuple[float, Any]] = []
    for pole in (north, south):
        if point_on_minor_arc(pole, a, b, length):
            pole_events.append((_arc_t(pole, a, length), pole))
    pole_events.sort(key=lambda e: e[0])
    ptimes = {t for t, _ in pole_events}

    bounds = [0.0]
    for t, _ in pole_events:
        if abs(t - bounds[-1]) > 1e-11:
            bounds.append(t)
    if bounds[-1] != 1.0:
        bounds.append(1.0)

    def boundary_is_pole(t: float) -> bool:
        return any(abs(t - pt) < 1e-11 for pt in ptimes)

    segs: list[ArcSegment] = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        mid = slerp(a, b, (lo + hi) * 0.5, length)
        p_lo = a if lo == 0.0 else slerp(a, b, lo, length)
        p_hi = b if hi == 1.0 else slerp(a, b, hi, length)
        if mid[0] < -_SEAM_DOT_EPS:
            # 子弧在 ±180° 半子午线上：dateline，不给经度、不跨缝连线。
            segs.append(_segment(i, "dateline", lo, hi, p_lo, p_hi, None, None,
                                 note="subarc_on_seam"))
        else:
            # 0° 子午线一侧：连续经度（趋近极点时极限为 0）。
            lon_lo: float | None
            lon_hi: float | None
            if lo == 0.0 and not _on_seam_half(a):
                lon_lo = va.lon
            elif boundary_is_pole(lo):
                lon_lo = _pole_approach_lon(a, b, length, lo, +1)
            else:
                lon_lo = None  # 端点在切缝侧（与 dateline 子段相接），无 0 侧经度
            if hi == 1.0 and not _on_seam_half(b):
                lon_hi = vb.lon
            elif boundary_is_pole(hi):
                lon_hi = _pole_approach_lon(a, b, length, hi, -1)
            else:
                lon_hi = None
            segs.append(_segment(i, "normal", lo, hi, p_lo, p_hi, lon_lo, lon_hi,
                                 note="span"))

    for t, p in pole_events:
        segs.append(_segment(
            i, "ambiguous", t, t, p, p, None, None,
            note="pole_intersection", pole="north" if p[2] > 0 else "south",
        ))
    segs.sort(key=lambda s: (s.t0, 0 if s.kind != "ambiguous" else 1))
    return segs


def _initial_azimuth(va: Vertex, vb: Vertex) -> float:
    # 经度差必须折叠到 [-pi,pi]：两端分居 +pi/-pi 两侧（同一缝附近）时，
    # 直接相减会得到 ~±2pi 的假差，方位角随之反转。
    dlon = math.remainder(vb.lon - va.lon, 2.0 * math.pi)
    phi1, phi2 = va.lat, vb.lat
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return math.atan2(y, x) % (2.0 * math.pi)


# --------------------------------------------------------------------------- #
# 公开入口
# --------------------------------------------------------------------------- #
def unwrap_ring(vertices: Sequence[tuple[float, float]]) -> RingResult:
    """对一条单位球面闭环求有向面积并按日期变更线展开。

    参数:
        vertices: ``[(lon, lat), ...]``，弧度，3..100 个点，首尾自动闭合。
            每条边取两顶点间**唯一较短大圆弧**；禁止重合/对跖相邻点。

    返回:
        :class:`~dateline_unwrap.models.RingResult`。

    异常（均可定位，不静默纠正）:
        RingSizeError: 顶点数越界或容器类型错误。
        InvalidCoordinateError: 分量缺失、NaN/Infinity、bool、经纬度越界。
        DegenerateEdgeError: 相邻顶点重合。
        AntipodalEndpointError: 相邻顶点对跖。
        SelfIntersectionError: 非相邻边相交（含共面重叠、非相邻顶点接触）。
    """
    vv = _validate_vertices(vertices)
    n = len(vv)
    pts = [unit(v.lon, v.lat) for v in vv]

    lengths: list[float] = []
    normals: list[tuple[float, float, float]] = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        # 近重合/近对跖时 acos(dot) 不稳定（dot=1 可对应 2e-8 弧度的短边），
        # 用 atan2(|a x b|, a.b) 直接给出稳定角距；重合点经 unit() 归一化后
        # bit 级一致，|cross| 恰为 0。
        cr = cross(a, b)
        length = math.atan2(norm(cr), dot(a, b))
        if length < LEN_EPS:
            raise DegenerateEdgeError(
                f"边 {i} 的两个端点重合，唯一较短大圆弧不存在", i
            )
        if math.pi - length < _ANTIPODAL_DOT_EPS:
            raise AntipodalEndpointError(
                f"边 {i} 的两个端点对跖，大圆弧不唯一（两条半大圆），契约禁止", i
            )
        lengths.append(length)
        normals.append(normalize(cr))

    _check_simple(pts, normals, lengths)
    area = _signed_area(pts, normals, lengths)

    edges: list[SplitEdge] = []
    seam_n = (0.0, 1.0, 0.0)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        va, vb = vv[i], vv[(i + 1) % n]
        length, nrm = lengths[i], normals[i]
        coplanar = abs(abs(dot(nrm, seam_n)) - 1.0) < EPS
        if coplanar:
            segs = _split_coplanar(i, a, b, length, va, vb)
        else:
            segs = _split_generic(i, a, b, length, nrm, va, vb)

        arc = ArcParams(
            edge_index=i,
            start=va,
            end=vb,
            length=length,
            azimuth=_initial_azimuth(va, vb),
            plane_normal=nrm,
        )
        edges.append(SplitEdge(arc=arc, segments=tuple(segs)))

    return RingResult(vertices=tuple(vv), edges=tuple(edges), area=area)


def area_of_ring(vertices: Sequence[tuple[float, float]]) -> float:
    """只取有向面积的便捷函数；校验/拒绝语义与 :func:`unwrap_ring` 完全相同。"""
    return unwrap_ring(vertices).area
