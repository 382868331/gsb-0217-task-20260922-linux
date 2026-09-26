"""Public orchestration: validate a closed loop and unfold it at the seam.

The single entry point is :func:`unfold_loop` (radians) with a thin
:func:`unfold_loop_degrees` wrapper.  It is a pure, stateless function:
validation completes before any result is built, so a call either returns a
fully consistent :class:`~antimeridian.models.UnfoldedLoop` or raises and
changes nothing.

Unwrapping strategy
-------------------
Per edge we first derive everything in *canonical* coordinates
(``lon in (-pi, pi]``), locating real seam intersections and pole events in
3-D.  A second pass walks the loop once and assigns continuous *unwrapped*
longitudes, adding ``+/- 2*pi`` only at genuine interior seam crossings.
Regular shared vertices therefore carry one globally consistent longitude;
a vertex sitting exactly on the seam whose incident arcs leave on opposite
canonical sides is the limiting case of a seam crossing (a zero-length seam
piece) and is allowed to show ``+pi`` on one side and ``-pi`` on the other.
Pole joins are never connected: they are emitted ``ambiguous``.
"""

from __future__ import annotations

import math

from . import geometry as g
from .errors import (
    DegenerateLoopError,
    InvalidInputError,
    SelfIntersectionError,
)
from .models import (
    AmbiguousMarker,
    ArcParam,
    Fragment,
    GeoPoint,
    RegionMoments,
    SeamCrossing,
    UnfoldedLoop,
)

MIN_VERTICES = 3
MAX_VERTICES = 100
POLE_LAT_EPS = 1e-9
SEAM_LON_EPS = 1e-9
UNKNOWN = object()


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def _as_point(value, index: int, scale: float = 1.0) -> GeoPoint:
    if isinstance(value, GeoPoint):
        lon0, lat0 = value.lon, value.lat
    else:
        if isinstance(value, (str, bytes)):
            raise InvalidInputError(
                f"vertex {index} must be a (lon, lat) pair, not a string",
                f"vertex {index}",
            )
        try:
            lon0, lat0 = value
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(
                f"vertex {index} must be a (lon, lat) pair",
                f"vertex {index}",
            ) from exc
    # Validate the raw numbers first (bool, NaN, Infinity rejected) and only
    # then convert degrees -> radians.
    lon0 = g.finite_number(lon0, f"vertex {index}.lon")
    lat0 = g.finite_number(lat0, f"vertex {index}.lat")
    lon, lat = lon0 * scale, lat0 * scale
    if lat < -math.pi / 2 - 1e-12 or lat > math.pi / 2 + 1e-12:
        raise InvalidInputError(
            f"vertex {index}.lat={lat:.12g} outside [-pi/2, pi/2]",
            f"vertex {index}.lat",
        )
    if lon < -math.pi - 1e-12 or lon > math.pi + 1e-12:
        raise InvalidInputError(
            f"vertex {index}.lon={lon:.12g} outside [-pi, pi]",
            f"vertex {index}.lon",
        )
    lat = min(math.pi / 2, max(-math.pi / 2, lat))
    lon = ((lon + math.pi) % (2.0 * math.pi)) - math.pi
    if lon <= -math.pi:
        lon = math.pi
    return GeoPoint(lon, lat)


def _is_pole(point: GeoPoint) -> bool:
    return abs(abs(point.lat) - math.pi / 2) <= POLE_LAT_EPS


def _is_on_seam(point: GeoPoint) -> bool:
    return not _is_pole(point) and abs(point.lon - math.pi) <= SEAM_LON_EPS


def _nearest_branch(canon_lon: float, reference: float) -> float:
    k = round((reference - canon_lon) / (2.0 * math.pi))
    return canon_lon + 2.0 * math.pi * k


def _approach_sign(p, q, alpha, t, toward) -> int:
    """Canonical longitude sign just off t: +1 near +pi, -1 near -pi."""
    for eps in (1e-7, 1e-6, 1e-5, 1e-4):
        tt = min(1.0, max(0.0, t + toward * eps))
        if tt in (0.0, 1.0):
            continue
        lon, _ = g.xyz_to_lonlat(g.slerp(p, q, alpha, tt))
        if abs(abs(lon) - math.pi) > 1e-7:
            return 1 if lon >= 0 else -1
    return 1


# Samples used only to track the *integer winding* (a +/- 2*pi offset) along
# an exact-SLERP arc.  Seam intersection positions themselves come from the
# closed-form plane construction, never from this grid, so no planar
# interpolation and no synthetic intersection can arise.
_PATH_SAMPLES = 1024


def _unwrapped_path(vs, ve, alpha, anchor_start: float):
    """Continuous longitude along a normal arc as a piecewise-linear table.

    Samples exact SLERP points, canonicalizes longitude and adds an integral
    +/- 2*pi whenever two adjacent samples straddle the seam (|jump| > pi).
    The whole table is shifted so its first longitude equals ``anchor_start``.
    Returns a list of ``(t, lon)`` with ``lon[0] == anchor_start``.
    """
    prev = None
    table = []
    offset = 0.0
    for k in range(_PATH_SAMPLES + 1):
        t = k / _PATH_SAMPLES
        lon, _ = g.xyz_to_lonlat(g.slerp(vs, ve, alpha, t))
        if prev is not None:
            d = lon - prev
            if d > math.pi:
                offset -= 2.0 * math.pi
            elif d < -math.pi:
                offset += 2.0 * math.pi
        table.append((t, lon + offset))
        prev = lon
    shift = anchor_start - table[0][1]
    return [(t, lon + shift) for t, lon in table]


def _path_lon_at(table, t: float) -> float:
    x = t * _PATH_SAMPLES
    k = int(math.floor(x))
    if k >= _PATH_SAMPLES:
        return table[-1][1]
    frac = x - k
    return table[k][1] * (1.0 - frac) + table[k + 1][1] * frac


def _edge_descriptor(i, ps: GeoPoint, pe: GeoPoint, vs, ve, alpha):
    """Classify edge i in canonical coordinates.

    Returns a dict; the unwrapped pass consumes it.
    """
    desc = {"index": i, "ps": ps, "pe": pe, "vs": vs, "ve": ve,
            "alpha": alpha, "kind": "normal", "pole": None,
            "seam_hit": None}

    if g.arc_is_seam_coincident(vs, ve, alpha):
        desc["kind"] = "coincident"
        return desc

    crossing = g.interior_pole(vs, ve, alpha)
    if crossing is not None:
        desc["kind"] = "pole_cross"
        desc["pole"] = crossing  # (name, t)
        return desc

    if _is_pole(ps) or _is_pole(pe):
        desc["kind"] = "pole_vertex"
        return desc

    hits = [h for h in g.seam_plane_hits(vs, ve, alpha) if not h[2]]
    if hits:
        t, cut_lat, _ = hits[0]
        direction = g.seam_crossing_direction(vs, ve, alpha, t)
        desc["seam_hit"] = (t, cut_lat, direction)
    return desc


def _unwrapped_fragments(desc, lon_start):
    """Build unwrapped fragments for a descriptor given its start longitude.

    ``lon_start`` is None for an edge whose start vertex is a pole.  Returns
    ``(fragments, crossings, lon_end)`` where ``lon_end`` is the regular end
    vertex longitude (or None at a pole).
    """
    i = desc["index"]
    ps, pe = desc["ps"], desc["pe"]
    vs, ve, alpha = desc["vs"], desc["ve"], desc["alpha"]

    if desc["kind"] == "coincident":
        return ([Fragment(i, 0, math.pi, math.pi, ps.lat, pe.lat,
                          kind="coincident")],
                [], math.pi)

    if desc["kind"] == "pole_cross":
        name, t = desc["pole"]
        pole_lat = math.pi / 2 if name == "north" else -math.pi / 2
        lon_s = math.pi if _is_on_seam(ps) else ps.lon
        lon_e = math.pi if _is_on_seam(pe) else pe.lon
        fr = [
            Fragment(i, 0, lon_s, lon_s, ps.lat, pole_lat,
                     kind="ambiguous", pole=name),
            Fragment(i, 1, lon_e, lon_e, pole_lat, pe.lat,
                     kind="ambiguous", pole=name),
        ]
        end = None if _is_pole(pe) else lon_e
        return fr, [], end

    if desc["kind"] == "pole_vertex":
        if _is_pole(ps):
            regular, pole_end = pe, False
            name = "north" if ps.lat > 0 else "south"
            lon = math.pi if _is_on_seam(pe) else pe.lon
            fr = [Fragment(i, 0, lon, lon, ps.lat, pe.lat,
                           kind="ambiguous", pole=name)]
            return fr, [], lon
        regular, name = ps, ("north" if pe.lat > 0 else "south")
        lon = math.pi if _is_on_seam(ps) else ps.lon
        fr = [Fragment(i, 0, lon, lon, ps.lat, pe.lat,
                       kind="ambiguous", pole=name)]
        return fr, [], None

    # normal arc
    if _is_on_seam(ps):
        # A seam vertex's two incident arcs may leave on opposite canonical
        # sides (limiting case of a seam crossing); each edge takes the
        # branch dictated by its own departure geometry.
        u_start = math.pi * _approach_sign(vs, ve, alpha, 0.0, +1)
    elif lon_start is None or lon_start is UNKNOWN:
        u_start = ps.lon
    else:
        u_start = _nearest_branch(ps.lon, lon_start)

    # Continuous longitude is read off the exact-SLERP path (the grid only
    # fixes the integer +/- 2*pi winding); this distinguishes a true seam cut
    # from a short arc whose endpoints sit on opposite canonical sides but
    # which actually goes around via the 0-meridian (no intersection).
    table = _unwrapped_path(vs, ve, alpha, u_start)

    hit = desc["seam_hit"]
    if hit is None:
        u_end = table[-1][1]
        if _is_on_seam(pe):
            branch = math.pi * _approach_sign(vs, ve, alpha, 1.0, -1)
            if abs(abs(u_end) - math.pi) > 1e-7:
                u_end = branch
        return ([Fragment(i, 0, u_start, u_end, ps.lat, pe.lat,
                          kind="normal")], [], u_end)

    t, cut_lat, direction = hit
    # The cut point is physically on the seam, so its longitude is exactly
    # +/- pi; only the integer 2*pi frame (accumulated unwrap offset) is read
    # from the continuous path table.  Interpolating longitude near a pole
    # would otherwise be inaccurate, since longitude changes rapidly there.
    eps = 1.0 / (_PATH_SAMPLES * 4)
    near = 0.5 * (_path_lon_at(table, max(0.0, t - eps))
                  + _path_lon_at(table, min(1.0, t + eps)))
    seam0 = direction * math.pi
    frame = round((near - seam0) / (2.0 * math.pi))
    cut_lon = seam0 + 2.0 * math.pi * frame
    u_end = table[-1][1]
    fr = [
        Fragment(i, 0, u_start, cut_lon, ps.lat, cut_lat, kind="normal"),
        Fragment(i, 1, cut_lon, u_end, cut_lat, pe.lat, kind="normal"),
    ]
    crossing = SeamCrossing(i, t, cut_lat, seam0)
    return fr, [crossing], u_end


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _validate_loop(points, units: str):
    """Shared validation for the public entry points.

    Returns ``(pts, xyz, alphas)``: validated points, unit vectors and
    per-edge central angles.  Raises before returning anything on bad
    input, so callers never see partial state.
    """
    if units == "deg":
        scale = math.pi / 180.0
    elif units == "rad":
        scale = 1.0
    else:
        raise InvalidInputError('units must be "rad" or "deg"', "units")

    try:
        raw = list(points)
    except TypeError as exc:
        raise InvalidInputError(
            "points must be an iterable of (lon, lat) pairs", "vertices"
        ) from exc

    if not (MIN_VERTICES <= len(raw) <= MAX_VERTICES):
        raise InvalidInputError(
            f"vertex count {len(raw)} outside [{MIN_VERTICES}, {MAX_VERTICES}]",
            "vertices",
        )

    pts = [_as_point(v, k, scale) for k, v in enumerate(raw)]
    xyz = [g.lonlat_to_xyz(p.lon, p.lat) for p in pts]
    n = len(pts)
    alphas = [g.prepare_edge(xyz[i], xyz[(i + 1) % n], i) for i in range(n)]

    # Reject non-adjacent self-intersection / overlap (closed-form, complete:
    # two great-circle planes have at most two common points, both tested).
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            kind = g.arcs_intersect(
                xyz[i], xyz[(i + 1) % n], alphas[i],
                xyz[j], xyz[(j + 1) % n], alphas[j],
            )
            if kind != "none":
                raise SelfIntersectionError(i, j, kind)

    if g.great_circle_degenerate(xyz):
        raise DegenerateLoopError(
            "all vertices lie on one great circle; signed area undefined",
            "loop",
        )
    return pts, xyz, alphas


def unfold_loop(points, *, units: str = "rad") -> UnfoldedLoop:
    """Validate and unfold one closed spherical loop.

    Parameters
    ----------
    points:
        Iterable of 3..100 ``(lon, lat)`` pairs (or :class:`GeoPoint`).
        The loop is closed implicitly: an edge from the last vertex back to
        the first is added.  Angles are radians by default; pass
        ``units="deg"`` for degrees.
    units:
        ``"rad"`` (default) or ``"deg"``.

    Returns
    -------
    UnfoldedLoop
        Original arc parameters, continuous-longitude fragments, real seam
        crossings, pole markers and the signed area.

    Raises
    ------
    InvalidInputError
        Bad count/type/range, NaN/Infinity, bool, or zero-length edge.
    AntipodalAdjacentError
        Consecutive vertices are (near) antipodal.
    SelfIntersectionError
        Two non-adjacent edges cross or overlap.
    DegenerateLoopError
        All vertices lie on one great circle (signed area undefined).
    """
    pts, xyz, alphas = _validate_loop(points, units)
    area = g.signed_area(xyz)
    n = len(pts)

    descs = [
        _edge_descriptor(i, pts[i], pts[(i + 1) % n],
                         xyz[i], xyz[(i + 1) % n], alphas[i])
        for i in range(n)
    ]

    # Global unwrapped pass: propagate one longitude per regular vertex.
    vertex_lon = [UNKNOWN] * n
    for k, p in enumerate(pts):
        if _is_pole(p):
            vertex_lon[k] = None
    fragments, crossings, markers = [], [], []
    for i, desc in enumerate(descs):
        start_lon = vertex_lon[i]
        fr, cr, end_lon = _unwrapped_fragments(desc, start_lon)
        fragments.extend(fr)
        crossings.extend(cr)
        k = (i + 1) % n
        if vertex_lon[k] is UNKNOWN:
            vertex_lon[k] = end_lon

    # Pole events: interior crossings and pole *vertices* (reported on the
    # arriving edge, t=1.0), each once.
    for desc in descs:
        if desc["kind"] == "pole_cross":
            name, t = desc["pole"]
            lat = math.pi / 2 if name == "north" else -math.pi / 2
            markers.append(AmbiguousMarker(desc["index"], name, t, lat))
    for k, p in enumerate(pts):
        if _is_pole(p):
            inc = (k - 1) % n
            already = any(
                m.edge_index == inc and m.t >= 1.0 - 1e-9 for m in markers
            )
            if not already:
                markers.append(AmbiguousMarker(
                    inc, "north" if p.lat > 0 else "south", 1.0, p.lat))
    markers.sort(key=lambda m: (m.edge_index, m.t))

    arcs = [
        ArcParam(
            edge_index=i, start=pts[i], end=pts[(i + 1) % n],
            central_angle=alphas[i],
            pole_coincident=descs[i]["kind"] == "coincident",
            contains_pole=descs[i]["kind"] == "pole_cross",
        )
        for i in range(n)
    ]

    winding = "ccw" if area > 0 else ("cw" if area < 0 else "zero")
    return UnfoldedLoop(
        arcs=tuple(arcs),
        fragments=tuple(fragments),
        seam_crossings=tuple(crossings),
        ambiguous_markers=tuple(markers),
        signed_area=area,
        area_winding=winding,
    )


def unfold_loop_degrees(points) -> UnfoldedLoop:
    """Convenience wrapper: longitudes/latitudes given in degrees."""
    return unfold_loop(points, units="deg")


# Threshold below which a first-moment vector is treated as zero (no
# well-defined centroid direction).  A region inside an open hemisphere
# always has |M| > 0; a zero moment requires a balanced (e.g. symmetric)
# region.
ZERO_MOMENT_EPS = 1e-12


def region_moments(points, *, units: str = "rad") -> RegionMoments:
    """Signed 3-D area moments of the region enclosed by a simple loop.

    Same input contract and error semantics as :func:`unfold_loop`; the
    loop is validated identically (NaN/Infinity/bool rejected, antipodal
    adjacent edges, self-intersection and great-circle degeneracy all
    raise before anything is computed).

    Returns
    -------
    RegionMoments
        ``signed_area`` (same value as :func:`unfold_loop`), the signed
        first-moment vector ``M = integral of x dA``, the signed symmetric
        second-moment matrix ``S = integral of x x^T dA`` and, when
        ``|M|`` is nonzero, the normalized direction centroid ``M/|M|``.

    Notes
    -----
    Everything is integrated on 3-D unit vectors, so loops crossing the
    date line need no special handling.  Reversing the loop negates the
    area and both moments.  ``trace(S) == signed_area`` and under a rigid
    rotation ``R`` the moments transform as ``R M`` and ``R S R^T``.
    """
    pts, xyz, alphas = _validate_loop(points, units)
    area, mom, sec = g.signed_moments(xyz)
    mag = g.norm(mom)
    centroid = None
    if mag > ZERO_MOMENT_EPS:
        centroid = (mom[0] / mag, mom[1] / mag, mom[2] / mag)
    winding = "ccw" if area > 0 else ("cw" if area < 0 else "zero")
    return RegionMoments(
        signed_area=area,
        area_winding=winding,
        first_moment=mom,
        second_moment=sec,
        direction_centroid=centroid,
    )


def region_moments_degrees(points) -> RegionMoments:
    """Convenience wrapper: longitudes/latitudes given in degrees."""
    return region_moments(points, units="deg")
