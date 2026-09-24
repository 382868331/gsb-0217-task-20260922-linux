"""Clip a convex spherical polygon against a great-circle hemisphere.

Public entry points: :func:`clip_loop_hemisphere` (radians) and
:func:`clip_loop_hemisphere_degrees`.  Pure and stateless like
:func:`antimeridian.core.unfold_loop`: all validation finishes before any
result is built, so a call either returns a complete
:class:`~antimeridian.models.ClippedLoop` or raises and changes nothing.

Method: Sutherland-Hodgman on the sphere.  Every edge is the unique short
great-circle arc between its endpoints; a short arc whose endpoints both lie
strictly outside the kept hemisphere stays outside (every arc point is a
positive combination of the endpoints), so clipping edge-by-edge against the
clip plane is exact.  Edge/circle intersections are found in 3-D by
intersecting the edge's great-circle plane with the clip plane -- never by
lon/lat interpolation.
"""

from __future__ import annotations

import math

from . import geometry as g
from .core import _units_scale, _validate_loop, _as_point
from .errors import NonConvexLoopError
from .models import ClippedLoop, GeoPoint

CLIP_EPS = 1e-12   # |x.n| below this counts as "on the clip great circle"
TURN_EPS = 1e-9    # turning-angle tolerance for the convexity test
INSIDE_EPS = 1e-9  # tolerance for the point-in-convex-polygon tie-break


# --------------------------------------------------------------------------- #
# Convexity
# --------------------------------------------------------------------------- #
def _convex_orientation(xyz) -> int:
    """+1 for a left-turning (ccw) convex loop, -1 for a clockwise one.

    Raises :class:`NonConvexLoopError` at the first vertex whose turn sign
    contradicts the orientation established by earlier vertices.  The loop
    is already known to be simple and non-degenerate, so equal turn signs
    are exactly the convexity condition.
    """
    n = len(xyz)
    orient = 0
    for i in range(n):
        t = g.turning_angle(xyz[i - 1], xyz[i], xyz[(i + 1) % n])
        if t > TURN_EPS:
            s = 1
        elif t < -TURN_EPS:
            s = -1
        else:
            continue  # (nearly) straight vertex: compatible with both
        if orient == 0:
            orient = s
        elif s != orient:
            raise NonConvexLoopError(i)
    return orient if orient else 1


def _inside_convex(x, edge_normals, orient: int) -> bool:
    """True iff unit point x is in the (closed) convex polygon interior."""
    return all(
        orient * g.dot(en, x) >= -INSIDE_EPS for en in edge_normals
    )


# --------------------------------------------------------------------------- #
# Spherical Sutherland-Hodgman
# --------------------------------------------------------------------------- #
def _circle_hit(p, q, nvec):
    """Intersection of the short arc p->q with the clip great circle, or None.

    The clip great circle is the plane through the origin with normal
    ``nvec``; intersecting it with the arc's great-circle plane gives two
    antipodal candidates, of which at most one lies on the *short* arc.
    """
    m = g.cross(p, q)
    axis = g.cross(m, nvec)
    if g.norm(axis) < 1e-14:
        return None  # arc's great circle IS the clip circle (edge on circle)
    x = g.normalize(axis)
    alpha = g.central_angle(p, q)
    for cand in (x, (-x[0], -x[1], -x[2])):
        t = g.arc_parameter(p, q, alpha, cand)
        if t is not None and -g.T_EPS < t < 1.0 + g.T_EPS:
            return cand
    return None


def _clip_ring_xyz(xyz, nvec):
    """Sutherland-Hodgman clip of the vertex list against x.nvec >= 0."""
    s = [g.dot(v, nvec) for v in xyz]
    out = []
    m = len(xyz)
    for i in range(m):
        j = (i + 1) % m
        p, q = xyz[i], xyz[j]
        in_p = s[i] >= -CLIP_EPS
        in_q = s[j] >= -CLIP_EPS
        if in_q:
            if not in_p:
                hit = _circle_hit(p, q, nvec)
                if hit is not None:
                    out.append(hit)
            out.append(q)
        elif in_p:
            hit = _circle_hit(p, q, nvec)
            if hit is not None:
                out.append(hit)
        # both outside: a short arc between two outside points stays
        # outside, so nothing is emitted.
    return out


def _dedup(ring):
    """Drop consecutive (incl. wrap-around) duplicate unit vectors."""
    out = []
    for v in ring:
        if out and g.dot(out[-1], v) >= 1.0 - g.DUP_VERTEX_EPS:
            continue
        out.append(v)
    while len(out) > 1 and g.dot(out[0], out[-1]) >= 1.0 - g.DUP_VERTEX_EPS:
        out.pop()
    return out


def _split_antipodal_gaps(ring, nvec, edge_normals, orient):
    """Split any antipodal consecutive pair by a midpoint on the clip circle.

    The clip result is convex, so an antipodal pair can only occur on the
    boundary chain that runs along the clip great circle; the half-circle
    arc between them would not be a unique short arc.  The midpoint of the
    semicircle lying inside the polygon is inserted, keeping every output
    edge strictly shorter than pi.  (Defensive: generic inputs never trigger
    this, but the output contract is enforced rather than assumed.)
    """
    if len(ring) < 2:
        return ring
    out = []
    m = len(ring)
    for i in range(m):
        cur = ring[i]
        out.append(cur)
        nxt = ring[(i + 1) % m]
        if g.dot(cur, nxt) <= -1.0 + g.ANTIPODAL_EPS:
            mid = g.normalize(g.cross(nvec, cur))
            if not _inside_convex(mid, edge_normals, orient):
                mid = (-mid[0], -mid[1], -mid[2])
            out.append(mid)
    return out


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def clip_loop_hemisphere(points, pole, *, units: str = "rad") -> ClippedLoop:
    """Intersect a convex spherical polygon with a great-circle hemisphere.

    Parameters
    ----------
    points:
        Iterable of 3..100 ``(lon, lat)`` pairs (or :class:`GeoPoint`),
        implicitly closed, forming a **convex** simple loop.  Angles are
        radians by default; pass ``units="deg"`` for degrees.
    pole:
        ``(lon, lat)`` of the clip hemisphere's centre; the kept region is
        ``{x : x . pole >= 0}`` (the boundary great circle is included).
    units:
        ``"rad"`` (default) or ``"deg"``.

    Returns
    -------
    ClippedLoop
        ``status`` is ``"full"`` / ``"empty"`` / ``"clipped"``; ``ring`` holds
        the clipped ring vertices (empty when ``"empty"``); ``area`` is the
        non-negative intersection area and ``signed_area`` its oriented
        counterpart (sign follows the input winding).

    Raises
    ------
    InvalidInputError
        Bad count/type/range, NaN/Infinity, bool, zero-length edge, or an
        invalid ``pole``/``units``.
    AntipodalAdjacentError
        Consecutive vertices are (near) antipodal.
    SelfIntersectionError
        Two non-adjacent edges cross or overlap.
    DegenerateLoopError
        All vertices lie on one great circle.
    NonConvexLoopError
        The loop has a reflex vertex; clipping requires convexity.
    """
    pts, xyz, alphas = _validate_loop(points, units)
    pl = _as_point(pole, "pole", _units_scale(units))
    n = len(pts)
    nvec = g.lonlat_to_xyz(pl.lon, pl.lat)
    orient = _convex_orientation(xyz)

    svals = [g.dot(v, nvec) for v in xyz]
    if all(s >= -CLIP_EPS for s in svals):
        # No vertex outside: every short edge stays inside, nothing is cut.
        signed = g.signed_area(xyz)
        return ClippedLoop(
            pole=pl, status="full", ring=tuple(pts),
            area=abs(signed), signed_area=signed,
        )

    ring = _dedup(_clip_ring_xyz(xyz, nvec))
    if len(ring) >= 2:
        edge_normals = [
            g.cross(xyz[i], xyz[(i + 1) % n]) for i in range(n)
        ]
        ring = _split_antipodal_gaps(ring, nvec, edge_normals, orient)
    if len(ring) < 3:
        return ClippedLoop(
            pole=pl, status="empty", ring=(), area=0.0, signed_area=0.0
        )
    signed = g.signed_area(ring)
    if abs(signed) <= 1e-12:
        # Only a measure-zero touch on the clip circle remains.
        return ClippedLoop(
            pole=pl, status="empty", ring=(), area=0.0, signed_area=0.0
        )
    ring_pts = tuple(GeoPoint(*g.xyz_to_lonlat(v)) for v in ring)
    return ClippedLoop(
        pole=pl, status="clipped", ring=ring_pts,
        area=abs(signed), signed_area=signed,
    )


def clip_loop_hemisphere_degrees(points, pole) -> ClippedLoop:
    """Convenience wrapper: longitudes/latitudes given in degrees."""
    return clip_loop_hemisphere(points, pole, units="deg")
