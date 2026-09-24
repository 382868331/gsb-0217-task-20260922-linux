"""Standalone independent reference used ONLY by the test-suite.

This module deliberately does **not** import :mod:`antimeridian`.  Expected
values are computed with methods that differ from the core's implementation:

* signed area  -- the Van Oosterom solid-angle fan formula
                  (``2*atan2(det, 1 + p.q + q.r + r.p)`` per fan triangle),
                  whereas the core sums Gauss-Bonnet turning angles;
* seam events  -- dense exact-SLERP sampling of the defining condition
                  ``y == 0 and x <= 0``, whereas the core intersects
                  great-circle planes algebraically;
* pole events  -- additive-arc-length test, whereas the core inverts the
                  SLERP basis.

Agreement of two independently derived answers is the point; round-trip
consistency alone is never used as a correctness oracle.
"""

from __future__ import annotations

import math

# Dense reference grid.  Only used to detect the *number and direction* of
# seam events, never to assert tight numerical cut positions.
SAMPLES = 6_000


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def to_xyz(lon, lat):
    cl = math.cos(lat)
    return (cl * math.cos(lon), cl * math.sin(lon), math.sin(lat))


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def slerp(p, q, alpha, t):
    s = math.sin(alpha)
    A = math.sin((1 - t) * alpha) / s
    B = math.sin(t * alpha) / s
    v = (A * p[0] + B * q[0], A * p[1] + B * q[1],
         A * p[2] + B * q[2])
    m = norm(v)
    return (v[0] / m, v[1] / m, v[2] / m)


def central_angle(p, q):
    return math.acos(max(-1.0, min(1.0, dot(p, q))))


# --------------------------------------------------------------------------- #
# Independent signed area (Van Oosterom oriented solid angle, fan from v0)
# --------------------------------------------------------------------------- #
def _triangle_solid_angle(a, b, c):
    det = dot(a, cross(b, c))
    denom = 1.0 + dot(a, b) + dot(b, c) + dot(c, a)
    return 2.0 * math.atan2(det, denom)


def reference_signed_area(points_rad):
    """Expected directed area in (-2*pi, 2*pi], left side positive."""
    v = [to_xyz(lo, la) for lo, la in points_rad]
    o = v[0]
    total = 0.0
    for i in range(1, len(v) - 1):
        total += _triangle_solid_angle(o, v[i], v[i + 1])
    two_pi = 2.0 * math.pi
    while total > two_pi:
        total -= 2.0 * two_pi
    while total <= -two_pi:
        total += 2.0 * two_pi
    return total


# --------------------------------------------------------------------------- #
# Independent seam/pole events on one edge by dense exact-SLERP sampling
# --------------------------------------------------------------------------- #
def reference_seam_events(p, q, alpha):
    """Return (n_crossings, coincident, east_count, west_count).

    A seam point satisfies ``y == 0`` and ``x <= 0`` (the 180 half-meridian,
    including the poles).  We compress the per-sample signs of y into maximal
    zero-runs; a run whose nonzero signs on either side are opposite is a
    genuine transverse crossing (this also catches crossings landing exactly
    on a sample, e.g. symmetric arcs).  Eastward = y goes + -> -.
    """
    pts = [slerp(p, q, alpha, k / SAMPLES) for k in range(SAMPLES + 1)]

    def sgn(y):
        return 1 if y > 0 else (-1 if y < 0 else 0)

    signs = [sgn(w[1]) for w in pts]
    east = west = 0

    def count_at(tzero, before, after):
        nonlocal east, west
        xz = slerp(p, q, alpha, min(1.0, max(0.0, tzero)))
        if xz[0] > 1e-9:
            return  # zero of y on the 0-degree half-meridian
        if before > 0 > after:
            east += 1
        elif before < 0 < after:
            west += 1

    k = 1
    while k < SAMPLES:
        if signs[k] == 0:
            # maximal zero run [lo, hi)
            lo = k
            while k <= SAMPLES and signs[k] == 0:
                k += 1
            hi = k
            before = signs[lo - 1]
            after = signs[hi] if hi <= SAMPLES else 0
            if before != 0 and after != 0 and before != after:
                count_at((lo + hi) / 2.0 / SAMPLES, before, after)
            continue
        # strict sign change between adjacent samples (generic crossing)
        if signs[k - 1] * signs[k] < 0:
            ya, yb = pts[k - 1][1], pts[k][1]
            frac = ya / (ya - yb)
            count_at((k - 1 + frac) / SAMPLES, signs[k - 1], signs[k])
        k += 1

    check = pts[1:-1:max(1, SAMPLES // 400)]
    coincident = bool(check) and all(
        abs(w[1]) < 1e-8 and w[0] <= 1e-8 for w in check
    )
    return east + west, coincident, east, west


def reference_edge_passes_pole(p, q, alpha):
    """Independent interior-pole test via additive arc lengths."""
    out = []
    for name, pole in (("north", (0, 0, 1)), ("south", (0, 0, -1))):
        if abs(dot(pole, cross(p, q))) > 1e-8:
            continue
        via = central_angle(p, pole) + central_angle(pole, q)
        if abs(via - alpha) < 1e-8 and 1e-9 < central_angle(p, pole) \
                < alpha - 1e-9:
            out.append(name)
    return out


# --------------------------------------------------------------------------- #
# Independent point-in-loop test for CONVEX loops: all-edge-normals test
# --------------------------------------------------------------------------- #
def reference_contains(points_rad, lon, lat):
    """True iff (lon, lat) lies inside the convex loop.

    Independent of the core's Sutherland-Hodgman walk: a convex loop is the
    intersection of the hemispheres left of every edge, so a point is inside
    iff it lies on the same side of *every* edge great circle.  The loop's
    orientation is taken from the independent Van Oosterom signed area.
    Boundary counts as inside.  (A tangent-plane winding sum was considered
    and rejected: it cannot distinguish a region from its antipodal image.)
    """
    x = to_xyz(lon, lat)
    vs = [to_xyz(lo, la) for lo, la in points_rad]
    orient = 1.0 if reference_signed_area(points_rad) > 0.0 else -1.0
    n = len(vs)
    for i in range(n):
        edge_normal = cross(vs[i], vs[(i + 1) % n])
        if orient * dot(x, edge_normal) < -1e-9:
            return False
    return True
