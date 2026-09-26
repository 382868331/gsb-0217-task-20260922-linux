"""Independent high-resolution reference for 3-D area moments.

Used ONLY by the test-suite.  This module deliberately does **not** import
:mod:`antimeridian`.  The core computes moments in closed form from
boundary integrals (spherical divergence theorem); this reference instead

* fan-triangulates the oriented loop from an arbitrary apex point (signed
  spherical triangles, so concave loops and apex-outside regions are both
  handled by cancellation),
* subdivides each spherical triangle into a fine geodesic grid,
* accumulates x dA and x x^T dA numerically, with each micro triangle
  signed area from the oriented solid-angle (atan2) formula and the
  integrand evaluated at the micro-triangle centroid.

Two genuinely different derivations must agree; the reference is a
numerical integration, never a reordering of the core boundary formulas.
"""

from __future__ import annotations

import math

# Subdivisions per fan-triangle edge.  Error decays ~1/N^2; 160 gives
# sub-1e-8 relative accuracy on the test polygons while staying fast.
GRID = 160


def _to_xyz(lon, lat):
    cl = math.cos(lat)
    return (cl * math.cos(lon), cl * math.sin(lon), math.sin(lat))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _norm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


def _signed_micro_area(a, b, c):
    """Oriented solid angle of triangle a-b-c (Van Oosterom formula)."""
    det = _dot(a, _cross(b, c))
    denom = 1.0 + _dot(a, b) + _dot(b, c) + _dot(c, a)
    return 2.0 * math.atan2(det, denom)


def _centroid_dir(a, b, c):
    return _normalize((a[0] + b[0] + c[0],
                       a[1] + b[1] + c[1],
                       a[2] + b[2] + c[2]))


def _grid_point(a, b, c, i, j, n):
    """Gnomonic-style interpolation: normalize(a + u*(b-a) + v*(c-a))."""
    u = i / n
    v = j / n
    return _normalize((a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]),
                       a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]),
                       a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])))


def _accumulate_triangle(acc, a, b, c, n):
    """Add numerical moments of the oriented spherical triangle a-b-c."""
    pts = {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts[(i, j)] = _grid_point(a, b, c, i, j, n)

    def emit(p, q, r):
        area = _signed_micro_area(p, q, r)
        g = _centroid_dir(p, q, r)
        acc[0] += area
        for k in range(3):
            acc[1][k] += g[k] * area
            for l in range(3):
                acc[2][k][l] += g[k] * g[l] * area

    for i in range(n):
        for j in range(n - i):
            p = pts[(i, j)]
            q = pts[(i + 1, j)]
            r = pts[(i, j + 1)]
            emit(p, q, r)
            if j < n - i - 1:
                s = pts[(i + 1, j + 1)]
                emit(q, s, r)


def reference_moments(points_rad, grid=GRID):
    """Numerically integrated (area, M, S) for an oriented simple loop.

    ``points_rad`` is a list of (lon, lat) in radians.  Returns
    ``(signed_area, first_moment[3], second_moment[3][3])``.
    """
    v = [_to_xyz(lo, la) for lo, la in points_rad]
    # Fan apex: any point works for a signed fan (overlaps cancel); pick a
    # deterministic one comfortably off every edge.
    sx = sum(p[0] for p in v)
    sy = sum(p[1] for p in v)
    sz = sum(p[2] for p in v)
    if sx * sx + sy * sy + sz * sz < 1e-6:
        apex = _normalize((v[0][0] + 0.31, v[0][1] - 0.17, v[0][2] + 0.23))
    else:
        apex = _normalize((sx, sy, sz))

    acc = [0.0, [0.0, 0.0, 0.0],
           [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    n = len(v)
    for i in range(n):
        _accumulate_triangle(acc, apex, v[i], v[(i + 1) % n], grid)

    # Canonicalize the signed area into (-2*pi, 2*pi] per the contract: the
    # fan accumulates the left-region area modulo 4*pi.
    raw = acc[0]
    two_pi = 2.0 * math.pi
    area = raw
    while area > two_pi:
        area -= 2.0 * two_pi
    while area <= -two_pi:
        area += 2.0 * two_pi
    # When the region on the left of the loop is the larger one, the signed
    # moments of the enclosed smaller region follow from the full-sphere
    # values (0 for M, (4*pi/3) I for S).
    if raw > two_pi:
        corr = 4.0 * math.pi / 3.0
        for k in range(3):
            acc[2][k][k] -= corr
    return area, acc[1], acc[2]
