"""Spherical geometry primitives on the unit sphere.

All geometric work is done on 3-D unit vectors converted from lon/lat.  No
flat lon/lat interpolation is ever performed.  A point on a short arc is
evaluated with the exact SLERP formula

    X(t) = sin((1-t)*a)/sin(a) * P + sin(t*a)/sin(a) * Q,

the unique shortest great-circle arc between non-antipodal endpoints P, Q
separated by central angle a in (0, pi).
"""

from __future__ import annotations

import math

from .errors import AntipodalAdjacentError, InvalidInputError

# --- numerical tolerances (documented in README) ---------------------------
ANTIPODAL_EPS = 1e-12        # |P.Q + 1| below this -> reject edge
DUP_VERTEX_EPS = 1e-12       # |P.Q - 1| above this -> repeated vertex
MERIDIAN_EPS = 1e-9          # |lon - pi| tolerance for the seam meridian
POLE_EPS = 1e-9              # |z| tolerance identifying a pole
T_EPS = 1e-9                 # arc-parameter interior tolerance
COPLANAR_EPS = 1e-12         # great-circle coplanarity / degeneracy test


# --------------------------------------------------------------------------- #
# Validation and conversions
# --------------------------------------------------------------------------- #
def finite_number(value, what: str) -> float:
    """Reject bool, NaN and +/-Infinity; return value as float."""
    if isinstance(value, bool):
        raise InvalidInputError(f"{what} must be a number, got bool", what)
    if not isinstance(value, (int, float)):
        raise InvalidInputError(
            f"{what} must be a real number, got {type(value).__name__}", what
        )
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise InvalidInputError(f"{what} must be finite, got {fv}", what)
    return fv


def lonlat_to_xyz(lon: float, lat: float):
    cl = math.cos(lat)
    return (cl * math.cos(lon), cl * math.sin(lon), math.sin(lat))


def xyz_to_lonlat(v):
    """Unit vector -> (lon, lat); lon in (-pi, pi].

    At a pole (x = y = 0) longitude is physically undefined and ``atan2``
    returns 0; callers detect poles via |z| and never rely on that longitude.
    """
    x, y, z = v
    lat = math.asin(max(-1.0, min(1.0, z)))
    lon = math.atan2(y, x)
    if lon <= -math.pi:
        lon = math.pi
    return lon, lat


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #
def cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def dot(u, v) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def norm(u) -> float:
    return math.sqrt(dot(u, u))


def normalize(u):
    n = norm(u)
    return (u[0] / n, u[1] / n, u[2] / n)


def central_angle(p, q) -> float:
    return math.acos(max(-1.0, min(1.0, dot(p, q))))


def slerp(p, q, alpha: float, t: float):
    """Exact unit-sphere point at parameter t in [0,1] on the short arc."""
    sa = math.sin(alpha)
    a = math.sin((1.0 - t) * alpha) / sa
    b = math.sin(t * alpha) / sa
    return normalize(
        (a * p[0] + b * q[0], a * p[1] + b * q[1], a * p[2] + b * q[2])
    )


def prepare_edge(p, q, edge_index: int) -> float:
    """Validate one consecutive pair; return its central angle in (0, pi)."""
    c = dot(p, q)
    if c >= 1.0 - DUP_VERTEX_EPS:
        raise InvalidInputError(
            "zero-length edge (duplicate consecutive vertices)",
            f"edge {edge_index}",
        )
    if c <= -1.0 + ANTIPODAL_EPS:
        raise AntipodalAdjacentError(edge_index)
    return math.acos(max(-1.0, min(1.0, c)))


def arc_basis(p, q, alpha: float, x):
    """SLERP coefficients (A, B) of x in the great-circle plane of p->q.

    x = A*P + B*Q (both must be >= 0 for x to lie on the *short* arc).
    Returns None when x is not on that great-circle plane.
    """
    c = math.cos(alpha)
    denom = 1.0 - c * c
    if denom < 1e-14:
        return None
    xp, xq = dot(x, p), dot(x, q)
    a = (xp - c * xq) / denom
    b = (xq - c * xp) / denom
    if abs(dot(cross(p, q), x)) > 1e-8:
        return None
    return a, b


def arc_parameter(p, q, alpha: float, x):
    """Parameter t in [0,1] of x on the short arc p->q, or None.

    Recovered directly from the SLERP basis (no lon/lat interpolation):
    with u = t*alpha, sin(u) = B*sin(alpha), cos(u) = A + B*cos(alpha).
    """
    ab = arc_basis(p, q, alpha, x)
    if ab is None:
        return None
    a, b = ab
    if a < -1e-8 or b < -1e-8:
        return None  # on the complementary (long) arc
    u = math.atan2(b * math.sin(alpha), a + b * math.cos(alpha))
    if u < -1e-9 or u > alpha + 1e-9:
        return None
    return min(1.0, max(0.0, u / alpha))


# --------------------------------------------------------------------------- #
# Signed area: Gauss-Bonnet exterior-turn sum (left side positive)
# --------------------------------------------------------------------------- #
def turning_angle(prev_v, cur, next_v) -> float:
    """Signed exterior turn at vertex cur; positive for a left turn.

    Tangent directions are taken in the tangent plane at the vertex from the
    exact great-circle arcs, so the sign is the loop's true orientation (left
    side positive) with no planar approximation.
    """
    c_in = dot(prev_v, cur)
    c_out = dot(cur, next_v)
    v_in = normalize((c_in * cur[0] - prev_v[0],
                      c_in * cur[1] - prev_v[1],
                      c_in * cur[2] - prev_v[2]))
    v_out = normalize((next_v[0] - c_out * cur[0],
                       next_v[1] - c_out * cur[1],
                       next_v[2] - c_out * cur[2]))
    return math.atan2(dot(cur, cross(v_in, v_out)), dot(v_in, v_out))


def great_circle_degenerate(vertices_xyz) -> bool:
    """True iff all vertices lie (to tolerance) on one great circle.

    Consecutive vertices are never equal or antipodal (validated earlier), so
    every edge normal ``v_i x v_{i+1}`` is nonzero.  All vertices are
    coplanar with the origin iff all edge normals are parallel up to sign.
    """
    n = len(vertices_xyz)
    ref = normalize(cross(vertices_xyz[0], vertices_xyz[1]))
    for i in range(1, n):
        ni = cross(vertices_xyz[i], vertices_xyz[(i + 1) % n])
        if norm(ni) <= 1e-12:
            return True  # defensive: treated as degenerate
        if abs(dot(ref, normalize(ni))) < 1.0 - COPLANAR_EPS:
            return False
    return True


def signed_area(vertices_xyz) -> float:
    """Directed area in (-2*pi, 2*pi]; Gauss-Bonnet, left side positive.

    For the oriented simple loop, area = 2*pi - sum(exterior turns).  The raw
    value is defined modulo 4*pi and is canonicalized into (-2*pi, 2*pi].
    """
    n = len(vertices_xyz)
    total_turn = 0.0
    for i in range(n):
        total_turn += turning_angle(
            vertices_xyz[i - 1], vertices_xyz[i], vertices_xyz[(i + 1) % n]
        )
    area = 2.0 * math.pi - total_turn
    while area > 2.0 * math.pi:
        area -= 4.0 * math.pi
    while area <= -2.0 * math.pi:
        area += 4.0 * math.pi
    return area


# --------------------------------------------------------------------------- #
# Signed 3-D first and second area moments of the enclosed region
# --------------------------------------------------------------------------- #
def signed_moments(vertices_xyz):
    """Signed area moments of the region enclosed by a simple loop.

    Returns (signed_area, first_moment, second_moment) where

    * signed_area   -- directed area in (-2*pi, 2*pi] (left side positive),
      identical to signed_area();
    * first_moment  -- the signed integral of the position vector,
      M = integral of x dA (a 3-tuple);
    * second_moment -- the signed integral of the outer product,
      S = integral of x x^T dA (a symmetric 3x3 matrix, tuple of 3 rows).

    Both moments are *signed*: reversing the loop orientation negates them,
    while the geometric region size is unchanged.  Because |x| = 1 on the
    unit sphere, trace(S) == signed_area exactly (up to rounding).

    Derivation (spherical divergence theorem; exact boundary integrals, no
    lon/lat unwrapping involved):

    * With the tangential projection P c = c - (c.x) x of a constant vector,
      div_S(P c) = -2 (c.x).  The outward conormal of a great-circle edge
      is the constant -m with m = (p x q)/|p x q|, so
      M = (1/2) sum_e alpha_e m_e.
    * div_S((c.x) P d) = (c.d) - 3 (c.x)(d.x); symmetrizing in (c, d) and
      integrating c.x along the arc x(s) = cos(s) p + sin(s) w with
      w = (q - cos(alpha) p)/sin(alpha) gives
      S[c,d] = (c.d) A/3 + (1/6) sum_e [(d.m) I_c + (c.m) I_d],
      I_c = (c.p) sin(alpha) + (c.w) (1 - cos(alpha)).

    The boundary formulas integrate the region on the *left* of the
    oriented loop.  When that left region is the larger one (area
    A_left > 2*pi), the signed moments of the enclosed smaller region
    follow from integral-of-x-over-sphere = 0 and
    integral-of-xx^T-over-sphere = (4*pi/3) I: the first moment is
    unchanged and the second moment loses (4*pi/3) I.
    """
    n = len(vertices_xyz)
    total_turn = 0.0
    for i in range(n):
        total_turn += turning_angle(
            vertices_xyz[i - 1], vertices_xyz[i], vertices_xyz[(i + 1) % n]
        )
    two_pi = 2.0 * math.pi
    four_pi = 2.0 * two_pi
    a_left = (two_pi - total_turn) % four_pi
    big = a_left > two_pi
    area = a_left - four_pi if big else a_left

    mom = [0.0, 0.0, 0.0]
    sec = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(n):
        p = vertices_xyz[i]
        q = vertices_xyz[(i + 1) % n]
        alpha = central_angle(p, q)
        sa = math.sin(alpha)
        ca = math.cos(alpha)
        pq = cross(p, q)
        m = (pq[0] / sa, pq[1] / sa, pq[2] / sa)  # unit edge normal
        w = ((q[0] - ca * p[0]) / sa,
             (q[1] - ca * p[1]) / sa,
             (q[2] - ca * p[2]) / sa)
        for c in range(3):
            mom[c] += 0.5 * alpha * m[c]
        # I_c = integral of (e_c . x) ds along the arc, per component.
        arc_i = (
            p[0] * sa + w[0] * (1.0 - ca),
            p[1] * sa + w[1] * (1.0 - ca),
            p[2] * sa + w[2] * (1.0 - ca),
        )
        for c in range(3):
            ic = arc_i[c]
            for d in range(3):
                sec[c][d] += (m[d] * ic + m[c] * arc_i[d]) / 6.0
    # The boundary sum integrates the region on the LEFT of the loop, so
    # the isotropic term uses the raw left-region area a_left (which may
    # exceed 2*pi); the 4*pi correction below then maps to the enclosed
    # smaller region whose area is the canonicalized signed area.
    third_a = a_left / 3.0
    for c in range(3):
        sec[c][c] += third_a
    if big:
        # The boundary integral covers the left (larger) region; subtract
        # the full-sphere second moment (4*pi/3) I to obtain the signed
        # moment of the enclosed smaller region.  The first moment needs
        # no correction because the full-sphere first moment is zero.
        corr = four_pi / 3.0
        for c in range(3):
            sec[c][c] -= corr
    # Symmetrize away residual rounding asymmetry; the exact value is
    # symmetric by construction.
    second = tuple(
        tuple(0.5 * (sec[r][c] + sec[c][r]) for c in range(3))
        for r in range(3)
    )
    return area, (mom[0], mom[1], mom[2]), second


# --------------------------------------------------------------------------- #
# Seam (lon = +/- pi half-meridian) and pole events of one short arc
# --------------------------------------------------------------------------- #
def seam_plane_hits(p, q, alpha: float):
    """Interior intersections of arc p->q with the 180-degree half-meridian.

    The seam meridian (lon = +pi == -pi) runs through (-1, 0, 0) and both
    poles; the full great circle containing it is the plane ``y = 0``.
    Intersecting that plane with the arc's great-circle plane gives two
    antipodal candidates; each is checked for lying on (a) the physical
    +180 *half*-meridian (``x <= 0``, excluding the opposite 0-meridian) and
    (b) the short arc.  This is a true 3-D construction -- planar lon/lat
    interpolation is never used.

    Returns a list of ``(t, lat, is_pole)`` sorted by t, without duplicates.
    """
    n = cross(p, q)
    axis = cross(n, (0.0, 1.0, 0.0))
    if norm(axis) < 1e-14:
        return []  # arc great circle IS the seam great circle (coincident)
    x = normalize(axis)
    hits = []
    for cand in (x, (-x[0], -x[1], -x[2])):
        t = arc_parameter(p, q, alpha, cand)
        if t is None or not (T_EPS < t < 1.0 - T_EPS):
            continue
        cx, cy, cz = cand
        # Angular distance to the nearer pole is the horizontal radius;
        # use the *linear* quantity (not 1-|z|, which is quadratic and
        # mislabels near-pole crossings as poles).
        polar_r = math.sqrt(max(0.0, 1.0 - cz * cz))
        is_pole = polar_r < POLE_EPS
        if is_pole:
            lat = math.copysign(math.pi / 2, cz)
        else:
            lon, lat = xyz_to_lonlat(cand)
            if abs(lon - math.pi) > MERIDIAN_EPS:
                continue
            # physical half-meridian: the x <= 0 side of the y=0 plane
            if cx > MERIDIAN_EPS or abs(cy) > MERIDIAN_EPS:
                continue
        hits.append((t, lat, is_pole))
    hits.sort(key=lambda h: h[0])
    out = []
    for h in hits:
        if not out or abs(h[0] - out[-1][0]) > 1e-10:
            out.append(h)
    return out


def seam_crossing_direction(p, q, alpha: float, t: float) -> int:
    """+1 = arc crosses eastward (lon wraps pi -> -pi), -1 westward.

    Decided from exact SLERP samples on either side of the crossing.
    """
    d = 1e-7
    lo, _ = xyz_to_lonlat(slerp(p, q, alpha, max(0.0, t - d)))
    hi, _ = xyz_to_lonlat(slerp(p, q, alpha, min(1.0, t + d)))
    # Near the seam the canonical longitude is either close to +pi or -pi.
    near_plus_before = abs(lo - math.pi) < abs(lo + math.pi)
    near_minus_after = abs(hi + math.pi) < abs(hi - math.pi)
    return 1 if (near_plus_before and near_minus_after) else -1


def interior_pole(p, q, alpha: float):
    """``("north"|"south", t)`` if the short arc crosses a pole internally."""
    n = cross(p, q)
    for name, pole in (("north", (0.0, 0.0, 1.0)),
                       ("south", (0.0, 0.0, -1.0))):
        if abs(dot(n, pole)) > 1e-9:
            continue
        t = arc_parameter(p, q, alpha, pole)
        if t is not None and T_EPS < t < 1.0 - T_EPS:
            return name, t
    return None


def on_seam_meridian(v) -> bool:
    """True if unit point v lies on the physical 180 half-meridian.

    Poles count (every meridian meets them); otherwise lon must be ~+pi.
    """
    if abs(abs(v[2]) - 1.0) <= POLE_EPS:
        return True
    lon, _ = xyz_to_lonlat(v)
    return abs(lon - math.pi) <= MERIDIAN_EPS


def arc_is_seam_coincident(p, q, alpha: float) -> bool:
    """True iff the whole short arc lies on the 180 half-meridian."""
    if not (on_seam_meridian(p) and on_seam_meridian(q)):
        return False
    # Sample the midpoint: a short arc between two points of this meridian
    # follows the meridian; verify rather than assume.
    mid = slerp(p, q, alpha, 0.5)
    return on_seam_meridian(mid)


# --------------------------------------------------------------------------- #
# Non-adjacent arc intersection (transversal crossing or overlap)
# --------------------------------------------------------------------------- #
def arcs_intersect(p1, q1, a1, p2, q2, a2) -> str:
    """``"none"`` | ``"cross"`` | ``"overlap"`` for two short arcs.

    Only interior-to-both contacts count; shared endpoints (adjacent edges)
    are excluded by the caller, and endpoint touches here are ignored.
    """
    n1, n2 = cross(p1, q1), cross(p2, q2)
    axis = cross(n1, n2)
    if norm(axis) >= 1e-12:
        x = normalize(axis)
        for cand in (x, (-x[0], -x[1], -x[2])):
            t1 = arc_parameter(p1, q1, a1, cand)
            t2 = arc_parameter(p2, q2, a2, cand)
            if (t1 is not None and t2 is not None
                    and T_EPS < t1 < 1 - T_EPS and T_EPS < t2 < 1 - T_EPS):
                return "cross"
        return "none"

    # Coplanar great circles: look for interior overlap via endpoints.
    for x in (p2, q2, slerp(p2, q2, a2, 0.5)):
        t1 = arc_parameter(p1, q1, a1, x)
        if t1 is not None and T_EPS < t1 < 1 - T_EPS:
            t2 = arc_parameter(p2, q2, a2, x)
            if t2 is not None and T_EPS < t2 < 1 - T_EPS:
                return "overlap"
    for x in (p1, q1):
        t1 = arc_parameter(p1, q1, a1, x)
        t2 = arc_parameter(p2, q2, a2, x)
        if (t1 is not None and t2 is not None
                and T_EPS < t1 < 1 - T_EPS and T_EPS < t2 < 1 - T_EPS):
            return "overlap"
    return "none"
