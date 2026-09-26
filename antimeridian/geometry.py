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
# Spherical area moments
# --------------------------------------------------------------------------- #
def lies_in_open_hemisphere(vertices_xyz, eps: float = 1e-12) -> bool:
    """True iff some unit axis has strictly positive dot with every vertex.

    A cell of the arrangement of the vertex great circles either contains a
    vertex or an intersection of two such great circles.  Candidate axes at
    those arrangement vertices are perturbed along their active constraints.
    """
    unit = [normalize(v) for v in vertices_xyz]
    candidates = list(unit)
    for i in range(len(unit)):
        for j in range(i + 1, len(unit)):
            axis = cross(unit[i], unit[j])
            length = norm(axis)
            if length > eps:
                axis = (axis[0] / length, axis[1] / length, axis[2] / length)
                candidates.extend((axis, (-axis[0], -axis[1], -axis[2])))

    def feasible(axis):
        length = norm(axis)
        if length <= eps:
            return False
        u = (axis[0] / length, axis[1] / length, axis[2] / length)
        return min(dot(u, v) for v in unit) > eps

    for axis in candidates:
        dots = [dot(axis, v) for v in unit]
        if min(dots) > eps:
            return True
        if min(dots) < -eps:
            continue
        active = [v for v, value in zip(unit, dots) if abs(value) <= eps]
        inward = (sum(v[0] for v in active),
                  sum(v[1] for v in active),
                  sum(v[2] for v in active))
        trials = [inward,
                  (axis[k] + inward[k] for k in range(3)),
                  (axis[k] - inward[k] for k in range(3))]
        for v in active:
            trials.append((axis[k] + v[k] for k in range(3)))
            side = cross(axis, v)
            if norm(side) > eps:
                trials.append(side)
                trials.append((-side[0], -side[1], -side[2]))
        for trial in trials:
            if feasible(tuple(trial)):
                return True
    return False


def first_area_moment(vertices_xyz):
    """Signed integral vector ``integral_R x dA`` for a geodesic polygon."""
    total = [0.0, 0.0, 0.0]
    n = len(vertices_xyz)
    for i in range(n):
        p = vertices_xyz[i]
        q = vertices_xyz[(i + 1) % n]
        edge_normal = cross(p, q)
        scale = central_angle(p, q) / norm(edge_normal)
        for k in range(3):
            total[k] += 0.5 * scale * edge_normal[k]
    return (total[0], total[1], total[2])


def second_area_moment(vertices_xyz, area: float):
    """Signed integral matrix ``integral_R x x^T dA``.

    The spherical divergence theorem reduces each entry to an edge integral.
    Diagonal entries use ``integral_edge (c.x) x ds``; each off-diagonal
    entry is the polarized symmetrized boundary term.  This implementation
    accumulates the diagonal and the half-symmetric off-diagonal value
    separately, then ``3M = area*I - S``.
    The returned matrix is symmetric and its trace equals the signed area.
    """
    s_xx = s_yy = s_zz = 0.0
    s_xy = s_xz = s_yz = 0.0
    n = len(vertices_xyz)
    for i in range(n):
        p = vertices_xyz[i]
        q = vertices_xyz[(i + 1) % n]
        edge_normal = cross(p, q)
        edge_norm = norm(edge_normal)
        alpha = math.acos(max(-1.0, min(1.0, dot(p, q))))
        conormal = (-edge_normal[0] / edge_norm,
                    -edge_normal[1] / edge_norm,
                    -edge_normal[2] / edge_norm)
        integral_x_scale = math.tan(alpha / 2.0)
        integral_x = tuple(
            integral_x_scale * (p[k] + q[k]) for k in range(3)
        )
        s_xx += conormal[0] * integral_x[0]
        s_yy += conormal[1] * integral_x[1]
        s_zz += conormal[2] * integral_x[2]
        s_xy += 0.5 * (conormal[0] * integral_x[1]
                       + conormal[1] * integral_x[0])
        s_xz += 0.5 * (conormal[0] * integral_x[2]
                       + conormal[2] * integral_x[0])
        s_yz += 0.5 * (conormal[1] * integral_x[2]
                       + conormal[2] * integral_x[1])
    return (
        (area - s_xx) / 3.0,
        (area - s_yy) / 3.0,
        (area - s_zz) / 3.0,
        (0.0 - s_xy) / 3.0,
        (0.0 - s_xz) / 3.0,
        (0.0 - s_yz) / 3.0,
    )


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
