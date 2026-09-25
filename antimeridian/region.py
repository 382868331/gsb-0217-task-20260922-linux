"""Split a hemispherical simple loop into seam-closed fillable loops.

:func:`split_region_at_seam` takes a loop that already passes the ordinary
simplicity validation and cuts it at its genuine transversal crossings of
the 180-degree half-meridian.  The pieces on each side of the seam are
closed into fillable loops by synthetic edges *along* the seam.  Seam
crossings are ordered by latitude and the seam intervals that actually
bound the region are found by an exact point-in-polygon test in a gnomonic
projection about the caller's hemisphere centre -- not by naive pairwise
input-order connection.  This is the unique pairing for which every closed
loop is simple, the directed areas sum to the input signed area, and the
synthetic seam edges cancel as a 1-chain (original seam-coincident edges
are kept exactly once, never duplicated).

Scope (explicitly unsupported otherwise): the region *and its boundary*
must lie inside one caller-given open hemisphere, and the loop must not
pass through a pole.  Both conditions are verified up front; any violation
raises :class:`~antimeridian.errors.UnsupportedError`.  No general
polygon-boolean or hemisphere-clipping machinery is provided.
"""

from __future__ import annotations

import math

from . import geometry as g
from .core import _as_point, _is_pole, _validate_loop
from .errors import UnsupportedError
from .models import (
    ArcParam,
    GeoPoint,
    RegionLoop,
    RegionSegment,
    SplitRegion,
)

# Guard band for the open-hemisphere test (documented in README).
HEMISPHERE_EPS = 1e-12
# Arc-parameter guard for the interior edge-minimum test.
_EDGE_MIN_EPS = 1e-12
# Minimum latitude gap accepted between distinct seam-contact points.
_LAT_EPS = 1e-10


class _Piece:
    """One maximal off-seam (or wholly on-seam) slice of the loop."""

    __slots__ = ("side", "edge", "t0", "t1", "start", "end")

    def __init__(self, side, edge, t0, t1, start, end):
        self.side = side  # "east" | "west" | "seam"
        self.edge = edge
        self.t0 = t0
        self.t1 = t1
        self.start = start  # GeoPoint
        self.end = end      # GeoPoint


def _side_of(v) -> str:
    lon, _ = g.xyz_to_lonlat(v)
    return "east" if lon > 0 else "west"


def _check_hemisphere(pts, xyz, alphas, center_xyz) -> None:
    """Verify the loop (and hence its region) lies in the open hemisphere.

    An open hemisphere is geodesically convex, so checking every vertex is
    strictly inside plus the closed-form interior minimum of every edge is
    a complete verification -- not a sampling heuristic.
    """
    n = len(pts)
    for k, v in enumerate(xyz):
        if g.dot(v, center_xyz) <= HEMISPHERE_EPS:
            raise UnsupportedError(
                "loop is not contained in the given open hemisphere",
                f"vertex {k}",
            )
    for i in range(n):
        p, q = xyz[i], xyz[(i + 1) % n]
        alpha = alphas[i]
        # X(u) = cos(u) P + sin(u) W on the arc; X.C = a cos u + b sin u.
        # Its only interior minimum is at u* = atan2(b, a) + pi with value
        # -sqrt(a^2+b^2) <= 0, i.e. on or beyond the hemisphere boundary.
        ca = math.cos(alpha)
        w = g.normalize((q[0] - ca * p[0], q[1] - ca * p[1],
                         q[2] - ca * p[2]))
        a = g.dot(p, center_xyz)
        b = g.dot(w, center_xyz)
        u_star = math.atan2(b, a) + math.pi
        if u_star > math.pi:
            u_star -= 2.0 * math.pi
        if _EDGE_MIN_EPS < u_star < alpha - _EDGE_MIN_EPS:
            raise UnsupportedError(
                "edge leaves the given open hemisphere",
                f"edge {i}",
            )


def _check_poles(pts, xyz, alphas) -> None:
    n = len(pts)
    for k, p in enumerate(pts):
        if _is_pole(p):
            raise UnsupportedError(
                "loop passes through a pole", f"vertex {k}"
            )
    for i in range(n):
        hit = g.interior_pole(xyz[i], xyz[(i + 1) % n], alphas[i])
        if hit is not None:
            raise UnsupportedError(
                f"edge crosses the {hit[0]} pole", f"edge {i}"
            )


def _build_pieces(pts, xyz, alphas, coincident_flags):
    """Slice every edge at its genuine interior seam crossing.

    Tangential endpoint touches produce no cut; wholly seam-coincident
    edges become "seam" pieces, kept once and untouched afterwards.
    """
    n = len(pts)
    pieces = []
    for i in range(n):
        j = (i + 1) % n
        vs, ve, alpha = xyz[i], xyz[j], alphas[i]
        ps, pe = pts[i], pts[j]
        if coincident_flags[i]:
            pieces.append(_Piece("seam", i, 0.0, 1.0, ps, pe))
            continue
        hits = [h for h in g.seam_plane_hits(vs, ve, alpha) if not h[2]]
        if hits:
            t, cut_lat, _ = hits[0]
            cut = GeoPoint(math.pi, cut_lat)
            first = _side_of(g.slerp(vs, ve, alpha, 0.5 * t))
            second = "west" if first == "east" else "east"
            pieces.append(_Piece(first, i, 0.0, t, ps, cut))
            pieces.append(_Piece(second, i, t, 1.0, cut, pe))
        else:
            side = _side_of(g.slerp(vs, ve, alpha, 0.5))
            pieces.append(_Piece(side, i, 0.0, 1.0, ps, pe))
    return pieces


def _build_chains(pieces):
    """Maximal cyclic same-side runs of *adjacent* off-seam pieces.

    A chain breaks at every side change and at every seam-coincident
    piece (those become seam links of their own).  A tangent touch (same
    side at a shared seam vertex) stays inside one chain.  Returns an
    empty list only when every piece lies on the seam (impossible after
    the great-circle degeneracy check).
    """
    m = len(pieces)
    off = [k for k in range(m) if pieces[k].side != "seam"]
    if not off:
        return []
    chains = [[off[0]]]
    for k in off[1:]:
        prev = chains[-1][-1]
        if k == prev + 1 and pieces[k].side == pieces[prev].side:
            chains[-1].append(k)
        else:
            chains.append([k])
    if len(chains) > 1:
        last = chains[-1][-1]
        first = chains[0][0]
        if ((last + 1) % m == first
                and pieces[last].side == pieces[first].side):
            chains[0] = chains[-1] + chains[0]
            chains.pop()
    return chains


def _count_crossings(pieces, chains):
    """Transversal crossings at consecutive-chain shared seam points where
    the sides differ.  Tangent touches are not such boundaries."""
    if len(chains) <= 1:
        return 0
    count = 0
    m = len(chains)
    for i in range(m):
        a = pieces[chains[i][-1]]
        b = pieces[chains[(i + 1) % m][0]]
        if a.end is b.start and a.side != b.side:
            count += 1
    return count


def _seam_points(pieces, chains):
    """All distinct seam-contact points, ascending in latitude."""
    points = []
    for ch in chains:
        for gp in (pieces[ch[0]].start, pieces[ch[-1]].end):
            if all(gp is not q for q in points):
                points.append(gp)
    for pc in pieces:  # coincident endpoints (also between two such edges)
        if pc.side == "seam":
            for gp in (pc.start, pc.end):
                if all(gp is not q for q in points):
                    points.append(gp)
    points.sort(key=lambda gp: gp.lat)
    for a, b in zip(points, points[1:]):
        if b.lat - a.lat < _LAT_EPS:
            raise UnsupportedError(
                "distinct seam-contact points coincide within the safety "
                "band", "seam")
    return points


def _coincident_covers(pieces, lo_pt, hi_pt):
    """The seam piece whose span is exactly interval [lo_pt, hi_pt]."""
    for pc in pieces:
        if pc.side != "seam":
            continue
        a, b = pc.start, pc.end
        if (a is lo_pt and b is hi_pt) or (a is hi_pt and b is lo_pt):
            return pc
    return None


def _interval_inside_gnomonic(pieces, center_xyz, lat):
    """Exact interior test of a seam-meridian point in the gnomonic plane.

    The gnomonic projection maps great-circle arcs to straight segments
    and is a homeomorphism of the open hemisphere, so the even-odd ray
    cast on the projected original loop gives an exact answer.
    """
    e1, e2 = g.gnomonic_frame(center_xyz)
    target = g.gnomonic_project(g.lonlat_to_xyz(math.pi, lat),
                                center_xyz, e1, e2)
    poly = [
        g.gnomonic_project(
            g.lonlat_to_xyz(pc.start.lon, pc.start.lat), center_xyz, e1, e2)
        for pc in pieces
    ]
    return g.planar_point_inside(poly, target)


class _SeamLink:
    """A directed link along the seam consumable by one side's loops."""

    __slots__ = ("start", "end", "side", "segment", "used")

    def __init__(self, start, end, side, segment):
        self.start = start
        self.end = end
        self.side = side
        self.segment = segment
        self.used = False


def _seam_segment(lo_pt, hi_pt, northward, *, edge_index=None,
                  t0=None, t1=None, central_angle=None, kind="seam"):
    start = lo_pt if northward else hi_pt
    end = hi_pt if northward else lo_pt
    if central_angle is None:
        central_angle = abs(hi_pt.lat - lo_pt.lat)
    return RegionSegment(
        kind=kind, start=start, end=end, edge_index=edge_index,
        t_start=t0, t_end=t1, central_angle=central_angle,
    )


def _build_seam_links(pieces, levels, center_xyz, ccw):
    """Directed seam links for every latitude interval bounding the region.

    * Interval covered by an original coincident edge: exactly one link,
      reusing that edge once, on the side whose required traversal matches.
    * Otherwise, the interval bounds the region iff its gnomonic midpoint
      is inside the projected original loop; then one synthetic link per
      side in opposite directions (they cancel in the directed sum).
    """
    links = []
    for lo_pt, hi_pt in zip(levels, levels[1:]):
        coin = _coincident_covers(pieces, lo_pt, hi_pt)
        if coin is not None:
            northward = coin.end.lat > coin.start.lat
            if ccw:
                side = "east" if northward else "west"
            else:
                side = "west" if northward else "east"
            seg = _seam_segment(lo_pt, hi_pt, northward,
                                edge_index=coin.edge, t0=0.0, t1=1.0,
                                kind="coincident")
            links.append(_SeamLink(
                lo_pt if northward else hi_pt,
                hi_pt if northward else lo_pt, side, seg))
            continue
        mid_lat = 0.5 * (lo_pt.lat + hi_pt.lat)
        if not _interval_inside_gnomonic(pieces, center_xyz, mid_lat):
            continue
        # ccw: east fill goes northward, west fill southward; reversed
        # input swaps both -- the region stays on the left of each loop.
        for side, northward in (("east", ccw), ("west", not ccw)):
            seg = _seam_segment(lo_pt, hi_pt, northward)
            links.append(_SeamLink(
                lo_pt if northward else hi_pt,
                hi_pt if northward else lo_pt, side, seg))
    return links


def _arc_segment(pc, alphas):
    return RegionSegment(
        kind="arc",
        start=pc.start,
        end=pc.end,
        edge_index=pc.edge,
        t_start=pc.t0,
        t_end=pc.t1,
        central_angle=(pc.t1 - pc.t0) * alphas[pc.edge],
    )


def _loop_area(segments) -> float:
    """Directed area (Gauss-Bonnet); each segment contributes its start
    and its exact SLERP midpoint, so digons whose arc and connector follow
    different great circles are evaluated correctly."""
    verts = []
    for seg in segments:
        p0 = g.lonlat_to_xyz(seg.start.lon, seg.start.lat)
        p1 = g.lonlat_to_xyz(seg.end.lon, seg.end.lat)
        alpha = g.central_angle(p0, p1)
        verts.append(p0)
        verts.append(g.slerp(p0, p1, alpha, 0.5))
    return g.signed_area(verts)


def _assemble_loops(pieces, chains, links, alphas):
    """Walk chains, alternating with seam links at crossing points.

    At a seam point the walk prefers an unvisited same-side chain; if
    none starts there it follows another same-side seam link (an original
    coincident edge can chain straight into a synthetic fill edge).
    """
    loops = []
    consumed = set()
    for first in chains:
        if id(first) in consumed:
            continue
        segments = []
        visiting = set()
        chain = first
        start_point = pieces[chain[0]].start
        side = pieces[chain[0]].side
        closed = False
        while True:
            if id(chain) in visiting:
                raise UnsupportedError("seam pairing did not terminate",
                                       "seam")
            visiting.add(id(chain))
            for k in chain:
                segments.append(_arc_segment(pieces[k], alphas))
            point = pieces[chain[-1]].end
            while True:
                nxt = next(
                    (ch for ch in chains
                     if id(ch) not in visiting
                     and pieces[ch[0]].start is point
                     and pieces[ch[0]].side == side),
                    None)
                if nxt is not None:
                    chain = nxt
                    break
                link = next((ln for ln in links
                             if not ln.used and ln.side == side
                             and ln.start is point), None)
                if link is None:
                    if point is start_point:
                        closed = True
                        break
                    raise UnsupportedError("seam pairing could not be "
                                           "completed", "seam")
                link.used = True
                segments.append(link.segment)
                point = link.end
            if closed:
                break
        consumed |= visiting
        loops.append(RegionLoop(
            index=len(loops),
            side=pieces[first[0]].side,
            segments=tuple(segments),
            signed_area=_loop_area(segments),
        ))
    unused = [ln for ln in links if not ln.used]
    if unused:
        # Every genuine seam link must bound exactly one assembled loop;
        # anything left over means the pairing rejected an input the
        # geometry cannot represent in this subset.
        raise UnsupportedError("seam pairing left unused seam links",
                               "seam")
    return loops


def split_region_at_seam(points, hemisphere_center, *, units: str = "rad"):
    """Split a simple hemispherical loop into seam-closed fillable loops.

    Parameters
    ----------
    points:
        Iterable of 3..100 ``(lon, lat)`` pairs (or :class:`GeoPoint`),
        closed implicitly -- the same contract as
        :func:`antimeridian.unfold_loop`.
    hemisphere_center:
        ``(lon, lat)`` of the centre of an open hemisphere containing the
        whole region and its boundary.
    units:
        ``"rad"`` (default) or ``"deg"``.

    Returns
    -------
    SplitRegion
        Closed fillable loops (original sub-arcs plus seam fill edges),
        the original arc parameters and the total signed area.

    Raises
    ------
    InvalidInputError, AntipodalAdjacentError, SelfIntersectionError,
    DegenerateLoopError
        The same structural violations as :func:`unfold_loop`.
    UnsupportedError
        The loop (or its region) is not contained in the given open
        hemisphere, or the loop passes through a pole.
    """
    from .errors import InvalidInputError
    if units not in ("rad", "deg"):
        raise InvalidInputError('units must be "rad" or "deg"', "units")
    scale = math.pi / 180.0 if units == "deg" else 1.0
    center_pt = _as_point(hemisphere_center, 0, scale,
                          label="hemisphere_center")
    center_xyz = g.lonlat_to_xyz(center_pt.lon, center_pt.lat)

    pts, xyz, alphas = _validate_loop(points, units)
    n = len(pts)

    _check_hemisphere(pts, xyz, alphas, center_xyz)
    _check_poles(pts, xyz, alphas)

    coincident_flags = [
        g.arc_is_seam_coincident(xyz[i], xyz[(i + 1) % n], alphas[i])
        for i in range(n)
    ]
    pieces = _build_pieces(pts, xyz, alphas, coincident_flags)
    chains = _build_chains(pieces)
    levels = _seam_points(pieces, chains)
    area = g.signed_area(xyz)
    ccw = area >= 0.0

    if not chains:
        raise UnsupportedError("loop lies entirely on the seam", "seam")

    crossings = _count_crossings(pieces, chains)
    if crossings == 0 and not any(pc.side == "seam" for pc in pieces):
        # No seam contact at all: the loop itself is the single fillable
        # polygon; no pairing or gnomonic test is needed.
        loops = _assemble_loops(pieces, chains, [], alphas)
    else:
        links = _build_seam_links(pieces, levels, center_xyz, ccw)
        loops = _assemble_loops(pieces, chains, links, alphas)

    arcs = tuple(
        ArcParam(
            edge_index=i,
            start=pts[i],
            end=pts[(i + 1) % n],
            central_angle=alphas[i],
            pole_coincident=coincident_flags[i],
            contains_pole=False,
        )
        for i in range(n)
    )
    winding = "ccw" if area > 0 else ("cw" if area < 0 else "zero")
    return SplitRegion(
        loops=tuple(loops),
        arcs=arcs,
        hemisphere_center=center_pt,
        signed_area=area,
        area_winding=winding,
        seam_crossing_count=crossings,
    )


def split_region_at_seam_degrees(points, hemisphere_center) -> SplitRegion:
    """Convenience wrapper: longitudes/latitudes given in degrees."""
    return split_region_at_seam(points, hemisphere_center, units="deg")
