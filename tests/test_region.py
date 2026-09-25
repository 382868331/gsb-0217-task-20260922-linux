"""Tests for split_region_at_seam: seam-closed fillable region loops.

Independent verification strategy (no antimeridian code is used as an
oracle):

* interior coverage is checked in a **gnomonic projection** about the
  caller's hemisphere centre, where great-circle arcs map to straight
  lines: projected output loops must tile exactly the projected original
  loop on a deterministic sample grid (even-odd point-in-polygon);
* per-loop directed spherical areas are checked against an independent
  Van Oosterom fan sum on the loop vertices, and their sum is checked
  against the original loop's signed area;
* synthetic seam fill edges are checked to cancel as a directed 1-chain
  on the 180-degree meridian.
"""

import math
import os
import sys
import unittest

from antimeridian import (
    InvalidInputError,
    UnsupportedError,
    split_region_at_seam,
    unfold_loop,
)

sys.path.insert(0, os.path.dirname(__file__))
import reference as R  # noqa: E402

D = math.pi / 180.0


# --------------------------------------------------------------------------- #
# Independent sphere / gnomonic-projection helpers (no core imports)
# --------------------------------------------------------------------------- #
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(_dot(a, a))


def _normv(a):
    m = _norm(a)
    return (a[0] / m, a[1] / m, a[2] / m)


def _xyz(lon, lat):
    cl = math.cos(lat)
    return (cl * math.cos(lon), cl * math.sin(lon), math.sin(lat))


def _gnomonic_basis(center_xyz):
    # An orthonormal tangent frame (e1, e2) at the projection centre.
    if abs(center_xyz[2]) < 0.9:
        e1 = _normv(_cross((0.0, 0.0, 1.0), center_xyz))
    else:
        e1 = _normv(_cross((0.0, 1.0, 0.0), center_xyz))
    e2 = _cross(center_xyz, e1)
    return e1, e2


def _gnomonic(x, center_xyz, e1, e2):
    s = _dot(x, center_xyz)
    return (_dot(x, e1) / s, _dot(x, e2) / s)


def _point_in_polygon(poly, pt):
    """Even-odd ray cast to the right in the gnomonic plane."""
    x, y = pt
    inside = False
    m = len(poly)
    for k in range(m):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % m]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xc > x:
                inside = not inside
    return inside


def _loop_xyzs(res_loop):
    return [_xyz(s.start.lon, s.start.lat) for s in res_loop.segments]


def _covers_identical(points_rad, res, center_rad, margin=0.02, step=0.04):
    """Projected interior coverage of output loops == original loop."""
    c = _xyz(*center_rad)
    e1, e2 = _gnomonic_basis(c)
    orig_poly = [_gnomonic(_xyz(lo, la), c, e1, e2)
                 for lo, la in points_rad]
    split_polys = [
        [_gnomonic(v, c, e1, e2) for v in _loop_xyzs(lp)]
        for lp in res.loops
    ]
    coords = [q for p in [orig_poly] + split_polys for q in p]
    xs = [q[0] for q in coords]
    ys = [q[1] for q in coords]
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    checked = 0
    u = x0
    while u <= x1:
        v = y0
        while v <= y1:
            in_orig = _point_in_polygon(orig_poly, (u, v))
            flags = [_point_in_polygon(p, (u, v)) for p in split_polys]
            if in_orig != any(flags):
                return False, checked
            # Output loops are interior-disjoint: at most one covers.
            if sum(1 for f in flags if f) > 1:
                return False, checked
            checked += 1
            v += step
        u += step
    return True, checked


def _seam_fill_cancels(res):
    """Synthetic seam edges form a zero directed 1-chain on the meridian."""
    seam = [s for lp in res.loops for s in lp.segments if s.kind == "seam"]
    if abs(sum(s.end.lat - s.start.lat for s in seam)) > 1e-12:
        return False
    levels = sorted({s.start.lat for s in seam}
                    | {s.end.lat for s in seam})
    for a, b in zip(levels, levels[1:]):
        lat = 0.5 * (a + b)

        def updown(s):
            lo = min(s.start.lat, s.end.lat)
            hi = max(s.start.lat, s.end.lat)
            return lo < lat < hi

        up = sum(1 for s in seam if updown(s) and s.end.lat > s.start.lat)
        down = sum(1 for s in seam if updown(s)
                   and s.end.lat < s.start.lat)
        if up != down:
            return False
    return True


def _independent_loop_area(res_loop):
    """Van Oosterom fan sum on the closed loop's exact corner vertices."""
    verts = _loop_xyzs(res_loop)
    o = verts[0]
    total = 0.0
    for k in range(1, len(verts) - 1):
        a, b, cc = o, verts[k], verts[k + 1]
        det = _dot(a, _cross(b, cc))
        total += 2.0 * math.atan2(
            det, 1.0 + _dot(a, b) + _dot(b, cc) + _dot(cc, a))
    two_pi = 2.0 * math.pi
    while total > two_pi:
        total -= 2.0 * two_pi
    while total <= -two_pi:
        total += 2.0 * two_pi
    return total

# Fixed deterministic shapes (degrees).
# Convex triangle straddling the seam (2 crossings).
CONVEX_DEG = [(170.0, 10.0), (-170.0, 10.0), (-175.0, 40.0)]
# Concave hexagon crossing the seam four times (a notch in a band).
CONCAVE_DEG = [(170.0, 10.0), (-170.0, 10.0), (-170.0, 50.0),
               (170.0, 50.0), (175.0, 30.0), (-175.0, 30.0)]
# Triangle that only touches the seam tangentially at a vertex.
TOUCH_DEG = [(170.0, 10.0), (180.0, 30.0), (170.0, 50.0)]
# Quadrilateral with one edge wholly on the seam, region on the east side.
COINCIDENT_DEG = [(170.0, 10.0), (180.0, 20.0), (180.0, 40.0),
                  (170.0, 50.0)]


def _rad(deg):
    return [(lo * D, la * D) for lo, la in deg]


def _check_closure(test, res):
    for lp in res.loops:
        segs = lp.segments
        for k, s in enumerate(segs):
            nxt = segs[(k + 1) % len(segs)]
            test.assertIs(s.end, nxt.start)
            test.assertEqual(s.end.lon, nxt.start.lon)
            test.assertEqual(s.end.lat, nxt.start.lat)
        test.assertGreaterEqual(len(segs), 2)


def _check_areas(test, points_rad, res, tol=1e-9):
    total = unfold_loop(points_rad).signed_area
    test.assertAlmostEqual(sum(lp.signed_area for lp in res.loops),
                           total, delta=tol)
    test.assertAlmostEqual(res.signed_area,
                           R.reference_signed_area(points_rad), delta=1e-9)
    for lp in res.loops:
        test.assertAlmostEqual(lp.signed_area,
                               _independent_loop_area(lp), delta=1e-9)


class SplitRegionTests(unittest.TestCase):
    def test_convex_loop_two_loops_and_coverage(self):
        pts = _rad(CONVEX_DEG)
        res = split_region_at_seam(pts, (math.pi, 25 * D))
        self.assertEqual(res.seam_crossing_count, 2)
        self.assertEqual(len(res.loops), 2)
        self.assertEqual(sorted(lp.side for lp in res.loops),
                         ["east", "west"])
        _check_closure(self, res)
        _check_areas(self, pts, res)
        self.assertTrue(_seam_fill_cancels(res))
        for lp in res.loops:
            kinds = [s.kind for s in lp.segments]
            self.assertEqual(kinds.count("seam"), 1)
            for s in lp.segments:
                if s.kind == "arc":
                    self.assertIsNotNone(s.edge_index)
                    self.assertGreaterEqual(s.t_start, 0.0)
                    self.assertLessEqual(s.t_end, 1.0)
                else:
                    self.assertIsNone(s.edge_index)
        ok, n = _covers_identical(pts, res, (math.pi, 25 * D))
        self.assertTrue(ok, f"coverage mismatch over {n} samples")
        self.assertGreater(n, 0)

    def test_concave_loop_four_crossings(self):
        pts = _rad(CONCAVE_DEG)
        res = split_region_at_seam(pts, (math.pi, 30 * D))
        self.assertEqual(res.seam_crossing_count, 4)
        # A concave 4-crossing loop (band with a notch) yields three
        # components after the cut: two on one side, one on the other --
        # the correct topology, not four pairwise-closed pieces.
        self.assertEqual(len(res.loops), 3)
        sides = sorted(lp.side for lp in res.loops)
        self.assertEqual(sides, ["east", "east", "west"])
        _check_closure(self, res)
        _check_areas(self, pts, res)
        self.assertTrue(_seam_fill_cancels(res))
        ok, _ = _covers_identical(pts, res, (math.pi, 30 * D))
        self.assertTrue(ok)
        # The single west loop carries two fill edges joining different
        # crossing pairs (correct parity pairing, not input order); the
        # two east loops carry one fill edge each.
        west_loop = [lp for lp in res.loops if lp.side == "west"][0]
        west_fills = {
            (round(s.start.lat, 9), round(s.end.lat, 9))
            for s in west_loop.segments if s.kind == "seam"
        }
        self.assertEqual(len(west_fills), 2)
        east_fills = [
            len([s for s in lp.segments if s.kind == "seam"])
            for lp in res.loops if lp.side == "east"
        ]
        self.assertEqual(sorted(east_fills), [1, 1])

    def test_tangent_touch_is_not_a_crossing(self):
        pts = _rad(TOUCH_DEG)
        res = split_region_at_seam(pts, (math.pi, 30 * D))
        self.assertEqual(res.seam_crossing_count, 0)
        self.assertEqual(len(res.loops), 1)
        self.assertTrue(all(s.kind == "arc"
                            for s in res.loops[0].segments))
        _check_closure(self, res)
        _check_areas(self, pts, res)
        ok, _ = _covers_identical(pts, res, (math.pi, 30 * D))
        self.assertTrue(ok)

    def test_coincident_edge_kept_once(self):
        pts = _rad(COINCIDENT_DEG)
        res = split_region_at_seam(pts, (math.pi, 30 * D))
        self.assertEqual(res.seam_crossing_count, 0)
        self.assertEqual(len(res.loops), 1)
        kinds = [s.kind for s in res.loops[0].segments]
        self.assertEqual(kinds.count("coincident"), 1)
        self.assertEqual(kinds.count("seam"), 0)
        coin = [s for s in res.loops[0].segments
                if s.kind == "coincident"][0]
        self.assertEqual(coin.edge_index, 1)
        _check_areas(self, pts, res)

    def test_crossing_plus_coincident_edge(self):
        # One interior crossing, one crossing at a seam vertex and one
        # coincident edge: the coincident link chains into a synthetic
        # fill edge inside one output loop.
        deg = [(170.0, 10.0), (-170.0, 10.0), (-170.0, 50.0),
               (180.0, 50.0), (180.0, 30.0)]
        pts = _rad(deg)
        res = split_region_at_seam(pts, (math.pi, 30 * D))
        self.assertEqual(res.seam_crossing_count, 1)
        self.assertEqual(len(res.loops), 2)
        flat = [s for lp in res.loops for s in lp.segments]
        self.assertEqual([s.kind for s in flat].count("coincident"), 1)
        _check_closure(self, res)
        _check_areas(self, pts, res)
        self.assertTrue(_seam_fill_cancels(res))
        ok, _ = _covers_identical(pts, res, (math.pi, 30 * D))
        self.assertTrue(ok)

    def test_reversed_traversal_negates_areas(self):
        pts = _rad(CONCAVE_DEG)
        fwd = split_region_at_seam(pts, (math.pi, 30 * D))
        rev = split_region_at_seam(list(reversed(pts)), (math.pi, 30 * D))
        self.assertEqual(fwd.area_winding, "ccw")
        self.assertEqual(rev.area_winding, "cw")
        self.assertAlmostEqual(
            sum(lp.signed_area for lp in rev.loops),
            -sum(lp.signed_area for lp in fwd.loops), delta=1e-9)
        ok, _ = _covers_identical(list(reversed(pts)), rev,
                                  (math.pi, 30 * D))
        self.assertTrue(ok)

    def test_degrees_entry_matches_radians(self):
        res_d = split_region_at_seam(CONVEX_DEG, (180.0, 25.0),
                                     units="deg")
        res_r = split_region_at_seam(_rad(CONVEX_DEG), (math.pi, 25 * D))
        self.assertEqual(len(res_d.loops), len(res_r.loops))
        self.assertAlmostEqual(res_d.signed_area, res_r.signed_area,
                               delta=1e-12)


class UnsupportedConditionsTests(unittest.TestCase):
    def test_outside_hemisphere_is_unsupported(self):
        with self.assertRaises(UnsupportedError) as ctx:
            split_region_at_seam(_rad(CONVEX_DEG), (0.0, 0.0))
        self.assertTrue(ctx.exception.location.startswith("vertex"))

    def test_pole_vertex_is_unsupported(self):
        deg = [(0.0, 60.0), (120.0, 60.0), (0.0, 90.0)]
        with self.assertRaises(UnsupportedError) as ctx:
            split_region_at_seam(_rad(deg), (0.0, 70 * D))
        self.assertIn("pole", str(ctx.exception))

    def test_edge_through_pole_is_unsupported(self):
        deg = [(0.0, 80.0), (180.0, 80.0), (90.0, 70.0)]
        with self.assertRaises(UnsupportedError) as ctx:
            split_region_at_seam(_rad(deg), (90 * D, 80 * D))
        self.assertEqual(ctx.exception.location, "edge 0")

    def test_bad_hemisphere_center_is_invalid_input(self):
        pts = _rad(CONVEX_DEG)
        with self.assertRaises(InvalidInputError):
            split_region_at_seam(pts, (float("nan"), 0.0))
        with self.assertRaises(InvalidInputError):
            split_region_at_seam(pts, (1, True))
        with self.assertRaises(InvalidInputError):
            split_region_at_seam(pts, (0.0, 0.0), units="furlong")

    def test_self_intersection_still_rejected(self):
        bowtie = _rad([(-20.0, 0.0), (20.0, 0.0),
                       (-20.0, 20.0), (20.0, 20.0)])
        with self.assertRaises(Exception):
            split_region_at_seam(bowtie, (0.0, 10 * D))


if __name__ == "__main__":
    unittest.main()
