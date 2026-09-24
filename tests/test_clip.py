"""Tests for hemisphere clipping of convex spherical polygons.

Expected values come from analytic geometry (the pi/4 half-octant, exact
symmetry halving, complement-to-whole relations) and from the independent
reference fan formula in ``tests/reference.py`` -- never from the clip
implementation itself.
"""

import math
import random
import unittest

from antimeridian import (
    AntipodalAdjacentError,
    DegenerateLoopError,
    InvalidInputError,
    NonConvexLoopError,
    SelfIntersectionError,
    clip_loop_hemisphere,
    clip_loop_hemisphere_degrees,
)
from antimeridian.clip import _split_antipodal_gaps
from antimeridian.geometry import central_angle, dot, lonlat_to_xyz

from reference import reference_signed_area

D = math.pi / 180.0

# Canonical one-eighth sphere triangle: vertices (1,0,0), (0,1,0), (0,0,1).
OCTANT = [(0.0, 0.0), (math.pi / 2, 0.0), (0.0, math.pi / 2)]


def ring_xyz(result):
    return [lonlat_to_xyz(p.lon, p.lat) for p in result.ring]


class FullKeepTests(unittest.TestCase):
    def test_octant_fully_inside(self):
        res = clip_loop_hemisphere(OCTANT, (0.0, 0.0))
        self.assertEqual(res.status, "full")
        self.assertAlmostEqual(res.area, math.pi / 2, places=14)
        self.assertAlmostEqual(res.signed_area, math.pi / 2, places=14)
        # ring is the input loop, unchanged and in order
        self.assertEqual(len(res.ring), 3)
        for got, (lo, la) in zip(res.ring, OCTANT):
            self.assertAlmostEqual(got.lon, lo, places=15)
            self.assertAlmostEqual(got.lat, la, places=15)

    def test_full_keep_degrees_wrapper(self):
        quad = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
        res = clip_loop_hemisphere_degrees(quad, (3.0, 4.0))
        self.assertEqual(res.status, "full")
        expect = reference_signed_area([(lo * D, la * D) for lo, la in quad])
        self.assertAlmostEqual(res.area, abs(expect), delta=1e-12)


class EmptyTests(unittest.TestCase):
    def test_all_vertices_outside(self):
        tri = [(0.0, -10 * D), (20 * D, -20 * D), (-20 * D, -20 * D)]
        res = clip_loop_hemisphere(tri, (0.0, math.pi / 2))  # north hemisphere
        self.assertEqual(res.status, "empty")
        self.assertEqual(res.ring, ())
        self.assertEqual(res.area, 0.0)
        self.assertEqual(res.signed_area, 0.0)

    def test_antipodal_hemisphere_of_octant(self):
        # Octant lies in x >= 0; the hemisphere x <= 0 keeps nothing
        # (two vertices sit exactly on the boundary circle: a measure-zero
        # touch, reported as empty).
        res = clip_loop_hemisphere(OCTANT, (math.pi, 0.0))
        self.assertEqual(res.status, "empty")
        self.assertEqual(res.ring, ())
        self.assertEqual(res.area, 0.0)


class AnalyticClipTests(unittest.TestCase):
    def test_octant_half_clip_area_pi_over_4(self):
        # Clip circle: plane x == y.  The kept piece has vertices
        # (0,1,0), (1,1,0)/sqrt(2), (0,0,1): a tri-rectangular triangle
        # with angles 90, 90, 45 degrees -> area pi/4 exactly.
        res = clip_loop_hemisphere(OCTANT, (135 * D, 0.0))
        self.assertEqual(res.status, "clipped")
        self.assertAlmostEqual(res.area, math.pi / 4, places=14)
        got = sorted((round(p.lon / D, 9), round(p.lat / D, 9))
                     for p in res.ring)
        self.assertEqual(got, [(0.0, 90.0), (45.0, 0.0), (90.0, 0.0)])

    def test_symmetric_square_is_halved(self):
        # Quad symmetric about lon 0; the lon>0 hemisphere keeps exactly
        # half of it (symmetry is the analytic relation, not the library).
        quad = [(-10 * D, -10 * D), (10 * D, -10 * D),
                (10 * D, 10 * D), (-10 * D, 10 * D)]
        res = clip_loop_hemisphere(quad, (90 * D, 0.0))
        self.assertEqual(res.status, "clipped")
        full = reference_signed_area(quad)
        self.assertAlmostEqual(res.area, abs(full) / 2, delta=1e-12)
        # ring: the two lon=+10 vertices plus two cut points on lon == 0
        self.assertEqual(len(res.ring), 4)
        lons = sorted(p.lon for p in res.ring)
        self.assertAlmostEqual(lons[0], 0.0, delta=1e-12)
        self.assertAlmostEqual(lons[1], 0.0, delta=1e-12)
        self.assertAlmostEqual(lons[2], 10 * D, delta=1e-12)
        self.assertAlmostEqual(lons[3], 10 * D, delta=1e-12)

    def test_clipped_area_matches_independent_reference(self):
        quad = [(-10 * D, -10 * D), (10 * D, -10 * D),
                (10 * D, 10 * D), (-10 * D, 10 * D)]
        res = clip_loop_hemisphere(quad, (90 * D, 0.0))
        ring_ll = [(p.lon, p.lat) for p in res.ring]
        expect = reference_signed_area(ring_ll)
        self.assertAlmostEqual(res.signed_area, expect, delta=1e-12)


class EdgeOnClipCircleTests(unittest.TestCase):
    # Quad whose western edge (lon == 0 meridian) lies exactly on the clip
    # great circle of the hemisphere centred at (90, 0).
    QUAD = [(0.0, -10 * D), (0.0, 10 * D), (20 * D, 10 * D), (20 * D, -10 * D)]

    def test_edge_on_circle_kept_side(self):
        res = clip_loop_hemisphere(self.QUAD, (90 * D, 0.0))
        self.assertEqual(res.status, "full")
        self.assertEqual(len(res.ring), 4)
        expect = reference_signed_area(self.QUAD)
        self.assertAlmostEqual(res.area, abs(expect), delta=1e-12)

    def test_edge_on_circle_excluded_side(self):
        # Only the boundary edge touches the kept hemisphere: empty.
        res = clip_loop_hemisphere(self.QUAD, (-90 * D, 0.0))
        self.assertEqual(res.status, "empty")
        self.assertEqual(res.ring, ())
        self.assertEqual(res.area, 0.0)

    def test_clip_circle_meridian_becomes_ring_edge(self):
        quad = [(-20 * D, -10 * D), (20 * D, -10 * D),
                (20 * D, 10 * D), (-20 * D, 10 * D)]
        res = clip_loop_hemisphere(quad, (90 * D, 0.0))
        self.assertEqual(res.status, "clipped")
        # two ring vertices lie on the clip circle (y == 0, x > 0)
        on_circle = [v for v in ring_xyz(res) if abs(v[1]) < 1e-9]
        self.assertEqual(len(on_circle), 2)
        for v in on_circle:
            self.assertGreater(v[0], 0.0)


class PolarRegionTests(unittest.TestCase):
    # Triangle around the north pole (contains it).
    CAP = [(0.0, 80 * D), (120 * D, 80 * D), (-120 * D, 80 * D)]

    def test_cap_inside_polar_hemisphere(self):
        res = clip_loop_hemisphere(self.CAP, (0.0, 90 * D))
        self.assertEqual(res.status, "full")
        expect = reference_signed_area(self.CAP)
        self.assertAlmostEqual(res.area, abs(expect), delta=1e-12)

    def test_cap_outside_opposite_hemisphere(self):
        res = clip_loop_hemisphere(self.CAP, (0.0, -90 * D))
        self.assertEqual(res.status, "empty")
        self.assertEqual(res.area, 0.0)

    def test_cap_split_by_meridian_hemisphere(self):
        a = clip_loop_hemisphere(self.CAP, (0.0, 0.0))
        b = clip_loop_hemisphere(self.CAP, (math.pi, 0.0))
        self.assertEqual(a.status, "clipped")
        self.assertEqual(b.status, "clipped")
        # complementary hemispheres partition the polygon (analytic relation)
        full = reference_signed_area(self.CAP)
        self.assertAlmostEqual(a.area + b.area, abs(full), delta=1e-12)
        self.assertLess(a.area, abs(full))
        self.assertGreater(a.area, 0.0)


class ComplementRelationTests(unittest.TestCase):
    """Deterministic small-scale enumeration: for random convex caps and
    random clip poles, the two complementary clips must partition the
    polygon, every ring vertex must lie in the kept closed hemisphere, and
    every output edge must be a unique short arc (< pi)."""

    def test_complement_partition_and_ring_invariants(self):
        rng = random.Random(20260924)
        for case in range(40):
            clon = rng.uniform(-150.0, 150.0) * D
            clat = rng.uniform(-60.0, 60.0) * D
            radius = rng.uniform(5.0, 25.0) * D
            k = rng.randint(3, 8)
            # vertices on a small circle around the centre, sorted by
            # angle -> convex spherical polygon
            poly = []
            for i in range(k):
                ang = 2.0 * math.pi * i / k
                la = clat + radius * math.sin(ang)
                lo = clon + radius * math.cos(ang) / max(0.2, math.cos(clat))
                lo = ((lo + math.pi) % (2 * math.pi)) - math.pi
                poly.append((lo, la))
            plon = rng.uniform(-180.0, 180.0) * D
            plat = rng.uniform(-90.0, 90.0) * D
            a = clip_loop_hemisphere(poly, (plon, plat))
            anti_lon = plon - math.pi if plon > 0.0 else plon + math.pi
            b = clip_loop_hemisphere(poly, (anti_lon, -plat))
            full = abs(reference_signed_area(poly))
            self.assertAlmostEqual(
                a.area + b.area, full, delta=1e-9,
                msg=f"case {case}: {a.area}+{b.area} != {full}")
            nvec = lonlat_to_xyz(plon, plat)
            for v in ring_xyz(a):
                self.assertGreaterEqual(dot(v, nvec), -1e-9)
            ring = ring_xyz(a)
            for i in range(len(ring)):
                ang = central_angle(ring[i - 1], ring[i])
                self.assertLess(ang, math.pi - 1e-9,
                                msg=f"case {case}: non-unique output arc")


class DefensiveSplitTests(unittest.TestCase):
    def test_antipodal_gap_is_split_on_clip_circle(self):
        # Defensive safeguard: an antipodal output pair (half-circle clip
        # boundary) must be split so every edge stays a unique short arc.
        nvec = (0.0, 0.0, 1.0)
        ring = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
        # polygon interior = y >= 0 half, so the +y midpoint must be chosen
        edge_normals = [(0.0, 0.0, 1.0)]
        out = _split_antipodal_gaps(ring, nvec, edge_normals, orient=1)
        self.assertEqual(len(out), 4)
        for i in range(len(out)):
            self.assertGreater(dot(out[i - 1], out[i]), -1.0 + 1e-9)
            self.assertAlmostEqual(dot(out[i], nvec), 0.0, places=12)
        ys = sorted(v[1] for v in out)
        self.assertAlmostEqual(ys[0], -1.0, places=12)
        self.assertAlmostEqual(ys[-1], 1.0, places=12)


class ClipErrorTests(unittest.TestCase):
    QUAD = [(-10 * D, -10 * D), (10 * D, -10 * D),
            (10 * D, 10 * D), (-10 * D, 10 * D)]

    def test_non_convex_rejected(self):
        concave = [(0.0, 0.0), (20 * D, 0.0), (10 * D, 5 * D),
                   (20 * D, 20 * D), (0.0, 20 * D)]
        with self.assertRaises(NonConvexLoopError) as ctx:
            clip_loop_hemisphere(concave, (0.0, 0.0))
        self.assertEqual(ctx.exception.vertex_index, 2)
        self.assertEqual(ctx.exception.location, "vertex 2")

    def test_clockwise_convex_accepted(self):
        cw = list(reversed(self.QUAD))
        res = clip_loop_hemisphere(cw, (0.0, 0.0))
        self.assertEqual(res.status, "full")
        self.assertLess(res.signed_area, 0.0)
        self.assertGreater(res.area, 0.0)

    def test_antipodal_adjacent_rejected(self):
        bad = [(0.0, 0.0), (math.pi, 0.0), (0.0, 10 * D)]
        with self.assertRaises(AntipodalAdjacentError):
            clip_loop_hemisphere(bad, (0.0, 0.0))

    def test_self_intersection_rejected(self):
        bowtie = [(-20 * D, 0.0), (20 * D, 0.0),
                  (-20 * D, 20 * D), (20 * D, 20 * D)]
        with self.assertRaises(SelfIntersectionError):
            clip_loop_hemisphere(bowtie, (0.0, 0.0))

    def test_great_circle_degenerate_rejected(self):
        line = [(0.0, 0.0), (60 * D, 0.0), (120 * D, 0.0)]
        with self.assertRaises(DegenerateLoopError):
            clip_loop_hemisphere(line, (0.0, 0.0))

    def test_bad_vertex_count(self):
        with self.assertRaises(InvalidInputError):
            clip_loop_hemisphere([(0.0, 0.0), (1.0, 1.0)], (0.0, 0.0))

    def test_bad_units(self):
        with self.assertRaises(InvalidInputError):
            clip_loop_hemisphere(self.QUAD, (0.0, 0.0), units="grad")

    def test_bad_pole_values(self):
        for bad in [(float("nan"), 0.0), (True, 0.0), (0.0, 2.0),
                    (4.0, 0.0), "x", (0.0,)]:
            with self.assertRaises(InvalidInputError, msg=f"pole={bad}"):
                clip_loop_hemisphere(self.QUAD, bad)

    def test_bad_vertex_values(self):
        with self.assertRaises(InvalidInputError):
            clip_loop_hemisphere(
                [(0.0, 0.0), (float("inf"), 0.0), (0.0, 1.0)], (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
