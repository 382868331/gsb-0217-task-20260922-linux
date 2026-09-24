"""Hemisphere clipping of a convex spherical loop.

Expected values come from hand-derived analytic geometry (the pi/6 triangle
below: interior angles 30/90/90 degrees), from the independent reference in
``tests/reference.py`` (Van Oosterom area, all-edge-normals containment), and
from the exact partition identity ``area(P ∩ H) + area(P ∩ -H) = area(P)``.
Nothing here calls the implementation under test to produce an expectation.
"""

import math
import os
import random
import sys
import unittest

from antimeridian import (
    AntipodalAdjacentError,
    DegenerateLoopError,
    InvalidInputError,
    NonConvexLoopError,
    SelfIntersectionError,
    clip_loop_to_hemisphere,
    clip_loop_to_hemisphere_degrees,
    unfold_loop,
)
from antimeridian.geometry import dot, lonlat_to_xyz

sys.path.insert(0, os.path.dirname(__file__))
from reference import reference_contains, reference_signed_area  # noqa: E402

D = math.pi / 180.0

# Canonical one-eighth-sphere triangle: area pi/2.
OCTANT = [(0.0, 0.0), (math.pi / 2, 0.0), (0.0, math.pi / 2)]

# Triangle enclosing the north pole (all vertices at latitude 80).
POLAR_CAP = [(0.0, 80.0 * D), (120.0 * D, 80.0 * D), (-120.0 * D, 80.0 * D)]


def deg(points):
    return [(lo * D, la * D) for lo, la in points]


def random_convex_polygon(rng):
    """Convex loop: vertices on a small circle around a random centre,
    sorted by azimuth (locally a convex planar polygon, rho < pi/2)."""
    z = rng.uniform(-0.8, 0.8)
    phi = rng.uniform(-math.pi, math.pi)
    r = math.sqrt(1.0 - z * z)
    c = (r * math.cos(phi), r * math.sin(phi), z)
    # local tangent basis at c
    ref = (0.0, 0.0, 1.0) if abs(z) < 0.9 else (1.0, 0.0, 0.0)
    e1 = (c[1] * ref[2] - c[2] * ref[1],
          c[2] * ref[0] - c[0] * ref[2],
          c[0] * ref[1] - c[1] * ref[0])
    m = math.sqrt(sum(v * v for v in e1))
    e1 = tuple(v / m for v in e1)
    e2 = (c[1] * e1[2] - c[2] * e1[1],
          c[2] * e1[0] - c[0] * e1[2],
          c[0] * e1[1] - c[1] * e1[0])
    n = rng.randint(3, 8)
    rho = rng.uniform(0.2, 1.1)
    pts = []
    for a in sorted(rng.uniform(0.0, 2.0 * math.pi) for _ in range(n)):
        v = tuple(
            math.cos(rho) * c[k]
            + math.sin(rho) * (math.cos(a) * e1[k] + math.sin(a) * e2[k])
            for k in range(3)
        )
        nv = math.sqrt(sum(x * x for x in v))
        x, y, zz = (v[k] / nv for k in range(3))
        pts.append((math.atan2(y, x), math.asin(max(-1.0, min(1.0, zz)))))
    return pts


def random_direction(rng):
    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(-math.pi, math.pi)
    r = math.sqrt(1.0 - z * z)
    return (r * math.cos(phi), r * math.sin(phi), z)


class AnalyticClipTests(unittest.TestCase):
    def test_full_keep_octant_northern_hemisphere(self):
        res = clip_loop_to_hemisphere(OCTANT, (0.0, math.pi / 2))
        self.assertEqual(res.status, "full")
        self.assertEqual(len(res.points), 3)
        for got, exp in zip(res.points, OCTANT):
            self.assertAlmostEqual(got.lon, exp[0], delta=1e-15)
            self.assertAlmostEqual(got.lat, exp[1], delta=1e-15)
        self.assertAlmostEqual(res.area, math.pi / 2, delta=1e-12)
        self.assertAlmostEqual(res.signed_area, math.pi / 2, delta=1e-12)
        self.assertEqual(res.winding, "ccw")

    def test_full_exclude_southern_triangle(self):
        tri = deg([(0.0, -10.0), (90.0, -10.0), (45.0, -60.0)])
        res = clip_loop_to_hemisphere(tri, (0.0, math.pi / 2))
        self.assertEqual(res.status, "empty")
        self.assertEqual(res.points, ())
        self.assertEqual(res.area, 0.0)
        self.assertEqual(res.signed_area, 0.0)

    def test_partial_octant_analytic_pi_over_6(self):
        # Hemisphere pole at lon 180, lat 30: horizon z = sqrt(3) x cuts the
        # octant triangle into the triangle (90,0)-(0,90)-(0,60) whose
        # interior angles are 30/90/90 degrees -> area pi/6 (hand-derived,
        # re-verified via dihedral angles above this test file's claims).
        res = clip_loop_to_hemisphere(OCTANT, (math.pi, math.pi / 6))
        self.assertEqual(res.status, "partial")
        self.assertEqual(len(res.points), 3)
        expected = [(math.pi / 2, 0.0), (0.0, math.pi / 2), (0.0, math.pi / 3)]
        for got, (elo, ela) in zip(res.points, expected):
            self.assertAlmostEqual(got.lon, elo, delta=1e-9)
            self.assertAlmostEqual(got.lat, ela, delta=1e-9)
        self.assertAlmostEqual(res.area, math.pi / 6, delta=1e-11)
        self.assertAlmostEqual(res.signed_area, math.pi / 6, delta=1e-11)
        # The closing edge (0,60) -> (90,0) lies exactly on the clip circle.
        pv = lonlat_to_xyz(math.pi, math.pi / 6)
        for p in (res.points[2], res.points[0]):
            self.assertAlmostEqual(
                dot(lonlat_to_xyz(p.lon, p.lat), pv), 0.0, delta=1e-12)

    def test_partition_identity_on_analytic_case(self):
        # Clipping by a hemisphere and its opposite partitions the polygon.
        north_cut = clip_loop_to_hemisphere(OCTANT, (math.pi, math.pi / 6))
        south_cut = clip_loop_to_hemisphere(OCTANT, (0.0, -math.pi / 6))
        self.assertAlmostEqual(north_cut.area + south_cut.area,
                               math.pi / 2, delta=1e-11)
        self.assertAlmostEqual(south_cut.area, math.pi / 3, delta=1e-11)

    def test_edge_on_clip_circle_kept(self):
        # Edge (0,0)->(90,0) is the equator, exactly on the horizon of the
        # northern hemisphere; the whole triangle is inside.
        tri = deg([(0.0, 0.0), (90.0, 0.0), (45.0, 60.0)])
        res = clip_loop_to_hemisphere(tri, (0.0, math.pi / 2))
        self.assertEqual(res.status, "full")
        self.assertEqual(len(res.points), 3)
        self.assertAlmostEqual(
            res.area, abs(reference_signed_area(tri)), delta=1e-12)

    def test_edge_on_clip_circle_excluded(self):
        # Same triangle against the southern hemisphere: only the equator
        # edge touches -> zero-area contact -> empty.
        tri = deg([(0.0, 0.0), (90.0, 0.0), (45.0, 60.0)])
        res = clip_loop_to_hemisphere(tri, (0.0, -math.pi / 2))
        self.assertEqual(res.status, "empty")
        self.assertEqual(res.area, 0.0)

    def test_tangent_vertex_contact_is_empty(self):
        # Octant vs southern hemisphere: two vertices sit on the equator
        # (the clip circle), the third is outside -> degenerate contact.
        res = clip_loop_to_hemisphere(OCTANT, (0.0, -math.pi / 2))
        self.assertEqual(res.status, "empty")
        self.assertEqual(res.points, ())

    def test_reversed_orientation_sign(self):
        cw = list(reversed(OCTANT))
        res = clip_loop_to_hemisphere(cw, (math.pi, math.pi / 6))
        self.assertEqual(res.winding, "cw")
        self.assertAlmostEqual(res.signed_area, -math.pi / 6, delta=1e-11)
        self.assertAlmostEqual(res.area, math.pi / 6, delta=1e-11)


class PolarClipTests(unittest.TestCase):
    def test_polar_cap_fully_inside_northern(self):
        res = clip_loop_to_hemisphere(POLAR_CAP, (0.0, math.pi / 2))
        self.assertEqual(res.status, "full")
        self.assertAlmostEqual(
            res.area, abs(reference_signed_area(POLAR_CAP)), delta=1e-12)

    def test_polar_cap_fully_outside_southern(self):
        res = clip_loop_to_hemisphere(POLAR_CAP, (0.0, -math.pi / 2))
        self.assertEqual(res.status, "empty")

    def test_polar_cap_cut_by_tilted_hemisphere(self):
        # Horizon through both poles (pole on the equator): cuts the cap.
        pole = (10.0 * D, 0.0)
        res = clip_loop_to_hemisphere(POLAR_CAP, pole)
        self.assertEqual(res.status, "partial")
        # Independent area formula must agree on the clipped ring.
        ring = [(p.lon, p.lat) for p in res.points]
        self.assertAlmostEqual(res.signed_area,
                               reference_signed_area(ring), delta=1e-9)
        # Partition identity with the opposite hemisphere.
        opp = clip_loop_to_hemisphere(POLAR_CAP, (pole[0] + math.pi
                                                  if pole[0] <= 0
                                                  else pole[0] - math.pi,
                                                  0.0))
        total = abs(reference_signed_area(POLAR_CAP))
        self.assertAlmostEqual(res.area + opp.area, total, delta=1e-9)
        # Independent all-edge-normals containment must agree with
        # "inside input AND inside hemisphere" on random probes.
        rng = random.Random(778899)
        pv = lonlat_to_xyz(*pole)
        for _ in range(300):
            v = random_direction(rng)
            lon = math.atan2(v[1], v[0])
            lat = math.asin(max(-1.0, min(1.0, v[2])))
            expect = (reference_contains(POLAR_CAP, lon, lat)
                      and dot(v, pv) >= 0.0)
            got = reference_contains(ring, lon, lat)
            self.assertEqual(got, expect,
                             f"containment mismatch at {(lon, lat)}")


class RandomClipPropertyTests(unittest.TestCase):
    def test_partition_identity_and_reference_area(self):
        rng = random.Random(20260924)
        for case in range(30):
            poly = random_convex_polygon(rng)
            pv = random_direction(rng)
            pole = (math.atan2(pv[1], pv[0]),
                    math.asin(max(-1.0, min(1.0, pv[2]))))
            anti = (pole[0] - math.pi if pole[0] > 0 else pole[0] + math.pi,
                    -pole[1])
            a = clip_loop_to_hemisphere(poly, pole)
            b = clip_loop_to_hemisphere(poly, anti)
            total = abs(reference_signed_area(poly))
            with self.subTest(case=case, status=(a.status, b.status)):
                self.assertAlmostEqual(a.area + b.area, total, delta=1e-9)
                for res in (a, b):
                    if res.status == "empty":
                        continue
                    ring = [(p.lon, p.lat) for p in res.points]
                    self.assertAlmostEqual(
                        res.signed_area, reference_signed_area(ring),
                        delta=1e-9)

    def test_clipped_vertices_are_input_or_on_clip_circle(self):
        rng = random.Random(13579)
        for case in range(20):
            poly = random_convex_polygon(rng)
            pv = random_direction(rng)
            pole = (math.atan2(pv[1], pv[0]),
                    math.asin(max(-1.0, min(1.0, pv[2]))))
            res = clip_loop_to_hemisphere(poly, pole)
            if res.status != "partial":
                continue
            in_xyz = [lonlat_to_xyz(*p) for p in poly]
            for p in res.points:
                v = lonlat_to_xyz(p.lon, p.lat)
                near_vertex = any(dot(v, w) > 1.0 - 1e-12 for w in in_xyz)
                on_circle = abs(dot(v, pv)) < 1e-9
                self.assertTrue(near_vertex or on_circle,
                                f"case {case}: stray clipped vertex")

    def test_containment_matches_reference(self):
        rng = random.Random(24680)
        for case in range(6):
            poly = random_convex_polygon(rng)
            pv = random_direction(rng)
            pole = (math.atan2(pv[1], pv[0]),
                    math.asin(max(-1.0, min(1.0, pv[2]))))
            res = clip_loop_to_hemisphere(poly, pole)
            if res.status == "empty":
                continue
            ring = [(p.lon, p.lat) for p in res.points]
            for _ in range(200):
                v = random_direction(rng)
                if abs(dot(v, pv)) < 1e-6:
                    continue  # keep probes off the clip boundary
                lon = math.atan2(v[1], v[0])
                lat = math.asin(max(-1.0, min(1.0, v[2])))
                expect = (reference_contains(poly, lon, lat)
                          and dot(v, pv) > 0.0)
                with self.subTest(case=case, probe=(lon, lat)):
                    self.assertEqual(reference_contains(ring, lon, lat),
                                     expect)


class ClipErrorTests(unittest.TestCase):
    def test_non_convex_rejected(self):
        concave = deg([(0.0, 0.0), (40.0, 0.0), (10.0, 10.0),
                       (40.0, 20.0), (0.0, 20.0)])
        with self.assertRaises(NonConvexLoopError) as cm:
            clip_loop_to_hemisphere(concave, (0.0, math.pi / 2))
        self.assertEqual(cm.exception.vertex_index, 2)
        self.assertEqual(cm.exception.location, "vertex 2")

    def test_bad_pole_rejected(self):
        for pole, loc in [((math.nan, 0.0), "pole.lon"),
                          ((0.0, math.inf), "pole.lat"),
                          ((True, 0.0), "pole.lon"),
                          ((0.0, 2.0), "pole.lat"),
                          ((4.0, 0.0), "pole.lon")]:
            with self.subTest(pole=pole):
                with self.assertRaises(InvalidInputError) as cm:
                    clip_loop_to_hemisphere(OCTANT, pole)
                self.assertEqual(cm.exception.location, loc)

    def test_loop_errors_propagate(self):
        with self.assertRaises(AntipodalAdjacentError):
            clip_loop_to_hemisphere(
                [(0.0, 0.0), (math.pi, 0.0), (0.5, 0.5)], (0.0, 1.0))
        with self.assertRaises(SelfIntersectionError):
            clip_loop_to_hemisphere(
                deg([(-20.0, 0.0), (20.0, 0.0),
                     (-20.0, 20.0), (20.0, 20.0)]), (0.0, 1.0))
        with self.assertRaises(DegenerateLoopError):
            clip_loop_to_hemisphere(
                [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], (0.0, 1.0))
        with self.assertRaises(InvalidInputError):
            clip_loop_to_hemisphere([(0.0, 0.0), (1.0, 0.0)], (0.0, 1.0))
        with self.assertRaises(InvalidInputError):
            clip_loop_to_hemisphere(OCTANT, (0.0, 1.0), units="grad")

    def test_degrees_mode(self):
        res = clip_loop_to_hemisphere_degrees(
            [(0.0, 0.0), (90.0, 0.0), (0.0, 90.0)], (180.0, 30.0))
        self.assertEqual(res.status, "partial")
        self.assertAlmostEqual(res.area, math.pi / 6, delta=1e-11)
        full = clip_loop_to_hemisphere_degrees(
            [(0.0, 0.0), (90.0, 0.0), (0.0, 90.0)], (0.0, 90.0))
        self.assertEqual(full.status, "full")
        self.assertAlmostEqual(full.area, math.pi / 2, delta=1e-12)

    def test_pure_and_repeatable(self):
        poly = [list(p) for p in deg([(10.0, 5.0), (80.0, 10.0),
                                      (40.0, 50.0)])]
        snapshot = [list(p) for p in poly]
        r1 = clip_loop_to_hemisphere(poly, (0.3, 0.4))
        r2 = clip_loop_to_hemisphere(poly, (0.3, 0.4))
        self.assertEqual(poly, snapshot)  # input untouched
        self.assertEqual(r1, r2)          # deterministic, stateless
        # The pre-existing entry point is unaffected by the new one.
        self.assertAlmostEqual(unfold_loop(OCTANT).signed_area,
                               math.pi / 2, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
