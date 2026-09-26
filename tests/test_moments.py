"""Tests for the signed 3-D first/second area moments (region_moments).

Independent expectations come from

* exact analytic values for the spherical right (octant) triangle,
* rigid-rotation covariance (M -> R M, S -> R S R^T),
* additivity (a quad split into two triangles),
* tests/moment_reference.py, a high-resolution subdivision integrator
  that does not import the package under test.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import moment_reference as mr  # noqa: E402  (independent, no antimeridian)

from antimeridian import (  # noqa: E402
    AntipodalAdjacentError,
    DegenerateLoopError,
    InvalidInputError,
    SelfIntersectionError,
    region_moments,
    region_moments_degrees,
    unfold_loop,
)

PI = math.pi


def rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def rot_y(t):
    c, s = math.cos(t), math.sin(t)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def mat_mul(A, B):
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def mat_vec(A, v):
    return tuple(sum(A[i][k] * v[k] for k in range(3)) for i in range(3))


def transpose(A):
    return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))


def to_xyz(lon, lat):
    c = math.cos(lat)
    return (c * math.cos(lon), c * math.sin(lon), math.sin(lat))


def to_lonlat(v):
    x, y, z = v
    return (math.atan2(y, x), math.asin(max(-1.0, min(1.0, z))))


def rotate_points(points, R):
    return [to_lonlat(mat_vec(R, to_xyz(lo, la))) for lo, la in points]


OCTANT = [(0.0, 0.0), (PI / 2, 0.0), (0.0, PI / 2)]
QUAD = [(0.1, 0.2), (1.0, 0.15), (1.1, 0.8), (0.2, 0.9)]
DATELINE = [(2.9, 0.1), (-2.9, 0.1), (-2.9, 0.9), (2.9, 0.9)]
CONCAVE = [(0.1, 0.1), (1.0, 0.1), (1.0, 0.5),
           (0.5, 0.5), (0.5, 1.0), (0.1, 1.0)]


class OctantAnalytic(unittest.TestCase):
    def test_octant_exact(self):
        r = region_moments(OCTANT)
        self.assertAlmostEqual(r.signed_area, PI / 2, places=14)
        self.assertEqual(r.area_winding, "ccw")
        for k in range(3):
            self.assertAlmostEqual(r.first_moment[k], PI / 4, places=14)
        for i in range(3):
            for j in range(3):
                want = PI / 6 if i == j else 1.0 / 3.0
                self.assertAlmostEqual(r.second_moment[i][j], want, places=14)
        inv = 1.0 / math.sqrt(3.0)
        for k in range(3):
            self.assertAlmostEqual(r.direction_centroid[k], inv, places=14)

    def test_trace_equals_area(self):
        r = region_moments(OCTANT)
        trace = sum(r.second_moment[i][i] for i in range(3))
        self.assertAlmostEqual(trace, r.signed_area, places=14)

    def test_degrees_wrapper(self):
        r = region_moments_degrees([(0.0, 0.0), (90.0, 0.0), (0.0, 90.0)])
        self.assertAlmostEqual(r.signed_area, PI / 2, places=13)


class Reversal(unittest.TestCase):
    def test_reversal_negates_all_signed_quantities(self):
        fwd = region_moments(QUAD)
        rev = region_moments(list(reversed(QUAD)))
        self.assertAlmostEqual(fwd.signed_area, -rev.signed_area, places=13)
        for k in range(3):
            self.assertAlmostEqual(
                fwd.first_moment[k], -rev.first_moment[k], places=13)
            self.assertAlmostEqual(
                fwd.direction_centroid[k], -rev.direction_centroid[k],
                places=13)
            for l in range(3):
                self.assertAlmostEqual(
                    fwd.second_moment[k][l], -rev.second_moment[k][l],
                    places=13)
        self.assertEqual(fwd.area_winding, "ccw")
        self.assertEqual(rev.area_winding, "cw")


class RotationCovariance(unittest.TestCase):
    def test_moments_rotate_as_rm_and_rsrt(self):
        R = mat_mul(rot_z(math.radians(37.0)), rot_y(math.radians(51.0)))
        base = region_moments(OCTANT)
        rot = region_moments(rotate_points(OCTANT, R))
        self.assertAlmostEqual(base.signed_area, rot.signed_area, places=12)
        rm = mat_vec(R, base.first_moment)
        for k in range(3):
            self.assertAlmostEqual(rm[k], rot.first_moment[k], places=12)
        rsrt = mat_mul(mat_mul(R, base.second_moment), transpose(R))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    rsrt[i][j], rot.second_moment[i][j], places=12)
        rc = mat_vec(R, base.direction_centroid)
        for k in range(3):
            self.assertAlmostEqual(rc[k], rot.direction_centroid[k], places=12)


class Additivity(unittest.TestCase):
    def test_quad_equals_sum_of_two_triangles(self):
        a, b, c, d = QUAD
        whole = region_moments(QUAD)
        t1 = region_moments([a, b, c])
        t2 = region_moments([a, c, d])
        self.assertAlmostEqual(
            whole.signed_area, t1.signed_area + t2.signed_area, places=12)
        for k in range(3):
            self.assertAlmostEqual(
                whole.first_moment[k],
                t1.first_moment[k] + t2.first_moment[k], places=12)
            for l in range(3):
                self.assertAlmostEqual(
                    whole.second_moment[k][l],
                    t1.second_moment[k][l] + t2.second_moment[k][l],
                    places=12)


class IndependentReference(unittest.TestCase):
    """Cross-check against the subdivision integrator (no shared code)."""

    def check(self, points, tol=2e-5):
        got = region_moments(points)
        area, mom, sec = mr.reference_moments(points)
        self.assertAlmostEqual(got.signed_area, area, delta=tol)
        for k in range(3):
            self.assertAlmostEqual(got.first_moment[k], mom[k], delta=tol)
            for l in range(3):
                self.assertAlmostEqual(
                    got.second_moment[k][l], sec[k][l], delta=tol)

    def test_octant(self):
        self.check(OCTANT)

    def test_quad(self):
        self.check(QUAD)

    def test_concave(self):
        self.check(CONCAVE)

    def test_dateline_crossing(self):
        # Computation is on 3-D directions; the date line is only a
        # coordinate representation and needs no special handling.
        self.check(DATELINE)

    def test_rotated_quad(self):
        R = mat_mul(rot_z(1.1), rot_y(0.4))
        self.check(rotate_points(QUAD, R))


class CentroidMeaning(unittest.TestCase):
    def test_centroid_is_not_vertex_average(self):
        # Guard against faking the area centroid by averaging vertices.
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.6), (0.0, 0.05)]
        r = region_moments(pts)
        vs = [to_xyz(lo, la) for lo, la in pts]
        avg = tuple(sum(v[k] for v in vs) / len(vs) for k in range(3))
        n = math.sqrt(sum(a * a for a in avg))
        avg = tuple(a / n for a in avg)
        diff = max(abs(avg[k] - r.direction_centroid[k]) for k in range(3))
        self.assertGreater(diff, 1e-3)
        # ... while the true centroid still matches the independent
        # numerical integral of x dA.
        _, mom, _ = mr.reference_moments(pts)
        mag = math.sqrt(sum(m * m for m in mom))
        for k in range(3):
            self.assertAlmostEqual(
                r.direction_centroid[k], mom[k] / mag, delta=2e-5)

    def test_zero_moment_region_has_no_centroid(self):
        # Dumbbell-shaped region (two lobes joined by a neck, symmetric
        # about lon=0 and lat=0) tuned so the first moment vanishes: the
        # lobe span b below is the exact root of M_x(b) = 0, and M_y = M_z
        # = 0 by the two reflection symmetries.  No centroid direction
        # exists, so direction_centroid must be None.  The region is NOT
        # inside an open hemisphere (|area| > pi) -- inside one the first
        # moment is always nonzero.
        b = math.radians(154.39798664667245)
        pts_deg = [
            (60, 10), (60, 30), (math.degrees(b), 30), (math.degrees(b), -30),
            (60, -30), (60, -10), (-60, -10), (-60, -30),
            (-math.degrees(b), -30), (-math.degrees(b), 30), (-60, 30),
            (-60, 10),
        ]
        pts = [(math.radians(lo), math.radians(la)) for lo, la in pts_deg]
        r = region_moments(pts)
        mag = math.sqrt(sum(m * m for m in r.first_moment))
        self.assertLess(mag, 1e-12)
        self.assertIsNone(r.direction_centroid)
        # trace(S) == signed_area still holds (this loop also exercises
        # the left-area > 2*pi correction branch).
        trace = sum(r.second_moment[i][i] for i in range(3))
        self.assertAlmostEqual(trace, r.signed_area, places=10)
        # Independent numerical integration agrees the moment vanishes.
        _, mom, _ = mr.reference_moments(pts)
        self.assertLess(max(abs(m) for m in mom), 1e-4)


class ConsistencyWithUnfold(unittest.TestCase):
    def test_signed_area_matches_unfold_loop(self):
        for pts in (OCTANT, QUAD, CONCAVE, DATELINE):
            self.assertAlmostEqual(
                region_moments(pts).signed_area,
                unfold_loop(pts).signed_area,
                places=13)


class ErrorPaths(unittest.TestCase):
    def test_nan_rejected(self):
        with self.assertRaises(InvalidInputError):
            region_moments([(0.0, 0.0), (float("nan"), 0.1), (0.2, 0.3)])

    def test_bool_rejected(self):
        with self.assertRaises(InvalidInputError):
            region_moments([(0.0, 0.0), (True, 0.1), (0.2, 0.3)])

    def test_antipodal_rejected(self):
        with self.assertRaises(AntipodalAdjacentError):
            region_moments([(0.0, 0.0), (PI, 0.0), (0.0, 0.5)])

    def test_self_intersection_rejected(self):
        bowtie = [(0.0, 0.0), (1.0, 0.5), (1.0, 0.0), (0.0, 0.5)]
        with self.assertRaises(SelfIntersectionError):
            region_moments(bowtie)

    def test_degenerate_great_circle_rejected(self):
        with self.assertRaises(DegenerateLoopError):
            region_moments([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    def test_vertex_count_rejected(self):
        with self.assertRaises(InvalidInputError):
            region_moments([(0.0, 0.0), (1.0, 0.0)])

    def test_failure_leaves_no_partial_result(self):
        # Pure function: a raising call returns nothing and mutates nothing.
        try:
            region_moments([(0.0, 0.0), (PI, 0.0), (0.0, 0.5)])
        except AntipodalAdjacentError as exc:
            self.assertIn("edge 0", str(exc))
        else:
            self.fail("expected AntipodalAdjacentError")


if __name__ == "__main__":
    unittest.main()
