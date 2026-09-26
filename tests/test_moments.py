"""Analytic, covariance, rotation, additivity and rejection tests for moments."""

import math
import os
import sys
import unittest

from antimeridian import NotInOpenHemisphereError, unfold_loop
from antimeridian.geometry import lonlat_to_xyz

sys.path.insert(0, os.path.dirname(__file__))
import reference as R  # noqa: E402

D = math.pi / 180.0


def mat_rows(result):
    return list(result.second_moment)


def rotation(axis, angle):
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return (
        (c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s),
        (y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s),
        (z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)),
    )


def rotate_vector(rot, v):
    return tuple(sum(rot[i][j] * v[j] for j in range(3)) for i in range(3))


def rotate_matrix(rot, matrix):
    out = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sum(
                rot[i][a] * matrix[a][b] * rot[j][b]
                for a in range(3) for b in range(3)
            )
    return out


def to_lonlat(v):
    return math.atan2(v[1], v[0]), math.asin(max(-1.0, min(1.0, v[2])))


def assert_vector_close(test, actual, expected, delta=1e-10):
    for a, e in zip(actual, expected):
        test.assertAlmostEqual(a, e, delta=delta)


def assert_matrix_close(test, actual, expected, delta=1e-10):
    for i in range(3):
        for j in range(3):
            test.assertAlmostEqual(actual[i][j], expected[i][j], delta=delta)


class RightTriangleMomentTests(unittest.TestCase):
    def setUp(self):
        self.phi = math.pi / 3.0
        self.points = [
            (0.0, 0.0),
            (self.phi, 0.0),
            (0.0, math.pi / 2.0),
        ]
        self.result = unfold_loop(self.points)

    def test_analytic_area_and_first_moment(self):
        phi = self.phi
        expected_moment = (
            math.pi / 4.0 * math.sin(phi),
            math.pi / 4.0 * (1.0 - math.cos(phi)),
            phi / 2.0,
        )
        self.assertAlmostEqual(self.result.signed_area, phi, delta=1e-14)
        assert_vector_close(self, self.result.first_moment, expected_moment,
                            delta=1e-14)
        centroid = tuple(v / math.sqrt(sum(x*x for x in expected_moment))
                         for v in expected_moment)
        assert_vector_close(self, self.result.centroid_direction, centroid)

    def test_analytic_second_moment_and_trace(self):
        phi = self.phi
        s2 = math.sin(2 * phi)
        expected = [
            [phi/3.0 + s2/6.0,
             math.sin(phi)**2/3.0, math.sin(phi)/3.0],
            [math.sin(phi)**2/3.0,
             phi/3.0 - s2/6.0,
             (1.0 - math.cos(phi)) / 3.0],
            [math.sin(phi)/3.0, (1.0 - math.cos(phi)) / 3.0, phi/3.0],
        ]
        assert_matrix_close(self, mat_rows(self.result), expected,
                            delta=1e-14)
        trace = sum(row[i] for i, row in enumerate(mat_rows(self.result)))
        self.assertAlmostEqual(trace, self.result.signed_area, delta=1e-14)

    def test_matches_independent_subdivision_reference(self):
        vertices = [R.to_xyz(*p) for p in self.points]
        area, first, second = R.richardson_triangle_moments(*vertices)
        self.assertAlmostEqual(self.result.signed_area, area, delta=1e-10)
        assert_vector_close(self, self.result.first_moment, first, 1e-7)
        assert_matrix_close(self, mat_rows(self.result),
                            [[second[0], second[3], second[4]],
                             [second[3], second[1], second[5]],
                             [second[4], second[5], second[2]]],
                            2e-7)


class CovarianceAndReversalTests(unittest.TestCase):
    def test_rotation_transforms_moments(self):
        points = [(0.1, 0.0), (0.7, 0.2), (0.2, 0.7)]
        original = unfold_loop(points)
        axis = tuple(a / math.sqrt(14.0) for a in (1.0, -2.0, 3.0))
        rot = rotation(axis, 0.987654321)
        rotated_points = [
            to_lonlat(rotate_vector(rot, lonlat_to_xyz(*p)))
            for p in points
        ]
        moved = unfold_loop(rotated_points)

        expected_first = rotate_vector(rot, original.first_moment)
        expected_second = rotate_matrix(rot, mat_rows(original))
        assert_vector_close(self, moved.first_moment, expected_first, 1e-11)
        assert_matrix_close(self, mat_rows(moved), expected_second, 1e-11)
        assert_vector_close(self, moved.centroid_direction,
                            rotate_vector(rot, original.centroid_direction),
                            1e-11)

    def test_reversal_negates_signed_moments_but_keeps_region_centroid(self):
        points = [(0.0, 0.0), (math.pi/3, 0.0), (0.0, math.pi/2)]
        forward = unfold_loop(points)
        backward = unfold_loop(list(reversed(points)))
        self.assertAlmostEqual(backward.signed_area, -forward.signed_area,
                               delta=1e-14)
        assert_vector_close(self, backward.first_moment,
                            tuple(-x for x in forward.first_moment), 1e-14)
        assert_matrix_close(
            self, mat_rows(backward),
                            [[-x for x in row] for row in mat_rows(forward)],
            1e-14)
        assert_vector_close(self, backward.centroid_direction,
                            forward.centroid_direction, 1e-14)


class AdditivityAndConcavityTests(unittest.TestCase):
    def test_two_known_wedges_sum_to_octant(self):
        a = unfold_loop([(0, 0), (math.pi/4, 0), (0, math.pi/2)])
        b = unfold_loop([(math.pi/4, 0), (math.pi/2, 0),
                         (0, math.pi/2)])
        total = unfold_loop([(0, 0), (math.pi/2, 0), (0, math.pi/2)])
        assert_vector_close(
            self, total.first_moment,
            tuple(x + y for x, y in zip(a.first_moment,
                                        b.first_moment)), 1e-13)
        assert_matrix_close(
            self, mat_rows(total),
            [[mat_rows(a)[i][j] + mat_rows(b)[i][j] for j in range(3)]
             for i in range(3)], 1e-13)

    def test_concave_polygon_matches_explicit_triangle_decomposition(self):
        def sphere_point(x, y):
            v = (x, y, 1.0)
            n = math.sqrt(x*x + y*y + 1.0)
            return (v[0]/n, v[1]/n, v[2]/n)

        planar = [(-1.0, 0.0), (1.0, 0.0), (0.2, 0.2), (0.0, 1.0)]
        vertices = [sphere_point(x, y) for x, y in planar]
        points = [to_lonlat(v) for v in vertices]
        result = unfold_loop(points)

        t1 = R.richardson_triangle_moments(
            vertices[0], vertices[1], vertices[2])
        t2 = R.richardson_triangle_moments(
            vertices[0], vertices[2], vertices[3])
        area = t1[0] + t2[0]
        first = tuple(x + y for x, y in zip(t1[1], t2[1]))
        second = tuple(x + y for x, y in zip(t1[2], t2[2]))
        self.assertAlmostEqual(result.signed_area, area, delta=1e-7)
        assert_vector_close(self, result.first_moment, first, 1e-7)
        assert_matrix_close(
            self, mat_rows(result),
            [[second[0], second[3], second[4]],
             [second[3], second[1], second[5]],
             [second[4], second[5], second[2]]], 1e-7)


class MomentRejectionTests(unittest.TestCase):
    def test_closed_hemisphere_is_rejected(self):
        points = [
            (0.0, 0.0),
            (90.0 * D, 0.0),
            (180.0 * D, 0.0),
            (0.0, 90.0 * D),
        ]
        with self.assertRaises(NotInOpenHemisphereError) as ctx:
            unfold_loop(points)
        self.assertEqual(ctx.exception.location, "loop")

    def test_antipodal_edge_still_rejected_before_moment(self):
        with self.assertRaisesRegex(Exception, "antipodal"):
            unfold_loop([(0.0, 0.0), (180.0 * D, 0.0),
                         (0.0, 30.0 * D)])


if __name__ == "__main__":
    unittest.main()
