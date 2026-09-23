"""Error semantics: locatable, non-silent rejection of illegal input."""

import math
import unittest

from antimeridian import (
    AntipodalAdjacentError,
    DegenerateLoopError,
    InvalidInputError,
    SelfIntersectionError,
    unfold_loop,
)

D = math.pi / 180.0


def expect_invalid(testcase, points, location=None, units="rad"):
    with testcase.assertRaises(InvalidInputError) as cm:
        unfold_loop(points, units=units)
    if location is not None:
        testcase.assertEqual(cm.exception.location, location)
    return cm.exception


class NumericValidationTests(unittest.TestCase):
    def test_nan_rejected(self):
        expect_invalid(self, [(math.nan, 0.0), (0.0, 0.0), (1.0, 1.0)],
                       "vertex 0.lon")

    def test_infinity_rejected(self):
        expect_invalid(self, [(0.0, math.inf), (0.0, 0.0), (1.0, 1.0)],
                       "vertex 0.lat")
        expect_invalid(self, [(-math.inf, 0.0), (0.0, 0.0), (1.0, 1.0)],
                       "vertex 0.lon")

    def test_bool_rejected_as_number(self):
        expect_invalid(self, [(True, 0.0), (0.0, 0.0), (1.0, 1.0)],
                       "vertex 0.lon")
        expect_invalid(self, [(0.0, False), (0.0, 0.0), (1.0, 1.0)],
                       "vertex 0.lat")

    def test_bool_rejected_in_degrees_too(self):
        expect_invalid(self, [(True, 50.0), (0.0, 0.0), (10.0, 10.0)],
                       "vertex 0.lon", units="deg")

    def test_lat_out_of_range(self):
        expect_invalid(self, [(0.0, 1.6), (0.0, 0.0), (1.0, 1.0)],
                       "vertex 0.lat")
        expect_invalid(self, [(0.0, -1.6), (0.0, 0.0), (1.0, 1.0)],
                       "vertex 0.lat")

    def test_lon_out_of_range(self):
        expect_invalid(self, [(3.2, 0.0), (0.0, 0.0), (1.0, 1.0)],
                       "vertex 0.lon")

    def test_degree_range_checked(self):
        expect_invalid(self, [(0.0, 91.0), (0.0, 0.0), (10.0, 10.0)],
                       "vertex 0.lat", units="deg")

    def test_string_vertex_rejected(self):
        expect_invalid(self, ["ab", (0.0, 0.0), (1.0, 1.0)], "vertex 0")

    def test_wrong_arity_vertex(self):
        expect_invalid(self, [(0.0,), (1.0, 1.0), (2.0, 2.0)], "vertex 0")

    def test_bad_units(self):
        expect_invalid(self, [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
                       "units", units="mm")


class CountValidationTests(unittest.TestCase):
    def test_too_few(self):
        expect_invalid(self, [(0.0, 0.0), (1.0, 1.0)], "vertices")

    def test_too_many(self):
        expect_invalid(self, [(0.0, 0.0)] * 101, "vertices")

    def test_boundary_counts_accepted_structurally(self):
        # 3 and 100 vertices pass the count check (may later raise on
        # antipodal/degenerate geometry, but not on count).  Use a small,
        # strictly convex spherical polygon so the loops are valid.
        for k in (3, 100):
            pts = []
            for j in range(k):
                th = 2 * math.pi * j / k
                pts.append((0.05 * math.cos(th), 0.05 * math.sin(th)))
            try:
                unfold_loop(pts)
            except InvalidInputError as exc:
                self.assertNotEqual(exc.location, "vertices")


class GeometryRejectionTests(unittest.TestCase):
    def test_antipodal_adjacent(self):
        with self.assertRaises(AntipodalAdjacentError) as cm:
            unfold_loop([(0.0, 0.0), (math.pi, 0.0), (1.0, 1.0)])
        self.assertEqual(cm.exception.edge_index, 0)
        self.assertEqual(cm.exception.location, "edge 0")

    def test_duplicate_consecutive_vertices(self):
        expect_invalid(self, [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)], "edge 0")

    def test_great_circle_degenerate(self):
        with self.assertRaises(DegenerateLoopError):
            unfold_loop([(0.0, 10 * D), (0.0, 40 * D), (0.0, -20 * D)])

    def test_self_intersection_rejected(self):
        bowtie = [(-20 * D, 0.0), (20 * D, 0.0),
                  (-20 * D, 20 * D), (20 * D, 20 * D)]
        with self.assertRaises(SelfIntersectionError) as cm:
            unfold_loop(bowtie)
        self.assertEqual(cm.exception.edge_indices, (1, 3))
        self.assertIn("1", cm.exception.location)


class PurityTests(unittest.TestCase):
    def test_failure_leaves_no_partial_state(self):
        # The API is a pure function: a rejection simply returns no result.
        with self.assertRaises(InvalidInputError):
            unfold_loop([(math.nan, 0.0), (0.0, 0.0), (1.0, 1.0)])
        # A subsequent valid call behaves exactly like the first (no cache).
        tri = [(0.0, 0.0), (math.pi / 2, 0.0), (0.0, math.pi / 2)]
        self.assertAlmostEqual(unfold_loop(tri).signed_area, math.pi / 2)

    def test_input_list_not_mutated(self):
        pts = [(0.0, 0.0), (math.pi / 2, 0.0), (0.0, math.pi / 2)]
        snapshot = list(pts)
        unfold_loop(pts)
        self.assertEqual(pts, snapshot)


if __name__ == "__main__":
    unittest.main()
