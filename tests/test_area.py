"""Analytic and rotation-invariance tests for signed area (acceptance)."""

import math
import random
import unittest

from antimeridian import unfold_loop
from antimeridian.geometry import lonlat_to_xyz

D = math.pi / 180.0


def rodrigues(v, axis, ang):
    x, y, z = v
    ux, uy, uz = axis
    c, s = math.cos(ang), math.sin(ang)
    return (
        (c + ux * ux * (1 - c)) * x
        + (ux * uy * (1 - c) - uz * s) * y
        + (ux * uz * (1 - c) + uy * s) * z,
        (uy * ux * (1 - c) + uz * s) * x
        + (c + uy * uy * (1 - c)) * y
        + (uy * uz * (1 - c) - ux * s) * z,
        (uz * ux * (1 - c) - uy * s) * x
        + (uz * uy * (1 - c) + ux * s) * y
        + (c + uz * uz * (1 - c)) * z,
    )


def to_lonlat(v):
    x, y, z = v
    return math.atan2(y, x), math.asin(max(-1.0, min(1.0, z)))


# Canonical one-eighth sphere tri-rectangular triangle:
# vertices (1,0,0), (0,1,0), (0,0,1); three right angles -> area pi/2.
OCTANT = [(0.0, 0.0), (math.pi / 2, 0.0), (0.0, math.pi / 2)]


class AreaAnalyticTests(unittest.TestCase):
    def test_octant_area_pi_over_2(self):
        res = unfold_loop(OCTANT)
        self.assertAlmostEqual(res.signed_area, math.pi / 2, places=14)
        self.assertEqual(res.area_winding, "ccw")
        self.assertGreater(res.signed_area, 0.0)

    def test_area_interval(self):
        res = unfold_loop(OCTANT)
        self.assertGreater(res.signed_area, -2 * math.pi)
        self.assertLessEqual(res.signed_area, 2 * math.pi)

    def test_reversal_negates_area(self):
        fwd = unfold_loop(OCTANT).signed_area
        rev = unfold_loop(list(reversed(OCTANT))).signed_area
        self.assertAlmostEqual(rev, -fwd, places=14)
        self.assertEqual(unfold_loop(list(reversed(OCTANT))).area_winding, "cw")

    def test_rotation_preserves_area(self):
        rng = random.Random(20260923)
        base = unfold_loop(OCTANT).signed_area
        for _ in range(24):
            ax = tuple(rng.uniform(-1, 1) for _ in range(3))
            m = math.sqrt(sum(c * c for c in ax))
            ax = tuple(c / m for c in ax)
            ang = rng.uniform(0.0, 2 * math.pi)
            rotated = [
                to_lonlat(rodrigues(lonlat_to_xyz(lo, la), ax, ang))
                for lo, la in OCTANT
            ]
            got = unfold_loop(rotated).signed_area
            self.assertAlmostEqual(got, base, places=11,
                                   msg=f"rotation axis={ax} ang={ang}")

    def test_larger_polygon_area_in_range(self):
        # A small geodesic quadrilateral around lon 0 must have small area.
        poly = [(-10 * D, -10 * D), (10 * D, -10 * D),
                (10 * D, 10 * D), (-10 * D, 10 * D)]
        res = unfold_loop(poly)
        # planar approximation ~ (20deg)^2 in radians ~ 0.122; allow loose.
        self.assertAlmostEqual(res.signed_area, 0.122, delta=0.01)


if __name__ == "__main__":
    unittest.main()
