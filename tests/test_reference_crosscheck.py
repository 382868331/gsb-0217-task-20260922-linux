"""Cross-validation against the independent reference (tests/reference.py).

The reference never imports the core.  For every random *valid* loop we check:

* signed area agrees with the reference Van Oosterom solid-angle sum;
* per edge, the number/direction of seam crossings, coincidence and interior
  poles agree with the reference dense exact-SLERP sampler.

Round-trip consistency is not used as an oracle anywhere.
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
    SelfIntersectionError,
    unfold_loop,
)
from antimeridian import geometry as g

sys.path.insert(0, os.path.dirname(__file__))
import reference as R  # noqa: E402

D = math.pi / 180.0


def _valid_loop(rng):
    """Build a guaranteed-valid simple loop around a random small centre."""
    clon = rng.uniform(-math.pi, math.pi)
    clat = rng.uniform(-1.0, 1.0)
    radius = rng.uniform(0.05, 0.5)
    m = rng.randint(3, 8)
    pts = []
    for k in range(m):
        theta = 2 * math.pi * k / m + rng.uniform(-0.1, 0.1)
        r = radius * rng.uniform(0.6, 1.0)
        lat = max(-math.pi / 2 + 1e-6,
                  min(math.pi / 2 - 1e-6, clat + r * math.sin(theta)))
        lon = clon + r * math.cos(theta) / max(0.2, math.cos(lat))
        lon = max(-math.pi + 1e-6, min(math.pi - 1e-6, lon))
        pts.append((lon, lat))
    return pts


def _try_loops(rng, count):
    ok = []
    attempts = 0
    while len(ok) < count and attempts < count * 40:
        attempts += 1
        pts = _valid_loop(rng)
        try:
            res = unfold_loop(pts)
        except (InvalidInputError, AntipodalAdjacentError,
                SelfIntersectionError, DegenerateLoopError):
            continue
        ok.append((pts, res))
    return ok


class AreaAgainstReferenceTests(unittest.TestCase):
    def test_many_loops_area_matches_reference(self):
        rng = random.Random(424242)
        cases = _try_loops(rng, 120)
        self.assertGreaterEqual(len(cases), 60)
        worst = 0.0
        for pts, res in cases:
            expected = R.reference_signed_area(pts)
            worst = max(worst, abs(res.signed_area - expected))
            self.assertAlmostEqual(
                res.signed_area, expected, delta=1e-9,
                msg=f"area mismatch pts={pts}",
            )
        # The two independent formulas agree tightly; document the observed
        # order of magnitude for the README error budget.
        self.assertLess(worst, 1e-10)

    def test_winding_sign_matches_reference(self):
        rng = random.Random(7)
        for pts, res in _try_loops(rng, 40):
            expected = R.reference_signed_area(pts)
            if abs(expected) < 1e-11:
                continue
            self.assertEqual(res.area_winding,
                             "ccw" if expected > 0 else "cw")


class SeamEventsAgainstReferenceTests(unittest.TestCase):
    def test_edge_events_match_dense_reference(self):
        rng = random.Random(99)
        checked = 0
        for pts, res in _try_loops(rng, 80):
            xyz = [g.lonlat_to_xyz(lo, la) for lo, la in pts]
            n = len(pts)
            for i in range(n):
                p, q = xyz[i], xyz[(i + 1) % n]
                alpha = g.central_angle(p, q)
                n_cross, coinc, east, west = R.reference_seam_events(
                    p, q, alpha)
                ref_poles = R.reference_edge_passes_pole(p, q, alpha)

                core_cross = [c for c in res.seam_crossings
                              if c.edge_index == i]
                core_arc = res.arcs[i]
                core_poles = [m for m in res.ambiguous_markers
                              if m.edge_index == i and 0.0 < m.t < 1.0]

                # Interior pole arcs are reported ambiguous, not crossed.
                if ref_poles:
                    self.assertTrue(
                        core_arc.contains_pole,
                        msg=f"edge {i} missed interior pole {pts}",
                    )
                    self.assertEqual(len(core_poles), len(ref_poles))
                    continue

                self.assertEqual(
                    len(core_cross), n_cross,
                    msg=f"edge {i} crossing count pts={pts}",
                )
                for c in core_cross:
                    exp_sign = 1 if east else -1
                    self.assertAlmostEqual(c.lon_unwrapped,
                                           exp_sign * math.pi)
                # Coincidence only compareable when neither endpoint is a pole.
                if not any(abs(abs(la) - math.pi / 2) < 1e-9
                           for _, la in (pts[i], pts[(i + 1) % n])):
                    self.assertEqual(core_arc.pole_coincident, coinc,
                                     msg=f"edge {i} coincidence pts={pts}")
                checked += 1
        self.assertGreater(checked, 100)


class DatelineSpecificReferenceTests(unittest.TestCase):
    def test_straddling_loop_area_reference(self):
        pts = [(170 * D, 10 * D), (-170 * D, 10 * D), (-175 * D, 40 * D)]
        res = unfold_loop(pts)
        self.assertAlmostEqual(
            res.signed_area, R.reference_signed_area(pts), delta=1e-12)

    def test_straddling_loop_crossing_counts(self):
        pts = [(170 * D, 10 * D), (-170 * D, 10 * D), (-175 * D, 40 * D)]
        res = unfold_loop(pts)
        xyz = [g.lonlat_to_xyz(*p) for p in pts]
        for i in range(3):
            alpha = g.central_angle(xyz[i], xyz[(i + 1) % 3])
            n_cross, _, east, west = R.reference_seam_events(
                xyz[i], xyz[(i + 1) % 3], alpha)
            core = [c for c in res.seam_crossings if c.edge_index == i]
            self.assertEqual(len(core), n_cross)


if __name__ == "__main__":
    unittest.main()
