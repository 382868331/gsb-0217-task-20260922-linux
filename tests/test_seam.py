"""Date-line crossing, splitting and coincidence tests."""

import math
import unittest

from antimeridian import unfold_loop

D = math.pi / 180.0


class DatelineSplitTests(unittest.TestCase):
    def setUp(self):
        # Triangle straddling the seam, no poles involved.
        self.poly = [(170 * D, 10 * D), (-170 * D, 10 * D),
                     (-175 * D, 40 * D)]
        self.res = unfold_loop(self.poly)

    def test_two_genuine_crossings(self):
        # Edges 0 and 2 cross the seam; edge 1 stays in the east (unwrapped).
        crossed_edges = sorted(c.edge_index for c in self.res.seam_crossings)
        self.assertEqual(crossed_edges, [0, 2])
        for c in self.res.seam_crossings:
            self.assertTrue(0.0 < c.t < 1.0)
            self.assertIn(round(c.lon_unwrapped / D), (180, -180))

    def test_fragments_have_continuous_longitude(self):
        fr = self.res.fragments
        # Every individual fragment stays within a band narrower than 2*pi.
        for f in fr:
            self.assertLess(abs(f.lon_end - f.lon_start), 2 * math.pi)
        # A crossing edge is split into two fragments that meet at +/- pi.
        edge0 = [f for f in fr if f.edge_index == 0]
        self.assertEqual(len(edge0), 2)
        self.assertAlmostEqual(edge0[0].lon_end, edge0[1].lon_start)
        self.assertAlmostEqual(abs(edge0[0].lon_end), math.pi)

    def test_unwrapped_fragments_cross_the_seam(self):
        # Edge 0 leaves lon +170 and must continue past +180 to ~+190,
        # i.e. it is NOT wrapped back to -170.
        edge0 = [f for f in self.res.fragments if f.edge_index == 0]
        self.assertGreater(edge0[1].lon_end, math.pi)

    def test_loop_chains_continuously_through_regular_vertex(self):
        # The shared vertex of edge 0 end / edge 1 start is the same
        # physical point; unwrapped longitudes must agree.
        fr = self.res.fragments
        end0 = [f for f in fr if f.edge_index == 0][-1].lon_end
        start1 = [f for f in fr if f.edge_index == 1][0].lon_start
        self.assertAlmostEqual(end0, start1, places=9)

    def test_no_ambiguous_without_poles(self):
        self.assertEqual(self.res.ambiguous_markers, ())
        self.assertTrue(all(f.pole is None for f in self.res.fragments))


class CoincidentSeamTests(unittest.TestCase):
    def setUp(self):
        # Edge 0 runs wholly along the 180 half-meridian (lat 10 -> 40).
        self.poly = [(180 * D, 10 * D), (180 * D, 40 * D),
                     (150 * D, 20 * D)]
        self.res = unfold_loop(self.poly)

    def test_coincident_edge_kept_whole(self):
        edge0 = [f for f in self.res.fragments if f.edge_index == 0]
        self.assertEqual(len(edge0), 1)
        self.assertEqual(edge0[0].kind, "coincident")
        self.assertTrue(self.res.arcs[0].pole_coincident)

    def test_coincident_edge_not_synthetically_split(self):
        # No seam crossing is reported for the coincident edge and no fake
        # cross-seam fragment is attached to it.
        self.assertFalse(any(c.edge_index == 0
                             for c in self.res.seam_crossings))
        f0 = [f for f in self.res.fragments if f.edge_index == 0][0]
        self.assertAlmostEqual(f0.lon_start, math.pi)
        self.assertAlmostEqual(f0.lon_end, math.pi)
        # The original arc is preserved: endpoints and central angle stand.
        self.assertAlmostEqual(self.res.arcs[0].central_angle, 30 * D)

    def test_other_edges_are_normal(self):
        self.assertTrue(all(
            f.kind == "normal"
            for f in self.res.fragments if f.edge_index != 0
        ))


class NoFakeCrossingTests(unittest.TestCase):
    def test_crossing_point_is_on_the_sphere_not_planar(self):
        # Edge from (170, 0) to (-170, 60).  A flat lon/lat "short way"
        # interpolation reaches the 180 line at t=0.5, latitude 30 deg.
        # The true great-circle arc meets the seam later (t ~= 0.683) at
        # latitude ~= 41.33 deg -- over 11 degrees away.  The core must
        # report the spherical intersection, not the planar one.
        poly = [(170 * D, 0.0), (-170 * D, 60 * D), (160 * D, 30 * D)]
        res = unfold_loop(poly)
        cut = [c for c in res.seam_crossings if c.edge_index == 0]
        self.assertEqual(len(cut), 1)
        self.assertAlmostEqual(cut[0].t, 0.6826592727, delta=1e-8)
        self.assertAlmostEqual(cut[0].lat, 41.327920477 * D, delta=1e-7)
        # The planar answers that must NOT be produced:
        self.assertNotAlmostEqual(cut[0].lat, 30 * D, delta=2 * D)
        self.assertNotAlmostEqual(cut[0].t, 0.5, delta=1e-3)

    def test_near_pole_crossing_not_mislabeled_pole(self):
        # This arc passes ~0.002 deg from the north pole and genuinely crosses
        # the seam there; a quadratic |z| test mislabels it as a pole.  It must
        # be an ordinary crossing with no ambiguous marker.
        poly = [(-101.20362715 * D, 67.65017269 * D),
                (78.80244463 * D, 15.89686847 * D),
                (-10 * D, 10 * D)]
        res = unfold_loop(poly)
        self.assertEqual(len([c for c in res.seam_crossings
                              if c.edge_index == 0]), 1)
        self.assertFalse(any(f.kind == "ambiguous"
                             for f in res.fragments if f.edge_index == 0))
        self.assertEqual(res.ambiguous_markers, ())

    def test_same_side_arc_stays_one_piece(self):
        # An arc entirely on the Atlantic side of the seam never touches it
        # and must remain a single normal fragment.
        poly = [(-70 * D, 40 * D), (-50 * D, 10 * D), (-60 * D, -10 * D)]
        res = unfold_loop(poly)
        self.assertEqual(res.seam_crossings, ())
        for i in range(3):
            edge = [f for f in res.fragments if f.edge_index == i]
            self.assertEqual(len(edge), 1)
            self.assertEqual(edge[0].kind, "normal")


if __name__ == "__main__":
    unittest.main()
