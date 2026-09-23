"""Pole handling: interior crossings and pole vertices are ambiguous."""

import math
import unittest

from antimeridian import unfold_loop

D = math.pi / 180.0


class PoleVertexTests(unittest.TestCase):
    def setUp(self):
        # Octant triangle with an explicit north-pole vertex.
        self.poly = [(0.0, 0.0), (math.pi / 2, 0.0), (0.0, math.pi / 2)]
        self.res = unfold_loop(self.poly)

    def test_area_pi_over_2(self):
        self.assertAlmostEqual(self.res.signed_area, math.pi / 2, places=14)

    def test_pole_vertex_marked_once(self):
        m = self.res.ambiguous_markers
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].pole, "north")
        self.assertAlmostEqual(m[0].t, 1.0)

    def test_incoming_and_outgoing_fragments_ambiguous(self):
        # Edge 1 arrives at the pole and edge 2 leaves it; both ambiguous.
        pole_frags = [f for f in self.res.fragments if f.pole == "north"]
        self.assertTrue(pole_frags)
        self.assertTrue(all(f.kind == "ambiguous" for f in pole_frags))

    def test_constant_meridian_to_pole(self):
        # Edge 0 (equator (0,0)->(90,0)) is normal; the two pole-touching
        # fragments each hold a fixed longitude (90 and 0 respectively).
        e1 = [f for f in self.res.fragments if f.edge_index == 1][0]
        e2 = [f for f in self.res.fragments if f.edge_index == 2][0]
        self.assertAlmostEqual(e1.lon_start, 90 * D)
        self.assertAlmostEqual(e1.lon_end, 90 * D)
        self.assertAlmostEqual(e2.lon_start, 0.0)
        self.assertAlmostEqual(e2.lon_end, 0.0)


class InteriorPoleCrossingTests(unittest.TestCase):
    def setUp(self):
        # Endpoints antipodal in longitude (170 and -10), equal latitude: the
        # short arc runs over the north pole in its interior.
        self.poly = [(170 * D, 20 * D), (-10 * D, 20 * D), (80 * D, 50 * D)]
        self.res = unfold_loop(self.poly)

    def test_interior_pole_marker(self):
        m = self.res.ambiguous_markers
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].edge_index, 0)
        self.assertEqual(m[0].pole, "north")
        self.assertAlmostEqual(m[0].t, 0.5, places=6)
        self.assertTrue(0.0 < m[0].t < 1.0)

    def test_arc_split_into_two_ambiguous_halves(self):
        halves = [f for f in self.res.fragments
                  if f.edge_index == 0 and f.kind == "ambiguous"]
        self.assertEqual(len(halves), 2)
        self.assertEqual([h.order for h in halves], [0, 1])
        self.assertAlmostEqual(halves[0].lat_end, math.pi / 2)
        self.assertAlmostEqual(halves[1].lat_start, math.pi / 2)
        # The two halves keep their own endpoint meridian; they are NOT joined
        self.assertNotAlmostEqual(halves[0].lon_end, halves[1].lon_end)

    def test_pole_crossing_not_a_seam_crossing(self):
        self.assertFalse(any(c.edge_index == 0
                             for c in self.res.seam_crossings))

    def test_arc_param_flags_pole(self):
        self.assertTrue(self.res.arcs[0].contains_pole)


class SouthPoleTests(unittest.TestCase):
    def test_south_pole_crossing(self):
        poly = [(100 * D, -20 * D), (-80 * D, -20 * D), (0.0, -50 * D)]
        res = unfold_loop(poly)
        m = res.ambiguous_markers
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].pole, "south")
        halves = [f for f in res.fragments
                  if f.edge_index == 0 and f.pole == "south"]
        self.assertEqual(len(halves), 2)


if __name__ == "__main__":
    unittest.main()
