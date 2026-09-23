"""日期变更线分裂、极点标注、coincident/dateline 语义与自交拒绝测试。"""

from __future__ import annotations

import math
import unittest

from dateline_unwrap import SelfIntersectionError, unwrap_ring
from dateline_unwrap.geometry import dot, normalize, slerp, unit


def D(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    return math.radians(lon_deg), math.radians(lat_deg)


def _kinds(edge):
    return [s.kind for s in edge.segments]


def _normals_ok(edge):
    a = unit(edge.arc.start.lon, edge.arc.start.lat)
    b = unit(edge.arc.end.lon, edge.arc.end.lat)
    L = edge.arc.length
    for s in edge.segments:
        if dot(slerp(a, b, s.t0, L), s.point_a) < 1.0 - 1e-9:
            return False
        if dot(slerp(a, b, s.t1, L), s.point_b) < 1.0 - 1e-9:
            return False
    return True


class TestArcParams(unittest.TestCase):
    def test_arc_length_and_plane_normal(self):
        r = unwrap_ring([D(0, 0), D(90, 0), D(0, 90)])
        for e in r.edges:
            # 三条边都是 90° 短大圆弧。
            self.assertAlmostEqual(e.arc.length, math.pi / 2, places=14)
            self.assertGreater(e.arc.length, 0.0)
            self.assertLess(e.arc.length, math.pi)
            nx, ny, nz = e.arc.plane_normal
            self.assertAlmostEqual(nx ** 2 + ny ** 2 + nz ** 2, 1.0, places=12)
            self.assertTrue(_normals_ok(e))
        # 首边 (0,0)->(90E,0) 沿赤道东行，法向指向北极，初始方位正东。
        self.assertAlmostEqual(r.edges[0].arc.plane_normal[2], 1.0, places=12)
        self.assertAlmostEqual(r.edges[0].arc.azimuth, math.pi / 2, places=12)

    def test_edge_indices_and_vertex_order(self):
        ring = [D(0, 0), D(10, 0), D(10, 10), D(0, 10)]
        r = unwrap_ring(ring)
        self.assertEqual([e.arc.edge_index for e in r.edges], [0, 1, 2, 3])
        self.assertAlmostEqual(r.edges[0].arc.start.lon, 0.0)
        self.assertAlmostEqual(r.edges[3].arc.end.lat, 0.0)
        # segments 扁平视图与各边一致。
        self.assertEqual(len(r.segments), sum(len(e.segments) for e in r.edges))

    def test_initial_azimuth_matches_3d_tangent(self):
        # 用独立的三维切向投影（东/北基）核验初始方位角，含跨 ±180° 缝的边
        # （经度差必须先折叠，否则方位角会反向）。
        rings = [
            [D(0, 0), D(90, 0), D(0, 90)],
            [D(170, 10), D(-170, 10), (-math.pi, math.radians(60))],
            [D(-179, 40), D(179, 50), D(0, -20)],
            [D(170, 70), D(-10, 70), D(-90, 10)],
            [D(180, 20), D(0, 60), D(90, 10)],
        ]
        for ring in rings:
            r = unwrap_ring(ring)
            for e in r.edges:
                a = unit(e.arc.start.lon, e.arc.start.lat)
                b = unit(e.arc.end.lon, e.arc.end.lat)
                tang = normalize(tuple(
                    b[k] - a[k] * math.cos(e.arc.length) for k in range(3)))
                lon, lat = e.arc.start.lon, e.arc.start.lat
                east = (-math.sin(lon), math.cos(lon), 0.0)
                north = (-math.sin(lat) * math.cos(lon),
                         -math.sin(lat) * math.sin(lon), math.cos(lat))
                ref = math.atan2(dot(tang, east), dot(tang, north)) % (2 * math.pi)
                d = abs((e.arc.azimuth - ref + math.pi) % (2 * math.pi) - math.pi)
                self.assertLess(d, 1e-12, msg=f"edge {e.arc.edge_index}")


class TestDatelineSplit(unittest.TestCase):
    def test_simple_crossing_splits_into_two(self):
        # 边 (170E,10N)->(170W,10N) 在北半球跨缝，分成东、西两段。
        r = unwrap_ring([D(170, 10), D(-170, 10), (-math.pi, math.radians(60))])
        edge0 = r.edges[0]
        self.assertEqual(len(edge0.segments), 2)
        lons = [(s.lon_a, s.lon_b) for s in edge0.segments]
        # 第一段止于 +pi（180E 侧），第二段起于 -pi（180W 侧）：不直接连线。
        self.assertAlmostEqual(lons[0][1], math.pi, places=9)
        self.assertAlmostEqual(lons[1][0], -math.pi, places=9)
        self.assertAlmostEqual(lons[0][0], math.radians(170), places=9)
        self.assertAlmostEqual(lons[1][1], math.radians(-170), places=9)
        for e in r.edges:
            self.assertTrue(_normals_ok(e))

    def test_no_fake_intersection_when_arc_merely_grazes_in_lonlat(self):
        # 两端都在东半球、且弧很短：即使 (lon,lat) 平面直线投影可能看似贴边，
        # 真实大圆弧不得出现缝交点。
        r = unwrap_ring([D(170, 0), D(175, 5), D(172, -5)])
        for e in r.edges:
            self.assertEqual(_kinds(e).count("normal"), len(e.segments))

    def test_segments_partition_the_arc(self):
        # 分裂参数区间 [t0,t1] 必须恰好覆盖 [0,1]，ambiguous 为零长度单点。
        rings = [
            [D(170, 10), D(-170, 10), (-math.pi, math.radians(60))],
            [D(170, 70), D(-10, 70), D(-90, 10)],
            [D(-170, -20), D(160, 30), D(120, -60)],
        ]
        for ring in rings:
            r = unwrap_ring(ring)
            for e in r.edges:
                spans = sorted(
                    (s.t0, s.t1) for s in e.segments if s.kind != "ambiguous"
                )
                self.assertAlmostEqual(spans[0][0], 0.0, places=9)
                self.assertAlmostEqual(spans[-1][1], 1.0, places=9)
                for (_, h), (lo, _) in zip(spans, spans[1:]):
                    self.assertAlmostEqual(h, lo, places=9)
                for s in e.segments:
                    if s.kind == "ambiguous":
                        self.assertEqual(s.t0, s.t1)
                    else:
                        self.assertLess(s.t0, s.t1)
                        # 每个正常片段经度跨度不超过 pi（连续单值支）。
                        self.assertLessEqual(abs(s.lon_b - s.lon_a), math.pi + 1e-8)


class TestPoleHandling(unittest.TestCase):
    def test_arc_over_pole_marks_ambiguous_and_does_not_connect(self):
        # (170E,70N)->(10W,70N) 的较短大圆弧越过北极附近（不经过极点本身，
        # 但同一大圆在缝侧有交点）；这里选大圆**恰好过北极**的构造：
        # 两端共子午圈（lon 相差 180），短弧经北极。
        r = unwrap_ring([D(170, 70), D(-10, 70), D(-90, 10)])
        over_pole_edge = r.edges[0]
        amb = [s for s in over_pole_edge.segments if s.kind == "ambiguous"]
        self.assertEqual(len(amb), 1)
        self.assertEqual(amb[0].pole, "north")
        self.assertAlmostEqual(amb[0].point_a[2], 1.0, places=9)
        # 极点单点不携带经度，渲染端不得拿它和另一侧直接拉平面直线。
        self.assertIsNone(amb[0].lon_a)
        self.assertIsNone(amb[0].lon_b)

    def test_pole_vertex_triangle(self):
        # 顶点恰为北极的八分之一球：两条邻边各在极点有一个 ambiguous 标注。
        r = unwrap_ring([D(-45, 0), D(45, 0), (0.0, math.pi / 2)])
        self.assertAlmostEqual(r.area, math.pi / 2, places=14)
        poles = [s for e in r.edges for s in e.segments if s.kind == "ambiguous"]
        self.assertEqual(len(poles), 2)
        self.assertTrue(all(s.pole == "north" for s in poles))

    def test_south_pole_vertex(self):
        r = unwrap_ring([(0.0, -math.pi / 2), D(45, 0), D(-45, 0)])
        self.assertAlmostEqual(r.area, math.pi / 2, places=14)
        poles = [s for e in r.edges for s in e.segments if s.kind == "ambiguous"]
        self.assertTrue(all(s.pole == "south" for s in poles))


class TestCoincidentSeam(unittest.TestCase):
    def test_entire_edge_on_seam_is_coincident(self):
        r = unwrap_ring([D(170, 0), D(180, 0), D(180, 50), D(150, 30)])
        seam_edge = r.edges[1]
        self.assertEqual(len(seam_edge.segments), 1)
        s = seam_edge.segments[0]
        self.assertEqual(s.kind, "coincident")
        # 保留原弧参数，不给经度，不造跨缝连接。
        self.assertEqual((s.t0, s.t1), (0.0, 1.0))
        self.assertIsNone(s.lon_a)
        self.assertIsNone(s.lon_b)
        self.assertTrue(_normals_ok(seam_edge))

    def test_coincident_edge_area_matches_neighbors(self):
        r = unwrap_ring([D(170, 0), D(180, 0), D(180, 50), D(150, 30)])
        # 面积为一个小的正量，且在合理量级。
        self.assertGreater(r.area, 0.0)
        self.assertLess(r.area, 1.0)

    def test_coplanar_arc_partly_on_seam_partly_zero_meridian(self):
        # 边大圆 == 缝所在子午线圈，从 180° 一侧过北极到 0° 一侧：
        # 缝侧子弧标 dateline，0° 侧为正常片段，极点 ambiguous。
        r = unwrap_ring([D(180, 20), D(0, 60), D(90, 10)])
        edge0 = r.edges[0]
        kinds = _kinds(edge0)
        self.assertIn("dateline", kinds)
        self.assertIn("normal", kinds)
        amb = [s for s in edge0.segments if s.kind == "ambiguous"]
        self.assertEqual(len(amb), 1)
        self.assertEqual(amb[0].pole, "north")
        # dateline 片段无经度；normal 片段经度在 0 附近且不跨缝。
        for s in edge0.segments:
            if s.kind == "dateline":
                self.assertIsNone(s.lon_a)
            if s.kind == "normal":
                self.assertIsNotNone(s.lon_a)
                self.assertTrue(all(abs(x) < 1e-6 for x in (s.lon_a, s.lon_b)))
        self.assertTrue(_normals_ok(edge0))


class TestSelfIntersectionRejection(unittest.TestCase):
    def test_bowtie_quadrilateral_rejected(self):
        # 经纬平面上的蝴蝶结：两条对角线边 B->C(边1) 与 D->A(边3) 真实相交。
        with self.assertRaises(SelfIntersectionError) as cm:
            unwrap_ring([D(0, 0), D(20, 0), D(0, 20), D(20, 20)])
        self.assertEqual({cm.exception.edge_i, cm.exception.edge_j}, {1, 3})
        self.assertEqual(len(cm.exception.point), 3)

    def test_nonadjacent_edge_touch_rejected(self):
        # 边3（经度15°，-15..15）在 (15E,0) 内点接触边0（赤道 0..30E）。
        ring = [D(0, 0), D(30, 0), D(30, 30), D(15, 15), D(15, -15), D(0, -15)]
        with self.assertRaises(SelfIntersectionError) as cm:
            unwrap_ring(ring)
        self.assertEqual(cm.exception.edge_i, 0)
        self.assertEqual(cm.exception.edge_j, 3)

    def test_nonadjacent_vertex_reuse_rejected(self):
        # 边2 的终点 D(20E,0) 是边0（赤道 0..40E）的内点：边对 (0,2) 拒绝。
        ring = [D(0, 0), D(40, 0), D(40, 40), D(20, 0), D(0, 40)]
        with self.assertRaises(SelfIntersectionError) as cm:
            unwrap_ring(ring)
        self.assertEqual({cm.exception.edge_i, cm.exception.edge_j}, {0, 2})

    def test_coplanar_overlapping_edges_rejected(self):
        # 尖刺：边0 (180,20N)->(0,60N) 与边2 (0,0)->(180,20N) 在同一
        # 子午线圈上，除公共顶点外还在极点两侧大段共面重叠 -> 拒绝。
        spike = [D(180, 20), D(0, 60), D(0, 0)]
        with self.assertRaises(SelfIntersectionError) as cm:
            unwrap_ring(spike)
        self.assertEqual({cm.exception.edge_i, cm.exception.edge_j}, {0, 2})

    def test_adjacent_edges_share_only_common_vertex(self):
        # 普通凸四边形必须通过简单性检查。
        r = unwrap_ring([D(0, 0), D(40, 0), D(40, 40), D(0, 40)])
        self.assertEqual(len(r.edges), 4)


if __name__ == "__main__":
    unittest.main()
