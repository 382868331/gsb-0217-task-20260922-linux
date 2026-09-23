"""输入校验与错误语义测试。"""

from __future__ import annotations

import math
import unittest

from dateline_unwrap import (
    AntipodalEndpointError,
    DegenerateEdgeError,
    InvalidCoordinateError,
    RingSizeError,
    SphericalLoopError,
    unwrap_ring,
)


def D(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    return math.radians(lon_deg), math.radians(lat_deg)


class TestValidation(unittest.TestCase):
    def test_nan_rejected_located(self):
        for bad in (float("nan"), float("-nan")):
            with self.assertRaises(InvalidCoordinateError) as cm:
                unwrap_ring([(bad, 0.0), D(1, 1), D(2, 2)])
            self.assertEqual(cm.exception.vertex_index, 0)
            self.assertEqual(cm.exception.field, "lon")

    def test_infinity_rejected_located(self):
        with self.assertRaises(InvalidCoordinateError) as cm:
            unwrap_ring([(0.0, float("inf")), D(1, 1), D(2, 2)])
        self.assertEqual(cm.exception.vertex_index, 0)
        self.assertEqual(cm.exception.field, "lat")
        with self.assertRaises(InvalidCoordinateError) as cm:
            unwrap_ring([D(0, 0), (float("-inf"), 0.0), D(2, 2)])
        self.assertEqual(cm.exception.vertex_index, 1)

    def test_bool_rejected_as_integer(self):
        # bool 虽是 int 子类，契约显式拒绝。
        with self.assertRaises(InvalidCoordinateError) as cm:
            unwrap_ring([(True, 0.0), D(1, 1), D(2, 2)])
        self.assertEqual(cm.exception.vertex_index, 0)
        self.assertEqual(cm.exception.field, "lon")
        with self.assertRaises(InvalidCoordinateError):
            unwrap_ring([D(0, 0), D(1, 1), (0.0, False)])

    def test_numeric_strings_rejected(self):
        with self.assertRaises(InvalidCoordinateError):
            unwrap_ring([("0", 0.0), D(1, 1), D(2, 2)])

    def test_range_violations_located(self):
        with self.assertRaises(InvalidCoordinateError) as cm:
            unwrap_ring([(3.2, 0.0), D(1, 1), D(2, 2)])
        self.assertEqual(cm.exception.field, "lon")
        with self.assertRaises(InvalidCoordinateError) as cm:
            unwrap_ring([D(0, 0), D(1, 1), (0.0, 1.58)])
        self.assertEqual(cm.exception.field, "lat")
        self.assertEqual(cm.exception.vertex_index, 2)

    def test_range_boundaries_accepted(self):
        # ±pi 经度（切缝同一条半子午线）、±pi/2 纬度都是合法边界。
        r = unwrap_ring([(math.pi, 0.1), (-math.pi, 0.2), (2.0, 0.3)])
        self.assertEqual(len(r.vertices), 3)
        r = unwrap_ring([(0.0, math.pi / 2), D(-45, 0), D(45, 0)])
        self.assertAlmostEqual(r.area, math.pi / 2, places=12)
        r = unwrap_ring([(0.0, -math.pi / 2), D(45, 0), D(-45, 0)])
        self.assertAlmostEqual(r.area, math.pi / 2, places=12)

    def test_vertex_count_bounds(self):
        with self.assertRaises(RingSizeError):
            unwrap_ring([])
        with self.assertRaises(RingSizeError):
            unwrap_ring([D(0, 0), D(1, 1)])
        with self.assertRaises(RingSizeError):
            unwrap_ring([D(float(k % 300), 0.0) for k in range(101)])

    def test_max_vertices_accepted(self):
        # 恰为 100 个顶点的小环：上界可构造、可计算（不要求穷举规模）。
        verts = []
        for k in range(100):
            th = 2.0 * math.pi * k / 100.0
            verts.append((th * 0.01, 0.2 + 0.1 * math.cos(th)))
        r = unwrap_ring(verts)
        self.assertEqual(len(r.vertices), 100)
        self.assertEqual(len(r.edges), 100)
        self.assertTrue(-2.0 * math.pi < r.area <= 2.0 * math.pi)

    def test_bad_containers(self):
        with self.assertRaises(RingSizeError):
            unwrap_ring(None)  # type: ignore[arg-type]
        with self.assertRaises(RingSizeError):
            unwrap_ring("0,0 1,1 2,2")  # type: ignore[arg-type]
        with self.assertRaises(InvalidCoordinateError):
            unwrap_ring([D(0, 0), D(1, 1), 3])  # type: ignore[list-item]
        with self.assertRaises(InvalidCoordinateError):
            unwrap_ring([D(0, 0), D(1, 1), (1.0,)])
        with self.assertRaises(InvalidCoordinateError):
            unwrap_ring([D(0, 0), D(1, 1), (1.0, 1.0, 1.0)])

    def test_duplicate_adjacent_vertices(self):
        with self.assertRaises(DegenerateEdgeError) as cm:
            unwrap_ring([D(10, 10), D(10, 10), D(20, 20)])
        self.assertEqual(cm.exception.edge_index, 0)
        with self.assertRaises(DegenerateEdgeError) as cm:
            unwrap_ring([D(10, 10), D(20, 20), D(20, 20)])
        self.assertEqual(cm.exception.edge_index, 1)
        # 首尾重合（闭合边）。
        with self.assertRaises(DegenerateEdgeError) as cm:
            unwrap_ring([D(10, 10), D(20, 20), D(10, 10)])
        self.assertEqual(cm.exception.edge_index, 2)

    def test_antipodal_adjacent_vertices(self):
        with self.assertRaises(AntipodalEndpointError) as cm:
            unwrap_ring([D(0, 0), D(180, 0), D(10, 10)])
        self.assertEqual(cm.exception.edge_index, 0)
        # 经度相同、纬度相反：同样对跖。
        with self.assertRaises(AntipodalEndpointError) as cm:
            unwrap_ring([D(45, 30), D(-135, -30), D(10, 10)])
        self.assertEqual(cm.exception.edge_index, 0)

    def test_all_library_errors_are_value_errors(self):
        for err in (
            RingSizeError("x"),
            InvalidCoordinateError("x"),
            DegenerateEdgeError("x", 0),
            AntipodalEndpointError("x", 0),
        ):
            self.assertIsInstance(err, ValueError)
            self.assertIsInstance(err, SphericalLoopError)

    def test_stateless_and_no_input_mutation(self):
        verts = [D(0, 0), D(90, 0), D(0, 90)]
        snapshot = tuple(verts)
        r1 = unwrap_ring(verts)
        r2 = unwrap_ring(verts)
        self.assertEqual(tuple(verts), snapshot)
        self.assertEqual(r1.area, r2.area)
        # 返回结构不可变。
        with self.assertRaises(Exception):
            r1.edges = ()  # type: ignore[misc]

    def test_budget_exhaustion_is_distinct_from_rejection(self):
        # 预算耗尽（RuntimeError 家族，非 ValueError/SphericalLoopError）与
        # “证明输入非法”的拒绝是两类语义：强制枢轴候选全部退化即可触发前者。
        from unittest import mock

        from dateline_unwrap import ComputationBudgetError
        from dateline_unwrap import unwrap as uw

        def empty_candidates(_points):
            # 不提供任何枢轴候选 -> 预算确定性耗尽（区别于输入被证明非法）。
            return iter(())

        with mock.patch.object(uw, "_pivot_candidates", empty_candidates):
            with self.assertRaises(ComputationBudgetError):
                unwrap_ring([D(0, 0), D(90, 0), D(0, 90)])
            self.assertFalse(issubclass(ComputationBudgetError, SphericalLoopError))
            self.assertTrue(issubclass(ComputationBudgetError, RuntimeError))
        # mock 退出后同一输入正常计算（无部分状态残留）。
        self.assertAlmostEqual(
            unwrap_ring([D(0, 0), D(90, 0), D(0, 90)]).area, math.pi / 2, places=14
        )


if __name__ == "__main__":
    unittest.main()
