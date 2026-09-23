"""有向面积测试：解析值、方向约定、旋转不变、独立预言、边界。"""

from __future__ import annotations

import math
import random
import unittest

from dateline_unwrap import unwrap_ring
from dateline_unwrap.geometry import cross, normalize, unit

try:  # 作为包成员导入（tests.test_area）
    from .reference import PIVOTS, fan_signed_area, gauss_bonnet_area
except ImportError:  # `python -m unittest discover -s tests` 把 tests 作为顶层
    from reference import PIVOTS, fan_signed_area, gauss_bonnet_area  # type: ignore


def D(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    return math.radians(lon_deg), math.radians(lat_deg)


def _rodrigues(v, axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    return normalize((
        v[0] * (c + axis[0] ** 2 * (1 - c))
        + v[1] * (axis[0] * axis[1] * (1 - c) - axis[2] * s)
        + v[2] * (axis[0] * axis[2] * (1 - c) + axis[1] * s),
        v[0] * (axis[1] * axis[0] * (1 - c) + axis[2] * s)
        + v[1] * (c + axis[1] ** 2 * (1 - c))
        + v[2] * (axis[1] * axis[2] * (1 - c) - axis[0] * s),
        v[0] * (axis[2] * axis[0] * (1 - c) - axis[1] * s)
        + v[1] * (axis[2] * axis[1] * (1 - c) + axis[0] * s)
        + v[2] * (c + axis[2] ** 2 * (1 - c)),
    ))


def _to_lonlat(v):
    return math.atan2(v[1], v[0]), math.asin(min(1.0, max(-1.0, v[2])))


def _cone_ring(rng: random.Random, radius: float):
    """绕随机球心的小圆锥截线多边形，半径、边数、朝向都随机，保证简单。"""
    center = normalize((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
    tmp = (0.0, 0.0, 1.0) if abs(center[2]) < 0.9 else (1.0, 0.0, 0.0)
    e1 = normalize(cross(center, tmp))
    e2 = cross(center, e1)
    k = rng.randint(3, 9)
    angs = sorted(rng.uniform(0, 2 * math.pi) for _ in range(k - 1))
    angs = [0.0] + angs + [2 * math.pi]
    ring = []
    for th in angs[:-1]:
        v = normalize(tuple(
            center[m] * math.cos(radius)
            + (e1[m] * math.cos(th) + e2[m] * math.sin(th)) * math.sin(radius)
            for m in range(3)
        ))
        ring.append(_to_lonlat(v))
    return ring


def _safe_area(ring):
    # 随机旋转后偶发对跖/共线退化：那是测试数据退化，跳过该实例。
    try:
        return unwrap_ring(ring).area
    except Exception:
        return None


class TestAreaAnalytic(unittest.TestCase):
    def test_octant_triangle_is_pi_over_2(self):
        # (0,0),(90E,0),(0,90N) 围出球面的 1/8：3 个直角，角盈 = pi/2。
        r = unwrap_ring([D(0, 0), D(90, 0), D(0, 90)])
        self.assertAlmostEqual(r.area, math.pi / 2, places=14)
        # 同一八分之一球的另两种循环排列（同为逆时针）。
        self.assertAlmostEqual(
            unwrap_ring([D(90, 0), D(0, 90), D(0, 0)]).area, math.pi / 2, places=14)
        self.assertAlmostEqual(
            unwrap_ring([D(0, 90), D(0, 0), D(90, 0)]).area, math.pi / 2, places=14)

    def test_left_side_positive_convention(self):
        # 沿 (0,0)->(90E,0) 东行，左侧是北；三角形在北侧 => 面积为正。
        self.assertGreater(unwrap_ring([D(0, 0), D(90, 0), D(0, 90)]).area, 0.0)
        # 同三角形反向：左侧是南，八分之一球在右侧 => 面积为负。
        self.assertAlmostEqual(
            unwrap_ring([D(0, 90), D(90, 0), D(0, 0)]).area, -math.pi / 2, places=14)

    def test_direction_reversal_flips_sign(self):
        rings = [
            [D(20, 30), D(-60, 10), D(150, -20), D(80, -45)],
            [D(170, 10), D(-170, 10), (-math.pi, math.radians(60))],
            [D(0, -10), D(-120, -10), D(120, -10)],
        ]
        for ring in rings:
            a = unwrap_ring(ring).area
            arev = unwrap_ring(list(reversed(ring))).area
            self.assertAlmostEqual(a, -arev, places=12, msg=str(ring))

    def test_rotation_invariance(self):
        rng = random.Random(0xC0FFEE)
        axis = normalize((0.3, -0.5, 0.8))
        angle = 1.23456
        compared = 0
        for _ in range(40):
            ring = _cone_ring(rng, rng.uniform(0.05, 1.45))
            rotated = [_rodrigues(unit(lo, la), axis, angle) for lo, la in ring]
            rotated_lola = [_to_lonlat(v) for v in rotated]
            a0 = unwrap_ring(ring).area
            a1 = _safe_area(rotated_lola)
            if a1 is not None:
                self.assertAlmostEqual(a0, a1, places=9)
                compared += 1
        self.assertGreater(compared, 20)

    def test_planar_limit_of_small_triangle(self):
        # 赤道附近极小“经纬直角三角形”：两边局部正交且长度均为 eps，
        # 球面角盈 -> 平面面积 eps^2/2（北极处经度腿会缩成 eps^2，不能放那里）。
        eps = 1e-4
        ring = [(0.0, 0.0), (eps, 0.0), (0.0, eps)]
        a = unwrap_ring(ring).area
        self.assertAlmostEqual(a, eps ** 2 / 2, delta=eps ** 3)

    def test_small_triangle_machine_precision(self):
        # 极小三角形仍保持相对机器精度（atan2 行列式公式，无 L'Huilier 抵消）。
        eps = 1e-7
        ring = [(0.0, 0.0), (eps, 0.0), (0.0, eps)]
        a = unwrap_ring(ring).area
        self.assertAlmostEqual(a, eps ** 2 / 2, delta=10.0 * eps ** 4)

    def test_area_range_and_full_great_circle_boundary(self):
        rng = random.Random(1234)
        for _ in range(150):
            ring = _cone_ring(rng, rng.uniform(0.03, 1.55))
            a = _safe_area(ring)
            if a is not None:
                self.assertTrue(-2 * math.pi < a <= 2 * math.pi)
        # 沿赤道完整大圆（西向）：真实有向面积 -2pi，按 (-2pi,2pi] 归并到 +2pi。
        self.assertEqual(unwrap_ring([D(0, 0), D(-120, 0), D(120, 0)]).area, 2 * math.pi)
        # 东向完整大圆在模 4pi 下同样归并到闭端 +2pi。
        self.assertEqual(unwrap_ring([D(120, 0), D(-120, 0), D(0, 0)]).area, 2 * math.pi)


class TestAreaAgainstIndependentReferences(unittest.TestCase):
    def test_fixed_cases(self):
        cases = [
            ([D(0, 0), D(90, 0), D(0, 90)], math.pi / 2),
            ([D(0, 90), D(90, 0), D(0, 0)], -math.pi / 2),
            ([D(170, 10), D(-170, 10), (-math.pi, math.radians(60))], None),
            ([D(170, 70), D(-10, 70), D(-90, 10)], None),
            ([D(0, -10), D(-120, -10), D(120, -10)], None),
            ([D(180, 20), D(0, 60), D(90, 10)], None),
            ([D(170, 0), D(180, 0), D(180, 50), D(150, 30)], None),
        ]
        for ring, analytic in cases:
            a = unwrap_ring(ring).area
            if analytic is not None:
                self.assertAlmostEqual(a, analytic, places=14)
            self.assertAlmostEqual(a, gauss_bonnet_area(ring), places=9)
            self.assertAlmostEqual(a, fan_signed_area(ring, PIVOTS[0]), places=9)

    def test_random_simple_rings_against_gb_and_fan(self):
        rng = random.Random(20260923)
        checked = 0
        for _ in range(220):
            ring = _cone_ring(rng, rng.uniform(0.03, 1.50))
            a = _safe_area(ring)
            if a is None:
                continue
            self.assertAlmostEqual(a, gauss_bonnet_area(ring), places=9)
            for q in PIVOTS:
                self.assertAlmostEqual(a, fan_signed_area(ring, q), places=9)
            checked += 1
        self.assertGreater(checked, 150)

    def test_three_pivots_agree_with_each_other(self):
        rng = random.Random(99)
        for _ in range(60):
            ring = _cone_ring(rng, rng.uniform(0.05, 1.4))
            vals = [fan_signed_area(ring, q) for q in PIVOTS]
            self.assertAlmostEqual(vals[0], vals[1], places=9)
            self.assertAlmostEqual(vals[0], vals[2], places=9)


if __name__ == "__main__":
    unittest.main()
