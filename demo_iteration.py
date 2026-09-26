"""Live demo for the signed 3-D area moments (this task's increment).

Everything is computed on the spot (well under 90 seconds; in practice
about one second).  Run:  python demo_iteration.py
"""

from __future__ import annotations

import math

from antimeridian import (
    SelfIntersectionError,
    region_moments,
    region_moments_degrees,
)

PI = math.pi


def show(title):
    print("=" * 68)
    print(title)
    print("=" * 68)


def fmt_vec(v):
    return "(" + ", ".join(f"{x:+.9f}" for x in v) + ")"


def fmt_mat(s):
    return "[" + ",\n         ".join(fmt_vec(row) for row in s) + "]"


# --------------------------------------------------------------------- #
show("1. Spherical right triangle (octant) -- exact closed-form check")
# Vertices (0,0), (90,0), (0,90) degrees: area pi/2,
# M = (pi/4)(1,1,1), S = diag(pi/6) + off-diagonal 1/3.
r = region_moments_degrees([(0.0, 0.0), (90.0, 0.0), (0.0, 90.0)])
print(f"signed area        : {r.signed_area:.12f}  (pi/2 = {PI/2:.12f})")
print(f"first moment M     : {fmt_vec(r.first_moment)}")
print(f"                   (pi/4 = {PI/4:.9f} on every axis)")
print(f"second moment S    : {fmt_mat(r.second_moment)}")
print(f"                   (diag pi/6 = {PI/6:.9f}, off-diag 1/3 = "
      f"{1/3:.9f})")
print(f"trace(S) - area    : {sum(r.second_moment[i][i] for i in range(3)) - r.signed_area:+.3e}")
print(f"direction centroid : {fmt_vec(r.direction_centroid)}")
print(f"                   (exact (1,1,1)/sqrt(3) = "
      f"({1/math.sqrt(3):.9f}, ...))")

# --------------------------------------------------------------------- #
show("2. Reversal flips area and both moments (region size unchanged)")
quad = [(10.0, 5.0), (55.0, 8.0), (60.0, 45.0), (15.0, 50.0)]
fwd = region_moments_degrees(quad)
rev = region_moments_degrees(list(reversed(quad)))
print(f"area  fwd/rev      : {fwd.signed_area:+.9f} / {rev.signed_area:+.9f}")
print(f"|M|   fwd/rev      : {math.dist((0,0,0), fwd.first_moment):.9f} / "
      f"{math.dist((0,0,0), rev.first_moment):.9f}")
dm = max(abs(fwd.first_moment[k] + rev.first_moment[k]) for k in range(3))
ds = max(abs(fwd.second_moment[i][j] + rev.second_moment[i][j])
         for i in range(3) for j in range(3))
print(f"M_fwd + M_rev      : max |sum| = {dm:.3e}")
print(f"S_fwd + S_rev      : max |sum| = {ds:.3e}")

# --------------------------------------------------------------------- #
show("3. Rotation covariance: M -> R M, S -> R S R^T")
def rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))

def rot_y(t):
    c, s = math.cos(t), math.sin(t)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))

def mat_mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))

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

R = mat_mul(rot_z(math.radians(37.0)), rot_y(math.radians(51.0)))
base = region_moments_degrees(quad)
rot_pts = [to_lonlat(mat_vec(R, to_xyz(math.radians(lo), math.radians(la))))
           for lo, la in quad]
rot = region_moments(rot_pts)
dm = max(abs(mat_vec(R, base.first_moment)[k] - rot.first_moment[k])
         for k in range(3))
rsrt = mat_mul(mat_mul(R, base.second_moment), transpose(R))
ds = max(abs(rsrt[i][j] - rot.second_moment[i][j])
         for i in range(3) for j in range(3))
print(f"|R M - M_rot|_inf  : {dm:.3e}")
print(f"|R S R^T - S_rot|  : {ds:.3e}")
print(f"area change        : {abs(base.signed_area) - abs(rot.signed_area):+.3e}")

# --------------------------------------------------------------------- #
show("4. Additivity: a quad equals the sum of its two triangles")
a, b, c, d = quad
whole = region_moments_degrees(quad)
t1 = region_moments_degrees([a, b, c])
t2 = region_moments_degrees([a, c, d])
da = whole.signed_area - (t1.signed_area + t2.signed_area)
dm = max(abs(whole.first_moment[k]
             - (t1.first_moment[k] + t2.first_moment[k])) for k in range(3))
ds = max(abs(whole.second_moment[i][j]
             - (t1.second_moment[i][j] + t2.second_moment[i][j]))
         for i in range(3) for j in range(3))
print(f"area residual      : {da:+.3e}")
print(f"M residual         : {dm:.3e}")
print(f"S residual         : {ds:.3e}")

# --------------------------------------------------------------------- #
show("5. Loop across the date line (3-D computation, no unwrapping)")
dl = [(170.0, 10.0), (-170.0, 10.0), (-170.0, 50.0), (170.0, 50.0)]
r = region_moments_degrees(dl)
print(f"signed area        : {r.signed_area:+.9f}")
print(f"first moment M     : {fmt_vec(r.first_moment)}")
print(f"direction centroid : {fmt_vec(r.direction_centroid)}")
lon = math.degrees(math.atan2(r.direction_centroid[1],
                              r.direction_centroid[0]))
print(f"centroid longitude : {lon:+.4f} deg (region straddles +/-180)")

# --------------------------------------------------------------------- #
show("6. Zero first moment -> no centroid direction")
# Dumbbell region tuned so that M = 0 exactly (see tests/test_moments.py).
bb = math.radians(154.39798664667245)
pts = [(math.radians(60), math.radians(10)),
       (math.radians(60), math.radians(30)),
       (bb, math.radians(30)), (bb, math.radians(-30)),
       (math.radians(60), math.radians(-30)),
       (math.radians(60), math.radians(-10)),
       (math.radians(-60), math.radians(-10)),
       (math.radians(-60), math.radians(-30)),
       (-bb, math.radians(-30)), (-bb, math.radians(30)),
       (math.radians(-60), math.radians(30)),
       (math.radians(-60), math.radians(10))]
r = region_moments(pts)
print(f"|M|                : {max(abs(x) for x in r.first_moment):.3e}")
print(f"direction centroid : {r.direction_centroid}")

# --------------------------------------------------------------------- #
show("7. Rejection path: self-intersecting loop is refused")
bowtie = [(0.0, 0.0), (1.0, 0.5), (1.0, 0.0), (0.0, 0.5)]
try:
    region_moments(bowtie)
except SelfIntersectionError as exc:
    print(f"raised SelfIntersectionError: {exc}")

print()
print("demo_iteration.py finished -- all values computed live.")
