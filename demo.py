#!/usr/bin/env python3
"""Fixed-input demonstration for the antimeridian unfolding module.

Run with::

    python demo.py

It performs real (deterministic, non-recorded) computation:

1. the analytic one-eighth-sphere triangle (signed area must be pi/2) and a
   rigid 3-D rotation of it (area must be invariant);
2. a loop that crosses the date line, with every render fragment and the real
   seam intersections listed;
3. an actually triggered rejection boundary -- a self-intersecting ("bow-tie")
   loop that is refused with a locatable error.

No network, no third-party packages, no sleeps.
"""

from __future__ import annotations

import math

from antimeridian import (
    NonConvexLoopError,
    SelfIntersectionError,
    clip_loop_to_hemisphere,
    unfold_loop,
)
from antimeridian.geometry import lonlat_to_xyz

D = math.pi / 180.0


def fmt_lon(x: float) -> str:
    return f"{x / D:8.3f} deg"


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


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Analytic area + rotation invariance.
    # ------------------------------------------------------------------ #
    section("1. One-eighth-sphere triangle  (analytic area = pi/2)")
    octant = [(0.0, 0.0), (math.pi / 2, 0.0), (0.0, math.pi / 2)]
    res = unfold_loop(octant)
    print("vertices (lon,lat) deg:", [(0, 0), (90, 0), (0, 90)])
    print(f"signed area          = {res.signed_area:.15f}")
    print(f"analytic pi/2        = {math.pi / 2:.15f}")
    print(f"absolute difference  = {abs(res.signed_area - math.pi / 2):.3e}")
    print(f"winding (left +)     = {res.area_winding}")
    rev = unfold_loop(list(reversed(octant)))
    print(f"reversed-loop area   = {rev.signed_area:.15f} (negates)")

    axis = tuple(a / math.sqrt(3.0) for a in (1.0, 1.0, 1.0))
    rotated = [to_lonlat(rodrigues(lonlat_to_xyz(*p), axis, 1.23456))
               for p in octant]
    rot_area = unfold_loop(rotated).signed_area
    print("rigid 3-D rotation by 1.23456 rad about axis (1,1,1)/sqrt(3):")
    print(f"  rotated area       = {rot_area:.15f}")
    print(f"  |delta|            = {abs(rot_area - math.pi / 2):.3e}")

    # ------------------------------------------------------------------ #
    # 2. Date-line crossing loop with explicit fragments.
    # ------------------------------------------------------------------ #
    section("2. Loop crossing the date line -- split render fragments")
    poly_deg = [(170.0, 10.0), (-170.0, 10.0), (-175.0, 40.0)]
    poly = [(lo * D, la * D) for lo, la in poly_deg]
    r = unfold_loop(poly)
    print("vertices (lon,lat) deg:", poly_deg)
    print(f"signed area          = {r.signed_area:.9f} rad^2 "
          f"({r.area_winding})")
    print()
    print("original arcs (short great-circle edges):")
    for a in r.arcs:
        tag = []
        if a.pole_coincident:
            tag.append("coincident")
        if a.contains_pole:
            tag.append("contains-pole")
        extra = f"  [{', '.join(tag)}]" if tag else ""
        print(f"  edge {a.edge_index}: ({a.start.lon/D:7.2f},"
              f"{a.start.lat/D:6.2f}) -> ({a.end.lon/D:7.2f},"
              f"{a.end.lat/D:6.2f})  central angle "
              f"{a.central_angle / D:7.3f} deg{extra}")
    print()
    print("render fragments (continuous unwrapped longitude bands):")
    for f in r.fragments:
        print(f"  edge {f.edge_index} #{f.order} [{f.kind:10s}] "
              f"lon {fmt_lon(f.lon_start)} -> {fmt_lon(f.lon_end)}   "
              f"lat {f.lat_start / D:7.3f} -> {f.lat_end / D:7.3f} deg")
    print()
    print("real seam intersections (found in 3-D, not planar):")
    for c in r.seam_crossings:
        print(f"  edge {c.edge_index}: t = {c.t:.6f}, "
              f"lat = {c.lat / D:.4f} deg, unwrapped seam lon = "
              f"{c.lon_unwrapped / D:.1f} deg")
    print(f"ambiguous pole markers: {len(r.ambiguous_markers)} (none here)")

    # ------------------------------------------------------------------ #
    # 3. An actually triggered rejection boundary.
    # ------------------------------------------------------------------ #
    section("3. Rejection boundary -- self-intersecting (bow-tie) loop")
    bowtie_deg = [(-20.0, 0.0), (20.0, 0.0), (-20.0, 20.0), (20.0, 20.0)]
    print("vertices (lon,lat) deg:", bowtie_deg)
    print("opposite edges cross each other -> must be refused")
    try:
        unfold_loop([(lo * D, la * D) for lo, la in bowtie_deg])
    except SelfIntersectionError as exc:
        print(f"raised {type(exc).__name__}: {exc}")
        print(f"locator: {exc.location}; edge pair: {exc.edge_indices}")
    else:  # pragma: no cover - defensive, must not happen
        raise AssertionError("self-intersecting loop was not rejected")

    # ------------------------------------------------------------------ #
    # 4. Hemisphere clipping of a convex loop.
    # ------------------------------------------------------------------ #
    section("4. Convex loop intersected with a great-circle hemisphere")
    cap_deg = [(0.0, 80.0), (120.0, 80.0), (-120.0, 80.0)]
    cap = [(lo * D, la * D) for lo, la in cap_deg]
    cap_area = unfold_loop(cap).signed_area
    print(f"polar cap vertices (lon,lat) deg: {cap_deg}")
    print(f"cap area (unsigned)      = {abs(cap_area):.9f} rad^2")

    pole_deg = (10.0, 0.0)
    pole = (pole_deg[0] * D, pole_deg[1] * D)
    res = clip_loop_to_hemisphere(cap, pole)
    print(f"\nclip by hemisphere centred on {pole_deg} (status={res.status}):")
    for i, p in enumerate(res.points):
        print(f"  vertex {i}: lon {p.lon / D:9.4f} deg, lat {p.lat / D:8.4f} deg")
    print(f"  clipped area           = {res.area:.9f} rad^2 "
          f"(signed {res.signed_area:+.9f}, {res.winding})")

    anti = clip_loop_to_hemisphere(cap, (pole[0] - math.pi, -pole[1]))
    print(f"opposite hemisphere area = {anti.area:.9f} rad^2 "
          f"(status={anti.status})")
    print(f"partition identity: {res.area:.9f} + {anti.area:.9f} "
          f"= {res.area + anti.area:.9f} (cap area {abs(cap_area):.9f}, "
          f"diff {abs(res.area + anti.area - abs(cap_area)):.2e})")

    full = clip_loop_to_hemisphere(cap, (0.0, math.pi / 2))
    print(f"\nclip by northern hemisphere: status={full.status}, "
          f"area={full.area:.9f} (whole cap kept)")
    empty = clip_loop_to_hemisphere(cap, (0.0, -math.pi / 2))
    print(f"clip by southern hemisphere: status={empty.status}, "
          f"area={empty.area:.1f} (whole cap excluded)")

    print("\nnon-convex (arrow) loop must be refused:")
    arrow = [(lo * D, la * D) for lo, la in
             [(0.0, 0.0), (40.0, 0.0), (10.0, 10.0), (40.0, 20.0),
              (0.0, 20.0)]]
    try:
        clip_loop_to_hemisphere(arrow, (0.0, math.pi / 2))
    except NonConvexLoopError as exc:
        print(f"raised {type(exc).__name__}: {exc}")
        print(f"locator: {exc.location}; reflex vertex: {exc.vertex_index}")
    else:  # pragma: no cover - defensive, must not happen
        raise AssertionError("non-convex loop was not rejected")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
