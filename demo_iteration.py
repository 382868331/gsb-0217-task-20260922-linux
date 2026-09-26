#!/usr/bin/env python3
"""Live demonstration for signed spherical area moments."""

from __future__ import annotations

import math

from antimeridian import NotInOpenHemisphereError, unfold_loop

D = math.pi / 180.0


def fmt_vector(v):
    return "(" + ", ".join(f"{x: .12f}" for x in v) + ")"


def fmt_matrix(matrix):
    return "\n".join(
        "    [" + ", ".join(f"{x: .12f}" for x in row) + "]"
        for row in matrix
    )


def rodrigues(v, axis, angle):
    x, y, z = v
    ux, uy, uz = axis
    c, s = math.cos(angle), math.sin(angle)
    return (
        (c + ux*ux*(1-c))*x + (ux*uy*(1-c)-uz*s)*y
        + (ux*uz*(1-c)+uy*s)*z,
        (uy*ux*(1-c)+uz*s)*x + (c+uy*uy*(1-c))*y
        + (uy*uz*(1-c)-ux*s)*z,
        (uz*ux*(1-c)-uy*s)*x + (uz*uy*(1-c)+ux*s)*y
        + (c+uz*uz*(1-c))*z,
    )


def to_lonlat(v):
    return math.atan2(v[1], v[0]), math.asin(max(-1.0, min(1.0, v[2])))


def xyz(point):
    lon, lat = point
    return (math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon),
            math.sin(lat))


def show(title, result):
    print(title)
    print("  signed area       :", result.signed_area)
    print("  integral x dA     :", fmt_vector(result.first_moment))
    print("  integral x x^T dA :")
    print(fmt_matrix(result.second_moment))
    trace = sum(row[i] for i, row in enumerate(result.second_moment))
    print("  matrix trace      :", trace)
    print("  centroid axis     :", fmt_vector(result.centroid_direction))


def main():
    print("Spherical area moments - live standard-library computation")
    triangle = [(0.0, 0.0), (math.pi/3.0, 0.0), (0.0, math.pi/2.0)]
    show("\n1. Spherical right triangle:", unfold_loop(triangle))

    axis = tuple(a / math.sqrt(14.0) for a in (1.0, -2.0, 3.0))
    rotated = [to_lonlat(rodrigues(xyz(p), axis, 0.987654321))
               for p in triangle]
    show("\n2. Same region after rigid 3-D rotation:", unfold_loop(rotated))

    first = unfold_loop([(0.0, 0.0), (math.pi/4.0, 0.0),
                         (0.0, math.pi/2.0)])
    second = unfold_loop([(math.pi/4.0, 0.0), (math.pi/2.0, 0.0),
                          (0.0, math.pi/2.0)])
    whole = unfold_loop([(0.0, 0.0), (math.pi/2.0, 0.0),
                         (0.0, math.pi/2.0)])
    print("\n3. Two known wedges are additive:")
    print("  wedge areas sum   :", first.signed_area + second.signed_area)
    print("  whole area        :", whole.signed_area)
    print("  first-moment sum  :",
          fmt_vector(tuple(a+b for a, b in
                           zip(first.first_moment, second.first_moment))))

    reversed_loop = unfold_loop(list(reversed(triangle)))
    print("\n4. Reversed contour:")
    print("  reversed area     :", reversed_loop.signed_area)
    print("  reversed moment   :", fmt_vector(reversed_loop.first_moment))
    print("  centroid axis     :",
          fmt_vector(reversed_loop.centroid_direction))

    print("\n5. Closed-hemisphere boundary is refused:")
    try:
        unfold_loop([(0, 0), (90*D, 0), (180*D, 0), (0, 90*D)])
    except NotInOpenHemisphereError as exc:
        print(" ", type(exc).__name__, "-", exc)
    print("\nDone.")


if __name__ == "__main__":
    main()
