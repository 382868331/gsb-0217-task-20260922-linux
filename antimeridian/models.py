"""Immutable public data structures.

All coordinates are angles in **radians**:

* longitude  in ``[-pi, pi]`` (input), canonicalized to ``(-pi, pi]`` inside;
  the antimeridian seam is the half-meridian ``lon == +pi`` (== ``-pi``),
* latitude   in ``[-pi/2, pi/2]``.

Points accept plain ``(lon, lat)`` pairs as well as :class:`GeoPoint`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeoPoint:
    """A point given by geographic longitude/latitude (radians)."""

    lon: float
    lat: float

    def __iter__(self):
        yield self.lon
        yield self.lat


@dataclass(frozen=True)
class ArcParam:
    """The *original* short great-circle arc for one loop edge.

    Attributes
    ----------
    edge_index:
        Zero-based index of the edge in the input loop.
    start, end:
        Endpoint geographic points exactly as supplied (validated input).
    central_angle:
        Length of the unique shortest arc in ``(0, pi)`` radians.
    pole_coincident:
        True iff the whole arc lies on the 180-degree half-meridian seam.
        Such an edge is reported as ``coincident`` and kept as one piece;
        no synthetic cross-seam connection is produced.
    contains_pole:
        True iff the arc passes through the north or south pole in its
        interior.  The pole is emitted as an :class:`AmbiguousMarker`.
    """

    edge_index: int
    start: GeoPoint
    end: GeoPoint
    central_angle: float
    pole_coincident: bool = False
    contains_pole: bool = False


@dataclass(frozen=True)
class Fragment:
    """One render fragment with a continuous unwrapped longitude range.

    Attributes
    ----------
    edge_index:
        Owning edge.
    order:
        Fragment order along the arc (0 = first from the edge start).
    lon_start, lon_end:
        Endpoint longitudes after date-line unwrapping.  Within one fragment
        the longitude never jumps across the seam: a fragment crossing the
        antimeridian eastward spans e.g. ``175 deg -> 190 deg`` (radians
        ``3.054 -> 3.316``), an east-to-west one ``-175 -> -190``.
        Consecutive fragments of the loop join continuously
        (``frag[k].lon_end == frag[k+1].lon_start``) except across a pole
        junction, which is flagged ``ambiguous`` instead of being connected.
    lat_start, lat_end:
        Endpoint latitudes.
    kind:
        ``"normal"``     -- ordinary fragment;
        ``"coincident"`` -- the whole original arc lies on the seam
                            (kept as a single piece, original arc preserved);
        ``"ambiguous"``  -- fragment touches an interior pole crossing.
    pole:
        ``"north"``/``"south"`` when the fragment touches a pole (crossing
        or pole vertex), else ``None``.  A renderer must not auto-connect
        across such a junction.
    """

    edge_index: int
    order: int
    lon_start: float
    lon_end: float
    lat_start: float
    lat_end: float
    kind: str = "normal"
    pole: str | None = None


@dataclass(frozen=True)
class SeamCrossing:
    """A genuine intersection of an arc with the 180-degree half-meridian.

    Computed in 3-D (intersection of great-circle planes), never by planar
    lon/lat interpolation.  ``t`` is the arc parameter in ``(0, 1)``;
    ``lon_unwrapped`` is the continuous longitude both neighbouring fragments
    share at the cut (``+pi`` or ``-pi`` plus the running unwrap offset).
    """

    edge_index: int
    t: float
    lat: float
    lon_unwrapped: float


@dataclass(frozen=True)
class AmbiguousMarker:
    """A pole event belonging to an edge, kept out of the fragment chain.

    The pole is a single physical point whose longitude is undefined; connecting
    it linearly in a lon/lat renderer would draw arbitrary spokes, so it is
    reported separately.  ``t`` is the arc parameter of the pole: interior
    for a crossing, ``1.0`` on the incoming edge for a pole *vertex*.
    """

    edge_index: int
    pole: str
    t: float
    lat: float


@dataclass(frozen=True)
class UnfoldedLoop:
    """Result of :func:`antimeridian.core.unfold_loop`.

    Attributes
    ----------
    arcs:
        Original arc parameters, in input edge order.
    fragments:
        Continuous-longitude render fragments, in input edge order and,
        within each edge, along-arc order.
    seam_crossings:
        Genuine arc/seam intersections (non-pole), in loop order.
    ambiguous_markers:
        Pole events flagged ``ambiguous`` (renderer decides how to join).
    signed_area:
        Directed area in ``(-2*pi, 2*pi]`` ``; the left side of the
        oriented loop is positive.  This is the oriented area modulo ``4*pi``
        represented in that interval (see README "Area convention").
    area_winding:
        ``"ccw"`` (positive signed area) or ``"cw"``.
    """

    arcs: tuple[ArcParam, ...]
    fragments: tuple[Fragment, ...]
    seam_crossings: tuple[SeamCrossing, ...]
    ambiguous_markers: tuple[AmbiguousMarker, ...]
    signed_area: float
    area_winding: str


@dataclass(frozen=True)
class ClippedLoop:
    """Result of :func:`antimeridian.clip.clip_loop_hemisphere`.

    Attributes
    ----------
    pole:
        The clip hemisphere's centre (kept side), as validated.
    status:
        ``"full"``    -- the whole polygon lies in the kept hemisphere;
                         ``ring`` is the input loop unchanged;
        ``"empty"``   -- the intersection is empty (at most a boundary
                         touch); ``ring`` is empty and both areas are 0;
        ``"clipped"`` -- a proper subset remains.
    ring:
        Vertices of the clipped ring as :class:`GeoPoint` (lon/lat radians),
        in the input's orientation, implicitly closed.  Every edge is again
        a unique short great-circle arc (an antipodal pair produced on the
        clip circle is split by an inserted midpoint, never left ambiguous).
        Empty when ``status == "empty"``.
    area:
        Non-negative area of the intersection in steradians, ``[0, 2*pi)``.
    signed_area:
        Oriented area of the clipped ring (sign follows the input winding);
        0 when empty.
    """

    pole: GeoPoint
    status: str
    ring: tuple[GeoPoint, ...]
    area: float
    signed_area: float
