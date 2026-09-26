"""antimeridian — date-line unfolding of spherical loops.

Public surface::

    from antimeridian import (
        unfold_loop, unfold_loop_degrees, GeoPoint,
        region_moments, region_moments_degrees, RegionMoments,
        UnfoldedLoop, ArcParam, Fragment, SeamCrossing, AmbiguousMarker,
        GeometryError, InvalidInputError, AntipodalAdjacentError,
        SelfIntersectionError, DegenerateLoopError,
    )
"""

from .core import (
    region_moments,
    region_moments_degrees,
    unfold_loop,
    unfold_loop_degrees,
)
from .errors import (
    AntipodalAdjacentError,
    DegenerateLoopError,
    GeometryError,
    InvalidInputError,
    SelfIntersectionError,
)
from .models import (
    AmbiguousMarker,
    ArcParam,
    Fragment,
    GeoPoint,
    RegionMoments,
    SeamCrossing,
    UnfoldedLoop,
)

__all__ = [
    "unfold_loop",
    "unfold_loop_degrees",
    "region_moments",
    "region_moments_degrees",
    "GeoPoint",
    "UnfoldedLoop",
    "RegionMoments",
    "ArcParam",
    "Fragment",
    "SeamCrossing",
    "AmbiguousMarker",
    "GeometryError",
    "InvalidInputError",
    "AntipodalAdjacentError",
    "SelfIntersectionError",
    "DegenerateLoopError",
]
