"""antimeridian — date-line unfolding of spherical loops.

Public surface::

    from antimeridian import (
        unfold_loop, unfold_loop_degrees,
        clip_loop_to_hemisphere, clip_loop_to_hemisphere_degrees,
        GeoPoint, UnfoldedLoop, ArcParam, Fragment, SeamCrossing,
        AmbiguousMarker, ClippedLoop,
        GeometryError, InvalidInputError, AntipodalAdjacentError,
        SelfIntersectionError, DegenerateLoopError, NonConvexLoopError,
    )
"""

from .core import (
    clip_loop_to_hemisphere,
    clip_loop_to_hemisphere_degrees,
    unfold_loop,
    unfold_loop_degrees,
)
from .errors import (
    AntipodalAdjacentError,
    DegenerateLoopError,
    GeometryError,
    InvalidInputError,
    NonConvexLoopError,
    SelfIntersectionError,
)
from .models import (
    AmbiguousMarker,
    ArcParam,
    ClippedLoop,
    Fragment,
    GeoPoint,
    SeamCrossing,
    UnfoldedLoop,
)

__all__ = [
    "unfold_loop",
    "unfold_loop_degrees",
    "clip_loop_to_hemisphere",
    "clip_loop_to_hemisphere_degrees",
    "GeoPoint",
    "UnfoldedLoop",
    "ArcParam",
    "Fragment",
    "SeamCrossing",
    "AmbiguousMarker",
    "ClippedLoop",
    "GeometryError",
    "InvalidInputError",
    "AntipodalAdjacentError",
    "SelfIntersectionError",
    "DegenerateLoopError",
    "NonConvexLoopError",
]
