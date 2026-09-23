"""antimeridian — date-line unfolding of spherical loops.

Public surface::

    from antimeridian import (
        unfold_loop, unfold_loop_degrees, GeoPoint,
        UnfoldedLoop, ArcParam, Fragment, SeamCrossing, AmbiguousMarker,
        GeometryError, InvalidInputError, AntipodalAdjacentError,
        SelfIntersectionError, DegenerateLoopError,
    )
"""

from .core import unfold_loop, unfold_loop_degrees
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
    SeamCrossing,
    UnfoldedLoop,
)

__all__ = [
    "unfold_loop",
    "unfold_loop_degrees",
    "GeoPoint",
    "UnfoldedLoop",
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
