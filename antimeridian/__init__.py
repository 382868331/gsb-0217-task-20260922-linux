"""antimeridian — date-line unfolding of spherical loops.

Public surface::

    from antimeridian import (
        unfold_loop, unfold_loop_degrees, GeoPoint,
        UnfoldedLoop, ArcParam, Fragment, SeamCrossing, AmbiguousMarker,
        split_region_at_seam, split_region_at_seam_degrees,
        SplitRegion, RegionLoop, RegionSegment,
        GeometryError, InvalidInputError, AntipodalAdjacentError,
        SelfIntersectionError, DegenerateLoopError, UnsupportedError,
    )
"""

from .core import unfold_loop, unfold_loop_degrees
from .errors import (
    AntipodalAdjacentError,
    DegenerateLoopError,
    GeometryError,
    InvalidInputError,
    SelfIntersectionError,
    UnsupportedError,
)
from .models import (
    AmbiguousMarker,
    ArcParam,
    Fragment,
    GeoPoint,
    RegionLoop,
    RegionSegment,
    SeamCrossing,
    SplitRegion,
    UnfoldedLoop,
)
from .region import split_region_at_seam, split_region_at_seam_degrees

__all__ = [
    "unfold_loop",
    "unfold_loop_degrees",
    "split_region_at_seam",
    "split_region_at_seam_degrees",
    "GeoPoint",
    "UnfoldedLoop",
    "ArcParam",
    "Fragment",
    "SeamCrossing",
    "AmbiguousMarker",
    "SplitRegion",
    "RegionLoop",
    "RegionSegment",
    "GeometryError",
    "InvalidInputError",
    "AntipodalAdjacentError",
    "SelfIntersectionError",
    "DegenerateLoopError",
    "UnsupportedError",
]
