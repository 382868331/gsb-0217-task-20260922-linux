"""Error semantics for the antimeridian unfolding package.

All errors derive from :class:`GeometryError` so callers can catch the whole
family with a single ``except``.  Every error carries a ``location`` string
that names the offending vertex/edge, satisfying the contract requirement
that illegal input is *locatable* rather than silently corrected.

The API is a single pure function (:func:`antimeridian.core.unfold_loop`):
it either returns a complete result or raises before returning anything, so
a failed call never leaves partial or undocumented state behind.  The
algorithm is closed-form (no iterative search), therefore "search budget
exhausted" and "proven unsatisfiable" outcomes do not exist here; every
rejection is a definitive, locatable input/geometry violation.
"""


class GeometryError(ValueError):
    """Base class for every error raised by this package.

    Attributes
    ----------
    location:
        Human-readable, machine-testable locator, e.g. ``"edge 3"`` or
        ``"vertex 1.lon"``.  ``None`` when no single location applies.
    """

    def __init__(self, message: str, location: str | None = None):
        self.location = location
        text = message if location is None else f"{message} [at {location}]"
        super().__init__(text)


class InvalidInputError(GeometryError):
    """A numeric/structural precondition of the public API was violated.

    Covers NaN/Infinity, booleans where numbers are required, lon/lat out of
    range, vertex count outside ``[3, 100]`` and zero-length edges.
    """


class AntipodalAdjacentError(GeometryError):
    """Two consecutive vertices are antipodal (or within the guarded band).

    Antipodal endpoints admit infinitely many shortest great-circle arcs, so
    the contract forbids them.  ``edge_index`` gives the offending edge.
    """

    def __init__(self, edge_index: int):
        self.edge_index = edge_index
        super().__init__(
            "consecutive vertices are antipodal; shortest arc is not unique",
            location=f"edge {edge_index}",
        )


class SelfIntersectionError(GeometryError):
    """Two non-adjacent arcs of the loop intersect.

    ``edge_indices`` is the sorted pair of zero-based edge indices.
    """

    def __init__(self, edge_a: int, edge_b: int, where: str):
        self.edge_indices = (edge_a, edge_b)
        super().__init__(
            f"non-adjacent edges {edge_a} and {edge_b} intersect ({where})",
            location=f"edges {edge_a},{edge_b}",
        )


class DegenerateLoopError(GeometryError):
    """The loop collapses onto a great circle; signed area is ambiguous.

    A loop whose vertices all lie on one great circle encloses an exact
    hemisphere on *both* sides, so a signed area in ``(-2*pi, 2*pi]`` cannot
    be assigned.  Raised instead of silently returning an arbitrary sign.
    """


class NonConvexLoopError(GeometryError):
    """The loop is not convex; hemisphere clipping requires convexity.

    Raised at the first vertex whose turning direction contradicts the
    orientation established by earlier vertices (a reflex vertex).
    ``vertex_index`` gives that zero-based vertex.
    """

    def __init__(self, vertex_index: int):
        self.vertex_index = vertex_index
        super().__init__(
            "loop is not convex (reflex turn); hemisphere clipping "
            "requires a convex loop",
            location=f"vertex {vertex_index}",
        )
