from _typeshed import Incomplete
from typing import Any, Callable, NamedTuple

__all__ = ['T', 'P', 'Shape']

P: Incomplete
T: Incomplete
Shape = list[dict[str, Any]]
ShapeSampler = Callable[..., Shape]

class Box(NamedTuple):
    """
    A 3D box.

    Examples:
        - `Box((0, 0, 0), 1)`
            A box with corner positions (-0.5, -0.5, -0.5) and (0.5, 0.5, 0.5).
        - `Box((0, 0, 0), (1, 2, 3))`
            A box with corner positions (-0.5, -1, -1.5) and (0.5, 1, 1.5).

    Attributes:
        center: P - Box center.
        size: float | P - Box size.
            If a float, the box is uniformly scaled; if a 3D vector, the box is non-uniformly scaled.
    """
    center: P
    size: float | P
