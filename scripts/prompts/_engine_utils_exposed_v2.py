from typing import Literal, Union, Tuple, Any
from type_utils import Shape, P
from _shape_utils import primitive_call as _primitive_call
from engine.constants import ENGINE_MODE
assert ENGINE_MODE in ['exposed_v2'], ENGINE_MODE


__all__ = ["primitive_call"]


def primitive_call(
        name: Literal["cube", "sphere", "cylinder", "cone"],
        shape_kwargs: dict[str, Any],
        color: Tuple[float, float, float] = (1., 1., 1.),
) -> Shape:
    """
    Constructs a primitive shape.

    Args:
        name: str - 'cube', 'sphere', 'cylinder', or 'cone'.
        shape_kwargs: dict[str, Any] - keyword arguments for the primitive shape.
            - For 'cube': {'scale': P} - 3-tuple of floats for scaling along x, y, z axes, relative to a unit cube centered at origin.
            - For 'sphere': {'radius': float} - radius of the sphere.
            - For 'cylinder': {'radius': float, 'p0': P, 'p1': P}
                - radius: float - radius of the cylinder.
                - p0: P - 3-tuple of floats for the start point of the cylinder's centerline.
                - p1: P - 3-tuple of floats for the end point of the cylinder's centerline.
            - For 'cone': {'radius': float, 'p0': P, 'p1': P}
                - radius: float - radius of the base of the cone at p0.
                - p0: P - 3-tuple of floats for the base center of the cone.
                - p1: P - 3-tuple of floats for the apex (tip) of the cone.
        color: Tuple[float, float, float] - RGB color in range [0, 1]^3.

    Returns:
        Shape - the primitive shape.

    Examples:
        - `primitive_call('cube', shape_kwargs={'scale': (1, 2, 1)})`
          Returns a cube with corners (-0.5, -1, -0.5) and (0.5, 1, 0.5).
        - `primitive_call('sphere', shape_kwargs={'radius': 0.5})`
          Returns a sphere with radius 0.5, with bounding box corners (-0.5, -0.5, -0.5) and (0.5, 0.5, 0.5).
        - `primitive_call('cylinder', shape_kwargs={'radius': 0.5, 'p0': (0, 0, 0), 'p1': (0, 1, 0)})`
          Returns a cylinder with bounding box corners (-0.5, 0, -0.5) and (0.5, 1, 0.5).
        - `primitive_call('cone', shape_kwargs={'radius': 0.5, 'p0': (0, 0, 0), 'p1': (0, 1, 0)})`
          Returns a cone with bounding box corners (-0.5, 0, -0.5) and (0.5, 1, 0.5).
    """
    shape = _primitive_call(name, color=color, **shape_kwargs)  # centered at origin
    return shape
