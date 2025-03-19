from type_utils import Shape
from typing import Any, Literal

__all__ = ['primitive_call']

def primitive_call(name: Literal['cube', 'sphere', 'cylinder'], shape_kwargs: dict[str, Any], color: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Shape:
    """
    Constructs a primitive shape.

    Args:
        name: str - 'cube', 'sphere', or 'cylinder'.
        shape_kwargs: dict[str, Any] - keyword arguments for the primitive shape.
            - For 'cube': {'scale': P} - 3-tuple of floats for scaling along x, y, z axes.
            - For 'sphere': {'radius': float} - radius of the sphere.
            - For 'cylinder': {'radius': float, 'p0': P, 'p1': P}
                - radius: float - radius of the cylinder.
                - p0: P - 3-tuple of floats for the start point of the cylinder's centerline.
                - p1: P - 3-tuple of floats for the end point of the cylinder's centerline.
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
    """
