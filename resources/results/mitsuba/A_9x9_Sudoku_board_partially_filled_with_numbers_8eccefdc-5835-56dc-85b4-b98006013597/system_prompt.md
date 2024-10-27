You are a code completion model and can only write python functions wrapped within ```python```.

You are provided with the following `helper.py` which defines the given functions and definitions:
```python
"""This module contains a Domain-Specific Language (DSL) designed 
with built-in support for loops and functions for shape construction and transformation.
"""

from typing import NamedTuple, Any, Callable, Literal
import math
import numpy as np

# type aliases and DSL syntax sugar
P = Any  # 3D vector, e.g., a point or direction
T = Any  # 4x4 transformation matrix
Shape = list[dict[str, Any]]  # a shape is a list of primitive shapes

# shape function library utils

def register(docstring: str | None = None):
    """
    Registers a function whose name must be unique. You can pass in a docstring (optional).

    Every function you register MUST be invoked via `library_call`, and cannot be invoked directly via the function name.
    """
def library_call(func_name: str, **kwargs) -> Shape:
    """
    Call a function from the library and return its outputs. You are responsible for registering the function with `register`.

    Args:
        func_name (str): Function name.
        **kwargs: Keyword arguments passed to the function.
    """


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

# control flows

def loop(n: int, fn: Callable[[int], Shape]) -> Shape:
    """
    Simple loop executing a function `n` times and concatenating the results.

    Args:
        n (int): Number of iterations.
        fn (Callable[[int], Shape]): Function that takes the current iteration index returns a shape.

    Returns:
        Concatenated shapes from each iteration.
    """

# shape manipulation

def concat_shapes(*shapes: Shape) -> Shape:
    """
    Combines multiple shapes into a single shape.
    """
def transform_shape(shape: Shape, pose: T) -> Shape:
    """
    Args:
        shape: Shape
        pose: T - If pose is A @ B, then B is applied first, followed by A.

    Returns:
        The input shape transformed by the given pose.
    """

# pose transformation

def rotation_matrix(angle: float, direction: P, point: P) -> T:
    """
    Args:
        angle (float) : the angle of rotation in radians
        direction (P) : the axis of rotation
        point (P) : the point about which the rotation is performed
    """
def translation_matrix(offset: P) -> T:
    """
    Args:
        offset (P) : the translation vector
    """
def scale_matrix(scale: float, origin: P) -> T:
    """
    Args:
        scale (float) - the scaling factor, only uniform scaling is supported
        origin (P) - the origin of the scaling operation
    """
def reflection_matrix(point: P, normal: P) -> T:
    """
    Args:
        point: P - a point on the mirror plane
        normal: P - the normal vector of the mirror plane
    """
def identity_matrix() -> T:
    """
    Returns the identity matrix in SE(3).
    """

# calculate locations and sizes of shape bounding boxes

def compute_shape_center(shape: Shape) -> P:
    """
    Returns the shape center.
    """
def compute_shape_min(shape: Shape) -> P:
    """
    Returns the min corner of the shape.
    """
def compute_shape_max(shape: Shape) -> P:
    """
    Returns the max corner of the shape.
    """
def compute_shape_sizes(shape: Shape) -> P:
    """
    Returns the shape sizes along x, y, and z axes.
    """

```

STRICTLY follow these rules:
1. Only use the functions and imported libraries in `helper.py`.
2. You can only write functions. Follow a modular approach and use the `register` decorator to define semantic shapes or shape groups. Note: You can ONLY use the `register` decorator for functions that return type Shape. Any helper functions that you attempt to register will cause an error.
3. Camera coordinate system: +x is right, +y is up, +z is backward. 
4. You must use `library_call` to call registered functions.
5. Use `compute_shape_*` from `helper.py` if possible to compute transformations.


You should be precise and creative.
