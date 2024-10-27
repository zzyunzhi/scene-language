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


def primitive_call(name: Literal['set_cuboid', 'delete_blocks'], **kwargs) -> Shape:
    """
    Args:
        name: str - the name of the primitive action
            support 'set_cuboid', 'delete_blocks'
        ...: Any - additional arguments for the primitive action
            For 'set_cuboid': 
                - block_type: a string that denotes the block type, e.g. 'oak_log'. THESE MUST BE VALID LITEMATIC BLOCK TYPES.
                - block_kwargs: a dict[str, str] of additional properties to define a block's state fully, e.g. for 'oak_log', we need to define the axis with possible values 'x', 'y', or 'z'
                - scale: a list of 3 elements, denoting the scaling along the positive x, y, and z axises respectively.  IMPORTANT: THESE CAN ONLY BE INTEGERS!
                - fill: a boolean, describing whether the cuboid should be filled, or be hollow. Hint: this can be useful for creating structures that should be hollow, such as a building.
            For 'delete_blocks': 
                - scale: a list of 3 elements, denoting the scaling along the positive x, y, and z axises respectively.  IMPORTANT: THESE CAN ONLY BE INTEGERS!
    Returns:
        Shape - 
            For 'set_cuboid': a cuboid composed of Minecraft blocks, with the closest block at (0, 0, 0) and furthest (right, back-most) block at (scale[0], scale[1], scale[2]).
            For 'delete_blocks': an empty cuboid-shaped space without any blocks, starting from the closest block at (0, 0, 0) and furthest (right, back-most) block at (scale[0], scale[1], scale[2]).
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

def translation_matrix(offset: P) -> T:
    """
    Args:
        offset (P) : the translation vector, which must be composed of integers
    """
def identity_matrix() -> T:
    """
    Returns the identity matrix in SE(3).
    """

```

STRICTLY follow these rules:
1. Only use the functions and imported libraries in `helper.py`.
2. You can only write functions. Follow a modular approach and use the `register` decorator to define semantic shapes or shape groups. Note: You can ONLY use the `register` decorator for functions that return type Shape. Any helper functions that you attempt to register will cause an error.
3. Camera coordinate system: +x is right, +y is up, +z is forward. 
4. Make sure to only pass in values into block_type that are supported by standard Minecraft engines. Pass in block_kwargs for blocks that need additional properties to define a block's state fully, such as stair blocks.
5. Pay attention that the objects are not too large so it can't be rendered.

You should be precise and creative.
