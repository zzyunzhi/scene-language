from typing import Literal
from type_utils import Shape, P
from _shape_utils import primitive_call as _primitive_call
from engine.constants import ENGINE_MODE
assert ENGINE_MODE == 'lmd', ENGINE_MODE


__all__ = ["primitive_call"]


def primitive_call(name: Literal['box'], prompt: str, scale: float | P) -> Shape:
    """
    Constructs a primitive shape.

    Args:
        name: str - only supports 'box'
        prompt: str - the text prompt to control the corresponding shape bounding box
            for a layout-conditioned text-to-image model.
        scale: float | P - the bounding box dimensions of the output shape

    Returns:
        Shape - a shape centered at the origin
    """

    return _primitive_call(name, prompt=prompt, scale=scale)
