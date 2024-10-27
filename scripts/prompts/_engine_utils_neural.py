from typing import Literal
from type_utils import Shape, P
from _shape_utils import primitive_call as _primitive_call
from engine.constants import ENGINE_MODE
assert ENGINE_MODE == 'neural', ENGINE_MODE


__all__ = ["primitive_call"]


def primitive_call(
        name: Literal['run'], prompt: str, scale: float | P,
        allow_rotate_y: bool = True,
        allow_rotate_x: bool = False,
        allow_rotate_z: bool = False,
) -> Shape:
    """
    Constructs a primitive shape.

    Args:
        name: str - only supports 'run'
        prompt: str - the text prompt to a pre-trained text-to-3D model Shap-E from OpenAI
            The model can only generate simple objects but not parts.
        scale: float | P - the bounding box dimensions of the output shape
        # TODO update

    Returns:
        Shape - a shape centered at the origin
    """

    return _primitive_call(name, prompt=prompt, scale=scale,
                           allow_rotate_y=allow_rotate_y,
                           allow_rotate_x=allow_rotate_x,
                           allow_rotate_z=allow_rotate_z)
