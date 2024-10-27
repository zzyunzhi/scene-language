from typing import Literal, Union, Tuple, Any, Optional
from type_utils import Shape, P
import numpy as np
from math_utils import translation_matrix
from _shape_utils import primitive_call as _primitive_call, transform_shape
from _engine_utils_exposed import primitive_call as _primitive_call_exposed
from engine.constants import ENGINE_MODE
from minecraft_types_to_color import minecraft_block_colors
assert ENGINE_MODE in ['mi', 'mi_material', 'mi_from_minecraft', 'minecraft'], ENGINE_MODE


__all__ = ["primitive_call"]


def primitive_call(name: Literal["cube", "sphere"],
                   scale: Union[float, P] = 1,
                   color: Tuple[float, float, float] = (1., 1., 1.),
                   set_mode: Literal["center", "min", "max"] = 'center',
                   set_to: P = (0, 0, 0),
                   shape_kwargs: Optional[dict[str, Any]] = None,
                   bsdf_kwargs: Optional[dict[str, Any]] = None,
                   prompt_kwargs_29fc3136: Optional[dict[str, Any]] = None,
                   ) -> Shape:
    # THE DOCSTRING WILL BE WRONG!!!!! Should not be exposed to GPT.
    """
    Constructs a primitive shape.
    NOTE: By default, the shape is centered at the origin and its bounding box has minimum corner `(-scale[0] / 2, -scale[1] / 2, -scale[2] / 2)`.

    Examples:
        - `primitive_call('cube', color=(1, 1, 1), scale=(1, 10, 1))`
          Returns a white, tall cube with corners (-0.5, -5, -0.5) and (0.5, 5, 0.5).
        - `primitive_call('sphere', color=(1, 1, 1), scale=1)`
          Returns a white sphere with radius 0.5 and centered at the origin.

    Args:
        name: str - 'cube' or 'sphere'.
        scale: float | P - float or 3-tuple of floats for `cube`, float for `sphere` (only uniform scaling supported!).
        color: Tuple[float, float, float] - RGB color in range [0, 1]^3.
        set_mode: str - 'center', 'min', or 'max'. Default is 'center'.
        set_to: P - A 3-tuple of floats, setting the `set_mode` (center / min / max) of bounding box to this point. Default is `(0, 0, 0)`.
    """
    if shape_kwargs is not None:  # to account for new API
        assert set_mode == 'center', f"[ERROR] {set_mode=} must be 'center' when using {shape_kwargs=}"
        assert set_to == (0, 0, 0), f"[ERROR] {set_to=} must be (0, 0, 0) when using {shape_kwargs=}"
        assert scale == 1, f"[ERROR] {scale=} must be 1 when using {shape_kwargs=}"

        if bsdf_kwargs is not None:  # to account for new API
            assert color == (1, 1, 1), (color, bsdf_kwargs, shape_kwargs)
            return _primitive_call_exposed(name=name, shape_kwargs=shape_kwargs, color=bsdf_kwargs['base_color'])
        return _primitive_call_exposed(name=name, shape_kwargs=shape_kwargs, color=color)

    if name == 'cylinder':
        name = 'cube'

    shape = _primitive_call(name, color=color, scale=scale)  # centered at origin
    scale = np.broadcast_to(scale, (3,))
    orig_min = -scale / 2
    if set_mode == 'center':
        set_min = [-scale[i] / 2 + set_to[i] if set_to[i] is not None else orig_min[i] for i in range(3)]
    elif set_mode == 'min':
        set_min = [set_to[i] if set_to[i] is not None else orig_min[i] for i in range(3)]
    elif set_mode == 'max':
        set_min = [-scale[i] + set_to[i] if set_to[i] is not None else orig_min[i] for i in range(3)]
    else:
        print(f'[ERROR] {set_mode=} must be "min", "max", or "center"')
        return shape
    shape = transform_shape(shape, translation_matrix(np.asarray(set_min) - orig_min))
    return shape


# def primitive_call_reduced(name: Literal['cube', 'sphere'],
#                    scale: Union[float, P] = 1,
#                    color: Tuple[float, float, float] = (1., 1., 1.)) -> Shape:
#     """
#     Constructs a primitive shape centered at the origin, with bounding box minimum corner at (0, 0, 0),
#     and max corner at (scale[0], scale[1], scale[2]) if scale is a 3-tuple or (scale, scale, scale) if scale is a float.
#
#     Args:
#         name: str - shape type, 'cube' or 'sphere'
#         scale: float | P - float or 3-tuple of floats for `cube`, float for `sphere` (only uniform scaling supported!).
#         color: Tuple[float, float, float] - RGB color in range [0, 1]^3.
#     """
#     return primitive_call(name=name, scale=scale, color=color, set_mode='min', set_to=(0, 0, 0))


def I0BpHzM2Xn_primitive_call_from_minecraft(name: Literal["set_cuboid", "spawn_entity"],
                   block_type: str, 
                   scale: Tuple[int, int, int] = (1, 1, 1),
                   fill: bool = True) -> Shape:
    # Derive color from block_type
    default_color = (1, 1, 1)
    color = minecraft_block_colors.get(block_type, default_color) # if name == 'set_cuboid' else minecraft_entity_colors.get(block_type, default_color)
    shape = _primitive_call('cube', color=color, scale=scale)  # centered at origin

    # Default to 'min' set_mode
    scale = np.broadcast_to(scale, (3,))
    orig_min = -scale / 2
    set_to = (0, 0, 0)
    set_min = [set_to[i] if set_to[i] is not None else orig_min[i] for i in range(3)]
    shape = transform_shape(shape, translation_matrix(np.asarray(set_min) - orig_min))

    return shape
