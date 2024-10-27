from typing import Literal, Union, Tuple, Optional
from type_utils import Shape, P
import numpy as np
from math_utils import translation_matrix
from _shape_utils import primitive_call as _primitive_call, transform_shape
from engine.constants import ENGINE_MODE
from minecraft_types_to_color import minecraft_block_colors
assert ENGINE_MODE in ['mi_from_minecraft'], ENGINE_MODE


__all__ = ["primitive_call"]


def primitive_call(name: Literal["set_cuboid", "spawn_entity"],
                   block_type: str,
                   scale: Tuple[int, int, int] = (1, 1, 1),
                   fill: bool = True,
                   block_kwargs: Optional[dict] = None,
                   prompt_kwargs_29fc3136: Optional[dict] = None) -> Shape:
    """
    THIS FUNCTION SHOULD NOT BE EXPOSED TO GPT.
    """
    # Derive color from block_type
    default_color = (1, 1, 1)
    if name == 'delete_blocks':
        print(f'[ERROR] CSG operation {name=} not supported')
    color = {
        'set_cuboid': minecraft_block_colors,
        # 'spawn_entity': minecraft_entity_colors,
    }[name].get(block_type, default_color)  # FIXME outdated

    shape = _primitive_call('cube', color=color, scale=scale)  # centered at origin

    # Default to 'min' set_mode
    scale = np.broadcast_to(scale, (3,))
    orig_min = -scale / 2
    set_to = (0, 0, 0)
    set_min = [set_to[i] if set_to[i] is not None else orig_min[i] for i in range(3)]
    shape = transform_shape(shape, translation_matrix(np.asarray(set_min) - orig_min))

    return shape
