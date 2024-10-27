from typing import Literal
from type_utils import Shape
from _shape_utils import primitive_call as _primitive_call
from engine.constants import ENGINE_MODE
assert ENGINE_MODE == 'minecraft', ENGINE_MODE


__all__ = ["primitive_call"]


def primitive_call(name: Literal["set_cuboid", "delete_blocks"], **kwargs) -> Shape:
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

    return _primitive_call(name, **kwargs)

