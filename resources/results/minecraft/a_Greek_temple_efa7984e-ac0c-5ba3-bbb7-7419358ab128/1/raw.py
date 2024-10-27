"""
Here's a program to create a Greek temple using the provided `helper.py`:

"""
from helper import *

"""
a Greek temple
"""

@register()
def greek_temple() -> Shape:
    return concat_shapes(
        library_call("temple_base"),
        library_call("temple_columns"),
        library_call("temple_roof"),
        library_call("temple_steps")
    )

@register()
def temple_base() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:smooth_stone",
        scale=(20, 2, 30),
        fill=True
    )

@register()
def temple_columns() -> Shape:
    def create_column(i):
        x_offset = 2 + i * 4
        return concat_shapes(
            transform_shape(
                library_call("column"),
                translation_matrix([x_offset, 2, 2])
            ),
            transform_shape(
                library_call("column"),
                translation_matrix([x_offset, 2, 27])
            )
        )
    return loop(5, create_column)

@register()
def column() -> Shape:
    return concat_shapes(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:quartz_pillar",
            block_kwargs={"axis": "y"},
            scale=(2, 10, 2),
            fill=True
        ),
        transform_shape(
            library_call("column_capital"),
            translation_matrix([0, 10, 0])
        )
    )

@register()
def column_capital() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:chiseled_quartz_block",
        scale=(2, 1, 2),
        fill=True
    )

@register()
def temple_roof() -> Shape:
    return concat_shapes(
        library_call("roof_base"),
        library_call("roof_pediment")
    )

@register()
def roof_base() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:smooth_quartz",
            scale=(22, 2, 32),
            fill=True
        ),
        translation_matrix([-1, 12, -1])
    )

@register()
def roof_pediment() -> Shape:
    def pediment_layer(i):
        width = 22 - i * 2
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:smooth_quartz",
                scale=(width, 1, 32),
                fill=True
            ),
            translation_matrix([i, 14 + i, -1])
        )
    return loop(5, pediment_layer)

@register()
def temple_steps() -> Shape:
    def step_layer(i):
        width = 24 + i * 2
        depth = 34 + i * 2
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:stone_brick_stairs",
                block_kwargs={"facing": "north", "half": "bottom"},
                scale=(width, 1, depth),
                fill=False
            ),
            translation_matrix([-2 - i, 1 - i, -2 - i])
        )
    return loop(3, step_layer)
"""

This program creates a Greek temple with the following components:

1. A stone base
2. Ten columns (five on each side)
3. A roof with a pediment
4. Steps leading up to the temple

The temple is constructed using various Minecraft blocks to approximate the look of a Greek temple. The main structure uses quartz blocks for a white, marble-like appearance, while the base and steps use stone variants for contrast.

The temple is modular, with separate functions for each major component. The `greek_temple()` function combines all these components to create the final structure.
"""