"""
Here's a program to create a simplified version of Stonehenge using the provided `helper.py`:

"""
from helper import *

"""
Stonehenge
"""

@register()
def stonehenge() -> Shape:
    return concat_shapes(
        library_call("outer_circle"),
        library_call("inner_horseshoe"),
        library_call("altar_stone"),
        library_call("ground")
    )

@register()
def outer_circle() -> Shape:
    def create_trilithon(i):
        angle = i * (360 / 30)  # 30 stones in the outer circle
        radius = 15
        x = int(radius * math.cos(math.radians(angle)))
        z = int(radius * math.sin(math.radians(angle)))
        return transform_shape(
            library_call("trilithon"),
            translation_matrix([x, 0, z])
        )
    return loop(30, create_trilithon)

@register()
def inner_horseshoe() -> Shape:
    def create_trilithon(i):
        angle = i * (360 / 5)  # 5 trilithons in the horseshoe
        radius = 8
        x = int(radius * math.cos(math.radians(angle)))
        z = int(radius * math.sin(math.radians(angle)))
        return transform_shape(
            library_call("large_trilithon"),
            translation_matrix([x, 0, z + 2])  # Shift slightly forward to create horseshoe shape
        )
    return loop(3, create_trilithon)  # Only create 3 out of 5 to form horseshoe

@register()
def trilithon() -> Shape:
    return concat_shapes(
        library_call("standing_stone"),
        transform_shape(library_call("standing_stone"), translation_matrix([3, 0, 0])),
        transform_shape(library_call("lintel"), translation_matrix([0, 4, 0]))
    )

@register()
def large_trilithon() -> Shape:
    return concat_shapes(
        library_call("large_standing_stone"),
        transform_shape(library_call("large_standing_stone"), translation_matrix([4, 0, 0])),
        transform_shape(library_call("large_lintel"), translation_matrix([0, 6, 0]))
    )

@register()
def standing_stone() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:stone",
        scale=(1, 4, 1),
        fill=True
    )

@register()
def large_standing_stone() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:stone",
        scale=(2, 6, 1),
        fill=True
    )

@register()
def lintel() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:stone",
        scale=(4, 1, 1),
        fill=True
    )

@register()
def large_lintel() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:stone",
        scale=(6, 1, 1),
        fill=True
    )

@register()
def altar_stone() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:stone",
            scale=(2, 1, 4),
            fill=True
        ),
        translation_matrix([0, 0, 10])
    )

@register()
def ground() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:grass_block",
            scale=(40, 1, 40),
            fill=True
        ),
        translation_matrix([-20, -1, -20])
    )
"""

This program creates a simplified version of Stonehenge with the following features:

1. An outer circle of 30 trilithons (pairs of vertical stones with a horizontal lintel on top).
2. An inner horseshoe of 3 larger trilithons.
3. An altar stone in the center.
4. A grass ground beneath the entire structure.

The structure is centered around the origin (0, 0, 0) and built on top of a grass block layer. The outer circle has a radius of 15 blocks, while the inner horseshoe has a radius of 8 blocks. The altar stone is placed 10 blocks forward from the center.

This simplified version captures the essence of Stonehenge's layout while keeping the structure at a manageable size for rendering in Minecraft.
"""