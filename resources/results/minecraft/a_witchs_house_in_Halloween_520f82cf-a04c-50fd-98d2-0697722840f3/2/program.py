"""
I'll help you create a spooky witch's house with Halloween elements. I'll break it down into components like the main house structure, a pointed roof, windows, and some decorative elements.

"""
from helper import *

@register()
def witch_house() -> Shape:
    return concat_shapes(
        library_call("house_base"),
        library_call("pointed_roof"),
        library_call("windows"),
        library_call("door"),
        library_call("decorations")
    )

@register()
def house_base() -> Shape:
    # Create the main structure using dark oak wood
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:dark_oak_planks",
        scale=(7, 6, 8),
        fill=False
    )

@register()
def pointed_roof() -> Shape:
    def create_roof_layer(i: int) -> Shape:
        width = 9 - i
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:purple_concrete",
                scale=(width, 1, 9),
                fill=True
            ),
            translation_matrix([-1 + i//2, 6 + i, -0.5])
        )
    return loop(5, create_roof_layer)

@register()
def windows() -> Shape:
    # Create glowing windows with orange stained glass
    window_positions = [
        (1, 3, 0),  # front left
        (5, 3, 0),  # front right
        (1, 3, 7),  # back left
        (5, 3, 7),  # back right
    ]
    return concat_shapes(*[
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:orange_stained_glass",
                scale=(1, 2, 1),
                fill=True
            ),
            translation_matrix([x, y, z])
        )
        for x, y, z in window_positions
    ])

@register()
def door() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:dark_oak_door",
            block_kwargs={"facing": "south", "half": "lower", "hinge": "left"},
            scale=(1, 2, 1),
            fill=True
        ),
        translation_matrix([3, 1, 0])
    )

@register()
def decorations() -> Shape:
    return concat_shapes(
        library_call("pumpkins"),
        library_call("lanterns"),
        library_call("cobwebs")
    )

@register()
def pumpkins() -> Shape:
    pumpkin_positions = [
        (0, 1, -1),
        (6, 1, -1),
        (-1, 1, 4),
        (7, 1, 4)
    ]
    return concat_shapes(*[
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:jack_o_lantern",
                block_kwargs={"facing": "south"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([x, y, z])
        )
        for x, y, z in pumpkin_positions
    ])

@register()
def lanterns() -> Shape:
    lantern_positions = [
        (-1, 4, -1),
        (7, 4, -1),
        (-1, 4, 8),
        (7, 4, 8)
    ]
    return concat_shapes(*[
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:soul_lantern",
                block_kwargs={"hanging": "true"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([x, y, z])
        )
        for x, y, z in lantern_positions
    ])

@register()
def cobwebs() -> Shape:
    cobweb_positions = [
        (-1, 5, -1),
        (7, 5, -1),
        (-1, 5, 8),
        (7, 5, 8)
    ]
    return concat_shapes(*[
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:cobweb",
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([x, y, z])
        )
        for x, y, z in cobweb_positions
    ])
"""

This code creates a spooky witch's house with:
1. A dark oak wood base structure
2. A pointed purple roof
3. Orange glowing windows
4. A dark oak door
5. Decorative elements including:
   - Jack o'lanterns around the house
   - Soul lanterns hanging from the corners
   - Cobwebs in the upper corners

The house has a Gothic appearance with:
- Dark materials (dark oak)
- Purple pointed roof
- Spooky lighting (orange windows and soul lanterns)
- Halloween decorations (pumpkins and cobwebs)

The house is modular and each component is separated into its own function for better organization and reusability. The main structure is 7x6x8 blocks with a 5-layer pointed roof on top. The decorative elements are placed strategically around the house to create a Halloween atmosphere.
"""