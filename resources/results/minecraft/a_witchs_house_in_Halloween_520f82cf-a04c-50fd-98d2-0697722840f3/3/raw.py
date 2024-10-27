"""
I'll help you create a spooky witch's house with Halloween elements. I'll break it down into modular components:

"""
from helper import *

@register()
def witch_house() -> Shape:
    return concat_shapes(
        library_call("house_base"),
        library_call("pointed_roof"),
        library_call("chimney"),
        library_call("windows"),
        library_call("door"),
        library_call("decorations")
    )

@register()
def house_base() -> Shape:
    # Create main structure with dark oak wood
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:dark_oak_planks",
        scale=(7, 5, 8),
        fill=False
    )

@register()
def pointed_roof() -> Shape:
    def roof_layer(i: int) -> Shape:
        width = 9 - (i * 2)
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:purple_concrete",
                scale=(width, 1, 9),
                fill=True
            ),
            translation_matrix([-1 + i, 5 + i, -1])
        )
    return loop(5, roof_layer)

@register()
def chimney() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:cobblestone",
            scale=(1, 4, 1),
            fill=True
        ),
        translation_matrix([5, 5, 2])
    )

@register()
def windows() -> Shape:
    return concat_shapes(
        # Front windows
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:orange_stained_glass",
                scale=(1, 2, 1),
                fill=True
            ),
            translation_matrix([2, 2, 0])
        ),
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:orange_stained_glass",
                scale=(1, 2, 1),
                fill=True
            ),
            translation_matrix([4, 2, 0])
        )
    )

@register()
def door() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:dark_oak_door",
            block_kwargs={"facing": "south", "half": "lower"},
            scale=(1, 2, 1),
            fill=True
        ),
        translation_matrix([3, 1, 0])
    )

@register()
def decorations() -> Shape:
    return concat_shapes(
        # Pumpkins around the house
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:jack_o_lantern",
                block_kwargs={"facing": "south"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([1, 1, -1])
        ),
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:jack_o_lantern",
                block_kwargs={"facing": "south"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([5, 1, -1])
        ),
        # Spider web in corner
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:cobweb",
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([0, 4, 0])
        ),
        # Cauldron
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:cauldron",
                block_kwargs={"level": "3"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([5, 1, 6])
        )
    )
"""

This code creates a spooky witch's house with:
1. A dark oak wood base structure
2. A pointed purple roof
3. A cobblestone chimney
4. Orange-tinted windows that give an eerie glow
5. A dark oak door
6. Halloween decorations including:
   - Jack o'lanterns at the front
   - Spider webs in the corner
   - A cauldron inside

The house is modular and each component is separated into its own function. The main structure is 7x5x8 blocks, with a pointed roof that gets narrower as it goes up. The decorations add to the Halloween theme with spooky elements typical of a witch's house.
"""