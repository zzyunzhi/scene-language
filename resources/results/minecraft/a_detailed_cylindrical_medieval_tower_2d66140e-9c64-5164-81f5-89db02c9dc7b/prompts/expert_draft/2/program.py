"""
Here's a program to create a detailed cylindrical medieval tower using the provided `helper.py`:

"""
from helper import *
import math

"""
a detailed cylindrical medieval tower
"""

@register()
def medieval_tower(height: int = 20, radius: int = 5) -> Shape:
    return concat_shapes(
        library_call("tower_base", height=height, radius=radius),
        library_call("tower_roof", radius=radius),
        library_call("tower_windows", height=height, radius=radius),
        library_call("tower_door"),
        library_call("tower_decorations", height=height, radius=radius)
    )

@register()
def tower_base(height: int, radius: int) -> Shape:
    def create_cylinder_layer(y):
        return loop(
            360,
            lambda angle: transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:stone_bricks",
                    scale=(1, 1, 1),
                    fill=True
                ),
                translation_matrix([
                    round(radius * math.cos(math.radians(angle))),
                    y,
                    round(radius * math.sin(math.radians(angle)))
                ])
            )
        )

    return loop(height, create_cylinder_layer)

@register()
def tower_roof(radius: int) -> Shape:
    def create_roof_layer(y):
        current_radius = max(0, radius - y)
        return loop(
            360,
            lambda angle: transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:dark_oak_planks",
                    scale=(1, 1, 1),
                    fill=True
                ),
                translation_matrix([
                    round(current_radius * math.cos(math.radians(angle))),
                    y + 20,  # Start above the tower base
                    round(current_radius * math.sin(math.radians(angle)))
                ])
            )
        )

    return loop(radius + 1, create_roof_layer)

@register()
def tower_windows(height: int, radius: int) -> Shape:
    def create_window(angle, y):
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:glass_pane",
                scale=(1, 2, 1),
                fill=True
            ),
            translation_matrix([
                round((radius - 1) * math.cos(math.radians(angle))),
                y,
                round((radius - 1) * math.sin(math.radians(angle)))
            ])
        )

    return concat_shapes(
        loop(4, lambda i: create_window(i * 90, 5)),
        loop(4, lambda i: create_window(i * 90, 10)),
        loop(4, lambda i: create_window(i * 90, 15))
    )

@register()
def tower_door() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:dark_oak_door",
            block_kwargs={"facing": "south", "half": "lower", "hinge": "left"},
            scale=(1, 2, 1),
            fill=True
        ),
        translation_matrix([0, 0, 5])
    )

@register()
def tower_decorations(height: int, radius: int) -> Shape:
    def create_flag(y):
        return concat_shapes(
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:oak_fence",
                    scale=(1, 3, 1),
                    fill=True
                ),
                translation_matrix([radius - 1, y, 0])
            ),
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:red_wool",
                    scale=(2, 1, 1),
                    fill=True
                ),
                translation_matrix([radius, y + 2, 0])
            )
        )

    return concat_shapes(
        create_flag(height + 1),
        loop(
            4,
            lambda i: transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:lantern",
                    block_kwargs={"hanging": "true"},
                    scale=(1, 1, 1),
                    fill=True
                ),
                translation_matrix([
                    round((radius - 1) * math.cos(math.radians(i * 90 + 45))),
                    3,
                    round((radius - 1) * math.sin(math.radians(i * 90 + 45)))
                ])
            )
        )
    )
"""

This program creates a detailed cylindrical medieval tower with the following components:

1. `medieval_tower`: The main function that combines all the tower elements.
2. `tower_base`: Creates the cylindrical stone base of the tower.
3. `tower_roof`: Adds a conical roof made of dark oak planks.
4. `tower_windows`: Places glass pane windows at regular intervals around the tower.
5. `tower_door`: Adds a dark oak door at the base of the tower.
6. `tower_decorations`: Adds decorative elements like a flag at the top and lanterns around the base.

The tower has a default height of 20 blocks and a radius of 5 blocks, but these can be adjusted when calling the `medieval_tower` function. The design includes multiple layers of windows, a conical roof, and decorative elements to give it a medieval appearance.
"""