

from pathlib import Path
import numpy as np

from helper import *
import mitsuba as mi
import traceback
import ipdb

import random
import math
import sys
import os

import mi_helper  # such that primitive call will be implemented


def main():
    from example_postprocess import parse_program
    from engine.utils.graph_utils import strongly_connected_components, get_root
    # from tu.train_setup import set_seed
    from engine.utils.train_utils import set_seed
    from dsl_utils import library, animation_func
    from minecraft_helper import execute, execute_animation

    set_seed(0)

    save_dir = Path(__file__).parent / 'renderings'
    save_dir.mkdir(exist_ok=True)

    if animation_func:
        frames = list(animation_func())
        name = animation_func.__name__
        execute_animation(frames, save_dir=(save_dir / name).as_posix(), description=name)
    else:
        exp_program_path = Path(__file__).parent / 'program.py'
        _, library_equiv = parse_program(exp_program_path.as_posix())
        scc = strongly_connected_components(library_equiv)
        print(f'{scc=}')

        try:
            root = get_root(library_equiv)
            print(f'{root=}')
        except Exception as e:
            # sometimes a function is implemented but never used, so there is no shared ancestor
            root = None
            print('[ERROR] cannot find root')
            for name, node in library_equiv.items():
                if len(node.parents) == 0 and len(node.children) > 0:
                    root = name
            if root is None:  # not sure, just pick anything?
                root = next(reversed(library.keys()))
            print(e)

        node = library_equiv[root]
        execute(node(), save_dir=(save_dir / node.name).as_posix(), description=node.name)

        save_dir = Path(__file__).parent / 'extra_renderings'
        for node in library_equiv.values():
            try:
                execute(node(), save_dir=(save_dir / node.name).as_posix(), description=node.name)
            except:
                import traceback; traceback.print_exc()
                pass


"""
Here's an improved version of the program for creating a detailed cylindrical medieval tower, addressing the points brought up in the review:

"""
from helper import *
import math

@register("A detailed cylindrical medieval tower")
def medieval_tower(height: int = 20, radius: int = 5) -> Shape:
    return concat_shapes(
        library_call("tower_base", height=height, radius=radius),
        library_call("tower_roof", height=height, radius=radius),
        library_call("tower_windows", height=height, radius=radius),
        library_call("tower_door", radius=radius),
        library_call("tower_battlements", height=height, radius=radius),
        library_call("tower_decorations", height=height, radius=radius)
    )

@register("Cylindrical stone brick base of the tower")
def tower_base(height: int, radius: int) -> Shape:
    def create_cylinder_layer(y: int) -> Shape:
        return concat_shapes(*[
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:stone_bricks" if (x + y + z) % 3 != 0 else "minecraft:mossy_stone_bricks",
                    scale=(1, 1, 1),
                    fill=True
                ),
                translation_matrix([x, y, z])
            )
            for x in range(-radius, radius + 1)
            for z in range(-radius, radius + 1)
            if x*x + z*z <= radius*radius
        ])

    return concat_shapes(*[create_cylinder_layer(y) for y in range(height)])

@register("Conical roof made of spruce planks")
def tower_roof(height: int, radius: int) -> Shape:
    roof_height = radius + 2
    def create_roof_layer(y: int) -> Shape:
        current_radius = max(1, int(radius * (1 - y / roof_height)))
        return concat_shapes(*[
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:spruce_planks",
                    scale=(1, 1, 1),
                    fill=True
                ),
                translation_matrix([x, y + height, z])
            )
            for x in range(-current_radius, current_radius + 1)
            for z in range(-current_radius, current_radius + 1)
            if x*x + z*z <= current_radius*current_radius
        ])

    return concat_shapes(*[create_roof_layer(y) for y in range(roof_height)])

@register("Windows placed at regular intervals")
def tower_windows(height: int, radius: int) -> Shape:
    def create_window(angle: float, y: int) -> Shape:
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:glass_pane",
                scale=(1, 2, 1),
                fill=True
            ),
            translation_matrix([
                int((radius - 1) * math.cos(math.radians(angle))),
                y,
                int((radius - 1) * math.sin(math.radians(angle)))
            ])
        )

    return concat_shapes(*[
        create_window(i * 90, y)
        for i in range(4)
        for y in range(3, height - 3, 5)
    ])

@register("Wooden door at the base of the tower")
def tower_door(radius: int) -> Shape:
    return concat_shapes(
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:oak_door",
                block_kwargs={"half": "lower", "hinge": "left", "facing": "south"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([0, 0, radius])
        ),
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:oak_door",
                block_kwargs={"half": "upper", "hinge": "left", "facing": "south"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([0, 1, radius])
        )
    )

@register("Battlements (crenellations) at the top of the tower")
def tower_battlements(height: int, radius: int) -> Shape:
    def create_battlement(angle: float) -> Shape:
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:stone_brick_wall",
                scale=(1, 2, 1),
                fill=True
            ),
            translation_matrix([
                int(radius * math.cos(math.radians(angle))),
                height,
                int(radius * math.sin(math.radians(angle)))
            ])
        )

    return concat_shapes(*[create_battlement(i * 22.5) for i in range(16)])

@register("Decorative elements for the tower")
def tower_decorations(height: int, radius: int) -> Shape:
    flag = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:red_banner",
            block_kwargs={"rotation": "1"},
            scale=(1, 1, 1),
            fill=True
        ),
        translation_matrix([0, height + radius + 2, 0])
    )

    torches = concat_shapes(*[
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:wall_torch",
                block_kwargs={"facing": "south"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([
                int(radius * math.cos(math.radians(angle))),
                height // 2,
                int(radius * math.sin(math.radians(angle)))
            ])
        )
        for angle in range(0, 360, 90)
    ])

    return concat_shapes(flag, torches)
"""

This improved version addresses the points mentioned in the review:

1. The `tower_base` function now uses a mix of regular and mossy stone bricks for a more realistic look. It also uses a more precise method to create the cylinder, avoiding gaps.

2. The `tower_roof` function has been adjusted to create a less steep roof by using a `roof_height` variable.

3. The `tower_windows` function now places windows at regular intervals vertically along the tower, providing more light and detail.

4. The `tower_battlements` function remains largely the same, as it was already well-implemented.

5. A new `tower_decorations` function has been added to include decorative elements like a flag at the top of the tower and torches along the walls.

6. The main `medieval_tower` function now includes the call to `tower_decorations`.

These changes result in a more detailed and realistic medieval tower while maintaining the cylindrical shape and key features. The tower now has varied textures, more windows, a better-proportioned roof, and additional decorative elements, making it more visually interesting and true to medieval architecture.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
