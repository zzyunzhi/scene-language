

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
        library_call("tower_roof", radius=radius),
        library_call("tower_windows", height=height, radius=radius),
        library_call("tower_entrance", radius=radius),
        library_call("tower_battlements", height=height, radius=radius),
        library_call("tower_arrow_slits", height=height, radius=radius),
        library_call("tower_flag", height=height, radius=radius),
        library_call("tower_spiral_staircase", height=height, radius=radius)
    )

@register()
def tower_base(height: int, radius: int) -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:stone_bricks",
        scale=(radius * 2 + 1, height, radius * 2 + 1),
        fill=False
    )

@register()
def tower_roof(radius: int) -> Shape:
    def create_roof_layer(y):
        current_radius = max(1, radius - y)
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:spruce_planks",
                scale=(current_radius * 2 + 1, 1, current_radius * 2 + 1),
                fill=False
            ),
            translation_matrix([0, y, 0])
        )

    return concat_shapes(*[create_roof_layer(y) for y in range(radius)])

@register()
def tower_windows(height: int, radius: int) -> Shape:
    def create_window(y):
        window_positions = [
            (radius, 0), (0, radius), (-radius, 0), (0, -radius)
        ]
        return concat_shapes(*[
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:glass_pane",
                    scale=(1, 2, 1),
                    fill=True,
                ),
                translation_matrix([x, y, z])
            ) for x, z in window_positions
        ])

    return concat_shapes(*[create_window(i * 4 + 2) for i in range(height // 4)])

@register()
def tower_entrance(radius: int) -> Shape:
    return concat_shapes(
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:oak_door",
                block_kwargs={"facing": "south", "half": "lower", "hinge": "left"},
                scale=(1, 1, 1),
                fill=True,
            ),
            translation_matrix([0, 0, radius])
        ),
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:oak_door",
                block_kwargs={"facing": "south", "half": "upper", "hinge": "left"},
                scale=(1, 1, 1),
                fill=True,
            ),
            translation_matrix([0, 1, radius])
        )
    )

@register()
def tower_battlements(height: int, radius: int) -> Shape:
    def create_battlement(angle):
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:stone_brick_wall",
                scale=(1, 2, 1),
                fill=True,
            ),
            translation_matrix([
                int(radius * math.cos(math.radians(angle))),
                height,
                int(radius * math.sin(math.radians(angle)))
            ])
        )

    return concat_shapes(*[create_battlement(i * 45) for i in range(8)])

@register()
def tower_arrow_slits(height: int, radius: int) -> Shape:
    def create_arrow_slit(y, angle):
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:air",
                scale=(1, 2, 1),
                fill=True,
            ),
            translation_matrix([
                int(radius * math.cos(math.radians(angle))),
                y,
                int(radius * math.sin(math.radians(angle)))
            ])
        )

    return concat_shapes(*[
        create_arrow_slit(y, angle)
        for y in range(2, height - 2, 3)
        for angle in [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
    ])

@register()
def tower_flag(height: int, radius: int) -> Shape:
    flagpole = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:oak_fence",
            scale=(1, 5, 1),
            fill=True,
        ),
        translation_matrix([radius - 1, height + radius, radius - 1])
    )

    flag = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:red_wool",
            scale=(3, 2, 1),
            fill=True,
        ),
        translation_matrix([radius - 3, height + radius + 2, radius - 1])
    )

    return concat_shapes(flagpole, flag)

@register()
def tower_spiral_staircase(height: int, radius: int) -> Shape:
    def create_stair(y, angle):
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:oak_stairs",
                block_kwargs={"facing": "east", "half": "bottom"},
                scale=(1, 1, 1),
                fill=True,
            ),
            translation_matrix([
                int((radius - 1) * math.cos(math.radians(angle))),
                y,
                int((radius - 1) * math.sin(math.radians(angle)))
            ])
        )

    return concat_shapes(*[
        create_stair(y, y * 20)
        for y in range(height)
    ])
"""

This improved version addresses the issues mentioned in the review and adds more details to the tower:

1. The code now uses `concat_shapes` correctly to combine shapes.
2. The `transform_shape` function is used appropriately.
3. The `primitive_call` function parameters have been corrected.
4. Additional features have been added:
   - Arrow slits for defense
   - A flag at the top of the tower
   - A spiral staircase inside the tower

The code is now more efficient and should execute properly within the given DSL constraints. It creates a more detailed and realistic cylindrical medieval tower with various architectural elements.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
