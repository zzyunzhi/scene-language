

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
        library_call("tower_entrance", radius=radius),
        library_call("tower_battlements", radius=radius),
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
                    fill=True,
                ),
                translation_matrix([
                    int(radius * math.cos(math.radians(angle))),
                    y,
                    int(radius * math.sin(math.radians(angle)))
                ])
            )
        )

    return loop(height, create_cylinder_layer)

@register()
def tower_roof(radius: int) -> Shape:
    def create_roof_layer(y):
        current_radius = max(1, radius - y)
        return loop(
            360,
            lambda angle: transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:spruce_planks",
                    scale=(1, 1, 1),
                    fill=True,
                ),
                translation_matrix([
                    int(current_radius * math.cos(math.radians(angle))),
                    y + 20,  # Start above the tower base
                    int(current_radius * math.sin(math.radians(angle)))
                ])
            )
        )

    return loop(radius, create_roof_layer)

@register()
def tower_windows(height: int, radius: int) -> Shape:
    def create_window(y):
        return concat_shapes(
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:glass_pane",
                    scale=(1, 2, 1),
                    fill=True,
                ),
                translation_matrix([radius, y, 0])
            ),
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:glass_pane",
                    scale=(1, 2, 1),
                    fill=True,
                ),
                translation_matrix([0, y, radius])
            ),
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:glass_pane",
                    scale=(1, 2, 1),
                    fill=True,
                ),
                translation_matrix([-radius, y, 0])
            ),
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:glass_pane",
                    scale=(1, 2, 1),
                    fill=True,
                ),
                translation_matrix([0, y, -radius])
            )
        )

    return loop(height // 4, lambda i: create_window(i * 4 + 2))

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
def tower_battlements(radius: int) -> Shape:
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
                20,  # Place at the top of the tower base
                int(radius * math.sin(math.radians(angle)))
            ])
        )

    return loop(8, lambda i: create_battlement(i * 45))

"""

This program creates a detailed cylindrical medieval tower with the following components:

1. A cylindrical stone brick base
2. A conical roof made of spruce planks
3. Windows placed at regular intervals around the tower
4. An entrance with an oak door
5. Battlements at the top of the tower

The tower is customizable with parameters for height and radius. The main `medieval_tower` function combines all these elements to create the final structure.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
