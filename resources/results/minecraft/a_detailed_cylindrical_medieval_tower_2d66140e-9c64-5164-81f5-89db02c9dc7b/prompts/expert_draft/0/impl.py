

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
        library_call("tower_door"),
        library_call("tower_battlements", radius=radius)
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
                    block_type="minecraft:spruce_planks",
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

    return loop(radius, create_roof_layer)

@register()
def tower_windows(height: int, radius: int) -> Shape:
    def create_window(angle):
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:glass_pane",
                scale=(1, 2, 1),
                fill=True
            ),
            translation_matrix([
                round((radius - 1) * math.cos(math.radians(angle))),
                height // 2,
                round((radius - 1) * math.sin(math.radians(angle)))
            ])
        )

    return loop(4, lambda i: create_window(i * 90))

@register()
def tower_door() -> Shape:
    return concat_shapes(
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:oak_door",
                block_kwargs={"half": "lower", "hinge": "left", "facing": "south"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([0, 0, 5])
        ),
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:oak_door",
                block_kwargs={"half": "upper", "hinge": "left", "facing": "south"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([0, 1, 5])
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
                fill=True
            ),
            translation_matrix([
                round(radius * math.cos(math.radians(angle))),
                20,  # Place on top of the tower base
                round(radius * math.sin(math.radians(angle)))
            ])
        )

    return loop(16, lambda i: create_battlement(i * 22.5))

"""

This program creates a detailed cylindrical medieval tower with the following components:

1. A cylindrical stone brick base
2. A conical roof made of spruce planks
3. Windows placed at regular intervals
4. A wooden door at the base
5. Battlements (crenellations) at the top of the tower

The tower is customizable with parameters for height and radius. The main `medieval_tower` function combines all these elements to create the final structure. Each component is implemented as a separate function for modularity and easier customization.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)