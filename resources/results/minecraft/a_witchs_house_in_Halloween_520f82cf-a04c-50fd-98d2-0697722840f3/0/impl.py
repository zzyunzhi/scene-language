

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
    # Create the main structure with dark oak wood
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:dark_oak_planks",
        scale=(7, 6, 8),
        fill=False
    )

@register()
def pointed_roof() -> Shape:
    def create_roof_layer(i: int) -> Shape:
        width = 9 - i * 2
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:purple_concrete",
                scale=(width, 1, 9),
                fill=True
            ),
            translation_matrix([-1 + i, 6 + i, -0.5])
        )
    return loop(4, create_roof_layer)

@register()
def windows() -> Shape:
    # Create glowing windows with soul lanterns
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
                block_type="minecraft:soul_lantern",
                block_kwargs={"hanging": "false"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([x, y, z])
        )
        for x, y, z in window_positions
    ])

@register()
def door() -> Shape:
    # Create a spooky door with iron door and frame
    return concat_shapes(
        # Door frame
        primitive_call(
            "set_cuboid",
            block_type="minecraft:dark_oak_log",
            block_kwargs={"axis": "y"},
            scale=(1, 3, 1),
            fill=True
        ),
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:dark_oak_log",
                block_kwargs={"axis": "y"},
                scale=(1, 3, 1),
                fill=True
            ),
            translation_matrix([2, 0, 0])
        ),
        # Door
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:iron_door",
                block_kwargs={"half": "lower", "facing": "south"},
                scale=(1, 2, 1),
                fill=True
            ),
            translation_matrix([1, 0, 0])
        )
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
            translation_matrix([-1, 0, -1])
        ),
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:jack_o_lantern",
                block_kwargs={"facing": "south"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([7, 0, -1])
        ),
        # Cobwebs on corners
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:cobweb",
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([0, 5, 0])
        ),
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:cobweb",
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([6, 5, 0])
        )
    )
"""

This code creates a spooky witch's house with:
1. A dark oak wood base structure
2. A pointed purple roof
3. Glowing soul lantern windows
4. An iron door with dark oak frame
5. Halloween decorations including:
   - Jack o'lanterns at the front corners
   - Cobwebs on the upper corners
   - A purple pointed roof for that classic witch house look

The house is relatively small (7x6x8 base) but includes all the essential spooky elements. The house features:
- Hollow interior (fill=False in house_base)
- Asymmetrical pointed roof made of purple concrete
- Glowing windows using soul lanterns for an eerie effect
- Iron door for that haunted feel
- Decorative elements like pumpkins and cobwebs
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
