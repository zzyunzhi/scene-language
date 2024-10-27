

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
Here's a program to create a simplified version of Stonehenge using the provided `helper.py`:

"""
from helper import *

"""
Stonehenge
"""

@register()
def stonehenge() -> Shape:
    return concat_shapes(
        library_call("stone_circle"),
        library_call("inner_horseshoe"),
        library_call("altar_stone"),
        library_call("heel_stone")
    )

@register()
def stone_circle() -> Shape:
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
    def create_inner_trilithon(i):
        angle = i * (360 / 5)  # 5 trilithons in the inner horseshoe
        radius = 8
        x = int(radius * math.cos(math.radians(angle)))
        z = int(radius * math.sin(math.radians(angle)))
        return transform_shape(
            library_call("large_trilithon"),
            translation_matrix([x, 0, z + 2])  # Shift slightly forward
        )
    return loop(3, create_inner_trilithon)  # Only create 3 out of 5 for horseshoe shape

@register()
def trilithon() -> Shape:
    return concat_shapes(
        library_call("standing_stone"),
        transform_shape(library_call("standing_stone"), translation_matrix([2, 0, 0])),
        transform_shape(library_call("lintel"), translation_matrix([0, 4, 0]))
    )

@register()
def large_trilithon() -> Shape:
    return concat_shapes(
        library_call("large_standing_stone"),
        transform_shape(library_call("large_standing_stone"), translation_matrix([3, 0, 0])),
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
        scale=(3, 1, 1),
        fill=True
    )

@register()
def large_lintel() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:stone",
        scale=(5, 1, 1),
        fill=True
    )

@register()
def altar_stone() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:smooth_stone",
            scale=(3, 1, 2),
            fill=True
        ),
        translation_matrix([0, 0, 0])  # Placed at the center
    )

@register()
def heel_stone() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:mossy_cobblestone",
            scale=(2, 3, 1),
            fill=True
        ),
        translation_matrix([0, 0, 20])  # Placed outside the main circle
    )
"""

This program creates a simplified version of Stonehenge with the following components:

1. An outer circle of 30 trilithons (pairs of vertical stones with a horizontal lintel on top).
2. An inner horseshoe of 3 larger trilithons.
3. An altar stone at the center.
4. A heel stone placed outside the main circle.

The structure is created using various stone types to differentiate between different elements. The outer circle uses regular stone, the altar uses smooth stone, and the heel stone uses mossy cobblestone for a more weathered look.

The positioning of the stones is approximated using trigonometric functions to create circular arrangements. The scale of the structure can be adjusted by modifying the radius values in the `stone_circle` and `inner_horseshoe` functions.

This simplified version captures the essence of Stonehenge's layout while adhering to the constraints of the Minecraft block system and the provided helper functions.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
