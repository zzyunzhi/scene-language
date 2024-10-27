

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
        library_call("tower_entrance", radius=radius),
        library_call("tower_battlements", height=height, radius=radius),
        library_call("tower_arrow_slits", height=height, radius=radius),
        library_call("tower_flag", height=height, radius=radius),
        library_call("tower_spiral_staircase", height=height, radius=radius),
        library_call("tower_floors", height=height, radius=radius)
    )

@register()
def tower_base(height: int, radius: int) -> Shape:
    def create_circle_layer(y):
        circle = []
        for x in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                if x*x + z*z <= radius*radius:
                    circle.append(transform_shape(
                        primitive_call("set_cuboid", block_type="minecraft:stone_bricks", scale=(1, 1, 1), fill=True),
                        translation_matrix([x, y, z])
                    ))
        return concat_shapes(*circle)

    return concat_shapes(*[create_circle_layer(y) for y in range(height)])

@register()
def tower_roof(height: int, radius: int) -> Shape:
    def create_roof_layer(y):
        current_radius = max(1, int(radius * (1 - y / (radius * 1.5))))
        return create_circle_layer(current_radius, height + y)

    def create_circle_layer(r, y):
        circle = []
        for x in range(-r, r + 1):
            for z in range(-r, r + 1):
                if x*x + z*z <= r*r:
                    circle.append(transform_shape(
                        primitive_call("set_cuboid", block_type="minecraft:spruce_planks", scale=(1, 1, 1), fill=True),
                        translation_matrix([x, y, z])
                    ))
        return concat_shapes(*circle)

    return concat_shapes(*[create_roof_layer(y) for y in range(int(radius * 1.5))])

@register()
def tower_windows(height: int, radius: int) -> Shape:
    def create_window(y, angle):
        x = int(radius * math.cos(math.radians(angle)))
        z = int(radius * math.sin(math.radians(angle)))
        return transform_shape(
            primitive_call("set_cuboid", block_type="minecraft:glass_pane", scale=(1, 2, 1), fill=True),
            translation_matrix([x, y, z])
        )

    return concat_shapes(*[
        create_window(y, angle)
        for y in range(3, height - 3, 4)
        for angle in range(0, 360, 90)
    ])

@register()
def tower_entrance(radius: int) -> Shape:
    return concat_shapes(
        transform_shape(
            primitive_call("set_cuboid", block_type="minecraft:oak_door",
                           block_kwargs={"facing": "south", "half": "lower", "hinge": "left"},
                           scale=(1, 1, 1), fill=True),
            translation_matrix([0, 0, radius])
        ),
        transform_shape(
            primitive_call("set_cuboid", block_type="minecraft:oak_door",
                           block_kwargs={"facing": "south", "half": "upper", "hinge": "left"},
                           scale=(1, 1, 1), fill=True),
            translation_matrix([0, 1, radius])
        )
    )

@register()
def tower_battlements(height: int, radius: int) -> Shape:
    def create_battlement(angle):
        x = int(radius * math.cos(math.radians(angle)))
        z = int(radius * math.sin(math.radians(angle)))
        return transform_shape(
            primitive_call("set_cuboid", block_type="minecraft:stone_brick_wall", scale=(1, 2, 1), fill=True),
            translation_matrix([x, height, z])
        )

    return concat_shapes(*[create_battlement(angle) for angle in range(0, 360, 45)])

@register()
def tower_arrow_slits(height: int, radius: int) -> Shape:
    def create_arrow_slit(y, angle):
        x = int(radius * math.cos(math.radians(angle)))
        z = int(radius * math.sin(math.radians(angle)))
        return transform_shape(
            primitive_call("set_cuboid", block_type="minecraft:air", scale=(1, 2, 1), fill=True),
            translation_matrix([x, y, z])
        )

    return concat_shapes(*[
        create_arrow_slit(y, angle)
        for y in range(2, height - 2, 3)
        for angle in range(22, 360, 45)
    ])

@register()
def tower_flag(height: int, radius: int) -> Shape:
    flagpole = transform_shape(
        primitive_call("set_cuboid", block_type="minecraft:oak_fence", scale=(1, 5, 1), fill=True),
        translation_matrix([0, height + 1, 0])
    )

    flag = transform_shape(
        primitive_call("set_cuboid", block_type="minecraft:red_wool", scale=(3, 2, 1), fill=True),
        translation_matrix([1, height + 3, 0])
    )

    return concat_shapes(flagpole, flag)

@register()
def tower_spiral_staircase(height: int, radius: int) -> Shape:
    def create_stair(y):
        angle = y * 20
        x = int((radius - 1) * math.cos(math.radians(angle)))
        z = int((radius - 1) * math.sin(math.radians(angle)))
        facing = ["east", "south", "west", "north"][int(angle / 90) % 4]
        return transform_shape(
            primitive_call("set_cuboid", block_type="minecraft:oak_stairs",
                           block_kwargs={"facing": facing, "half": "bottom"},
                           scale=(1, 1, 1), fill=True),
            translation_matrix([x, y, z])
        )

    return concat_shapes(*[create_stair(y) for y in range(height)])

@register()
def tower_floors(height: int, radius: int) -> Shape:
    def create_floor(y):
        floor = []
        for x in range(-radius + 1, radius):
            for z in range(-radius + 1, radius):
                if x*x + z*z <= (radius - 1)*(radius - 1):
                    floor.append(transform_shape(
                        primitive_call("set_cuboid", block_type="minecraft:oak_planks", scale=(1, 1, 1), fill=True),
                        translation_matrix([x, y, z])
                    ))
        return concat_shapes(*floor)

    return concat_shapes(*[create_floor(y) for y in range(4, height - 1, 4)])
"""

This improved version addresses the issues mentioned in the review and adds more details to the tower:

1. The `tower_base` function now creates a cylindrical shape by approximating a circle using multiple cuboids.
2. The `tower_roof` function creates a conical roof that matches the cylindrical shape of the tower.
3. The `tower_windows`, `tower_battlements`, and `tower_arrow_slits` functions now align with the circular shape of the tower.
4. The `tower_spiral_staircase` function adjusts the facing direction of each step based on its position.
5. A new `tower_floors` function has been added to create interior floors.

These changes result in a more realistic and detailed cylindrical medieval tower that better matches the task description and addresses the points raised in the review.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
