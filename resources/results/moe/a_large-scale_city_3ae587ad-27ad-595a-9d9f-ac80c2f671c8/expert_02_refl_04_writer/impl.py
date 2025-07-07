

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

from dsl_utils import register_animation
import mi_helper  # such that primitive call will be implemented
import argparse
from typing import Literal, Optional

EXTRA_ENGINE_MODE = ['box', 'interior', 'exterior', 'mesh',
                     'gala3d', 'lmd', 'migc', 'loosecontrol', 'omost', 'densediffusion', 'neural']  # `densediffusion` must be the last as it modifies diffusers library


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine-modes', nargs='*', default=[], choices=EXTRA_ENGINE_MODE)
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite existing renderings')
    parser.add_argument('--log-dir', type=str, default=(Path(__file__).parent / 'renderings').as_posix(), help='log directory')
    parser.add_argument('--dependency-path', type=str, default=None, help='dependency path')
    parser.add_argument('--program-path', type=str, default=None, help='program path')
    return parser


def main():
    args = get_parser().parse_args()
    core(engine_modes=args.engine_modes, overwrite=args.overwrite, save_dir=args.log_dir,
         dependency_path=args.dependency_path, program_path=args.program_path)


def core(engine_modes: list[Literal['neural', 'lmd', 'omost', 'loosecontrol', 'densediffusion', 'mesh']], overwrite: bool, save_dir: str,
         dependency_path: Optional[str] = None, program_path: Optional[str] = None, root: Optional[str] = None,
         tree_depths: Optional[list[int]] = None):
    try:
        import torch
        cuda_is_available = torch.cuda.is_available()
    except:
        cuda_is_available = False

    from PIL import Image
    from dsl_utils import library, animation_func, set_seed
    from impl_utils import create_nodes, run, redirect_logs
    from engine.utils.graph_utils import strongly_connected_components, get_root, calculate_node_depths
    from impl_helper import make_new_library
    from prompt_helper import load_program
    from impl_parse_dependency import parse_dependency
    from engine.constants import ENGINE_MODE
    if ENGINE_MODE == "exposed_v2":
        import scripts.prompts.mesh_helper  # requires manual import to rewrite primitive call implementations from mi_helper!!
    try:
        from tu.loggers.utils import print_vcv_url
        from tu.loggers.utils import setup_vi
    except:
        print_vcv_url = lambda *args, **kwargs: print('[INFO]', str(args) + str(kwargs))

        class Helper:

            def dump_table(self, *args, **kwargs):
                print('[INFO]', str(args) + str(kwargs))

            def print_url(self, *args, **kwargs):
                print('[INFO]', str(args) + str(kwargs))
        setup_vi = lambda x: (None, Helper())

    from mi_helper import execute_from_preset
    import imageio
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    print_vcv_url(save_dir.as_posix())

    vi, vi_helper = setup_vi(save_dir)
    vi_debug, _ = setup_vi(save_dir / 'debug')
    vi_helper.dump_table(vi, [
        [vi_helper.print_url(vi_debug, verbose=False)],
        [print_vcv_url(save_dir.as_posix(), verbose=False)],
    ], col_type='text')

    if cuda_is_available and animation_func is not None:
        print(f'[INFO] skipping animation on cluster')
    elif animation_func is not None:
        print(f'[INFO] rendering animation...')
        frames = list(animation_func())
        name = animation_func.__name__
        final_frame_paths = []
        if len(frames) > 8:
            frame_skip = int(len(frames) / 8)
            frames = frames[::frame_skip]
        out = execute_from_preset(sum(frames, []), save_dir=None)
        for i in range(len(frames)):
            frame_save_dir = save_dir / name / f'{i:02d}'
            _ = execute_from_preset(frames[i], save_dir=frame_save_dir.as_posix(), prev_out=out)
            # TODO change `sensor_info`
            traj_paths = list(sorted(frame_save_dir.glob('rendering_traj_[0-9][0-9][0-9].png')))
            final_frame_paths.append(traj_paths[0])
            if i == 0:
                imageio.mimsave((save_dir / f'{name}_static.gif').as_posix(), [np.asarray(Image.open(p)) for p in traj_paths], fps=4, loop=0)
                out['sensors'] = {'rendering_traj_000': out['sensors']['rendering_traj_000']}
        imageio.mimsave((save_dir / f'{name}_animation.gif').as_posix(), [np.asarray(Image.open(p)) for p in final_frame_paths], fps=len(final_frame_paths) / 2, loop=0)

        return

    if root is not None:
        pass
    elif dependency_path is not None:
        root_node_ref, library_equiv_alt = parse_dependency(load_program(dependency_path))
        root = root_node_ref.name
    else:
        root = None
    library_equiv = create_nodes(roots=[root] if root is not None else None)
    success = True
    if success:
        if root is None:
            try:
                root = get_root(library_equiv)
                print(f'{root=}')
                vi_helper.dump_table(vi_debug, [['Parsed root from program.']])
            except Exception as e:
                # sometimes a function is implemented but never used, so there is no shared ancestor
                print('[ERROR] cannot find root', e)
                success = False
    if not success:
        if dependency_path is not None:
            from sketch_helper import transfer_dependency_to_library
            try:
                library_equiv = transfer_dependency_to_library(library_equiv_alt)
                root = get_root(library_equiv)
                print(f'{root=}')
                success = True
                vi_helper.dump_table(vi_debug, [['Parsed root from dependency.']])
            except Exception as e:
                print('[ERROR] cannot transfer dependency', e)
    if not success:
        root = None
        for name, node in library_equiv.items():  # do we need this? or just pick the last node?
            if len(node.parents) == 0 and len(node.children) > 0:
                root = name
        if root is not None:
            vi_helper.dump_table(vi_debug, [['Picked root with 0 parent and >=1 child from library.']])
        if root is None:
            root = next(reversed(library.keys()))
            vi_helper.dump_table(vi_debug, [['Last resort; picked last node from library.']])

    scc = strongly_connected_components(library_equiv)
    vi_helper.dump_table(vi, [[f'root function name: {root}'], [f'{scc=}']])
    vi_helper.dump_table(vi_debug, [[f'root function name: {root}'], [f'{scc=}']])
    vi_helper.dump_table(vi_debug, [[
        '' if dependency_path is None else load_program(dependency_path),
        '' if program_path is None else load_program(program_path)
    ]], col_names=['dependency', 'program'])

    print(f'[INFO] executing `{root}`...')
    # out = run(root, save_dir=save_dir.as_posix(), preset_id='table')
    new_library = make_new_library(library=library, library_equiv=library_equiv, tree_depth=float("inf"), engine_mode='interior', root=root)
    with set_seed(0):
        # frame = library_call(root)
        frame = new_library[root]['__target__']()
    out = execute_from_preset(frame, save_dir=None, preset_id='rover_background')  # compute normalization and sensors
    out = run(root, save_dir=save_dir.as_posix(), preset_id='rover_background', overwrite=overwrite, prev_out=out, new_library=new_library)
    print(f'[INFO] executing `{root}` done!')

    for name in library.keys():
        continue  # FIXME
        node_save_dir = Path(__file__).parent / 'nodes' / name
        node_save_dir.mkdir(parents=True, exist_ok=True)
        with redirect_logs((node_save_dir / f'log.txt').as_posix()):
            print(f'[INFO] executing `{name}`...')
            try:
                with set_seed(0):
                    frame = library_call(name)
            except Exception:
                print(f'[ERROR] failed to execute `{name}`')
                print(traceback.format_exc())
                continue
            _ = execute_from_preset(frame, save_dir=node_save_dir.as_posix(), preset_id='indoors_no_window', # preset_id='table',
                                    normalization=out['normalization'],
                                    sensors={k: v for k, v in out['sensors'].items() if 'traj' not in k})
            print(f'[INFO] executing `{name}` done!')

    # change the function implementation from `primitive_call` for mitsuba to for other engines
    try:
        node_depths = calculate_node_depths(library_equiv, root=root)
        print(f'{node_depths=}')
        max_tree_depth = max(node_depths.values())
    except Exception as e:
        print(e)
        import traceback; traceback.print_exc()
        max_tree_depth = -1
    if next(iter(library.values()))['docstring'].startswith('{'):
        tree_depths = [-1]
    elif tree_depths is None:
        tree_depths = list(range(max_tree_depth + 1))
    extra_frame_paths: dict[tuple[str, int], list[Path]] = {}

    def load_image(path: Path, resolution: int = 512):
        image = Image.open(path.as_posix())
        # image = image.resize((resolution, int(resolution * image.height / image.width)), resample=Image.BILINEAR)
        image = image.resize((resolution, resolution), resample=Image.BILINEAR).convert('RGB')
        return image

    for engine_mode in EXTRA_ENGINE_MODE:
        if engine_mode not in engine_modes:
            continue
        if engine_mode not in ['box', 'interior', 'exterior', 'mesh'] and not cuda_is_available:
            continue
        print(f'[INFO] running with {engine_mode}')
        for tree_depth in tree_depths:
            new_library = make_new_library(
                library=library,
                library_equiv=library_equiv,
                tree_depth=tree_depth,
                engine_mode=engine_mode,
                root=root,
            )

            print(f'[INFO] running with {tree_depth=} new library {new_library.keys()}')
            extra_out = run(root, save_dir=save_dir.as_posix(), preset_id='rover_background',
                            engine_mode=engine_mode, prev_out=out,
                            save_suffix=f'depth_{tree_depth:02d}',
                            new_library=new_library,
                            overwrite=overwrite)

            extra_frame_paths[(engine_mode, tree_depth)] = extra_out['final_frame_paths']

            for frame_ind, images_to_concat in enumerate(zip(*filter(None, extra_out['seq_name_to_frame_paths'].values()))):
                vi_helper.dump_table(vi_debug, [[f'engine_mode_{engine_mode}_tree_depth_{tree_depth}_viewpoint_{frame_ind:02d}']])
                vi_helper.dump_table(vi_debug, [list(map(load_image, images_to_concat))])

    # for tree_depth in np.linspace(0, max(max_tree_depth, 0), num=min(5, max(max_tree_depth, 0) + 1), dtype=int):
    # depth_candidates = list(range(max(max_tree_depth + 1, 1)))  # when max_tree_depth == -1, still execute the loop once
    depth_candidates = [0] if len(tree_depths) == 0 else tree_depths
    if len(depth_candidates) > 5:
        depth_candidates = depth_candidates[:4] + [depth_candidates[-1]]
    for tree_depth in depth_candidates:
        vi_helper.dump_table(vi, [[f'starting tree_depth={tree_depth:02d}']])
        runtime_engine_modes = [ENGINE_MODE]

        frame_paths_to_show = [out['final_frame_paths']]
        for engine_mode in EXTRA_ENGINE_MODE:
            if len(extra_frame_paths.get((engine_mode, tree_depth), [])) == 0:
                continue
            runtime_engine_modes.append(engine_mode)
            frame_paths_to_show.append(extra_frame_paths[(engine_mode, tree_depth)])
        for frame_ind, images_to_concat in enumerate(zip(*frame_paths_to_show)):
            vi_helper.dump_table(vi, [[f'tree_depth={tree_depth:02d}, viewpoint={frame_ind:02d}']])
            vi_helper.dump_table(vi, [list(map(load_image, images_to_concat))], col_names=[*runtime_engine_modes])

    # for p in sum(seq_name_to_frame_paths.values(), []):
    #     p.unlink()

    vi_helper.print_url(vi)
    vi_helper.print_url(vi_debug)


from helper import *

"""
a large-scale city
"""

@register("Creates a building with specified dimensions and color")
def building(width: float, height: float, depth: float, color: tuple = (0.7, 0.7, 0.7)) -> Shape:
    building_shape = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    # Position building so its base is at y=0
    return transform_shape(building_shape, translation_matrix((0, height/2, 0)))

@register("Creates a skyscraper with windows")
def skyscraper(width: float, height: float, depth: float, color: tuple = (0.5, 0.5, 0.6)) -> Shape:
    # Main building
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    main_building = transform_shape(main_building, translation_matrix((0, height/2, 0)))

    # Add windows - scale number of windows with building height
    windows = []
    window_width = width * 0.15
    window_height = height * 0.05
    window_depth = 0.01
    window_color = (0.9, 0.9, 0.7)

    # Calculate number of floors based on height
    num_floors = max(3, int(height * 2))
    windows_per_floor = 4

    def window_loop_fn(i):
        if i >= num_floors * windows_per_floor:
            return []

        row = i // windows_per_floor
        col = i % windows_per_floor
        window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
        x_pos = (col - 1.5) * (width * 0.22)
        y_pos = (row * height / num_floors) + (height * 0.1)
        z_pos = depth / 2 + window_depth/2  # Position windows on the surface
        return transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))

    windows = loop(num_floors * windows_per_floor, window_loop_fn)

    return concat_shapes(main_building, windows)

@register("Creates a house with a roof")
def house(width: float, height: float, depth: float) -> Shape:
    # Main house
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=(0.8, 0.6, 0.5))
    main_house = transform_shape(main_house, translation_matrix((0, height/2, 0)))

    # Roof
    roof_height = height * 0.5
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 1.2, roof_height, depth * 1.2)}, color=(0.6, 0.3, 0.2))
    roof = transform_shape(roof, translation_matrix((0, height + roof_height/2, 0)))

    # Door
    door_width = width * 0.2
    door_height = height * 0.4
    door = primitive_call('cube', shape_kwargs={'scale': (door_width, door_height, 0.01)}, color=(0.4, 0.2, 0.1))
    door = transform_shape(door, translation_matrix((0, door_height/2, depth/2 + 0.005)))  # Position door on the surface

    # Windows
    window1 = primitive_call('cube', shape_kwargs={'scale': (width * 0.15, height * 0.15, 0.01)}, color=(0.8, 0.8, 1.0))
    window1 = transform_shape(window1, translation_matrix((width * 0.25, height * 0.5, depth/2 + 0.005)))  # Position window on the surface

    window2 = primitive_call('cube', shape_kwargs={'scale': (width * 0.15, height * 0.15, 0.01)}, color=(0.8, 0.8, 1.0))
    window2 = transform_shape(window2, translation_matrix((-width * 0.25, height * 0.5, depth/2 + 0.005)))  # Position window on the surface

    return concat_shapes(main_house, roof, door, window1, window2)

@register("Creates a road segment with proper markings")
def road_segment(length: float, width: float = 1.0) -> Shape:
    road = primitive_call('cube', shape_kwargs={'scale': (width, 0.01, length)}, color=(0.2, 0.2, 0.2))
    road = transform_shape(road, translation_matrix((0, 0.005, 0)))  # Position road slightly above ground

    # Road markings - scale with road length
    marking_width = width * 0.05
    marking_length = length * 0.1
    num_markings = max(3, int(length / 0.5))  # Scale number of markings with road length

    def marking_loop_fn(i):
        if i >= num_markings:
            return []

        marking = primitive_call('cube', shape_kwargs={'scale': (marking_width, 0.02, marking_length)}, color=(1.0, 1.0, 1.0))
        z_pos = (i - num_markings/2) * (length / num_markings)
        return transform_shape(marking, translation_matrix((0, 0.01, z_pos)))  # Position markings slightly above road

    markings = loop(num_markings, marking_loop_fn)

    # Add sidewalks on both sides of the road
    sidewalk_width = width * 0.3
    sidewalk_left = primitive_call('cube', shape_kwargs={'scale': (sidewalk_width, 0.02, length)}, color=(0.8, 0.8, 0.8))
    sidewalk_left = transform_shape(sidewalk_left, translation_matrix((-width/2 - sidewalk_width/2, 0.01, 0)))

    sidewalk_right = primitive_call('cube', shape_kwargs={'scale': (sidewalk_width, 0.02, length)}, color=(0.8, 0.8, 0.8))
    sidewalk_right = transform_shape(sidewalk_right, translation_matrix((width/2 + sidewalk_width/2, 0.01, 0)))

    return concat_shapes(road, markings, sidewalk_left, sidewalk_right)

@register("Creates a traffic light")
def traffic_light() -> Shape:
    # Pole - increased height for better scale
    pole = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, 2.0, 0)}, color=(0.3, 0.3, 0.3))

    # Light housing
    housing_height = 0.5
    housing = primitive_call('cube', shape_kwargs={'scale': (0.2, housing_height, 0.2)}, color=(0.2, 0.2, 0.2))
    housing_y = 2.0 + housing_height/2
    housing = transform_shape(housing, translation_matrix((0, housing_y, 0)))

    # Lights - positioned relative to housing
    light_spacing = housing_height / 3

    red_light = primitive_call('sphere', shape_kwargs={'radius': 0.06}, color=(1.0, 0.0, 0.0))
    red_light = transform_shape(red_light, translation_matrix((0, housing_y + housing_height/3, 0.11)))

    yellow_light = primitive_call('sphere', shape_kwargs={'radius': 0.06}, color=(1.0, 1.0, 0.0))
    yellow_light = transform_shape(yellow_light, translation_matrix((0, housing_y, 0.11)))

    green_light = primitive_call('sphere', shape_kwargs={'radius': 0.06}, color=(0.0, 1.0, 0.0))
    green_light = transform_shape(green_light, translation_matrix((0, housing_y - housing_height/3, 0.11)))

    return concat_shapes(pole, housing, red_light, yellow_light, green_light)

@register("Creates a street lamp")
def street_lamp() -> Shape:
    # Pole - increased height for better scale
    pole = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, 2.5, 0)}, color=(0.3, 0.3, 0.3))

    # Lamp head
    lamp_head = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 0.9, 0.6))
    lamp_head = transform_shape(lamp_head, translation_matrix((0, 2.7, 0)))

    # Lamp arm
    lamp_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.03, 'p0': (0, 2.3, 0), 'p1': (0, 2.7, 0)}, color=(0.3, 0.3, 0.3))

    return concat_shapes(pole, lamp_head, lamp_arm)

@register("Creates a commercial building with distinctive features")
def commercial_building(width: float, height: float, depth: float) -> Shape:
    # Main building
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=(0.4, 0.6, 0.8))
    main_building = transform_shape(main_building, translation_matrix((0, height/2, 0)))

    # Glass facade
    facade = primitive_call('cube', shape_kwargs={'scale': (width * 0.95, height * 0.9, 0.05)}, color=(0.6, 0.8, 0.9))
    facade = transform_shape(facade, translation_matrix((0, height/2, depth/2 + 0.025)))

    # Entrance
    entrance_width = width * 0.3
    entrance_height = height * 0.2
    entrance = primitive_call('cube', shape_kwargs={'scale': (entrance_width, entrance_height, 0.1)}, color=(0.2, 0.2, 0.3))
    entrance = transform_shape(entrance, translation_matrix((0, entrance_height/2, depth/2 + 0.05)))

    return concat_shapes(main_building, facade, entrance)

@register("Creates a parking lot")
def parking_lot(width: float, depth: float) -> Shape:
    # Base asphalt
    base = primitive_call('cube', shape_kwargs={'scale': (width, 0.01, depth)}, color=(0.3, 0.3, 0.3))
    base = transform_shape(base, translation_matrix((0, 0.005, 0)))

    # Parking lines
    spaces_x = max(2, int(width / 0.8))
    spaces_z = max(1, int(depth / 2.0))

    parking_lines = []
    space_width = width / spaces_x
    space_depth = depth / spaces_z

    for i in range(spaces_x):
        for j in range(spaces_z):
            line = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.01, space_depth * 0.8)}, color=(1.0, 1.0, 1.0))
            x_pos = (i - spaces_x/2 + 0.5) * space_width
            z_pos = (j - spaces_z/2 + 0.5) * space_depth
            line = transform_shape(line, translation_matrix((x_pos, 0.01, z_pos)))
            parking_lines.append(line)

    return concat_shapes(base, *parking_lines)

@register("Creates a city block with buildings in a grid pattern")
def city_block(block_size: float, num_buildings: int, block_type: int = 1) -> Shape:
    # Calculate grid dimensions
    grid_size = int(math.sqrt(num_buildings))
    if grid_size * grid_size < num_buildings:
        grid_size += 1

    # Track occupied positions to prevent overlaps
    occupied_positions = []
    buildings = []

    def building_loop_fn(i):
        if i >= num_buildings:
            return []

        # Determine building type based on block type and position
        if block_type == 0:  # Downtown/commercial
            building_type = 0 if i % 5 < 3 else 3  # More skyscrapers and commercial buildings
        elif block_type == 1:  # Mixed use
            building_type = i % 4
        else:  # Residential
            building_type = 2 if i % 3 < 2 else 1  # More houses

        row = i // grid_size
        col = i % grid_size

        # Calculate position in grid
        spacing = block_size / (grid_size + 1)
        x_pos = (col - (grid_size-1)/2) * spacing
        z_pos = (row - (grid_size-1)/2) * spacing

        # Add slight randomness to position
        x_offset = math.sin(i * 0.5) * spacing * 0.1
        z_offset = math.cos(i * 0.5) * spacing * 0.1
        x_pos += x_offset
        z_pos += z_offset

        # Determine building size based on type
        if building_type == 0:  # Skyscraper
            width = 0.3 + (i % 5) * 0.05
            height = 4.0 + (i % 7) * 0.5  # Taller skyscrapers
            depth = 0.3 + (i % 4) * 0.05
            building_shape = library_call('skyscraper', width=width, height=height, depth=depth)
        elif building_type == 1:  # Regular building
            width = 0.4 + (i % 3) * 0.05
            height = 1.5 + (i % 4) * 0.2
            depth = 0.4 + (i % 3) * 0.05
            building_shape = library_call('building', width=width, height=height, depth=depth)
        elif building_type == 2:  # House
            width = 0.3 + (i % 2) * 0.05
            height = 0.5 + (i % 3) * 0.1
            depth = 0.3 + (i % 2) * 0.05
            building_shape = library_call('house', width=width, height=height, depth=depth)
        else:  # Commercial building
            width = 0.5 + (i % 3) * 0.1
            height = 2.0 + (i % 3) * 0.3
            depth = 0.5 + (i % 3) * 0.1
            building_shape = library_call('commercial_building', width=width, height=height, depth=depth)

        # Check for collisions with existing buildings
        building_footprint = (width, depth)
        for pos, size in occupied_positions:
            if (abs(pos[0] - x_pos) < (building_footprint[0]/2 + size[0]/2 + 0.05) and
                abs(pos[1] - z_pos) < (building_footprint[1]/2 + size[1]/2 + 0.05)):
                # Skip this building due to overlap
                return []

        # Add to occupied positions
        occupied_positions.append(((x_pos, z_pos), building_footprint))

        return transform_shape(building_shape, translation_matrix((x_pos, 0, z_pos)))

    buildings = loop(grid_size * grid_size, building_loop_fn)

    # Add parking lot if this is a commercial block
    if block_type == 0 or block_type == 1:
        parking = library_call('parking_lot', width=block_size * 0.3, depth=block_size * 0.2)
        parking = transform_shape(parking, translation_matrix((block_size * 0.3, 0, block_size * 0.3)))
        buildings = concat_shapes(buildings, parking)

    # Add street furniture
    street_furniture = []

    # Add traffic lights at corners
    traffic_light1 = library_call('traffic_light')
    traffic_light1 = transform_shape(traffic_light1, translation_matrix((block_size/2, 0, block_size/2)))

    traffic_light2 = library_call('traffic_light')
    traffic_light2 = transform_shape(traffic_light2, translation_matrix((-block_size/2, 0, block_size/2)))

    traffic_light3 = library_call('traffic_light')
    traffic_light3 = transform_shape(traffic_light3, translation_matrix((block_size/2, 0, -block_size/2)))

    traffic_light4 = library_call('traffic_light')
    traffic_light4 = transform_shape(traffic_light4, translation_matrix((-block_size/2, 0, -block_size/2)))

    # Add street lamps along block perimeter
    def lamp_loop_fn(i):
        if i >= 8:
            return []

        # Place lamps along the perimeter of the block
        if i < 2:  # North side
            x = (i - 0.5) * block_size
            z = -block_size/2 + 0.2
        elif i < 4:  # East side
            x = block_size/2 - 0.2
            z = (i - 3) * block_size
        elif i < 6:  # South side
            x = (i - 5.5) * block_size
            z = block_size/2 - 0.2
        else:  # West side
            x = -block_size/2 + 0.2
            z = (i - 8) * block_size

        lamp = library_call('street_lamp')
        return transform_shape(lamp, translation_matrix((x, 0, z)))

    street_lamps = loop(8, lamp_loop_fn)

    street_furniture = concat_shapes(traffic_light1, traffic_light2, traffic_light3, traffic_light4, street_lamps)

    return concat_shapes(buildings, street_furniture)

@register("Creates a park with trees and features")
def park(size: float) -> Shape:
    # Base grass
    grass = primitive_call('cube', shape_kwargs={'scale': (size, 0.01, size)}, color=(0.2, 0.7, 0.2))
    grass = transform_shape(grass, translation_matrix((0, 0.005, 0)))  # Position grass slightly above ground

    # Trees with more variety and natural placement
    def tree_loop_fn(i):
        if i >= 15:  # More trees for a denser park
            return []

        # Use polar coordinates for more natural tree placement
        angle = i * math.pi * 0.7  # Non-uniform angles
        distance = (0.2 + (i % 5) * 0.15) * size/2  # Varied distances from center

        x_pos = math.cos(angle) * distance
        z_pos = math.sin(angle) * distance

        # Tree trunk
        trunk_height = 0.5 + (i % 5) * 0.1  # Taller trees
        trunk = primitive_call('cylinder', shape_kwargs={
            'radius': 0.05 + (i % 3) * 0.01,  # Varied trunk thickness
            'p0': (0, 0, 0),
            'p1': (0, trunk_height, 0)
        }, color=(0.5, 0.3, 0.1))

        # Tree top - vary the shape based on tree type
        if i % 3 == 0:  # Conical tree
            top = primitive_call('cube', shape_kwargs={'scale': (0.3, 0.4, 0.3)}, color=(0.1, 0.5, 0.1))
            top = transform_shape(top, translation_matrix((0, trunk_height + 0.2, 0)))
        else:  # Round tree
            top_radius = 0.15 + (i % 3) * 0.05
            top = primitive_call('sphere', shape_kwargs={'radius': top_radius}, color=(0.1 + (i % 3) * 0.05, 0.5, 0.1))
            top = transform_shape(top, translation_matrix((0, trunk_height + top_radius * 0.5, 0)))

        tree = concat_shapes(trunk, top)
        return transform_shape(tree, translation_matrix((x_pos, 0, z_pos)))

    trees = loop(15, tree_loop_fn)

    # Add benches (more proportional)
    def bench_loop_fn(i):
        if i >= 4:
            return []

        angle = i * math.pi/2 + math.pi/4  # Place benches at 45°, 135°, 225°, 315°
        distance = size * 0.35

        x_pos = math.cos(angle) * distance
        z_pos = math.sin(angle) * distance

        # Bench seat
        seat = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.05, 0.15)}, color=(0.6, 0.4, 0.2))

        # Bench legs
        leg1 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.1, 0.15)}, color=(0.5, 0.3, 0.1))
        leg1 = transform_shape(leg1, translation_matrix((0.15, -0.05, 0)))

        leg2 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.1, 0.15)}, color=(0.5, 0.3, 0.1))
        leg2 = transform_shape(leg2, translation_matrix((-0.15, -0.05, 0)))

        bench = concat_shapes(seat, leg1, leg2)
        bench = transform_shape(bench, translation_matrix((x_pos, 0.1, z_pos)))
        bench = transform_shape(bench, rotation_matrix(angle + math.pi/2, (0, 1, 0), (x_pos, 0.1, z_pos)))

        return bench

    benches = loop(4, bench_loop_fn)

    # Add a larger fountain
    fountain_base = primitive_call('cylinder', shape_kwargs={'radius': 0.4, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)}, color=(0.7, 0.7, 0.7))
    fountain_rim = primitive_call('cylinder', shape_kwargs={'radius': 0.45, 'p0': (0, 0.1, 0), 'p1': (0, 0.15, 0)}, color=(0.6, 0.6, 0.6))
    fountain_water = primitive_call('cylinder', shape_kwargs={'radius': 0.35, 'p0': (0, 0.1, 0), 'p1': (0, 0.12, 0)}, color=(0.6, 0.8, 1.0))

    # Fountain center piece
    center_column = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (0, 0, 0), 'p1': (0, 0.3, 0)}, color=(0.6, 0.6, 0.6))
    water_spray = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(0.7, 0.9, 1.0))
    water_spray = transform_shape(water_spray, translation_matrix((0, 0.35, 0)))

    fountain = concat_shapes(fountain_base, fountain_rim, fountain_water, center_column, water_spray)
    fountain = transform_shape(fountain, translation_matrix((0, 0, 0)))

    # Add pathways
    def pathway_loop_fn(i):
        if i >= 4:
            return []

        angle = i * math.pi/2
        path = primitive_call('cube', shape_kwargs={'scale': (0.2, 0.02, size/2)}, color=(0.8, 0.7, 0.6))
        path = transform_shape(path, rotation_matrix(angle, (0, 1, 0), (0, 0, 0)))
        path = transform_shape(path, translation_matrix((0, 0.01, 0)))
        return path

    pathways = loop(4, pathway_loop_fn)

    return concat_shapes(grass, trees, benches, fountain, pathways)

@register("Creates a bridge over a road")
def bridge(length: float, width: float) -> Shape:
    # Bridge deck
    deck = primitive_call('cube', shape_kwargs={'scale': (width, 0.2, length)}, color=(0.5, 0.5, 0.5))
    deck = transform_shape(deck, translation_matrix((0, 1.5, 0)))  # Elevated above road

    # Support columns
    column1 = primitive_call('cylinder', shape_kwargs={'radius': 0.2, 'p0': (0, 0, -length/3), 'p1': (0, 1.5, -length/3)}, color=(0.4, 0.4, 0.4))
    column2 = primitive_call('cylinder', shape_kwargs={'radius': 0.2, 'p0': (0, 0, length/3), 'p1': (0, 1.5, length/3)}, color=(0.4, 0.4, 0.4))

    # Railings
    railing1 = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, 0.05)}, color=(0.3, 0.3, 0.3))
    railing1 = transform_shape(railing1, translation_matrix((0, 1.65, -length/2 + 0.025)))

    railing2 = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, 0.05)}, color=(0.3, 0.3, 0.3))
    railing2 = transform_shape(railing2, translation_matrix((0, 1.65, length/2 - 0.025)))

    railing3 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.1, length)}, color=(0.3, 0.3, 0.3))
    railing3 = transform_shape(railing3, translation_matrix((width/2 - 0.025, 1.65, 0)))

    railing4 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.1, length)}, color=(0.3, 0.3, 0.3))
    railing4 = transform_shape(railing4, translation_matrix((-width/2 + 0.025, 1.65, 0)))

    return concat_shapes(deck, column1, column2, railing1, railing2, railing3, railing4)

@register("Creates a connected road network for the city with intersections")
def road_network(size: int, block_size: float, spacing: float) -> Shape:
    total_size = size * (block_size + spacing)
    roads = []

    # Create horizontal roads
    for i in range(size + 1):
        pos = (i - size/2) * (block_size + spacing)
        road = library_call('road_segment', length=total_size, width=spacing)
        road = transform_shape(road, translation_matrix((0, 0, pos)))
        roads.append(road)

    # Create vertical roads
    for i in range(size + 1):
        pos = (i - size/2) * (block_size + spacing)
        road = library_call('road_segment', length=total_size, width=spacing)
        road = transform_shape(road, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        road = transform_shape(road, translation_matrix((pos, 0, 0)))
        roads.append(road)

    # Create intersections
    for i in range(size + 1):
        for j in range(size + 1):
            x_pos = (i - size/2) * (block_size + spacing)
            z_pos = (j - size/2) * (block_size + spacing)

            intersection = primitive_call('cube', shape_kwargs={'scale': (spacing, 0.015, spacing)}, color=(0.25, 0.25, 0.25))
            intersection = transform_shape(intersection, translation_matrix((x_pos, 0.007, z_pos)))
            roads.append(intersection)

    # Add a bridge over one of the roads
    if size >= 3:
        bridge_shape = library_call('bridge', length=block_size + spacing, width=spacing)
        bridge_shape = transform_shape(bridge_shape, translation_matrix((0, 0, (block_size + spacing))))
        roads.append(bridge_shape)

    return concat_shapes(*roads)

@register("Creates a large-scale city with multiple blocks and connected roads")
def city(size: int = 4) -> Shape:
    block_size = 3.0  # Increased block size for better proportions
    road_width = 1.0  # Wider roads
    block_spacing = road_width  # Space between blocks equals road width

    # Create ground
    ground_size = size * (block_size + block_spacing) + block_spacing
    ground = primitive_call('cube', shape_kwargs={'scale': (ground_size, 0.1, ground_size)}, color=(0.6, 0.6, 0.6))
    ground = transform_shape(ground, translation_matrix((0, -0.05, 0)))  # Position ground so its top is at y=0

    # Create connected road network
    roads = library_call('road_network', size=size, block_size=block_size, spacing=block_spacing)

    # Create city blocks with different types based on location
    def city_block_loop_fn(i):
        if i >= size * size:
            return []

        row = i // size
        col = i % size

        # Determine block type based on position
        # 0: Downtown/commercial, 1: Mixed use, 2: Residential, 3: Park
        distance_from_center = math.sqrt((row - size/2 + 0.5)**2 + (col - size/2 + 0.5)**2)

        if distance_from_center < size/4:
            block_type = 0  # Downtown
            num_buildings = 12
        elif distance_from_center < size/2:
            block_type = 1  # Mixed use
            num_buildings = 9
        elif i % 7 != 0:  # Most outer blocks are residential
            block_type = 2  # Residential
            num_buildings = 6
        else:  # Some outer blocks are parks
            block_type = 3  # Park

        if block_type < 3:
            block = library_call('city_block', block_size=block_size, num_buildings=num_buildings, block_type=block_type)
        else:
            block = library_call('park', size=block_size)

        x_pos = (col - size/2 + 0.5) * (block_size + block_spacing)
        z_pos = (row - size/2 + 0.5) * (block_size + block_spacing)

        return transform_shape(block, translation_matrix((x_pos, 0, z_pos)))

    city_blocks = loop(size * size, city_block_loop_fn)

    return concat_shapes(ground, roads, city_blocks)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
