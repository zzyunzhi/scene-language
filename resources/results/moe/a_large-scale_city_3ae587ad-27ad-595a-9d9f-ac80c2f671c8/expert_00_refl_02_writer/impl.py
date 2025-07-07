

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
    # Create building centered at its base
    building_shape = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    # Move up so bottom is at y=0
    return transform_shape(building_shape, translation_matrix((0, height/2, 0)))

@register("Creates a skyscraper with windows")
def skyscraper(width: float, height: float, depth: float, color: tuple = (0.5, 0.5, 0.6)) -> Shape:
    # Main building centered at its base
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    main_building = transform_shape(main_building, translation_matrix((0, height/2, 0)))

    # Window parameters
    window_width = width * 0.15
    window_height = window_width * 1.5
    window_depth = 0.01
    window_color = (0.9, 0.9, 1.0)

    # Calculate number of windows per side
    windows_per_width = max(2, int(width / (window_width * 1.5)))
    windows_per_height = max(3, int(height / (window_height * 1.5)))

    # Function to create windows on one face using loop
    def create_face_windows(face_direction: str) -> Shape:
        def window_fn(idx: int) -> Shape:
            i = idx % windows_per_width
            j = idx // windows_per_width

            if face_direction in ['front', 'back']:
                z_pos = depth/2 if face_direction == 'front' else -depth/2
                x_spacing = width / (windows_per_width + 1)
                y_spacing = height / (windows_per_height + 1)

                x_pos = ((i + 1) * x_spacing) - width/2
                y_pos = (j + 1) * y_spacing

                window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
                return transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
            else:  # 'left' or 'right'
                x_pos = width/2 if face_direction == 'right' else -width/2
                z_spacing = depth / (windows_per_width + 1)
                y_spacing = height / (windows_per_height + 1)

                z_pos = ((i + 1) * z_spacing) - depth/2
                y_pos = (j + 1) * y_spacing

                window = primitive_call('cube', shape_kwargs={'scale': (window_depth, window_height, window_width)}, color=window_color)
                return transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))

        return loop(windows_per_width * windows_per_height, window_fn)

    # Create windows for each face
    front_windows = create_face_windows('front')
    back_windows = create_face_windows('back')
    left_windows = create_face_windows('left')
    right_windows = create_face_windows('right')

    return concat_shapes(main_building, front_windows, back_windows, left_windows, right_windows)

@register("Creates a house with a roof")
def house(width: float, height: float, depth: float, color: tuple = (0.8, 0.7, 0.6)) -> Shape:
    # Main house centered at its base
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    main_house = transform_shape(main_house, translation_matrix((0, height/2, 0)))

    # Roof
    roof_height = height * 0.5
    roof_color = (0.6, 0.3, 0.2)

    # Create a roof using a scaled and rotated cube
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 1.2, roof_height, depth * 1.2)}, color=roof_color)

    # Position the roof on top of the house
    roof = transform_shape(roof, translation_matrix((0, height + roof_height/2, 0)))

    # Add a door
    door_width = width * 0.2
    door_height = height * 0.4
    door_depth = 0.01
    door_color = (0.4, 0.2, 0.1)

    door = primitive_call('cube', shape_kwargs={'scale': (door_width, door_height, door_depth)}, color=door_color)
    door = transform_shape(door, translation_matrix((0, door_height/2, depth/2)))

    return concat_shapes(main_house, roof, door)

@register("Creates a road segment with sidewalks")
def road(length: float, width: float = 1.0) -> Shape:
    road_color = (0.2, 0.2, 0.2)
    road_height = 0.02

    # Slightly elevate road to prevent z-fighting
    road_elevation = 0.01

    # Main road
    road_segment = primitive_call('cube', shape_kwargs={'scale': (width, road_height, length)}, color=road_color)
    road_segment = transform_shape(road_segment, translation_matrix((0, road_elevation, 0)))

    # Add road markings using loop
    marking_width = width * 0.05
    marking_length = length * 0.05
    marking_height = 0.025
    marking_color = (1.0, 1.0, 1.0)
    markings_count = int(length / (marking_length * 2))

    def marking_fn(i: int) -> Shape:
        marking = primitive_call('cube', shape_kwargs={'scale': (marking_width, marking_height, marking_length)}, color=marking_color)
        position = (0, road_elevation + road_height/2, -length/2 + marking_length/2 + i * marking_length * 2)
        return transform_shape(marking, translation_matrix(position))

    markings = loop(markings_count, marking_fn)

    # Add sidewalks
    sidewalk_width = width * 0.3
    sidewalk_height = 0.05
    sidewalk_color = (0.7, 0.7, 0.7)

    left_sidewalk = primitive_call('cube', shape_kwargs={'scale': (sidewalk_width, sidewalk_height, length)}, color=sidewalk_color)
    left_sidewalk = transform_shape(left_sidewalk, translation_matrix((-width/2 - sidewalk_width/2, sidewalk_height/2, 0)))

    right_sidewalk = primitive_call('cube', shape_kwargs={'scale': (sidewalk_width, sidewalk_height, length)}, color=sidewalk_color)
    right_sidewalk = transform_shape(right_sidewalk, translation_matrix((width/2 + sidewalk_width/2, sidewalk_height/2, 0)))

    return concat_shapes(road_segment, markings, left_sidewalk, right_sidewalk)

@register("Creates a street lamp")
def street_lamp(height: float = 2.5) -> Shape:
    pole_radius = 0.05
    pole_color = (0.3, 0.3, 0.3)

    # Create pole
    pole = primitive_call('cylinder', shape_kwargs={'radius': pole_radius, 'p0': (0, 0, 0), 'p1': (0, height, 0)}, color=pole_color)

    # Create lamp head
    lamp_radius = 0.15
    lamp_color = (0.9, 0.9, 0.6)

    lamp_head = primitive_call('sphere', shape_kwargs={'radius': lamp_radius}, color=lamp_color)
    lamp_head = transform_shape(lamp_head, translation_matrix((0, height, 0)))

    return concat_shapes(pole, lamp_head)

@register("Creates a traffic light")
def traffic_light(height: float = 3.0) -> Shape:
    pole_radius = 0.05
    pole_color = (0.3, 0.3, 0.3)

    # Create pole
    pole = primitive_call('cylinder', shape_kwargs={'radius': pole_radius, 'p0': (0, 0, 0), 'p1': (0, height, 0)}, color=pole_color)

    # Create traffic light box
    box_width = 0.2
    box_height = 0.6
    box_depth = 0.2
    box_color = (0.2, 0.2, 0.2)

    box = primitive_call('cube', shape_kwargs={'scale': (box_width, box_height, box_depth)}, color=box_color)
    box = transform_shape(box, translation_matrix((0, height - box_height/2, 0)))

    # Create lights
    light_radius = 0.06
    red_light = primitive_call('sphere', shape_kwargs={'radius': light_radius}, color=(1.0, 0.0, 0.0))
    yellow_light = primitive_call('sphere', shape_kwargs={'radius': light_radius}, color=(1.0, 1.0, 0.0))
    green_light = primitive_call('sphere', shape_kwargs={'radius': light_radius}, color=(0.0, 1.0, 0.0))

    red_light = transform_shape(red_light, translation_matrix((0, height - box_height/6, box_depth/2)))
    yellow_light = transform_shape(yellow_light, translation_matrix((0, height - box_height/2, box_depth/2)))
    green_light = transform_shape(green_light, translation_matrix((0, height - 5*box_height/6, box_depth/2)))

    return concat_shapes(pole, box, red_light, yellow_light, green_light)

@register("Creates a tree")
def tree(height: float = 2.0) -> Shape:
    # Create trunk
    trunk_radius = 0.1
    trunk_height = height * 0.4
    trunk_color = (0.5, 0.3, 0.2)

    trunk = primitive_call('cylinder', shape_kwargs={'radius': trunk_radius, 'p0': (0, 0, 0), 'p1': (0, trunk_height, 0)}, color=trunk_color)

    # Create foliage
    foliage_radius = height * 0.3
    foliage_color = (0.1, 0.6, 0.1)

    foliage = primitive_call('sphere', shape_kwargs={'radius': foliage_radius}, color=foliage_color)
    foliage = transform_shape(foliage, translation_matrix((0, trunk_height + foliage_radius * 0.7, 0)))

    return concat_shapes(trunk, foliage)

@register("Creates a park with trees and benches")
def park(width: float, depth: float) -> Shape:
    # Create grass base
    grass_height = 0.05
    grass_color = (0.2, 0.7, 0.2)

    grass = primitive_call('cube', shape_kwargs={'scale': (width, grass_height, depth)}, color=grass_color)
    grass = transform_shape(grass, translation_matrix((0, grass_height/2, 0)))

    # Add trees
    num_trees = int(width * depth / 10)

    def tree_fn(i: int) -> Shape:
        x = np.random.uniform(-width/2 + 1, width/2 - 1)
        z = np.random.uniform(-depth/2 + 1, depth/2 - 1)
        tree_height = np.random.uniform(1.5, 2.5)
        tree = library_call('tree', height=tree_height)
        return transform_shape(tree, translation_matrix((x, 0, z)))

    trees = loop(num_trees, tree_fn)

    # Add benches
    bench_width = 1.0
    bench_height = 0.4
    bench_depth = 0.4
    bench_color = (0.6, 0.4, 0.2)

    def bench_fn(i: int) -> Shape:
        # Place benches around the perimeter
        side = i % 4
        pos = (i // 4) / max(1, (num_benches // 4))

        if side == 0:  # Top
            x = -width/2 + width * pos
            z = -depth/2 + 1
        elif side == 1:  # Right
            x = width/2 - 1
            z = -depth/2 + depth * pos
        elif side == 2:  # Bottom
            x = -width/2 + width * pos
            z = depth/2 - 1
        else:  # Left
            x = -width/2 + 1
            z = -depth/2 + depth * pos

        bench = primitive_call('cube', shape_kwargs={'scale': (bench_width, bench_height, bench_depth)}, color=bench_color)
        bench = transform_shape(bench, translation_matrix((x, bench_height/2, z)))
        return bench

    num_benches = 8
    benches = loop(num_benches, bench_fn)

    return concat_shapes(grass, trees, benches)

@register("Creates a city block with buildings")
def city_block(width: float, depth: float, max_buildings: int = 6, block_type: str = 'mixed') -> Shape:
    # Set random seed for reproducibility
    np.random.seed(int(width * depth) % 1000)

    buildings_list = []

    # Divide the block into a grid
    grid_size = int(math.sqrt(max_buildings))
    cell_width = width / grid_size
    cell_depth = depth / grid_size

    def create_building(i: int, j: int) -> Shape:
        # Adjust building type probabilities based on block type
        if block_type == 'downtown':
            type_probs = [0.7, 0.3, 0.0]  # skyscraper, building, house
            height_range = (4.0, 8.0)
        elif block_type == 'residential':
            type_probs = [0.0, 0.3, 0.7]  # skyscraper, building, house
            height_range = (0.8, 2.0)
        elif block_type == 'commercial':
            type_probs = [0.2, 0.7, 0.1]  # skyscraper, building, house
            height_range = (1.5, 4.0)
        else:  # mixed
            type_probs = [0.3, 0.4, 0.3]  # skyscraper, building, house
            height_range = (1.0, 6.0)

        building_type = np.random.choice(['skyscraper', 'building', 'house'], p=type_probs)
        building_width = cell_width * np.random.uniform(0.6, 0.9)
        building_depth = cell_depth * np.random.uniform(0.6, 0.9)

        if building_type == 'skyscraper':
            building_height = np.random.uniform(height_range[0], height_range[1])
            color = (np.random.uniform(0.4, 0.6), np.random.uniform(0.4, 0.6), np.random.uniform(0.5, 0.7))
            building = library_call('skyscraper', width=building_width, height=building_height, depth=building_depth, color=color)
        elif building_type == 'building':
            building_height = np.random.uniform(height_range[0] * 0.7, height_range[1] * 0.7)
            color = (np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8))
            building = library_call('building', width=building_width, height=building_height, depth=building_depth, color=color)
        else:
            building_height = np.random.uniform(height_range[0] * 0.5, height_range[1] * 0.5)
            color = (np.random.uniform(0.7, 0.9), np.random.uniform(0.6, 0.8), np.random.uniform(0.5, 0.7))
            building = library_call('house', width=building_width, height=building_height, depth=building_depth, color=color)

        # Position the building in the grid
        x_pos = -width/2 + cell_width/2 + i * cell_width
        z_pos = -depth/2 + cell_depth/2 + j * cell_depth

        return transform_shape(building, translation_matrix((x_pos, 0, z_pos)))

    # Create buildings in a grid pattern
    for i in range(grid_size):
        for j in range(grid_size):
            # Randomly skip some positions to create variety
            if np.random.random() < 0.8:  # 80% chance to place a building
                buildings_list.append(create_building(i, j))

    # Add a small park in one corner if it's a residential or mixed block
    if block_type in ['residential', 'mixed'] and np.random.random() < 0.3:
        park_size = min(width, depth) * 0.3
        park_x = -width/2 + park_size/2
        park_z = -depth/2 + park_size/2
        park = library_call('park', width=park_size, depth=park_size)
        park = transform_shape(park, translation_matrix((park_x, 0, park_z)))
        buildings_list.append(park)

    return concat_shapes(*buildings_list)

@register("Creates a city district with blocks and roads")
def city_district(size: float, num_blocks: int = 3, district_type: str = 'mixed') -> Shape:
    # Set random seed for reproducibility
    np.random.seed(int(size * 100) % 1000)

    district = []

    # Calculate block and road dimensions
    road_width = 1.0
    block_size = (size - (num_blocks + 1) * road_width) / num_blocks

    # Create city blocks
    for i in range(num_blocks):
        for j in range(num_blocks):
            x_pos = -size/2 + road_width + block_size/2 + i * (block_size + road_width)
            z_pos = -size/2 + road_width + block_size/2 + j * (block_size + road_width)

            # Determine block type based on district type and position
            if district_type == 'downtown':
                block_type = 'downtown'
            elif district_type == 'residential':
                block_type = 'residential'
            elif district_type == 'commercial':
                block_type = 'commercial'
            else:  # mixed
                # Center blocks are more likely to be downtown
                distance_from_center = math.sqrt((i - num_blocks/2)**2 + (j - num_blocks/2)**2)
                if distance_from_center < num_blocks/4:
                    block_type = np.random.choice(['downtown', 'commercial', 'mixed'], p=[0.6, 0.3, 0.1])
                else:
                    block_type = np.random.choice(['residential', 'commercial', 'mixed'], p=[0.6, 0.3, 0.1])

            block = library_call('city_block', width=block_size, depth=block_size, block_type=block_type)
            block = transform_shape(block, translation_matrix((x_pos, 0, z_pos)))
            district.append(block)

    # Create horizontal roads
    for i in range(num_blocks + 1):
        z_pos = -size/2 + i * (block_size + road_width)
        road = library_call('road', length=size, width=road_width)
        road = transform_shape(road, translation_matrix((0, 0, z_pos)))
        district.append(road)

    # Create vertical roads
    for i in range(num_blocks + 1):
        x_pos = -size/2 + i * (block_size + road_width)
        road = library_call('road', length=size, width=road_width)
        road = transform_shape(road, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        road = transform_shape(road, translation_matrix((x_pos, 0, 0)))
        district.append(road)

    # Add street lamps and traffic lights at intersections
    for i in range(num_blocks + 1):
        for j in range(num_blocks + 1):
            x_pos = -size/2 + i * (block_size + road_width)
            z_pos = -size/2 + j * (block_size + road_width)

            # Add street lamp at corner of intersection
            lamp = library_call('street_lamp')
            lamp = transform_shape(lamp, translation_matrix((x_pos - road_width/3, 0, z_pos - road_width/3)))
            district.append(lamp)

            # Add traffic light at major intersections
            if i > 0 and i < num_blocks and j > 0 and j < num_blocks:
                if (i + j) % 2 == 0:  # Only at some intersections
                    traffic_light_ns = library_call('traffic_light')
                    traffic_light_ns = transform_shape(traffic_light_ns, translation_matrix((x_pos - road_width/2, 0, z_pos)))

                    traffic_light_ew = library_call('traffic_light')
                    traffic_light_ew = transform_shape(traffic_light_ew, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
                    traffic_light_ew = transform_shape(traffic_light_ew, translation_matrix((x_pos, 0, z_pos - road_width/2)))

                    district.append(traffic_light_ns)
                    district.append(traffic_light_ew)

    # Add trees around the district
    num_trees = int(size * 0.8)

    def tree_fn(i: int) -> Shape:
        # Place trees around the perimeter with even distribution
        side = i % 4
        pos_along_side = (i // 4) / max(1, (num_trees // 4))

        if side == 0:  # Top
            x_pos = -size/2 + pos_along_side * size
            z_pos = -size/2 - np.random.uniform(0.5, 1.5)
        elif side == 1:  # Right
            x_pos = size/2 + np.random.uniform(0.5, 1.5)
            z_pos = -size/2 + pos_along_side * size
        elif side == 2:  # Bottom
            x_pos = -size/2 + pos_along_side * size
            z_pos = size/2 + np.random.uniform(0.5, 1.5)
        else:  # Left
            x_pos = -size/2 - np.random.uniform(0.5, 1.5)
            z_pos = -size/2 + pos_along_side * size

        tree_height = np.random.uniform(1.5, 3.0)
        tree = library_call('tree', height=tree_height)
        return transform_shape(tree, translation_matrix((x_pos, 0, z_pos)))

    trees = loop(num_trees, tree_fn)
    district.append(trees)

    return concat_shapes(*district)

@register("Creates a complete city with multiple districts")
def city() -> Shape:
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a ground plane with subtle elevation
    ground_size = 100
    ground_height = 0.1
    ground_color = (0.3, 0.5, 0.3)

    ground = primitive_call('cube', shape_kwargs={'scale': (ground_size, ground_height, ground_size)}, color=ground_color)
    ground = transform_shape(ground, translation_matrix((0, -ground_height/2, 0)))

    # Create city districts
    districts = []

    # Central district with larger buildings (downtown)
    central_district = library_call('city_district', size=30, num_blocks=4, district_type='downtown')
    districts.append(central_district)

    # Surrounding districts with different types
    district_positions = [
        (35, 0, 0, 'commercial'),      # East
        (-35, 0, 0, 'residential'),    # West
        (0, 0, 35, 'commercial'),      # South
        (0, 0, -35, 'residential'),    # North
        (25, 0, 25, 'mixed'),          # Southeast
        (-25, 0, 25, 'residential'),   # Southwest
        (25, 0, -25, 'mixed'),         # Northeast
        (-25, 0, -25, 'residential')   # Northwest
    ]

    for pos in district_positions:
        district = library_call('city_district', size=20, num_blocks=3, district_type=pos[3])
        district = transform_shape(district, translation_matrix((pos[0], 0, pos[2])))
        districts.append(district)

    # Create a river through the city
    river_width = 5.0
    river_length = ground_size
    river_depth = 0.5
    river_color = (0.1, 0.4, 0.8)

    river = primitive_call('cube', shape_kwargs={'scale': (river_width, river_depth, river_length)}, color=river_color)
    river = transform_shape(river, translation_matrix((15, -ground_height/2, 0)))

    # Create bridges over the river
    bridge_width = 2.0
    bridge_height = 0.3
    bridge_length = river_width * 1.2
    bridge_color = (0.5, 0.5, 0.5)

    def bridge_fn(i: int) -> Shape:
        z_pos = -river_length/2 + river_length/(4+1) * (i+1)
        bridge = primitive_call('cube', shape_kwargs={'scale': (bridge_length, bridge_height, bridge_width)}, color=bridge_color)
        return transform_shape(bridge, translation_matrix((15, 0, z_pos)))

    bridges = loop(4, bridge_fn)

    return concat_shapes(ground, river, bridges, *districts)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
