

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
    return primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

@register("Creates a skyscraper with windows")
def skyscraper(width: float, height: float, depth: float, color: tuple = (0.5, 0.5, 0.6)) -> Shape:
    # Main building
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

    # Add windows with a reasonable limit
    window_width = width * 0.15
    window_height = height * 0.05
    window_depth = 0.01
    window_color = (0.9, 0.9, 0.7)

    # Calculate number of windows per side with limits to prevent excessive windows
    windows_per_row = min(max(int(width / (window_width * 1.5)), 2), 5)
    windows_per_column = min(max(int(height / (window_height * 1.5)), 4), 10)

    def window_loop_fn(i):
        row = i % windows_per_row
        col = (i // windows_per_row) % windows_per_column
        side = i // (windows_per_row * windows_per_column)

        if side < 4:  # 4 sides of the building
            window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)

            # Position for each side
            if side == 0:  # Front
                x_pos = (row - (windows_per_row - 1) / 2) * (width / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                z_pos = depth / 2 + window_depth / 2
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
            elif side == 1:  # Back
                x_pos = (row - (windows_per_row - 1) / 2) * (width / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                z_pos = -depth / 2 - window_depth / 2
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
            elif side == 2:  # Left
                z_pos = (row - (windows_per_row - 1) / 2) * (depth / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                x_pos = -width / 2 - window_depth / 2
                window = transform_shape(window, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
            elif side == 3:  # Right
                z_pos = (row - (windows_per_row - 1) / 2) * (depth / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                x_pos = width / 2 + window_depth / 2
                window = transform_shape(window, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))

            return window
        return []

    total_windows = windows_per_row * windows_per_column * 4
    windows = loop(total_windows, window_loop_fn)

    # Add roof structure
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 0.5, height * 0.1, depth * 0.5)}, color=(0.4, 0.4, 0.5))
    roof = transform_shape(roof, translation_matrix((0, height / 2 + height * 0.05, 0)))

    antenna = primitive_call('cylinder', shape_kwargs={'radius': width * 0.02, 'p0': (0, height / 2 + height * 0.1, 0), 'p1': (0, height / 2 + height * 0.3, 0)}, color=(0.3, 0.3, 0.3))

    return concat_shapes(main_building, windows, roof, antenna)

@register("Creates a house with a roof")
def house(width: float, height: float, depth: float, color: tuple = (0.8, 0.7, 0.6)) -> Shape:
    # Main house
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

    # Improved roof implementation
    roof_height = height * 0.4
    roof_color = (0.6, 0.3, 0.2)

    # Create a triangular roof using two rectangular prisms
    roof_left = primitive_call('cube', shape_kwargs={'scale': (width, roof_height, depth)}, color=roof_color)
    roof_right = primitive_call('cube', shape_kwargs={'scale': (width, roof_height, depth)}, color=roof_color)

    # Position and rotate the roof parts
    roof_left = transform_shape(roof_left, rotation_matrix(math.pi/4, (0, 0, 1), (0, 0, 0)))
    roof_left = transform_shape(roof_left, translation_matrix((-width/4, height/2 + roof_height/2, 0)))

    roof_right = transform_shape(roof_right, rotation_matrix(-math.pi/4, (0, 0, 1), (0, 0, 0)))
    roof_right = transform_shape(roof_right, translation_matrix((width/4, height/2 + roof_height/2, 0)))

    # Door
    door_width = width * 0.3
    door_height = height * 0.6
    door_depth = 0.01
    door = primitive_call('cube', shape_kwargs={'scale': (door_width, door_height, door_depth)}, color=(0.4, 0.2, 0.1))
    door = transform_shape(door, translation_matrix((0, -height/2 + door_height/2, depth/2 + door_depth/2)))

    # Windows
    window_width = width * 0.2
    window_height = height * 0.2
    window_depth = 0.01
    window_color = (0.9, 0.9, 1.0)

    window1 = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
    window1 = transform_shape(window1, translation_matrix((-width/4, 0, depth/2 + window_depth/2)))

    window2 = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
    window2 = transform_shape(window2, translation_matrix((width/4, 0, depth/2 + window_depth/2)))

    return concat_shapes(main_house, roof_left, roof_right, door, window1, window2)

@register("Creates a road segment with realistic markings")
def road(length: float, width: float = 1.0, color: tuple = (0.2, 0.2, 0.2)) -> Shape:
    road_height = 0.05
    road_shape = primitive_call('cube', shape_kwargs={'scale': (width, road_height, length)}, color=color)

    # Add road markings with more realistic spacing
    marking_width = width * 0.05
    marking_length = length * 0.05
    marking_color = (1.0, 1.0, 1.0)
    marking_gap = length * 0.07  # Gap between markings

    def marking_loop_fn(i):
        # Calculate position with gaps between markings
        z_pos = (i - 4.5) * (marking_length + marking_gap)

        # Add some randomness to make it more realistic
        z_pos += np.random.uniform(-length * 0.01, length * 0.01)

        marking = primitive_call('cube', shape_kwargs={'scale': (marking_width, road_height * 1.01, marking_length)}, color=marking_color)
        return transform_shape(marking, translation_matrix((0, 0, z_pos)))

    markings = loop(10, marking_loop_fn)

    return concat_shapes(road_shape, markings)

@register("Creates a park with trees and paths")
def park(width: float, depth: float) -> Shape:
    # Ground
    ground = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.2, 0.6, 0.2))

    # Add paths
    path_width = width * 0.15
    path_color = (0.8, 0.7, 0.6)

    # Horizontal path
    h_path = primitive_call('cube', shape_kwargs={'scale': (width, 0.12, path_width)}, color=path_color)

    # Vertical path
    v_path = primitive_call('cube', shape_kwargs={'scale': (path_width, 0.12, depth)}, color=path_color)

    # Add a central circular area
    center_radius = min(width, depth) * 0.15
    center_area = primitive_call('cylinder',
                                shape_kwargs={'radius': center_radius,
                                             'p0': (0, 0, 0),
                                             'p1': (0, 0.12, 0)},
                                color=path_color)
    center_area = transform_shape(center_area, rotation_matrix(math.pi/2, (1, 0, 0), (0, 0, 0)))

    def tree_loop_fn(i):
        # Calculate position
        row = i % 5
        col = i // 5
        x_pos = (row - 2) * (width / 5)
        z_pos = (col - 2) * (depth / 5)

        # Add some randomness to positions
        x_pos += np.random.uniform(-width/12, width/12)
        z_pos += np.random.uniform(-depth/12, depth/12)

        # Skip trees that would be on the paths
        if abs(x_pos) < path_width/2 or abs(z_pos) < path_width/2:
            return []

        # Skip trees that would be in the central area
        if (x_pos**2 + z_pos**2) < center_radius**2:
            return []

        # Create tree trunk - ensure it starts at the ground level
        trunk_height = np.random.uniform(0.3, 0.6)
        trunk_radius = np.random.uniform(0.05, 0.1)
        trunk = primitive_call('cylinder',
                              shape_kwargs={'radius': trunk_radius,
                                           'p0': (x_pos, 0.1, z_pos),  # Start at ground level
                                           'p1': (x_pos, 0.1 + trunk_height, z_pos)},
                              color=(0.5, 0.3, 0.1))

        # Create tree foliage
        foliage_radius = np.random.uniform(0.3, 0.5)
        foliage = primitive_call('sphere',
                                shape_kwargs={'radius': foliage_radius},
                                color=(0.0, np.random.uniform(0.5, 0.8), 0.0))
        foliage = transform_shape(foliage, translation_matrix((x_pos, 0.1 + trunk_height + foliage_radius * 0.7, z_pos)))

        # Add a bench near some trees (randomly)
        if np.random.random() < 0.2:  # 20% chance of adding a bench
            bench_seat = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.05, 0.2)}, color=(0.6, 0.4, 0.2))
            bench_back = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.2, 0.05)}, color=(0.6, 0.4, 0.2))

            bench_seat = transform_shape(bench_seat, translation_matrix((x_pos + 0.3, 0.15, z_pos)))
            bench_back = transform_shape(bench_back, translation_matrix((x_pos + 0.3, 0.25, z_pos - 0.075)))

            return concat_shapes(trunk, foliage, bench_seat, bench_back)

        return concat_shapes(trunk, foliage)

    trees = loop(25, tree_loop_fn)

    return concat_shapes(ground, h_path, v_path, center_area, trees)

@register("Creates a traffic light")
def traffic_light() -> Shape:
    # Pole
    pole = primitive_call('cylinder',
                         shape_kwargs={'radius': 0.05,
                                      'p0': (0, 0, 0),
                                      'p1': (0, 1.5, 0)},
                         color=(0.3, 0.3, 0.3))

    # Light housing
    housing = primitive_call('cube', shape_kwargs={'scale': (0.2, 0.5, 0.2)}, color=(0.1, 0.1, 0.1))
    housing = transform_shape(housing, translation_matrix((0, 1.5, 0)))

    # Lights
    red_light = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(1.0, 0.0, 0.0))
    red_light = transform_shape(red_light, translation_matrix((0, 1.65, 0.1)))

    yellow_light = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(1.0, 1.0, 0.0))
    yellow_light = transform_shape(yellow_light, translation_matrix((0, 1.5, 0.1)))

    green_light = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 1.0, 0.0))
    green_light = transform_shape(green_light, translation_matrix((0, 1.35, 0.1)))

    return concat_shapes(pole, housing, red_light, yellow_light, green_light)

@register("Creates a city block with buildings")
def city_block(width: float, depth: float, is_downtown: bool = False) -> Shape:
    # Create the base
    base = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.6, 0.6, 0.6))

    # Create a grid for building placement to prevent overlap
    grid_size = 3
    cell_width = width / grid_size
    cell_depth = depth / grid_size

    def building_loop_fn(i):
        # Calculate position using grid
        row = i % grid_size
        col = i // grid_size

        # Calculate center of the grid cell
        x_pos = (row - (grid_size-1)/2) * cell_width
        z_pos = (col - (grid_size-1)/2) * cell_depth

        # Add small randomness within the cell to avoid perfect alignment
        x_pos += np.random.uniform(-cell_width * 0.1, cell_width * 0.1)
        z_pos += np.random.uniform(-cell_depth * 0.1, cell_depth * 0.1)

        # Determine building size to fit within the cell
        building_width = cell_width * np.random.uniform(0.6, 0.8)
        building_depth = cell_depth * np.random.uniform(0.6, 0.8)

        # Adjust height based on whether it's downtown (taller) or suburbs
        if is_downtown:
            building_height = np.random.uniform(2.0, 5.0)
            # Higher chance of skyscrapers downtown
            building_type_probs = [0.7, 0.25, 0.05]  # [skyscraper, building, house]
        else:
            building_height = np.random.uniform(1.0, 2.5)
            # Higher chance of houses in suburbs
            building_type_probs = [0.1, 0.4, 0.5]  # [skyscraper, building, house]

        # Randomly choose between different building types based on probabilities
        building_type = np.random.choice(['skyscraper', 'building', 'house'], p=building_type_probs)

        if building_type == 'skyscraper':
            building_shape = library_call('skyscraper',
                                         width=building_width,
                                         height=building_height * 2,
                                         depth=building_depth,
                                         color=(np.random.uniform(0.4, 0.6), np.random.uniform(0.4, 0.6), np.random.uniform(0.5, 0.7)))
        elif building_type == 'building':
            building_shape = library_call('building',
                                         width=building_width,
                                         height=building_height,
                                         depth=building_depth,
                                         color=(np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8)))
        else:
            building_shape = library_call('house',
                                         width=building_width,
                                         height=building_height * 0.7,
                                         depth=building_depth,
                                         color=(np.random.uniform(0.7, 0.9), np.random.uniform(0.6, 0.8), np.random.uniform(0.5, 0.7)))

        # Position the building on top of the base
        building_bottom = compute_shape_min(building_shape)[1]
        base_top = compute_shape_max(base)[1]
        building_shape = transform_shape(building_shape, translation_matrix((x_pos, base_top - building_bottom, z_pos)))

        return building_shape

    buildings = loop(grid_size * grid_size, building_loop_fn)

    return concat_shapes(base, buildings)

@register("Creates a city intersection with traffic lights")
def intersection(road_width: float = 1.0) -> Shape:
    # Create the intersection base
    intersection_base = primitive_call('cube',
                                      shape_kwargs={'scale': (road_width * 2, 0.05, road_width * 2)},
                                      color=(0.2, 0.2, 0.2))

    # Add traffic lights at the corners
    traffic_light1 = library_call('traffic_light')
    traffic_light1 = transform_shape(traffic_light1,
                                    translation_matrix((road_width * 0.8, 0, road_width * 0.8)))

    traffic_light2 = library_call('traffic_light')
    traffic_light2 = transform_shape(traffic_light2,
                                    translation_matrix((-road_width * 0.8, 0, road_width * 0.8)))

    traffic_light3 = library_call('traffic_light')
    traffic_light3 = transform_shape(traffic_light3,
                                    translation_matrix((road_width * 0.8, 0, -road_width * 0.8)))

    traffic_light4 = library_call('traffic_light')
    traffic_light4 = transform_shape(traffic_light4,
                                    translation_matrix((-road_width * 0.8, 0, -road_width * 0.8)))

    # Add crosswalk markings
    crosswalk_color = (1.0, 1.0, 1.0)

    def crosswalk_marking_fn(i):
        # Create horizontal crosswalk markings
        if i < 5:
            marking = primitive_call('cube',
                                    shape_kwargs={'scale': (0.2, 0.06, road_width * 0.1)},
                                    color=crosswalk_color)
            x_pos = (i - 2) * 0.25
            return transform_shape(marking, translation_matrix((x_pos, 0, road_width * 0.6)))
        # Create vertical crosswalk markings
        else:
            marking = primitive_call('cube',
                                    shape_kwargs={'scale': (road_width * 0.1, 0.06, 0.2)},
                                    color=crosswalk_color)
            z_pos = ((i - 5) - 2) * 0.25
            return transform_shape(marking, translation_matrix((road_width * 0.6, 0, z_pos)))

    crosswalk_markings = loop(10, crosswalk_marking_fn)

    return concat_shapes(intersection_base, traffic_light1, traffic_light2,
                         traffic_light3, traffic_light4, crosswalk_markings)

@register("Creates a city grid with blocks and roads")
def city_grid(size: int = 5, block_size: float = 4.0, road_width: float = 1.0, is_downtown: bool = False) -> Shape:
    city = []

    # Calculate total unit size (block + road)
    total_unit_size = block_size + road_width

    # Create city blocks
    def block_loop_fn(i):
        row = i % size
        col = i // size

        # Calculate position with roads in between
        x_pos = (row - (size-1)/2) * total_unit_size
        z_pos = (col - (size-1)/2) * total_unit_size

        # Randomly choose between regular block and park
        if is_downtown:
            park_chance = 0.1  # Less parks downtown
        else:
            park_chance = 0.2  # More parks in suburbs

        if np.random.random() < park_chance:
            block = library_call('park', width=block_size, depth=block_size)
        else:
            block = library_call('city_block', width=block_size, depth=block_size, is_downtown=is_downtown)

        return transform_shape(block, translation_matrix((x_pos, 0, z_pos)))

    blocks = loop(size * size, block_loop_fn)
    city.append(blocks)

    # Create horizontal roads
    def h_road_loop_fn(i):
        row = i % size
        col = i // size

        total_unit_size = block_size + road_width
        x_pos = (row - (size-1)/2) * total_unit_size
        z_pos = (col - (size-1)/2) * total_unit_size + block_size/2 + road_width/2

        road = library_call('road', length=block_size, width=road_width)
        return transform_shape(road, translation_matrix((x_pos, 0, z_pos)))

    h_roads = loop(size * (size-1), h_road_loop_fn)
    city.append(h_roads)

    # Create vertical roads
    def v_road_loop_fn(i):
        row = i % (size-1)
        col = i // (size-1)

        total_unit_size = block_size + road_width
        x_pos = (row - (size-1)/2) * total_unit_size + block_size/2 + road_width/2
        z_pos = (col - (size-1)/2) * total_unit_size

        road = library_call('road', length=block_size, width=road_width)
        road = transform_shape(road, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        return transform_shape(road, translation_matrix((x_pos, 0, z_pos)))

    v_roads = loop((size-1) * size, v_road_loop_fn)
    city.append(v_roads)

    # Create intersections
    def intersection_loop_fn(i):
        row = i % (size-1)
        col = i // (size-1)

        total_unit_size = block_size + road_width
        x_pos = (row - (size-1)/2) * total_unit_size + block_size/2 + road_width/2
        z_pos = (col - (size-1)/2) * total_unit_size + block_size/2 + road_width/2

        intersection_shape = library_call('intersection', road_width=road_width)
        return transform_shape(intersection_shape, translation_matrix((x_pos, 0, z_pos)))

    intersections = loop((size-1) * (size-1), intersection_loop_fn)
    city.append(intersections)

    return concat_shapes(*city)

@register("Creates a parking lot")
def parking_lot(width: float, depth: float) -> Shape:
    # Base
    base = primitive_call('cube', shape_kwargs={'scale': (width, 0.05, depth)}, color=(0.3, 0.3, 0.3))

    # Parking spaces
    space_width = 0.8
    space_depth = 1.6
    spacing = 0.1

    def parking_space_fn(i):
        rows_per_side = int((depth - 2) / (space_depth + spacing))
        spaces_per_row = int((width - 2) / (space_width + spacing))

        row = i % rows_per_side
        col = (i // rows_per_side) % spaces_per_row
        side = i // (rows_per_side * spaces_per_row)

        if side < 2:  # Two sides of parking spaces
            # Create parking space
            space = primitive_call('cube', shape_kwargs={'scale': (space_width, 0.06, space_depth)}, color=(0.25, 0.25, 0.25))

            # Add parking line markings
            line1 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.07, space_depth)}, color=(1.0, 1.0, 1.0))
            line2 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.07, space_depth)}, color=(1.0, 1.0, 1.0))

            line1 = transform_shape(line1, translation_matrix((space_width/2, 0, 0)))
            line2 = transform_shape(line2, translation_matrix((-space_width/2, 0, 0)))

            space_with_lines = concat_shapes(space, line1, line2)

            # Position based on side
            if side == 0:  # Left side
                x_pos = -width/2 + 1 + space_width/2 + col * (space_width + spacing)
                z_pos = -depth/2 + 1 + space_depth/2 + row * (space_depth + spacing)
            else:  # Right side
                x_pos = -width/2 + 1 + space_width/2 + col * (space_width + spacing)
                z_pos = depth/2 - 1 - space_depth/2 - row * (space_depth + spacing)

            return transform_shape(space_with_lines, translation_matrix((x_pos, 0, z_pos)))
        return []

    # Calculate total number of parking spaces
    rows_per_side = int((depth - 2) / (space_depth + spacing))
    spaces_per_row = int((width - 2) / (space_width + spacing))
    total_spaces = rows_per_side * spaces_per_row * 2

    parking_spaces = loop(total_spaces, parking_space_fn)

    # Central driving lane
    lane = primitive_call('cube', shape_kwargs={'scale': (width - 2, 0.06, depth - 2 - 2*space_depth - 2*spacing)}, color=(0.2, 0.2, 0.2))

    return concat_shapes(base, parking_spaces, lane)

@register("Creates a complete city with downtown and suburbs")
def large_scale_city() -> Shape:
    # Create ground
    ground = primitive_call('cube', shape_kwargs={'scale': (50, 0.1, 50)}, color=(0.3, 0.3, 0.3))
    ground = transform_shape(ground, translation_matrix((0, -0.05, 0)))  # Position at y=-0.05 so top is at y=0

    # Create downtown area with tall buildings
    downtown_size = 3
    downtown_block_size = 5.0
    downtown_road_width = 1.2
    downtown = library_call('city_grid', size=downtown_size, block_size=downtown_block_size, road_width=downtown_road_width, is_downtown=True)

    # Calculate the total size of downtown for proper positioning of suburbs
    downtown_total_size = downtown_size * downtown_block_size + (downtown_size - 1) * downtown_road_width

    # Create suburban areas with proper positioning
    suburb_size = 2
    suburb_block_size = 6.0
    suburb_road_width = 1.0
    suburb_total_size = suburb_size * suburb_block_size + (suburb_size - 1) * suburb_road_width

    # Calculate positions for suburbs to connect with downtown
    suburb_offset = (downtown_total_size + suburb_total_size) / 2

    suburb1 = library_call('city_grid', size=suburb_size, block_size=suburb_block_size, road_width=suburb_road_width)
    suburb1 = transform_shape(suburb1, translation_matrix((suburb_offset, 0, suburb_offset)))

    suburb2 = library_call('city_grid', size=suburb_size, block_size=suburb_block_size, road_width=suburb_road_width)
    suburb2 = transform_shape(suburb2, translation_matrix((-suburb_offset, 0, suburb_offset)))

    suburb3 = library_call('city_grid', size=suburb_size, block_size=suburb_block_size, road_width=suburb_road_width)
    suburb3 = transform_shape(suburb3, translation_matrix((suburb_offset, 0, -suburb_offset)))

    suburb4 = library_call('city_grid', size=suburb_size, block_size=suburb_block_size, road_width=suburb_road_width)
    suburb4 = transform_shape(suburb4, translation_matrix((-suburb_offset, 0, -suburb_offset)))

    # Create connecting roads with proper lengths
    # Calculate the gap between downtown and suburbs
    gap_size = suburb_offset - downtown_total_size/2

    # Horizontal connecting roads
    road1 = library_call('road', length=gap_size, width=1.5)
    road1 = transform_shape(road1, translation_matrix((downtown_total_size/2 + gap_size/2, 0, 0)))

    road2 = library_call('road', length=gap_size, width=1.5)
    road2 = transform_shape(road2, translation_matrix((-downtown_total_size/2 - gap_size/2, 0, 0)))

    # Vertical connecting roads
    road3 = library_call('road', length=gap_size, width=1.5)
    road3 = transform_shape(road3, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    road3 = transform_shape(road3, translation_matrix((0, 0, downtown_total_size/2 + gap_size/2)))

    road4 = library_call('road', length=gap_size, width=1.5)
    road4 = transform_shape(road4, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    road4 = transform_shape(road4, translation_matrix((0, 0, -downtown_total_size/2 - gap_size/2)))

    # Add parking lots near suburbs
    parking1 = library_call('parking_lot', width=4.0, depth=6.0)
    parking1 = transform_shape(parking1, translation_matrix((suburb_offset - suburb_total_size/2 - 3, 0, suburb_offset)))

    parking2 = library_call('parking_lot', width=4.0, depth=6.0)
    parking2 = transform_shape(parking2, translation_matrix((-suburb_offset + suburb_total_size/2 + 3, 0, -suburb_offset)))

    return concat_shapes(ground, downtown, suburb1, suburb2, suburb3, suburb4,
                         road1, road2, road3, road4, parking1, parking2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
