

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

@register()
def building(width: float, height: float, depth: float, color: tuple = (0.7, 0.7, 0.7)) -> Shape:
    """Creates a building with specified dimensions and color"""
    return primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

@register()
def skyscraper(width: float, height: float, depth: float, color: tuple = (0.5, 0.5, 0.6)) -> Shape:
    """Creates a skyscraper with windows"""
    # Main building
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    all_shapes = [main_building]

    # Add windows
    window_width = width * 0.1
    window_height = width * 0.15
    window_depth = 0.01
    window_spacing_h = width / 8
    window_spacing_v = height / 15

    # Calculate how many windows can fit
    num_windows_h = max(1, int(width / window_spacing_h) - 1)
    num_windows_v = max(1, int(height / window_spacing_v) - 1)

    # Create windows on front face
    for i in range(num_windows_h):
        for j in range(num_windows_v):
            x_pos = (i - (num_windows_h - 1) / 2) * window_spacing_h
            y_pos = (j - (num_windows_v - 1) / 2) * window_spacing_v
            window = primitive_call('cube',
                                   shape_kwargs={'scale': (window_width, window_height, window_depth)},
                                   color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((x_pos, y_pos, depth/2 + window_depth/2)))
            all_shapes.append(window)

    # Create windows on back face
    for i in range(num_windows_h):
        for j in range(num_windows_v):
            x_pos = (i - (num_windows_h - 1) / 2) * window_spacing_h
            y_pos = (j - (num_windows_v - 1) / 2) * window_spacing_v
            window = primitive_call('cube',
                                   shape_kwargs={'scale': (window_width, window_height, window_depth)},
                                   color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((x_pos, y_pos, -depth/2 - window_depth/2)))
            all_shapes.append(window)

    # Add an antenna on top
    antenna_height = height * 0.2
    antenna = primitive_call('cylinder',
                           shape_kwargs={'radius': 0.05, 'p0': (0, height/2, 0), 'p1': (0, height/2 + antenna_height, 0)},
                           color=(0.3, 0.3, 0.3))
    all_shapes.append(antenna)

    return concat_shapes(*all_shapes)

@register()
def house(width: float, height: float, depth: float) -> Shape:
    """Creates a house with a roof"""
    # Main house
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=(0.8, 0.7, 0.6))

    # Roof
    roof_height = height * 0.5
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 1.2, roof_height, depth * 1.2)}, color=(0.6, 0.3, 0.2))

    # Position roof on top of house
    house_max = compute_shape_max(main_house)
    roof = transform_shape(roof, translation_matrix((0, house_max[1] + roof_height/2, 0)))

    # Add a door
    door_width = width * 0.2
    door_height = height * 0.4
    door_depth = 0.01
    door = primitive_call('cube', shape_kwargs={'scale': (door_width, door_height, door_depth)}, color=(0.4, 0.2, 0.1))

    # Position door at the bottom front of the house
    house_min = compute_shape_min(main_house)
    door = transform_shape(door, translation_matrix((0, house_min[1] + door_height/2, depth/2 + door_depth/2)))

    return concat_shapes(main_house, roof, door)

@register()
def road(length: float, width: float) -> Shape:
    """Creates a road segment"""
    road_base = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, length)}, color=(0.2, 0.2, 0.2))
    all_shapes = [road_base]

    # Add road markings
    marking_width = width * 0.05
    marking_length = length * 0.1
    marking_height = 0.01

    num_markings = int(length / (marking_length * 2))

    for i in range(num_markings):
        z_pos = (i - (num_markings - 1) / 2) * marking_length * 2
        marking = primitive_call('cube',
                               shape_kwargs={'scale': (marking_width, marking_height, marking_length)},
                               color=(1.0, 1.0, 1.0))
        marking = transform_shape(marking, translation_matrix((0, 0.05 + marking_height/2, z_pos)))
        all_shapes.append(marking)

    return concat_shapes(*all_shapes)

@register()
def street_lamp(height: float) -> Shape:
    """Creates a street lamp"""
    # Pole
    pole = primitive_call('cylinder',
                         shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, height, 0)},
                         color=(0.3, 0.3, 0.3))

    # Light bulb
    light = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 1.0, 0.8))
    light = transform_shape(light, translation_matrix((0, height + 0.15, 0)))

    # Light fixture - horizontal cylinder
    fixture = primitive_call('cylinder',
                           shape_kwargs={'radius': 0.1, 'p0': (-0.2, height, 0), 'p1': (0.2, height, 0)},
                           color=(0.2, 0.2, 0.2))

    return concat_shapes(pole, light, fixture)

@register()
def tree(height: float) -> Shape:
    """Creates a tree"""
    # Trunk
    trunk_height = height * 0.3
    trunk = primitive_call('cylinder',
                          shape_kwargs={'radius': 0.1, 'p0': (0, 0, 0), 'p1': (0, trunk_height, 0)},
                          color=(0.5, 0.3, 0.2))

    # Foliage
    foliage_radius = height * 0.2
    foliage = primitive_call('sphere', shape_kwargs={'radius': foliage_radius}, color=(0.1, 0.6, 0.1))
    foliage = transform_shape(foliage, translation_matrix((0, trunk_height + foliage_radius * 0.7, 0)))

    return concat_shapes(trunk, foliage)

@register()
def park_bench(width: float = 0.8) -> Shape:
    """Creates a park bench"""
    # Bench seat
    seat = primitive_call('cube', shape_kwargs={'scale': (width, 0.05, 0.3)}, color=(0.4, 0.2, 0.1))
    seat = transform_shape(seat, translation_matrix((0, 0.25, 0)))

    # Bench legs
    leg1 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.25, 0.3)}, color=(0.3, 0.3, 0.3))
    leg1 = transform_shape(leg1, translation_matrix((width/2 - 0.05, 0.125, 0)))

    leg2 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.25, 0.3)}, color=(0.3, 0.3, 0.3))
    leg2 = transform_shape(leg2, translation_matrix((-width/2 + 0.05, 0.125, 0)))

    # Bench back
    back = primitive_call('cube', shape_kwargs={'scale': (width, 0.3, 0.05)}, color=(0.4, 0.2, 0.1))
    back = transform_shape(back, translation_matrix((0, 0.4, -0.125)))

    return concat_shapes(seat, leg1, leg2, back)

@register()
def city_block(width: float, depth: float) -> Shape:
    """Creates a city block with buildings"""
    buildings_list = []

    # Number of buildings in each direction
    num_buildings_x = 3
    num_buildings_z = 3

    # Space between buildings
    spacing_x = width / num_buildings_x
    spacing_z = depth / num_buildings_z

    for i in range(num_buildings_x):
        for j in range(num_buildings_z):
            # Randomize building properties
            building_width = spacing_x * 0.7
            building_depth = spacing_z * 0.7
            building_height = np.random.uniform(1.0, 3.0)

            # Determine building type based on random value
            rand_val = np.random.random()
            if rand_val < 0.4:
                building_type = 'building'
            elif rand_val < 0.8:
                building_type = 'skyscraper'
            else:
                building_type = 'house'

            # Calculate position
            x_pos = (i - (num_buildings_x - 1) / 2) * spacing_x
            z_pos = (j - (num_buildings_z - 1) / 2) * spacing_z

            # Create building based on type
            if building_type == 'building':
                r = 0.5 + 0.3 * np.random.random()
                g = 0.5 + 0.3 * np.random.random()
                b = 0.5 + 0.3 * np.random.random()
                building = library_call('building', width=building_width, height=building_height, depth=building_depth, color=(r, g, b))
            elif building_type == 'skyscraper':
                building_height *= 2  # Skyscrapers are taller
                r = 0.4 + 0.2 * np.random.random()
                g = 0.4 + 0.2 * np.random.random()
                b = 0.5 + 0.2 * np.random.random()
                building = library_call('skyscraper', width=building_width, height=building_height, depth=building_depth, color=(r, g, b))
            else:  # house
                building_height *= 0.5  # Houses are shorter
                building = library_call('house', width=building_width, height=building_height, depth=building_depth)

            # Position the building
            building = transform_shape(building, translation_matrix((x_pos, building_height/2, z_pos)))
            buildings_list.append(building)

    return concat_shapes(*buildings_list)

@register()
def park(width: float, depth: float) -> Shape:
    """Creates a park with trees and benches"""
    # Base ground
    ground = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.2, 0.6, 0.2))
    all_shapes = [ground]

    # Add a pond in the center
    pond_radius = min(width, depth) * 0.2
    pond = primitive_call('cylinder',
                         shape_kwargs={'radius': pond_radius, 'p0': (0, 0.05, 0), 'p1': (0, 0.06, 0)},
                         color=(0.1, 0.4, 0.8))
    all_shapes.append(pond)

    # Add trees
    num_trees = 10
    placed_trees = 0
    attempts = 0
    while placed_trees < num_trees and attempts < 100:  # Prevent infinite loop
        attempts += 1
        x_pos = np.random.random() * (width - 1.0) - (width/2 - 0.5)
        z_pos = np.random.random() * (depth - 1.0) - (depth/2 - 0.5)

        # Avoid placing trees in the pond
        if x_pos*x_pos + z_pos*z_pos < pond_radius*pond_radius:
            continue

        tree_height = 0.8 + 0.7 * np.random.random()
        tree = library_call('tree', height=tree_height)
        tree = transform_shape(tree, translation_matrix((x_pos, 0.05, z_pos)))
        all_shapes.append(tree)
        placed_trees += 1

    # Add benches
    num_benches = 4
    for i in range(num_benches):
        angle = i * (2 * math.pi / num_benches)
        x_pos = (pond_radius + 0.5) * math.cos(angle)
        z_pos = (pond_radius + 0.5) * math.sin(angle)

        bench = library_call('park_bench')
        # Rotate bench to face the pond
        bench = transform_shape(bench, rotation_matrix(angle + math.pi, (0, 1, 0), (0, 0, 0)))
        bench = transform_shape(bench, translation_matrix((x_pos, 0.05, z_pos)))
        all_shapes.append(bench)

    # Add paths
    path_width = 0.5
    for i in range(4):
        angle = i * (math.pi / 2)
        path_length = max(width, depth) / 2

        path = primitive_call('cube',
                             shape_kwargs={'scale': (path_width, 0.02, path_length)},
                             color=(0.8, 0.7, 0.6))
        path = transform_shape(path, rotation_matrix(angle, (0, 1, 0), (0, 0, 0)))
        path = transform_shape(path, translation_matrix((0, 0.06, 0)))
        all_shapes.append(path)

    return concat_shapes(*all_shapes)

@register()
def landmark_building() -> Shape:
    """Creates a landmark building"""
    # Base
    base_width = 3.0
    base_height = 1.0
    base_depth = 3.0
    base = primitive_call('cube', shape_kwargs={'scale': (base_width, base_height, base_depth)}, color=(0.7, 0.7, 0.8))

    # Tower
    tower_width = 1.5
    tower_height = 6.0
    tower_depth = 1.5
    tower = primitive_call('cube', shape_kwargs={'scale': (tower_width, tower_height, tower_depth)}, color=(0.6, 0.6, 0.7))
    tower = transform_shape(tower, translation_matrix((0, base_height/2 + tower_height/2, 0)))

    # Spire
    spire_height = 2.0
    spire = primitive_call('cylinder',
                          shape_kwargs={'radius': 0.1, 'p0': (0, base_height/2 + tower_height, 0),
                                       'p1': (0, base_height/2 + tower_height + spire_height, 0)},
                          color=(0.8, 0.8, 0.9))

    # Windows
    windows = []
    window_size = 0.2
    for i in range(5):  # 5 floors of windows
        for j in range(-1, 2):  # 3 windows per side
            # Front windows
            window = primitive_call('cube',
                                  shape_kwargs={'scale': (window_size, window_size, 0.05)},
                                  color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((j*0.4, base_height/2 + i*1.0 + 0.5, tower_depth/2)))
            windows.append(window)

            # Back windows
            window = primitive_call('cube',
                                  shape_kwargs={'scale': (window_size, window_size, 0.05)},
                                  color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((j*0.4, base_height/2 + i*1.0 + 0.5, -tower_depth/2)))
            windows.append(window)

            # Side windows
            window = primitive_call('cube',
                                  shape_kwargs={'scale': (0.05, window_size, window_size)},
                                  color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((tower_width/2, base_height/2 + i*1.0 + 0.5, j*0.4)))
            windows.append(window)

            window = primitive_call('cube',
                                  shape_kwargs={'scale': (0.05, window_size, window_size)},
                                  color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((-tower_width/2, base_height/2 + i*1.0 + 0.5, j*0.4)))
            windows.append(window)

    return concat_shapes(base, tower, spire, *windows)

@register()
def city_district(size: float, seed: int = None) -> Shape:
    """Creates a city district with blocks, roads, parks, and landmarks"""
    if seed is not None:
        np.random.seed(seed)

    district_elements = []

    # Create a grid of city blocks and roads
    grid_size = 3
    block_size = size / grid_size
    road_width = block_size * 0.2  # Reduced from 0.3 for better proportions

    # Create city blocks
    for i in range(grid_size):
        for j in range(grid_size):
            # Center block is a landmark or plaza
            if i == grid_size // 2 and j == grid_size // 2:
                landmark = library_call('landmark_building')
                landmark = transform_shape(landmark, translation_matrix((0, 0, 0)))
                district_elements.append(landmark)
                continue

            # Determine if this should be a park (20% chance)
            is_park = np.random.random() < 0.2

            x_pos = (i - (grid_size - 1) / 2) * block_size
            z_pos = (j - (grid_size - 1) / 2) * block_size

            if is_park:
                block = library_call('park', width=block_size * 0.8, depth=block_size * 0.8)
            else:
                block = library_call('city_block', width=block_size * 0.8, depth=block_size * 0.8)

            block = transform_shape(block, translation_matrix((x_pos, 0, z_pos)))
            district_elements.append(block)

    # Create horizontal roads
    for i in range(grid_size + 1):
        z_pos = (i - grid_size / 2) * block_size
        road = library_call('road', length=size, width=road_width)
        road = transform_shape(road, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        road = transform_shape(road, translation_matrix((0, 0, z_pos)))
        district_elements.append(road)

    # Create vertical roads
    for i in range(grid_size + 1):
        x_pos = (i - grid_size / 2) * block_size
        road = library_call('road', length=size, width=road_width)
        road = transform_shape(road, translation_matrix((x_pos, 0, 0)))
        district_elements.append(road)

    # Add street lamps along roads
    lamp_height = 0.5  # Reduced from 1.0 for better proportions
    for i in range(grid_size + 1):
        for j in range(5):  # 5 lamps per road
            # Horizontal roads
            z_pos = (i - grid_size / 2) * block_size
            x_pos = (j - 2) * (size / 4)
            lamp = library_call('street_lamp', height=lamp_height)
            lamp = transform_shape(lamp, translation_matrix((x_pos, 0, z_pos + road_width/3)))
            district_elements.append(lamp)

            # Vertical roads
            x_pos = (i - grid_size / 2) * block_size
            z_pos = (j - 2) * (size / 4)
            lamp = library_call('street_lamp', height=lamp_height)
            lamp = transform_shape(lamp, translation_matrix((x_pos + road_width/3, 0, z_pos)))
            district_elements.append(lamp)

    return concat_shapes(*district_elements)

@register()
def highway(length: float, width: float) -> Shape:
    """Creates a highway segment"""
    # Main road
    highway_base = primitive_call('cube', shape_kwargs={'scale': (width, 0.2, length)}, color=(0.3, 0.3, 0.3))

    # Divider
    divider_width = 0.2
    divider = primitive_call('cube', shape_kwargs={'scale': (divider_width, 0.25, length)}, color=(0.7, 0.7, 0.7))

    # Road markings
    markings = []
    marking_width = 0.1
    marking_length = length * 0.05
    marking_spacing = length * 0.1
    num_markings = int(length / (marking_length + marking_spacing))

    for i in range(num_markings):
        z_pos = (i - (num_markings - 1) / 2) * (marking_length + marking_spacing)

        # Left lane marking
        left_marking = primitive_call('cube',
                                    shape_kwargs={'scale': (marking_width, 0.01, marking_length)},
                                    color=(1.0, 1.0, 0.0))
        left_marking = transform_shape(left_marking, translation_matrix((-width/4, 0.11, z_pos)))
        markings.append(left_marking)

        # Right lane marking
        right_marking = primitive_call('cube',
                                     shape_kwargs={'scale': (marking_width, 0.01, marking_length)},
                                     color=(1.0, 1.0, 0.0))
        right_marking = transform_shape(right_marking, translation_matrix((width/4, 0.11, z_pos)))
        markings.append(right_marking)

    # Guardrails
    left_rail = primitive_call('cube', shape_kwargs={'scale': (0.1, 0.3, length)}, color=(0.6, 0.6, 0.6))
    left_rail = transform_shape(left_rail, translation_matrix((-width/2 - 0.1, 0.15, 0)))

    right_rail = primitive_call('cube', shape_kwargs={'scale': (0.1, 0.3, length)}, color=(0.6, 0.6, 0.6))
    right_rail = transform_shape(right_rail, translation_matrix((width/2 + 0.1, 0.15, 0)))

    return concat_shapes(highway_base, divider, left_rail, right_rail, *markings)

@register()
def city(seed: int = 42) -> Shape:
    """Creates a complete city with multiple districts and highways"""
    np.random.seed(seed)
    city_elements = []

    # Create multiple districts with different seeds
    district_size = 10
    district_spacing = 11  # Reduced spacing to make city more continuous

    for i in range(-1, 2):
        for j in range(-1, 2):
            district = library_call('city_district', size=district_size, seed=seed + i*10 + j)
            district = transform_shape(district, translation_matrix((i * district_spacing, 0, j * district_spacing)))
            city_elements.append(district)

    # Add highways connecting districts
    highway_width = 2.0  # Reduced from 3.0 for better proportions

    # East-West highways
    for j in range(-1, 2):
        highway_segment = library_call('highway', length=district_spacing*3, width=highway_width)
        highway_segment = transform_shape(highway_segment, translation_matrix((0, 0, j * district_spacing - highway_width/2)))
        city_elements.append(highway_segment)

    # North-South highways
    for i in range(-1, 2):
        highway_segment = library_call('highway', length=district_spacing*3, width=highway_width)
        highway_segment = transform_shape(highway_segment, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        highway_segment = transform_shape(highway_segment, translation_matrix((i * district_spacing - highway_width/2, 0, 0)))
        city_elements.append(highway_segment)

    return concat_shapes(*city_elements)

def main() -> Shape:
    """Main function that returns the complete city"""
    return library_call('city')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
