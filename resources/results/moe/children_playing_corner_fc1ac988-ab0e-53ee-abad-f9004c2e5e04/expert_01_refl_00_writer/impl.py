

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
children playing corner
"""

@register()
def toy_block(scale: P, color: tuple[float, float, float] = (1.0, 0.5, 0.0)) -> Shape:
    """Create a toy block with given scale and color"""
    return primitive_call('cube', color=color, shape_kwargs={'scale': scale})

@register()
def ball(radius: float, color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> Shape:
    """Create a ball with given radius and color"""
    return primitive_call('sphere', color=color, shape_kwargs={'radius': radius})

@register()
def toy_blocks_stack(base_size: float, height: float, num_blocks: int) -> Shape:
    """Create a stack of toy blocks with random colors and slight offsets"""
    colors = [(1.0, 0.5, 0.0), (0.0, 0.7, 0.3), (0.3, 0.3, 1.0), (1.0, 0.8, 0.0), (0.8, 0.2, 0.8)]

    def loop_fn(i) -> Shape:
        # Randomize block size slightly
        size_variation = np.random.uniform(0.8, 1.0)
        block_size = (base_size * size_variation, height, base_size * size_variation)

        # Select random color
        color = colors[i % len(colors)]

        # Create block
        block = library_call('toy_block', scale=block_size, color=color)

        # Add random offset and rotation
        offset_x = np.random.uniform(-0.05, 0.05)
        offset_z = np.random.uniform(-0.05, 0.05)
        y_pos = i * height

        # Transform block
        block = transform_shape(block, translation_matrix([offset_x, y_pos, offset_z]))
        block_center = compute_shape_center(block)
        rotation_angle = np.random.uniform(-0.2, 0.2)
        return transform_shape(block, rotation_matrix(rotation_angle, direction=(0, 1, 0), point=block_center))

    return loop(num_blocks, loop_fn)

@register()
def toy_balls_pile(num_balls: int, radius_range: tuple[float, float] = (0.05, 0.1)) -> Shape:
    """Create a pile of colorful balls with random sizes and positions"""
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
              (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]

    def loop_fn(i) -> Shape:
        # Random radius
        radius = np.random.uniform(radius_range[0], radius_range[1])

        # Random color
        color = colors[i % len(colors)]

        # Create ball
        ball = library_call('ball', radius=radius, color=color)

        # Random position within a circular area
        angle = np.random.uniform(0, 2 * math.pi)
        distance = np.random.uniform(0, 0.2)
        x = distance * math.cos(angle)
        z = distance * math.sin(angle)
        y = radius + np.random.uniform(0, 0.05) * i  # Stack with some randomness

        return transform_shape(ball, translation_matrix([x, y, z]))

    return loop(num_balls, loop_fn)

@register()
def toy_train(length: float, color: tuple[float, float, float] = (0.7, 0.0, 0.0)) -> Shape:
    """Create a simple toy train with engine and cars"""
    # Engine body
    engine_body = primitive_call('cube', color=color, shape_kwargs={'scale': (0.15, 0.1, 0.2)})

    # Engine cabin
    engine_cabin = primitive_call('cube', color=(0.3, 0.3, 0.3),
                                 shape_kwargs={'scale': (0.12, 0.08, 0.1)})
    engine_cabin = transform_shape(engine_cabin, translation_matrix([0, 0.09, -0.05]))

    # Wheels
    def create_wheel(x: float, z: float) -> Shape:
        wheel = primitive_call('cylinder', color=(0.2, 0.2, 0.2),
                              shape_kwargs={'radius': 0.03, 'p0': (x, 0, z), 'p1': (x, 0.03, z)})
        return wheel

    wheels = concat_shapes(
        create_wheel(-0.05, -0.08),
        create_wheel(-0.05, 0.08),
        create_wheel(0.05, -0.08),
        create_wheel(0.05, 0.08)
    )

    # Smokestack
    smokestack = primitive_call('cylinder', color=(0.3, 0.3, 0.3),
                               shape_kwargs={'radius': 0.02, 'p0': (0, 0.1, -0.07), 'p1': (0, 0.18, -0.07)})

    # Combine engine parts
    engine = concat_shapes(engine_body, engine_cabin, wheels, smokestack)

    # Create cars
    def create_car(position: float) -> Shape:
        car_color = (np.random.uniform(0.3, 0.9), np.random.uniform(0.3, 0.9), np.random.uniform(0.3, 0.9))
        car_body = primitive_call('cube', color=car_color, shape_kwargs={'scale': (0.12, 0.08, 0.15)})
        car_body = transform_shape(car_body, translation_matrix([0, 0, position]))

        car_wheels = concat_shapes(
            create_wheel(-0.04, position - 0.05),
            create_wheel(-0.04, position + 0.05),
            create_wheel(0.04, position - 0.05),
            create_wheel(0.04, position + 0.05)
        )

        return concat_shapes(car_body, car_wheels)

    # Create cars based on train length
    num_cars = max(1, int(length / 0.2) - 1)
    cars = concat_shapes(*[create_car(0.25 + i * 0.2) for i in range(num_cars)])

    return concat_shapes(engine, cars)

@register()
def play_mat(width: float, length: float, color: tuple[float, float, float] = (0.0, 0.7, 0.2)) -> Shape:
    """Create a play mat for the children's corner"""
    mat = primitive_call('cube', color=color, shape_kwargs={'scale': (width, 0.01, length)})
    return mat

@register()
def toy_shelf(width: float, height: float, depth: float) -> Shape:
    """Create a toy shelf with multiple compartments"""
    # Main shelf body
    shelf_body = primitive_call('cube', color=(0.8, 0.8, 0.8),
                               shape_kwargs={'scale': (width, height, depth)})

    # Shelf dividers
    num_dividers = 2
    divider_width = 0.02

    dividers = []
    for i in range(1, num_dividers + 1):
        x_pos = (i / (num_dividers + 1) - 0.5) * width
        divider = primitive_call('cube', color=(0.75, 0.75, 0.75),
                                shape_kwargs={'scale': (divider_width, height - 0.05, depth - 0.05)})
        divider = transform_shape(divider, translation_matrix([x_pos, 0, 0]))
        dividers.append(divider)

    # Horizontal shelf
    shelf = primitive_call('cube', color=(0.75, 0.75, 0.75),
                          shape_kwargs={'scale': (width - 0.05, divider_width, depth - 0.05)})
    shelf = transform_shape(shelf, translation_matrix([0, height/4, 0]))

    return concat_shapes(shelf_body, *dividers, shelf)

@register()
def stuffed_animal(position: P, size: float, color: tuple[float, float, float]) -> Shape:
    """Create a simple stuffed animal (teddy bear)"""
    # Body
    body = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.15})
    body = transform_shape(body, translation_matrix([position[0], position[1], position[2]]))

    # Head
    head = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.1})
    head = transform_shape(head, translation_matrix([position[0], position[1] + size * 0.2, position[2]]))

    # Ears
    ear_left = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.04})
    ear_left = transform_shape(ear_left, translation_matrix([position[0] - size * 0.08,
                                                           position[1] + size * 0.28,
                                                           position[2]]))

    ear_right = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.04})
    ear_right = transform_shape(ear_right, translation_matrix([position[0] + size * 0.08,
                                                             position[1] + size * 0.28,
                                                             position[2]]))

    # Arms
    arm_left = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.06})
    arm_left = transform_shape(arm_left, translation_matrix([position[0] - size * 0.18,
                                                           position[1] + size * 0.05,
                                                           position[2]]))

    arm_right = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.06})
    arm_right = transform_shape(arm_right, translation_matrix([position[0] + size * 0.18,
                                                             position[1] + size * 0.05,
                                                             position[2]]))

    # Legs
    leg_left = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.07})
    leg_left = transform_shape(leg_left, translation_matrix([position[0] - size * 0.1,
                                                           position[1] - size * 0.15,
                                                           position[2]]))

    leg_right = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.07})
    leg_right = transform_shape(leg_right, translation_matrix([position[0] + size * 0.1,
                                                             position[1] - size * 0.15,
                                                             position[2]]))

    # Nose
    nose = primitive_call('sphere', color=(0.1, 0.1, 0.1), shape_kwargs={'radius': size * 0.02})
    nose = transform_shape(nose, translation_matrix([position[0],
                                                   position[1] + size * 0.2,
                                                   position[2] + size * 0.09]))

    return concat_shapes(body, head, ear_left, ear_right, arm_left, arm_right, leg_left, leg_right, nose)

@register()
def children_playing_corner() -> Shape:
    """Create a complete children's playing corner with toys and furniture"""
    # Create play mat as the base
    mat = library_call('play_mat', width=2.0, length=2.0, color=(0.2, 0.8, 0.3))

    # Create toy shelf
    shelf = library_call('toy_shelf', width=0.8, height=0.6, depth=0.3)
    shelf = transform_shape(shelf, translation_matrix([0.5, 0.3, -0.7]))

    # Create blocks stack
    blocks = library_call('toy_blocks_stack', base_size=0.15, height=0.05, num_blocks=7)
    blocks = transform_shape(blocks, translation_matrix([-0.5, 0.025, -0.3]))

    # Create ball pile
    balls = library_call('toy_balls_pile', num_balls=12)
    balls = transform_shape(balls, translation_matrix([0.4, 0.025, 0.4]))

    # Create toy train
    train = library_call('toy_train', length=0.8)
    train = transform_shape(train, translation_matrix([-0.3, 0.03, 0.5]))
    train_center = compute_shape_center(train)
    train = transform_shape(train, rotation_matrix(math.pi/4, direction=(0, 1, 0), point=train_center))

    # Create stuffed animals
    teddy1 = library_call('stuffed_animal', position=[-0.6, 0.15, 0.0], size=0.8, color=(0.6, 0.4, 0.2))
    teddy2 = library_call('stuffed_animal', position=[0.0, 0.15, -0.5], size=0.7, color=(0.8, 0.7, 0.3))
    teddy2_center = compute_shape_center(teddy2)
    teddy2 = transform_shape(teddy2, rotation_matrix(math.pi/3, direction=(0, 1, 0), point=teddy2_center))

    # Add some random blocks on the shelf
    shelf_blocks = library_call('toy_blocks_stack', base_size=0.1, height=0.05, num_blocks=3)
    shelf_blocks = transform_shape(shelf_blocks, translation_matrix([0.3, 0.6, -0.7]))

    # Add a ball on the shelf
    shelf_ball = library_call('ball', radius=0.07, color=(0.0, 0.5, 1.0))
    shelf_ball = transform_shape(shelf_ball, translation_matrix([0.7, 0.65, -0.7]))

    return concat_shapes(
        mat, shelf, blocks, balls, train, teddy1, teddy2, shelf_blocks, shelf_ball
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
