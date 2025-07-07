

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

# Set random seed for reproducibility
np.random.seed(42)

@register()
def toy_block(scale: P, color: tuple[float, float, float] = (1.0, 0.5, 0.0)) -> Shape:
    """Create a toy block with given scale and color"""
    return primitive_call('cube', color=color, shape_kwargs={'scale': scale})

@register()
def ball(radius: float, color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> Shape:
    """Create a ball with given radius and color"""
    return primitive_call('sphere', color=color, shape_kwargs={'radius': radius})

@register()
def toy_blocks_stack(num_blocks: int = 5) -> Shape:
    """Create a stack of toy blocks with random colors and slight offsets"""
    colors = [(1.0, 0.5, 0.0), (0.0, 0.7, 0.3), (0.3, 0.3, 1.0),
              (1.0, 0.8, 0.0), (0.8, 0.2, 0.8), (0.2, 0.8, 0.8)]

    def loop_fn(i) -> Shape:
        block_size = (0.15, 0.05, 0.1)
        color_idx = i % len(colors)
        block = library_call('toy_block', scale=block_size, color=colors[color_idx])
        offset_x = np.random.uniform(-0.03, 0.03)
        offset_z = np.random.uniform(-0.03, 0.03)
        block = transform_shape(block, translation_matrix([offset_x, i * block_size[1], offset_z]))
        return block

    return loop(num_blocks, loop_fn)

@register()
def toy_blocks_pyramid(levels: int = 4) -> Shape:
    """Create a pyramid of toy blocks"""
    blocks = []

    for level in range(levels):
        num_blocks_in_level = levels - level
        for i in range(num_blocks_in_level):
            color = (np.random.uniform(0.3, 1.0),
                     np.random.uniform(0.3, 1.0),
                     np.random.uniform(0.3, 1.0))
            block = library_call('toy_block', scale=(0.1, 0.05, 0.1), color=color)

            # Position blocks in a row for this level
            offset_x = (i - (num_blocks_in_level - 1) / 2) * 0.12
            offset_y = level * 0.06
            offset_z = 0

            block = transform_shape(block, translation_matrix([offset_x, offset_y, offset_z]))
            blocks.append(block)

    return concat_shapes(*blocks)

@register()
def ball_pile(num_balls: int = 6) -> Shape:
    """Create a pile of colorful balls"""
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
              (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]

    def loop_fn(i) -> Shape:
        radius = np.random.uniform(0.04, 0.07)
        color_idx = i % len(colors)
        ball = library_call('ball', radius=radius, color=colors[color_idx])

        # Arrange balls in a tighter pile for more cohesion
        if i < 3:  # First layer
            angle = i * (2 * math.pi / 3)
            offset_x = 0.06 * math.cos(angle)  # Reduced from 0.08
            offset_z = 0.06 * math.sin(angle)  # Reduced from 0.08
            offset_y = radius
        else:  # Second layer
            angle = (i-3) * (2 * math.pi / 3) + (math.pi/3)
            offset_x = 0.03 * math.cos(angle)  # Reduced from 0.05
            offset_z = 0.03 * math.sin(angle)  # Reduced from 0.05
            offset_y = 0.10 + radius  # Reduced from 0.12

        return transform_shape(ball, translation_matrix([offset_x, offset_y, offset_z]))

    return loop(num_balls, loop_fn)

@register()
def toy_train(length: float = 0.3) -> Shape:
    """Create a simple toy train"""
    # Train body
    body = primitive_call('cube', color=(0.7, 0.0, 0.0), shape_kwargs={'scale': (length, 0.1, 0.12)})

    # Train cabin
    cabin = primitive_call('cube', color=(0.8, 0.0, 0.0),
                          shape_kwargs={'scale': (0.12, 0.15, 0.12)})
    cabin = transform_shape(cabin, translation_matrix([length/4, 0.125, 0]))

    # Wheels - fixed to be perpendicular to the train body (along x-axis)
    wheels = []
    wheel_positions = [(-length/3, -0.05, 0.07), (-length/3, -0.05, -0.07),
                       (length/3, -0.05, 0.07), (length/3, -0.05, -0.07)]

    for pos in wheel_positions:
        wheel = primitive_call('cylinder', color=(0.2, 0.2, 0.2),
                              shape_kwargs={'radius': 0.03, 'p0': (pos[0]-0.02, pos[1], pos[2]),
                                           'p1': (pos[0]+0.02, pos[1], pos[2])})
        wheels.append(wheel)

    # Chimney - ensure it's vertical (along y-axis)
    chimney = primitive_call('cylinder', color=(0.3, 0.3, 0.3),
                            shape_kwargs={'radius': 0.02, 'p0': (length/4, 0.2, 0),
                                         'p1': (length/4, 0.3, 0)})

    return concat_shapes(body, cabin, chimney, *wheels)

@register()
def play_mat() -> Shape:
    """Create a colorful play mat"""
    return primitive_call('cube', color=(0.0, 0.6, 0.2),
                         shape_kwargs={'scale': (1.5, 0.02, 1.5)})

@register()
def toy_shelf() -> Shape:
    """Create a simple toy shelf with proper structure"""
    # Back panel
    back_panel = primitive_call('cube', color=(0.8, 0.7, 0.6),
                              shape_kwargs={'scale': (0.8, 0.6, 0.05)})
    back_panel = transform_shape(back_panel, translation_matrix([0, 0, -0.125]))

    # Side panels
    left_panel = primitive_call('cube', color=(0.8, 0.7, 0.6),
                              shape_kwargs={'scale': (0.05, 0.6, 0.3)})
    left_panel = transform_shape(left_panel, translation_matrix([-0.375, 0, 0]))

    right_panel = primitive_call('cube', color=(0.8, 0.7, 0.6),
                               shape_kwargs={'scale': (0.05, 0.6, 0.3)})
    right_panel = transform_shape(right_panel, translation_matrix([0.375, 0, 0]))

    # Shelves
    shelf1 = primitive_call('cube', color=(0.75, 0.65, 0.55),
                          shape_kwargs={'scale': (0.75, 0.02, 0.28)})
    shelf1 = transform_shape(shelf1, translation_matrix([0, -0.1, 0]))

    shelf2 = primitive_call('cube', color=(0.75, 0.65, 0.55),
                          shape_kwargs={'scale': (0.75, 0.02, 0.28)})
    shelf2 = transform_shape(shelf2, translation_matrix([0, 0.1, 0]))

    # Add some decorative elements to the shelf
    book1 = primitive_call('cube', color=(0.9, 0.2, 0.2),
                          shape_kwargs={'scale': (0.08, 0.12, 0.04)})
    book1 = transform_shape(book1, translation_matrix([-0.3, 0.16, 0]))

    book2 = primitive_call('cube', color=(0.2, 0.2, 0.9),
                          shape_kwargs={'scale': (0.08, 0.10, 0.04)})
    book2 = transform_shape(book2, translation_matrix([-0.2, 0.15, 0]))

    toy_box = primitive_call('cube', color=(1.0, 0.8, 0.0),
                           shape_kwargs={'scale': (0.15, 0.08, 0.15)})
    toy_box = transform_shape(toy_box, translation_matrix([0.3, 0.14, 0]))

    return concat_shapes(back_panel, left_panel, right_panel, shelf1, shelf2, book1, book2, toy_box)

@register()
def teddy_bear() -> Shape:
    """Create a teddy bear with proper scaling"""
    # Reduced scale factor for better proportions
    scale_factor = 0.9

    # Body
    body = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.15 * scale_factor})
    body = transform_shape(body, translation_matrix([0, 0.15 * scale_factor, 0]))

    # Head
    head = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.1 * scale_factor})
    head = transform_shape(head, translation_matrix([0, 0.35 * scale_factor, 0]))

    # Ears
    ear1 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.03 * scale_factor})
    ear1 = transform_shape(ear1, translation_matrix([0.08 * scale_factor, 0.43 * scale_factor, 0]))

    ear2 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.03 * scale_factor})
    ear2 = transform_shape(ear2, translation_matrix([-0.08 * scale_factor, 0.43 * scale_factor, 0]))

    # Arms - fix transformation order: scale first, then translate
    arm1 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.05 * scale_factor})
    arm1 = transform_shape(arm1, scale_matrix(1.5, (0, 0, 0)))
    arm1 = transform_shape(arm1, translation_matrix([0.2 * scale_factor, 0.15 * scale_factor, 0]))

    arm2 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.05 * scale_factor})
    arm2 = transform_shape(arm2, scale_matrix(1.5, (0, 0, 0)))
    arm2 = transform_shape(arm2, translation_matrix([-0.2 * scale_factor, 0.15 * scale_factor, 0]))

    # Legs - fix transformation order: scale first, then translate
    leg1 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.05 * scale_factor})
    leg1 = transform_shape(leg1, scale_matrix(1.5, (0, 0, 0)))
    leg1 = transform_shape(leg1, translation_matrix([0.1 * scale_factor, -0.05 * scale_factor, 0]))

    leg2 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.05 * scale_factor})
    leg2 = transform_shape(leg2, scale_matrix(1.5, (0, 0, 0)))
    leg2 = transform_shape(leg2, translation_matrix([-0.1 * scale_factor, -0.05 * scale_factor, 0]))

    # Eyes
    eye1 = primitive_call('sphere', color=(0.0, 0.0, 0.0),
                         shape_kwargs={'radius': 0.015 * scale_factor})
    eye1 = transform_shape(eye1, translation_matrix([0.05 * scale_factor, 0.38 * scale_factor, -0.08 * scale_factor]))

    eye2 = primitive_call('sphere', color=(0.0, 0.0, 0.0),
                         shape_kwargs={'radius': 0.015 * scale_factor})
    eye2 = transform_shape(eye2, translation_matrix([-0.05 * scale_factor, 0.38 * scale_factor, -0.08 * scale_factor]))

    # Nose
    nose = primitive_call('sphere', color=(0.3, 0.2, 0.1),
                         shape_kwargs={'radius': 0.02 * scale_factor})
    nose = transform_shape(nose, translation_matrix([0, 0.33 * scale_factor, -0.09 * scale_factor]))

    return concat_shapes(body, head, ear1, ear2, arm1, arm2, leg1, leg2, eye1, eye2, nose)

@register()
def small_rug() -> Shape:
    """Create a small decorative rug for the play area"""
    rug = primitive_call('cube', color=(0.9, 0.3, 0.3),
                        shape_kwargs={'scale': (0.6, 0.01, 0.4)})
    return rug

@register()
def children_playing_corner() -> Shape:
    """Create a children's playing corner with toys and play area"""
    # Create the play mat as the base
    mat = library_call('play_mat')
    mat_height = 0.01  # Half height of the mat

    # Add a small decorative rug
    rug = library_call('small_rug')
    rug = transform_shape(rug, translation_matrix([0.2, mat_height + 0.005, 0.2]))

    # Add toy blocks in different arrangements - fix positioning to rest on mat
    blocks_stack = library_call('toy_blocks_stack')
    blocks_stack = transform_shape(blocks_stack, translation_matrix([0.4, mat_height + 0.025, 0.3]))

    blocks_pyramid = library_call('toy_blocks_pyramid')
    blocks_pyramid = transform_shape(blocks_pyramid, translation_matrix([-0.4, mat_height + 0.025, 0.4]))

    # Add balls
    balls = library_call('ball_pile')
    balls = transform_shape(balls, translation_matrix([0.3, mat_height + 0.04, -0.3]))

    # Improve toy train visibility
    train = library_call('toy_train')
    train = transform_shape(train, translation_matrix([-0.3, mat_height + 0.05, 0.0]))
    train = transform_shape(train, rotation_matrix(math.pi/4, direction=(0, 1, 0), point=(-0.3, mat_height + 0.05, 0.0)))

    # Fix shelf positioning to rest properly on the floor
    shelf = library_call('toy_shelf')
    shelf_height = 0.6  # Height of the shelf
    shelf = transform_shape(shelf, translation_matrix([0, shelf_height/2 + mat_height, -0.6]))

    # Add teddy bear with fixed proportions and positioning
    bear = library_call('teddy_bear')
    bear = transform_shape(bear, translation_matrix([-0.5, mat_height + 0.15, -0.4]))
    bear = transform_shape(bear, rotation_matrix(math.pi/6, direction=(0, 1, 0), point=(-0.5, mat_height + 0.15, -0.4)))

    # Create some additional balls scattered around with specific positions
    ball_positions = [(0.1, 0.04, 0.1), (-0.3, 0.04, -0.1), (0.5, 0.04, -0.5), (-0.6, 0.04, 0.2)]
    scattered_balls = []
    for i, pos in enumerate(ball_positions):
        ball_color = (np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0))
        ball = library_call('ball', radius=0.04, color=ball_color)
        # Adjust y-coordinate to rest on mat
        scattered_balls.append(transform_shape(ball, translation_matrix([pos[0], mat_height + pos[1], pos[2]])))

    # Add a toy box on the floor
    toy_box = primitive_call('cube', color=(0.3, 0.7, 0.9),
                            shape_kwargs={'scale': (0.2, 0.15, 0.2)})
    toy_box = transform_shape(toy_box, translation_matrix([0.6, mat_height + 0.075, 0.5]))

    return concat_shapes(
        mat,
        rug,
        blocks_stack,
        blocks_pyramid,
        balls,
        train,
        shelf,
        bear,
        toy_box,
        *scattered_balls
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
