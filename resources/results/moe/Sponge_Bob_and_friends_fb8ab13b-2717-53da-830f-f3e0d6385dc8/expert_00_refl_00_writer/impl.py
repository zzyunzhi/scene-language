

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

EXTRA_ENGINE_MODE = ['box', 'interior', 'exterior',
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


def core(engine_modes: list[Literal['neural', 'lmd', 'omost', 'loosecontrol', 'densediffusion']], overwrite: bool, save_dir: str,
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
        if engine_mode not in ['box', 'interior', 'exterior'] and not cuda_is_available:
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
Sponge Bob and friends
"""

@register()
def sponge_bob() -> Shape:
    # Create SpongeBob's body (yellow sponge)
    body = primitive_call('cube', shape_kwargs={'scale': (0.8, 1.0, 0.4)}, color=(1.0, 0.9, 0.0))

    # Create SpongeBob's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.2, 0.3, -0.21)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.2, 0.3, -0.21)))

    # Create SpongeBob's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.2, 0.3, -0.35)))

    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.2, 0.3, -0.35)))

    # Create SpongeBob's nose
    nose = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 0.8, 0.0))
    nose = transform_shape(nose, translation_matrix((0.0, 0.0, -0.25)))

    # Create SpongeBob's mouth
    mouth = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.3, -0.2, -0.21), 'p1': (0.3, -0.2, -0.21)}, color=(0.8, 0.0, 0.0))

    # Create SpongeBob's arms
    left_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.4, 0.0, 0.0), 'p1': (-0.8, -0.2, 0.0)}, color=(1.0, 0.9, 0.0))
    right_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0.4, 0.0, 0.0), 'p1': (0.8, -0.2, 0.0)}, color=(1.0, 0.9, 0.0))

    # Create SpongeBob's legs
    left_leg = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.25, -0.5, 0.0), 'p1': (-0.25, -1.0, 0.0)}, color=(1.0, 0.9, 0.0))
    right_leg = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0.25, -0.5, 0.0), 'p1': (0.25, -1.0, 0.0)}, color=(1.0, 0.9, 0.0))

    # Create SpongeBob's shoes
    left_shoe = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(0.0, 0.0, 0.0))
    left_shoe = transform_shape(left_shoe, translation_matrix((-0.25, -1.0, 0.0)))

    right_shoe = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(0.0, 0.0, 0.0))
    right_shoe = transform_shape(right_shoe, translation_matrix((0.25, -1.0, 0.0)))

    return concat_shapes(body, left_eye, right_eye, left_pupil, right_pupil, nose, mouth,
                         left_arm, right_arm, left_leg, right_leg, left_shoe, right_shoe)

@register()
def patrick() -> Shape:
    # Create Patrick's body (pink starfish)
    body = primitive_call('sphere', shape_kwargs={'radius': 0.5}, color=(1.0, 0.6, 0.6))
    body = transform_shape(body, scale_matrix(1.2, (0, 0, 0)))

    # Create Patrick's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.15, 0.2, -0.45)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.15, 0.2, -0.45)))

    # Create Patrick's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.15, 0.2, -0.55)))

    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.15, 0.2, -0.55)))

    # Create Patrick's mouth
    mouth = primitive_call('cylinder', shape_kwargs={'radius': 0.04, 'p0': (-0.2, -0.1, -0.45), 'p1': (0.2, -0.1, -0.45)}, color=(0.8, 0.0, 0.0))

    # Create Patrick's arms and legs (starfish points)
    limbs = []

    # Create 5 limbs in a star pattern
    for i in range(5):
        angle = 2 * math.pi * i / 5 + math.pi/10  # Offset to make it stand on two legs
        x = 0.7 * math.cos(angle)
        y = 0.7 * math.sin(angle)

        limb = primitive_call('cylinder', shape_kwargs={'radius': 0.1, 'p0': (0, 0, 0), 'p1': (x, y, 0)}, color=(1.0, 0.6, 0.6))
        limbs.append(limb)

    return concat_shapes(body, left_eye, right_eye, left_pupil, right_pupil, mouth, *limbs)

@register()
def squidward() -> Shape:
    # Create Squidward's head (turquoise)
    head = primitive_call('sphere', shape_kwargs={'radius': 0.4}, color=(0.0, 0.7, 0.7))
    head = transform_shape(head, scale_matrix(0.8, (0, 0, 0)))

    # Create Squidward's nose
    nose = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (0, 0, -0.3), 'p1': (0, 0, -0.7)}, color=(0.0, 0.7, 0.7))

    # Create Squidward's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.12}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.15, 0.15, -0.3)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.12}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.15, 0.15, -0.3)))

    # Create Squidward's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.15, 0.15, -0.42)))

    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.15, 0.15, -0.42)))

    # Create Squidward's body
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.25, 'p0': (0, -0.3, 0), 'p1': (0, -1.0, 0)}, color=(0.0, 0.7, 0.7))

    # Create Squidward's tentacles
    tentacles = []
    for i in range(6):
        angle = math.pi + (math.pi * i / 5)
        x = 0.3 * math.cos(angle)
        z = 0.3 * math.sin(angle)

        tentacle = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (x, -1.0, z), 'p1': (x*1.5, -1.5, z*1.5)}, color=(0.0, 0.7, 0.7))
        tentacles.append(tentacle)

    return concat_shapes(head, nose, left_eye, right_eye, left_pupil, right_pupil, body, *tentacles)

@register()
def mr_krabs() -> Shape:
    # Create Mr. Krabs' body (red)
    body = primitive_call('sphere', shape_kwargs={'radius': 0.4}, color=(0.9, 0.2, 0.1))

    # Create Mr. Krabs' eyes (on stalks)
    left_stalk = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.1, 0.2, 0), 'p1': (-0.1, 0.5, -0.2)}, color=(0.9, 0.2, 0.1))
    right_stalk = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0.1, 0.2, 0), 'p1': (0.1, 0.5, -0.2)}, color=(0.9, 0.2, 0.1))

    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.1, 0.5, -0.2)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.1, 0.5, -0.2)))

    # Create Mr. Krabs' pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.1, 0.5, -0.3)))

    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.1, 0.5, -0.3)))

    # Create Mr. Krabs' claws
    left_claw = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.9, 0.2, 0.1))
    left_claw = transform_shape(left_claw, translation_matrix((-0.6, 0.0, 0.0)))
    left_claw = transform_shape(left_claw, scale_matrix(1.2, (-0.6, 0.0, 0.0)))

    right_claw = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.9, 0.2, 0.1))
    right_claw = transform_shape(right_claw, translation_matrix((0.6, 0.0, 0.0)))
    right_claw = transform_shape(right_claw, scale_matrix(1.2, (0.6, 0.0, 0.0)))

    # Create Mr. Krabs' legs
    legs = []
    for i in range(4):
        x_offset = 0.2 if i % 2 == 0 else -0.2
        z_offset = 0.2 if i < 2 else -0.2

        leg = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (x_offset, -0.2, z_offset), 'p1': (x_offset*1.5, -0.8, z_offset*1.5)}, color=(0.9, 0.2, 0.1))
        legs.append(leg)

    return concat_shapes(body, left_stalk, right_stalk, left_eye, right_eye,
                         left_pupil, right_pupil, left_claw, right_claw, *legs)

@register()
def sandy() -> Shape:
    # Create Sandy's helmet (transparent sphere)
    helmet = primitive_call('sphere', shape_kwargs={'radius': 0.5}, color=(0.8, 0.8, 1.0))

    # Create Sandy's head (squirrel)
    head = primitive_call('sphere', shape_kwargs={'radius': 0.35}, color=(0.8, 0.6, 0.4))

    # Create Sandy's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.08}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.15, 0.1, -0.3)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.08}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.15, 0.1, -0.3)))

    # Create Sandy's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.15, 0.1, -0.38)))

    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.15, 0.1, -0.38)))

    # Create Sandy's nose
    nose = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.6, 0.3, 0.3))
    nose = transform_shape(nose, translation_matrix((0.0, 0.0, -0.35)))

    # Create Sandy's body
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.3, 'p0': (0, -0.5, 0), 'p1': (0, -1.2, 0)}, color=(1.0, 1.0, 1.0))

    # Create Sandy's arms
    left_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (-0.3, -0.6, 0), 'p1': (-0.6, -0.4, 0)}, color=(1.0, 1.0, 1.0))
    right_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (0.3, -0.6, 0), 'p1': (0.6, -0.4, 0)}, color=(1.0, 1.0, 1.0))

    # Create Sandy's tail
    tail = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.8, 0.6, 0.4))
    tail = transform_shape(tail, translation_matrix((0.0, -1.0, 0.4)))
    tail = transform_shape(tail, scale_matrix(1.5, (0.0, -1.0, 0.4)))

    return concat_shapes(helmet, head, left_eye, right_eye, left_pupil, right_pupil,
                         nose, body, left_arm, right_arm, tail)

@register()
def bikini_bottom_scene() -> Shape:
    # Create the characters
    spongebob = library_call('sponge_bob')
    patrick = library_call('patrick')
    squidward = library_call('squidward')
    mr_krabs = library_call('mr_krabs')
    sandy = library_call('sandy')

    # Position the characters
    spongebob = transform_shape(spongebob, translation_matrix((0, 0, 0)))
    patrick = transform_shape(patrick, translation_matrix((1.5, -0.2, 0.5)))
    squidward = transform_shape(squidward, translation_matrix((-1.5, 0.2, 0.3)))
    mr_krabs = transform_shape(mr_krabs, translation_matrix((0.8, 0, -1.5)))
    sandy = transform_shape(sandy, translation_matrix((-0.8, 0, -1.2)))

    # Create the ocean floor
    floor = primitive_call('cube', shape_kwargs={'scale': (10, 0.1, 10)}, color=(0.8, 0.7, 0.2))
    floor = transform_shape(floor, translation_matrix((0, -1.5, 0)))

    # Create some seaweed
    seaweeds = []
    for i in range(8):
        x = np.random.uniform(-4, 4)
        z = np.random.uniform(-4, 4)
        height = np.random.uniform(0.5, 1.5)

        seaweed = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (x, -1.45, z), 'p1': (x, -1.45 + height, z)}, color=(0.0, 0.6, 0.3))
        seaweeds.append(seaweed)

    # Create some rocks
    rocks = []
    for i in range(5):
        x = np.random.uniform(-4, 4)
        z = np.random.uniform(-4, 4)
        size = np.random.uniform(0.2, 0.5)

        rock = primitive_call('sphere', shape_kwargs={'radius': size}, color=(0.5, 0.5, 0.5))
        rock = transform_shape(rock, translation_matrix((x, -1.45 + size/2, z)))
        rocks.append(rock)

    return concat_shapes(spongebob, patrick, squidward, mr_krabs, sandy, floor, *seaweeds, *rocks)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
