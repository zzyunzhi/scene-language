

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


"""
Here's a program to create a chessboard with a full set of chess pieces:

"""
from helper import *

"""
a chessboard with a full set of chess pieces
"""

@register()
def chessboard_square(color: tuple[float, float, float]) -> Shape:
    return primitive_call('cube', color=color, shape_kwargs={'scale': (0.125, 0.02, 0.125)})

@register()
def chessboard() -> Shape:
    def loop_fn(i) -> Shape:
        row, col = divmod(i, 8)
        color = (0.9, 0.9, 0.9) if (row + col) % 2 == 0 else (0.1, 0.1, 0.1)
        square = library_call('chessboard_square', color=color)
        return transform_shape(square, translation_matrix((col * 0.125, 0, row * 0.125)))

    return loop(64, loop_fn)

@register()
def pawn() -> Shape:
    base = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.04, 'p0': (0, 0, 0), 'p1': (0, 0.05, 0)})
    body = primitive_call('sphere', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.03})
    body = transform_shape(body, translation_matrix((0, 0.08, 0)))
    return concat_shapes(base, body)

@register()
def rook() -> Shape:
    base = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.04, 'p0': (0, 0, 0), 'p1': (0, 0.05, 0)})
    body = primitive_call('cube', color=(0.8, 0.8, 0.8), shape_kwargs={'scale': (0.06, 0.08, 0.06)})
    body = transform_shape(body, translation_matrix((0, 0.09, 0)))
    return concat_shapes(base, body)

@register()
def knight() -> Shape:
    base = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.04, 'p0': (0, 0, 0), 'p1': (0, 0.05, 0)})
    body = primitive_call('sphere', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.04})
    body = transform_shape(body, translation_matrix((0, 0.09, 0)))
    head = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.02, 'p0': (0, 0.09, 0), 'p1': (0.04, 0.13, 0)})
    return concat_shapes(base, body, head)

@register()
def bishop() -> Shape:
    base = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.04, 'p0': (0, 0, 0), 'p1': (0, 0.05, 0)})
    body = primitive_call('sphere', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.035})
    body = transform_shape(body, translation_matrix((0, 0.085, 0)))
    top = primitive_call('sphere', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.015})
    top = transform_shape(top, translation_matrix((0, 0.14, 0)))
    return concat_shapes(base, body, top)

@register()
def queen() -> Shape:
    base = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.04, 'p0': (0, 0, 0), 'p1': (0, 0.05, 0)})
    body = primitive_call('sphere', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.045})
    body = transform_shape(body, translation_matrix((0, 0.095, 0)))
    crown = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.02, 'p0': (0, 0.14, 0), 'p1': (0, 0.16, 0)})
    return concat_shapes(base, body, crown)

@register()
def king() -> Shape:
    base = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.04, 'p0': (0, 0, 0), 'p1': (0, 0.05, 0)})
    body = primitive_call('sphere', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.045})
    body = transform_shape(body, translation_matrix((0, 0.095, 0)))
    cross_v = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.01, 'p0': (0, 0.14, 0), 'p1': (0, 0.18, 0)})
    cross_h = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 0.01, 'p0': (-0.02, 0.16, 0), 'p1': (0.02, 0.16, 0)})
    return concat_shapes(base, body, cross_v, cross_h)

@register()
def place_pieces() -> Shape:
    pieces = []
    piece_order = ['rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook']

    for i, piece_name in enumerate(piece_order):
        white_piece = library_call(piece_name)
        black_piece = transform_shape(library_call(piece_name), scale_matrix(1, (0, 0, 0)))
        black_piece = transform_shape(black_piece, translation_matrix((0, 0, 0.875)))

        white_piece = transform_shape(white_piece, translation_matrix((i * 0.125, 0.02, 0)))
        black_piece = transform_shape(black_piece, translation_matrix((i * 0.125, 0.02, 0)))

        pieces.extend([white_piece, black_piece])

    for i in range(8):
        white_pawn = library_call('pawn')
        black_pawn = transform_shape(library_call('pawn'), scale_matrix(1, (0, 0, 0)))

        white_pawn = transform_shape(white_pawn, translation_matrix((i * 0.125, 0.02, 0.125)))
        black_pawn = transform_shape(black_pawn, translation_matrix((i * 0.125, 0.02, 0.75)))

        pieces.extend([white_pawn, black_pawn])

    return concat_shapes(*pieces)

@register()
def chessboard_with_pieces() -> Shape:
    board = library_call('chessboard')
    pieces = library_call('place_pieces')
    return concat_shapes(board, pieces)
"""

This program creates a chessboard with a full set of chess pieces. Here's a breakdown of the functions:

1. `chessboard_square`: Creates a single square of the chessboard.
2. `chessboard`: Creates the entire chessboard by placing 64 squares in an 8x8 grid.
3. `pawn`, `rook`, `knight`, `bishop`, `queen`, `king`: Create individual chess pieces using primitive shapes.
4. `place_pieces`: Places all the chess pieces in their starting positions on the board.
5. `chessboard_with_pieces`: Combines the chessboard and the placed pieces into a single shape.

The chessboard is created with alternating light and dark squares, and the pieces are placed in their correct starting positions. The white pieces are on the bottom two rows, and the black pieces are on the top two rows. The pieces are simplified representations using basic shapes, but they should be recognizable as chess pieces.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
