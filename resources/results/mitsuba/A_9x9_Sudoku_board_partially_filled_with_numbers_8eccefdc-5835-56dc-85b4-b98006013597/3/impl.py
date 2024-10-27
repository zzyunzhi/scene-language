

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
I'll help you create a Sudoku board with numbers. I'll break it down into modular components:

"""
from helper import *

"""
A 9x9 Sudoku board partially filled with numbers
"""

@register()
def grid_cell(size: float, thickness: float) -> Shape:
    """Creates a single cell of the Sudoku grid"""
    return primitive_call('cube',
                        color=(0.9, 0.9, 0.9),
                        shape_kwargs={'scale': (size, thickness, size)})

@register()
def number_vertical_line(height: float, thickness: float) -> Shape:
    """Creates a vertical line for numbers"""
    return primitive_call('cylinder',
                        color=(0.1, 0.1, 0.1),
                        shape_kwargs={
                            'radius': thickness/2,
                            'p0': (0, 0, 0),
                            'p1': (0, height, 0)
                        })

@register()
def number_horizontal_line(width: float, thickness: float) -> Shape:
    """Creates a horizontal line for numbers"""
    return primitive_call('cylinder',
                        color=(0.1, 0.1, 0.1),
                        shape_kwargs={
                            'radius': thickness/2,
                            'p0': (0, 0, 0),
                            'p1': (width, 0, 0)
                        })

@register()
def number_1(height: float, thickness: float) -> Shape:
    """Creates number 1"""
    return library_call('number_vertical_line', height=height, thickness=thickness)

@register()
def number_4(size: float, thickness: float) -> Shape:
    """Creates number 4"""
    height = size
    width = size * 0.6

    vertical_right = library_call('number_vertical_line', height=height, thickness=thickness)
    vertical_left = library_call('number_vertical_line', height=height/2, thickness=thickness)
    horizontal = library_call('number_horizontal_line', width=width, thickness=thickness)

    vertical_right = transform_shape(vertical_right, translation_matrix((width, 0, 0)))
    vertical_left = transform_shape(vertical_left, translation_matrix((0, height/2, 0)))
    horizontal = transform_shape(horizontal, translation_matrix((0, height/2, 0)))

    return concat_shapes(vertical_right, vertical_left, horizontal)

@register()
def number_7(size: float, thickness: float) -> Shape:
    """Creates number 7"""
    height = size
    width = size * 0.6

    vertical = library_call('number_vertical_line', height=height, thickness=thickness)
    horizontal = library_call('number_horizontal_line', width=width, thickness=thickness)

    vertical = transform_shape(vertical,
                             translation_matrix((width*0.7, 0, 0)) @
                             rotation_matrix(math.pi/10, (0, 0, 1), (width*0.7, height, 0)))
    horizontal = transform_shape(horizontal, translation_matrix((0, height, 0)))

    return concat_shapes(vertical, horizontal)

@register()
def sudoku_grid() -> Shape:
    """Creates the main 9x9 Sudoku grid"""
    cell_size = 0.1
    thickness = 0.005

    def loop_fn(i: int) -> Shape:
        row = i // 9
        col = i % 9
        cell = library_call('grid_cell', size=cell_size, thickness=thickness)
        return transform_shape(cell,
                             translation_matrix((col * cell_size, -row * cell_size, 0)))

    return loop(81, loop_fn)

@register()
def sudoku_board() -> Shape:
    """Creates a complete Sudoku board with some numbers"""
    board = library_call('sudoku_grid')
    cell_size = 0.1
    number_size = cell_size * 0.8
    thickness = 0.003

    # Define some initial numbers for the Sudoku puzzle
    numbers = {
        (0, 0): '4', (0, 3): '7', (0, 5): '1',
        (1, 2): '7', (1, 6): '4',
        (2, 1): '1', (2, 4): '4', (2, 8): '7',
        (3, 0): '7', (3, 7): '1',
        (5, 1): '4', (5, 8): '1',
        (6, 0): '1', (6, 4): '7', (6, 7): '4',
        (7, 2): '4', (7, 6): '1',
        (8, 3): '1', (8, 5): '7', (8, 8): '4'
    }

    number_shapes = []
    for (row, col), num in numbers.items():
        if num == '1':
            number = library_call('number_1', height=number_size, thickness=thickness)
        elif num == '4':
            number = library_call('number_4', size=number_size, thickness=thickness)
        elif num == '7':
            number = library_call('number_7', size=number_size, thickness=thickness)

        pos_x = col * cell_size + cell_size * 0.2
        pos_y = -row * cell_size - cell_size * 0.1
        pos_z = -thickness

        number = transform_shape(number, translation_matrix((pos_x, pos_y, pos_z)))
        number_shapes.append(number)

    return concat_shapes(board, *number_shapes)
"""

This program creates a 9x9 Sudoku board with some numbers (1, 4, and 7) placed in various cells. Here's how it works:

1. `grid_cell`: Creates a single cell of the Sudoku grid using a cube primitive
2. `number_vertical_line` and `number_horizontal_line`: Helper functions to create lines for numbers
3. `number_1`, `number_4`, `number_7`: Create individual numbers using combinations of lines
4. `sudoku_grid`: Creates the complete 9x9 grid by placing 81 cells
5. `sudoku_board`: Combines the grid with numbers placed in specific cells

The board is created in the x-y plane with numbers slightly raised above the grid. The numbers are constructed using simple geometric primitives (cylinders for lines). For simplicity, I've only implemented three numbers (1, 4, and 7), but the same approach could be extended to create all digits.

The grid cells are white-ish (0.9, 0.9, 0.9) and the numbers are dark (0.1, 0.1, 0.1) for good contrast. The board is sized using small units (0.1 per cell) to keep it compact, and the numbers are scaled to fit nicely within the cells.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
