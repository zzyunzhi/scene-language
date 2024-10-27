

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
I'll help you create a Sudoku board with numbers represented as 3D shapes. Here's the implementation:

"""
from helper import *

"""
A 9x9 Sudoku board partially filled with numbers
"""

@register()
def grid_line(length: float, thickness: float, is_horizontal: bool = True) -> Shape:
    """Creates a single grid line"""
    if is_horizontal:
        scale = (length, thickness, thickness)
    else:
        scale = (thickness, length, thickness)
    return primitive_call('cube', color=(0.2, 0.2, 0.2), shape_kwargs={'scale': scale})

@register()
def number_1(height: float) -> Shape:
    """Creates number 1 using cylinders"""
    thickness = height * 0.1
    return primitive_call('cylinder',
                         color=(0.1, 0.1, 0.8),
                         shape_kwargs={'radius': thickness/2,
                                     'p0': (0, 0, 0),
                                     'p1': (0, height, 0)})

@register()
def number_2(height: float) -> Shape:
    """Creates number 2 using cylinders"""
    thickness = height * 0.1
    radius = thickness/2

    # Three horizontal and two vertical segments
    h1 = primitive_call('cylinder', color=(0.1, 0.1, 0.8),
                       shape_kwargs={'radius': radius, 'p0': (0, height, 0), 'p1': (height*0.5, height, 0)})
    h2 = primitive_call('cylinder', color=(0.1, 0.1, 0.8),
                       shape_kwargs={'radius': radius, 'p0': (0, height/2, 0), 'p1': (height*0.5, height/2, 0)})
    h3 = primitive_call('cylinder', color=(0.1, 0.1, 0.8),
                       shape_kwargs={'radius': radius, 'p0': (0, 0, 0), 'p1': (height*0.5, 0, 0)})
    v1 = primitive_call('cylinder', color=(0.1, 0.1, 0.8),
                       shape_kwargs={'radius': radius, 'p0': (height*0.5, height, 0), 'p1': (height*0.5, height/2, 0)})
    v2 = primitive_call('cylinder', color=(0.1, 0.1, 0.8),
                       shape_kwargs={'radius': radius, 'p0': (0, height/2, 0), 'p1': (0, 0, 0)})

    return concat_shapes(h1, h2, h3, v1, v2)

@register()
def sudoku_grid(size: float) -> Shape:
    """Creates the 9x9 Sudoku grid"""
    cell_size = size / 9
    thick_line = cell_size * 0.04  # Thicker lines for 3x3 boxes
    thin_line = cell_size * 0.02   # Thinner lines for cells

    def loop_fn(i: int) -> Shape:
        # Horizontal lines
        h_line = library_call('grid_line', length=size,
                            thickness=thick_line if i % 3 == 0 else thin_line,
                            is_horizontal=True)
        # Vertical lines
        v_line = library_call('grid_line', length=size,
                            thickness=thick_line if i % 3 == 0 else thin_line,
                            is_horizontal=False)

        return concat_shapes(
            transform_shape(h_line, translation_matrix((0, i * cell_size, 0))),
            transform_shape(v_line, translation_matrix((i * cell_size, 0, 0)))
        )

    # Create all grid lines
    grid = loop(10, loop_fn)  # 10 lines needed for 9x9 grid

    # Add base plate
    base = primitive_call('cube', color=(0.9, 0.9, 0.9),
                         shape_kwargs={'scale': (size, size, thick_line)})

    return concat_shapes(base, grid)

@register()
def place_number(num: int, row: int, col: int, cell_size: float) -> Shape:
    """Places a number in the specified cell"""
    number_height = cell_size * 0.6

    if num == 1:
        number = library_call('number_1', height=number_height)
    else:  # num == 2, add more numbers as needed
        number = library_call('number_2', height=number_height)

    # Center the number in its cell
    number_center = compute_shape_center(number)
    cell_center = (col * cell_size + cell_size/2, row * cell_size + cell_size/2, 0)
    offset = (cell_center[0] - number_center[0],
             cell_center[1] - number_center[1],
             cell_size * 0.1)  # Slight elevation above the board

    return transform_shape(number, translation_matrix(offset))

@register()
def sudoku_board() -> Shape:
    """Creates a complete Sudoku board with some numbers"""
    board_size = 2.0  # 2 units wide/tall
    board = library_call('sudoku_grid', size=board_size)

    # Example number placement (add more numbers as needed)
    cell_size = board_size / 9
    numbers = concat_shapes(
        library_call('place_number', num=1, row=0, col=0, cell_size=cell_size),
        library_call('place_number', num=2, row=1, col=1, cell_size=cell_size),
        library_call('place_number', num=1, row=2, col=2, cell_size=cell_size),
        library_call('place_number', num=2, row=3, col=3, cell_size=cell_size),
        library_call('place_number', num=1, row=4, col=4, cell_size=cell_size)
    )

    return concat_shapes(board, numbers)
"""

This implementation creates a 3D Sudoku board with the following features:

1. A 9x9 grid with thicker lines separating the 3x3 boxes and thinner lines for individual cells
2. A light gray base plate
3. Dark gray grid lines
4. Blue 3D numbers (currently implemented 1 and 2) that are slightly elevated above the board
5. Numbers are constructed using cylinders to create a 3D effect
6. The board is 2 units wide and tall, making it easy to scale and position
7. Example number placement in a diagonal pattern (can be modified to create any valid Sudoku puzzle)

The main components are:
- `grid_line`: Creates individual grid lines
- `number_1` and `number_2`: Create 3D representations of numbers
- `sudoku_grid`: Creates the complete 9x9 grid with appropriate line thicknesses
- `place_number`: Handles the placement of numbers in specific cells
- `sudoku_board`: Combines all elements into the final board

You can create the complete Sudoku board by calling:
```python
board = library_call('sudoku_board')
```

More numbers can be added by creating additional number functions (number_3 through number_9) following the same pattern as number_1 and number_2, and adding more number placements in the sudoku_board function.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
