

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
Basílica de la Sagrada Família
"""

@register()
def tower(height: float, base_radius: float, color: tuple[float, float, float] = (0.8, 0.8, 0.7)) -> Shape:
    """Creates a tower with a conical top and Gaudí-inspired details"""
    base_height = height * 0.6
    top_height = height * 0.4

    # Create the cylindrical base with more substantial width
    base = primitive_call('cylinder',
                         shape_kwargs={'radius': base_radius,
                                      'p0': (0, 0, 0),
                                      'p1': (0, base_height, 0)},
                         color=color)

    # Create the conical top with hyperboloid-inspired detailing
    def loop_fn(i) -> Shape:
        progress = i / 10
        y_pos = base_height + progress * top_height

        # Create hyperboloid effect with waist in the middle
        curve_factor = 1 - 4 * (progress - 0.5) * (progress - 0.5)
        radius = base_radius * (1 - progress) * (0.8 + 0.2 * curve_factor)

        # Add texture to the spire
        segment = primitive_call('cylinder',
                               shape_kwargs={'radius': radius,
                                            'p0': (0, y_pos, 0),
                                            'p1': (0, y_pos + top_height/10, 0)},
                               color=color)

        # Add decorative elements around each segment for more Gaudí-like appearance
        def detail_fn(j) -> Shape:
            angle = 2 * math.pi * j / 8
            x = radius * 1.1 * math.cos(angle)
            z = radius * 1.1 * math.sin(angle)
            detail = primitive_call('sphere',
                                  shape_kwargs={'radius': radius * 0.15},
                                  color=(color[0]*0.9, color[1]*0.9, color[2]*0.9))
            return transform_shape(detail, translation_matrix((x, y_pos, z)))

        details = loop(8, detail_fn)
        return concat_shapes(segment, details)

    cone_parts = loop(10, loop_fn)

    # Add a decorative top
    top_element = primitive_call('sphere', shape_kwargs={'radius': base_radius * 0.2}, color=color)
    top_element = transform_shape(top_element, translation_matrix((0, height, 0)))

    return concat_shapes(base, cone_parts, top_element)

@register()
def decorative_element(size: float, color: tuple[float, float, float] = (0.75, 0.75, 0.65)) -> Shape:
    """Creates a decorative element for the towers with spikes in all directions"""
    sphere = primitive_call('sphere', shape_kwargs={'radius': size/2}, color=color)

    # Add spikes in all directions (not just XY plane)
    def loop_fn(i) -> Shape:
        if i < 8:  # XY plane
            angle = 2 * math.pi * i / 8
            spike = primitive_call('cylinder',
                                 shape_kwargs={'radius': size/10,
                                              'p0': (0, 0, 0),
                                              'p1': (size * math.cos(angle), size * math.sin(angle), 0)},
                                 color=color)
        elif i < 12:  # XZ plane
            angle = 2 * math.pi * (i-8) / 4
            spike = primitive_call('cylinder',
                                 shape_kwargs={'radius': size/10,
                                              'p0': (0, 0, 0),
                                              'p1': (size * math.cos(angle), 0, size * math.sin(angle))},
                                 color=color)
        else:  # YZ plane
            angle = 2 * math.pi * (i-12) / 4
            spike = primitive_call('cylinder',
                                 shape_kwargs={'radius': size/10,
                                              'p0': (0, 0, 0),
                                              'p1': (0, size * math.cos(angle), size * math.sin(angle))},
                                 color=color)
        return spike

    spikes = loop(16, loop_fn)
    return concat_shapes(sphere, spikes)

@register()
def decorated_tower(height: float, base_radius: float, color: tuple[float, float, float] = (0.8, 0.8, 0.7)) -> Shape:
    """Creates a tower with decorative elements integrated into the structure"""
    tower_shape = library_call('tower', height=height, base_radius=base_radius, color=color)

    # Add decorative elements along the tower
    def loop_fn(i) -> Shape:
        y_pos = height * 0.2 + i * (height * 0.5) / 5
        element_size = base_radius * 0.5
        element = library_call('decorative_element', size=element_size, color=color)

        # Place elements around the tower, integrated with the structure
        def element_loop_fn(j) -> Shape:
            angle = 2 * math.pi * j / 4
            x = base_radius * 0.9 * math.cos(angle)
            z = base_radius * 0.9 * math.sin(angle)
            return transform_shape(element, translation_matrix((x, y_pos, z)))

        return loop(4, element_loop_fn)

    decorations = loop(5, loop_fn)

    # Add a decorative element at the top
    top_element = library_call('decorative_element', size=base_radius * 0.3, color=color)
    top_element = transform_shape(top_element, translation_matrix((0, height, 0)))

    return concat_shapes(tower_shape, decorations, top_element)

@register()
def parabolic_arch(width: float, height: float, depth: float, color: tuple[float, float, float] = (0.8, 0.8, 0.7)) -> Shape:
    """Creates a Gaudí-style parabolic arch"""
    def arch_segment_fn(i) -> Shape:
        # Create parabolic curve: y = 4*h*(x/w)*(1-x/w) where h=height, w=width
        progress = i / 20
        x = (progress - 0.5) * width
        # Parabolic equation
        y = 4 * height * (0.25 - (x/width) * (x/width))

        segment = primitive_call('cylinder',
                               shape_kwargs={'radius': depth/10,
                                            'p0': (x, 0, -depth/2),
                                            'p1': (x, y, -depth/2)},
                               color=color)

        # Add the other side of the arch
        segment2 = primitive_call('cylinder',
                                shape_kwargs={'radius': depth/10,
                                             'p0': (x, 0, depth/2),
                                             'p1': (x, y, depth/2)},
                                color=color)

        # Connect the two sides
        if i % 4 == 0:
            connector = primitive_call('cylinder',
                                     shape_kwargs={'radius': depth/15,
                                                  'p0': (x, y, -depth/2),
                                                  'p1': (x, y, depth/2)},
                                     color=color)
            return concat_shapes(segment, segment2, connector)

        return concat_shapes(segment, segment2)

    arch_segments = loop(21, arch_segment_fn)
    return arch_segments

@register()
def cruciform_body(width: float, length: float, height: float, color: tuple[float, float, float] = (0.85, 0.85, 0.75)) -> Shape:
    """Creates the main body of the basilica with a cruciform layout"""
    # Create a true cruciform structure by using cylinders for the main nave and transept
    # Main nave (longer part of the cross)
    main_nave = primitive_call('cylinder',
                             shape_kwargs={'radius': width/2,
                                          'p0': (0, 0, -length/2),
                                          'p1': (0, 0, length/2)},
                             color=color)

    # Transept (crossing section)
    transept = primitive_call('cylinder',
                            shape_kwargs={'radius': width/2,
                                         'p0': (-length*0.4, 0, 0),
                                         'p1': (length*0.4, 0, 0)},
                            color=color)

    # Add height to the structure
    def extrude_fn(i) -> Shape:
        y_pos = i * (height / 10)
        nave_slice = primitive_call('cylinder',
                                  shape_kwargs={'radius': width/2,
                                               'p0': (0, y_pos, -length/2),
                                               'p1': (0, y_pos, length/2)},
                                  color=color)

        transept_slice = primitive_call('cylinder',
                                      shape_kwargs={'radius': width/2,
                                                   'p0': (-length*0.4, y_pos, 0),
                                                   'p1': (length*0.4, y_pos, 0)},
                                      color=color)

        return concat_shapes(nave_slice, transept_slice)

    structure = loop(10, extrude_fn)

    # Add a roof
    roof_nave = primitive_call('cylinder',
                             shape_kwargs={'radius': width/2,
                                          'p0': (0, height, -length/2),
                                          'p1': (0, height, length/2)},
                             color=color)

    roof_transept = primitive_call('cylinder',
                                 shape_kwargs={'radius': width/2,
                                              'p0': (-length*0.4, height, 0),
                                              'p1': (length*0.4, height, 0)},
                                 color=color)

    # Add parabolic arches along the main nave
    def nave_arch_fn(i) -> Shape:
        z_pos = (i - 3) * (length / 7)
        arch = library_call('parabolic_arch', width=width*0.8, height=height*0.8, depth=width*0.1, color=color)
        return transform_shape(arch, translation_matrix((0, 0, z_pos)))

    nave_arches = loop(7, nave_arch_fn)

    # Add parabolic arches along the transept
    def transept_arch_fn(i) -> Shape:
        x_pos = (i - 2) * (length*0.8 / 5)
        arch = library_call('parabolic_arch', width=width*0.8, height=height*0.8, depth=width*0.1, color=color)
        arch = transform_shape(arch, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        return transform_shape(arch, translation_matrix((x_pos, 0, 0)))

    transept_arches = loop(5, transept_arch_fn)

    # Roof structure with hyperboloid-inspired curves
    def roof_detail_fn(i) -> Shape:
        # Add details along both x and z axes for proper cruciform coverage
        if i < 10:
            # X-axis details
            x_pos = (i - 4.5) * (width / 9)
            detail = primitive_call('cylinder',
                                   shape_kwargs={'radius': width/25,
                                                'p0': (x_pos, height*0.5, -length/2),
                                                'p1': (x_pos, height*0.5, length/2)},
                                   color=color)
        else:
            # Z-axis details
            z_pos = ((i-10) - 4.5) * (length / 9)
            detail = primitive_call('cylinder',
                                   shape_kwargs={'radius': width/25,
                                                'p0': (-length*0.4, height*0.5, z_pos),
                                                'p1': (length*0.4, height*0.5, z_pos)},
                                   color=color)
        return detail

    roof_details = loop(20, roof_detail_fn)

    return concat_shapes(main_nave, transept, structure, roof_nave, roof_transept, nave_arches, transept_arches, roof_details)

@register()
def stained_glass_window(width: float, height: float, color: tuple[float, float, float] = (0.6, 0.8, 0.9)) -> Shape:
    """Creates a church window with Gaudí-inspired stained glass effect"""
    # Window frame with more organic shape
    frame = primitive_call('cylinder',
                         shape_kwargs={'radius': width/2,
                                      'p0': (0, 0, 0),
                                      'p1': (0, 0, width*0.1)},
                         color=(0.7, 0.7, 0.6))

    # Glass with vibrant colors (more Gaudí-like)
    glass = primitive_call('cylinder',
                         shape_kwargs={'radius': width*0.45,
                                      'p0': (0, 0, width*0.05),
                                      'p1': (0, 0, width*0.15)},
                         color=color)

    # Create a rosette pattern typical of Gaudí's designs
    def rosette_fn(i) -> Shape:
        angle = 2 * math.pi * i / 12
        radius = width * 0.35
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        # Vary colors for stained glass effect
        hue_shift = i / 12
        glass_color = (
            (color[0] + hue_shift) % 1.0,
            (color[1] + hue_shift * 0.5) % 1.0,
            (color[2] + hue_shift * 0.25) % 1.0
        )

        petal = primitive_call('sphere',
                             shape_kwargs={'radius': width*0.12},
                             color=glass_color)

        return transform_shape(petal, translation_matrix((x, y, width*0.1)))

    rosette = loop(12, rosette_fn)

    # Center piece
    center = primitive_call('sphere',
                          shape_kwargs={'radius': width*0.15},
                          color=(0.9, 0.7, 0.3))
    center = transform_shape(center, translation_matrix((0, 0, width*0.1)))

    return concat_shapes(frame, glass, rosette, center)

@register()
def entrance(width: float, height: float, depth: float, color: tuple[float, float, float] = (0.75, 0.75, 0.65)) -> Shape:
    """Creates an elaborate entrance for the basilica"""
    # Main arch structure
    base = primitive_call('cube', shape_kwargs={'scale': (width, height * 0.5, depth)}, color=color)

    # Parabolic arch for the entrance
    arch = library_call('parabolic_arch', width=width*0.9, height=height*0.5, depth=depth, color=color)
    arch = transform_shape(arch, translation_matrix((0, height*0.5, 0)))

    # Door with more organic shape
    door = primitive_call('cylinder',
                        shape_kwargs={'radius': width*0.25,
                                     'p0': (0, 0, -depth*0.05),
                                     'p1': (0, 0, depth*0.05)},
                        color=(0.4, 0.3, 0.2))
    door = transform_shape(door, translation_matrix((0, height * 0.25, 0)))

    # Decorative elements around the entrance - more organic and Gaudí-like
    def decor_fn(i) -> Shape:
        angle = math.pi * i / 10
        x = width * 0.5 * math.cos(angle)
        y = height * 0.5 + width * 0.5 * math.sin(angle)

        # Use more organic shapes for decorations
        if i % 2 == 0:
            element = primitive_call('sphere', shape_kwargs={'radius': width * 0.05}, color=color)
        else:
            element = library_call('decorative_element', size=width*0.1,
                                 color=(color[0]*0.9, color[1]*0.9, color[2]*0.9))

        return transform_shape(element, translation_matrix((x, y, depth * 0.4)))

    decorations = loop(11, decor_fn)

    return concat_shapes(base, arch, door, decorations)

@register()
def ground_plane(size: float = 20.0, color: tuple[float, float, float] = (0.7, 0.7, 0.7)) -> Shape:
    """Creates a ground plane"""
    return primitive_call('cube', shape_kwargs={'scale': (size, 0.1, size)}, color=color)

@register()
def facade(width: float, height: float, depth: float, facade_type: str, color: tuple[float, float, float] = (0.8, 0.8, 0.7)) -> Shape:
    """Creates one of the three grand façades (Nativity, Passion, or Glory)"""
    # Base structure with more organic shape
    base = primitive_call('cylinder',
                        shape_kwargs={'radius': width/2,
                                     'p0': (0, 0, -depth/2),
                                     'p1': (0, 0, depth/2)},
                        color=color)

    # Extrude to create height
    def extrude_fn(i) -> Shape:
        y_pos = i * (height*0.7 / 10)
        segment = primitive_call('cylinder',
                               shape_kwargs={'radius': width/2,
                                            'p0': (0, y_pos, -depth/2),
                                            'p1': (0, y_pos, depth/2)},
                               color=color)
        return segment

    structure = loop(10, extrude_fn)

    # Upper part with parabolic arch shape
    def upper_fn(i) -> Shape:
        progress = i / 10
        y_offset = height * 0.7 + progress * height * 0.3
        # Parabolic curve for the top
        x_scale = width * (1 - progress*progress)

        segment = primitive_call('cylinder',
                                shape_kwargs={'radius': x_scale/2,
                                             'p0': (0, y_offset, -depth/2),
                                             'p1': (0, y_offset, depth/2)},
                                color=color)
        return segment

    upper = loop(10, upper_fn)

    # Different decorative elements based on façade type
    if facade_type == "nativity":
        # More organic, nature-inspired elements
        def detail_fn(i) -> Shape:
            x = (i % 5 - 2) * (width * 0.2)
            y = (i // 5) * (height * 0.25) + height * 0.2
            size = width * 0.08

            # Use more organic shapes
            if i % 3 == 0:
                detail = library_call('decorative_element', size=size,
                                    color=(0.7, 0.8, 0.6))  # Green-tinted for nature theme
            else:
                detail = primitive_call('sphere',
                                      shape_kwargs={'radius': size/2},
                                      color=(0.8, 0.7, 0.6))  # Earth tones

            return transform_shape(detail, translation_matrix((x, y, depth * 0.5)))

        details = loop(15, detail_fn)

    elif facade_type == "passion":
        # More angular, severe elements with darker colors
        def detail_fn(i) -> Shape:
            x = (i % 3 - 1) * (width * 0.3)
            y = (i // 3) * (height * 0.3) + height * 0.2

            # More angular shapes for Passion facade
            if i % 2 == 0:
                detail = primitive_call('cube',
                                      shape_kwargs={'scale': (width*0.15, width*0.15, depth*0.2)},
                                      color=(0.6, 0.5, 0.5))  # Darker, reddish tint
            else:
                detail = primitive_call('cylinder',
                                      shape_kwargs={'radius': width*0.06,
                                                   'p0': (0, 0, 0),
                                                   'p1': (0, width*0.2, 0)},
                                      color=(0.5, 0.5, 0.6))  # Bluish-gray

            return transform_shape(detail, translation_matrix((x, y, depth * 0.5)))

        details = loop(9, detail_fn)

    else:  # "glory"
        # Grand, triumphant elements with golden tones
        def detail_fn(i) -> Shape:
            angle = 2 * math.pi * i / 12
            radius = width * 0.4
            x = radius * math.cos(angle)
            y = radius * math.sin(angle) + height * 0.5

            # Radiant, glory-themed elements
            if i % 3 == 0:
                detail = primitive_call('sphere',
                                      shape_kwargs={'radius': width*0.06},
                                      color=(0.9, 0.8, 0.3))  # Golden
            else:
                detail = library_call('decorative_element',
                                    size=width*0.08,
                                    color=(0.9, 0.7, 0.4))  # Warm golden

            return transform_shape(detail, translation_matrix((x, y, depth * 0.5)))

        details = loop(12, detail_fn)

    # Add an entrance to each façade - properly integrated
    entrance_shape = library_call('entrance', width=width * 0.6, height=height * 0.6, depth=depth * 0.5)
    entrance_shape = transform_shape(entrance_shape, translation_matrix((0, 0, depth * 0.25)))

    return concat_shapes(base, structure, upper, details, entrance_shape)

@register()
def sagrada_familia() -> Shape:
    """Creates the Basílica de la Sagrada Família"""
    # Ground plane
    ground = library_call('ground_plane')

    # Main body - positioned at ground level
    main = library_call('cruciform_body', width=5.0, length=8.0, height=4.0)
    main = transform_shape(main, translation_matrix((0, 2.05, 0)))  # Raise above ground

    # Three grand façades - properly integrated with the main body
    nativity_facade = library_call('facade', width=5.0, height=6.0, depth=1.5, facade_type="nativity")
    nativity_facade = transform_shape(nativity_facade, translation_matrix((0, 2.05, 4.0)))

    passion_facade = library_call('facade', width=5.0, height=6.0, depth=1.5, facade_type="passion")
    passion_facade = transform_shape(passion_facade, rotation_matrix(math.pi, (0, 1, 0), (0, 0, 0)))
    passion_facade = transform_shape(passion_facade, translation_matrix((0, 2.05, -4.0)))

    glory_facade = library_call('facade', width=4.0, height=6.0, depth=1.5, facade_type="glory")
    glory_facade = transform_shape(glory_facade, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    glory_facade = transform_shape(glory_facade, translation_matrix((4.0, 2.05, 0)))

    # Towers - positioned according to the architectural plan
    # 4 towers at each facade (12 Apostles), 4 Evangelists, 1 Mary, 1 Jesus
    towers = []

    # Nativity facade towers (4 Apostles)
    tower_heights = [7.0, 7.2, 6.8, 7.1]  # Fixed heights instead of random
    for i in range(4):
        x = 1.5 * (1 if i % 2 == 0 else -1)
        z = 4.0 + (0.8 if i < 2 else -0.8)
        height = tower_heights[i]
        tower = library_call('decorated_tower', height=height, base_radius=0.5)
        towers.append(transform_shape(tower, translation_matrix((x, 2.05, z))))

    # Passion facade towers (4 Apostles)
    tower_heights = [7.2, 6.9, 7.1, 6.8]  # Fixed heights instead of random
    for i in range(4):
        x = 1.5 * (1 if i % 2 == 0 else -1)
        z = -4.0 + (0.8 if i < 2 else -0.8)
        height = tower_heights[i]
        tower = library_call('decorated_tower', height=height, base_radius=0.5)
        towers.append(transform_shape(tower, translation_matrix((x, 2.05, z))))

    # Glory facade towers (4 Apostles)
    tower_heights = [6.9, 7.1, 7.0, 6.8]  # Fixed heights instead of random
    for i in range(4):
        x = 4.0 + (0.8 if i < 2 else -0.8)
        z = 1.5 * (1 if i % 2 == 0 else -1)
        height = tower_heights[i]
        tower = library_call('decorated_tower', height=height, base_radius=0.5)
        towers.append(transform_shape(tower, translation_matrix((x, 2.05, z))))

    # Four Evangelist towers at the central crossing
    evangelist_heights = [8.5, 8.7, 8.6, 8.4]  # Fixed heights
    for i in range(4):
        x = 2.0 * (1 if i % 2 == 0 else -1)
        z = 2.0 * (1 if i < 2 else -1)
        height = evangelist_heights[i]
        tower = library_call('decorated_tower', height=height, base_radius=0.6)
        towers.append(transform_shape(tower, translation_matrix((x, 2.05, z))))

    # Mary's tower
    mary_tower = library_call('decorated_tower', height=9.0, base_radius=0.6)
    mary_tower = transform_shape(mary_tower, translation_matrix((0, 2.05, 2.0)))

    # Jesus's tower (central, tallest)
    jesus_tower = library_call('decorated_tower', height=12.0, base_radius=0.8)
    jesus_tower = transform_shape(jesus_tower, translation_matrix((0, 2.05, 0)))

    # Windows - properly positioned on the walls
    def window_fn(i) -> Shape:
        # Place windows along the sides of the main body
        side = i // 5
        pos = i % 5

        # Calculate window positions based on the cruciform structure
        if side == 0:  # Front side (z-positive)
            angle = math.pi * (pos - 2) / 5  # Distribute around the cylinder
            x = 2.5 * math.sin(angle)
            z = 4.0 - 0.1  # Slightly inset from the facade
            y = 2.05 + 1.0 + 0.5 * (pos % 3)  # Vary height
            rotation = 0
        elif side == 1:  # Back side (z-negative)
            angle = math.pi * (pos - 2) / 5
            x = 2.5 * math.sin(angle)
            z = -4.0 + 0.1
            y = 2.05 + 1.0 + 0.5 * (pos % 3)
            rotation = math.pi
        elif side == 2:  # Left side (x-negative)
            angle = math.pi * (pos - 2) / 5
            z = 2.5 * math.sin(angle)
            x = -2.5 + 0.1
            y = 2.05 + 1.0 + 0.5 * (pos % 3)
            rotation = math.pi / 2
        else:  # Right side (x-positive)
            angle = math.pi * (pos - 2) / 5
            z = 2.5 * math.sin(angle)
            x = 2.5 - 0.1
            y = 2.05 + 1.0 + 0.5 * (pos % 3)
            rotation = -math.pi / 2

        # Use the more vibrant stained glass windows
        window = library_call('stained_glass_window', width=0.6, height=1.2,
                            color=(0.3+pos*0.1, 0.4+pos*0.1, 0.8-pos*0.1))  # Varied colors
        window = transform_shape(window, rotation_matrix(rotation, (0, 1, 0), (0, 0, 0)))
        return transform_shape(window, translation_matrix((x, y, z)))

    windows = loop(20, window_fn)

    # Combine all elements
    all_towers = concat_shapes(*towers, mary_tower, jesus_tower)

    return concat_shapes(
        ground,
        main,
        all_towers,
        windows,
        nativity_facade,
        passion_facade,
        glory_facade
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
