import argparse
import sys
import ipdb
import traceback
import shlex
import os
from tu.loggers.utils import print_vcv_url
from typing import Literal, Optional
import logging
import subprocess
import json
import numpy as np
from helper import Shape
from sketch_helper import parse_program
from engine.utils.graph_utils import strongly_connected_components, get_root, calculate_node_depths
from engine.utils.argparse_utils import setup_save_dir, modify_string_for_file
from engine.utils.execute_utils import execute_command
from impl_helper import make_new_library
from impl_utils import execute_from_preset
from _shape_utils import compute_bbox
from engine.constants import ENGINE_MODE, DEBUG, PROJ_DIR
from pathlib import Path


root = Path(__file__).parent.parent

TRANSFORM_JSON_FN = 'transformations.json'
GEOMETRY_JSON_FN = 'geometry_convert_from.json'
NORMALIZE_JSON_FN = 'normalization.json'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--program-path', type=str, default=(root / 'assets/coke2.py').as_posix())
    parser.add_argument('--root', type=str, default=None, help='root node')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite existing cache')
    parser.add_argument('--log-dir', type=str, default=(root / 'outputs' / Path(__file__).stem).as_posix())
    parser.add_argument('--log-unique', action='store_true', help='append timestamp to logging dir')
    parser.add_argument('--depth', type=int, default=-1, help='replace shapes for nodes from this depth with neural assets')
    return parser


def load_shape(shape: Shape):
    # An example shape:
    # shape = [{
    #     'type': 'ply', 'filename': ply_save_path.as_posix(),
    #     'to_world': np.eye(4),
    #     'bsdf': {'type': 'diffuse',
    #              'reflectance': {'type': 'rgb', 'value': filename_to_color(ply_save_path.as_posix())}},
    #     'info': {'docstring': prompt, 'stack': []},
    # }]
    nets = []
    individual_info = {}
    for s in shape:
        # assert s['type'] == 'ply', s
        # mesh = load_net(s['filename'])  # or maybe load as gaussian splatting network
        # to_world = s['to_world']
        # mesh.apply_transform(to_world)  # transform all meshes to world space
        # if 'filename' in s.keys() and os.path.exists(s['filename']):
        #     optimize_init = (s['filename'], s['info']['docstring'], True)
        # else:
        #     optimize_init = (None, s['info']['docstring'], False)
        # s.update({'optimize_init_info': optimize_init})
        # nets.append(s)

        prompt = s['info']['docstring']

        # FIXME debug
        # prompt = 'a blue cube'
        if prompt in individual_info.keys():
            to_world_this = individual_info[prompt] # a Nx4x4 numpy array
            to_world = np.array(s['to_world'])[None]
            to_world_this = np.concatenate([to_world_this, to_world], axis=0)
            individual_info[prompt] = to_world_this
        
        else:
            to_world = np.array(s['to_world'])[None]
            individual_info.update({prompt: to_world})

    # TODO
    return individual_info


def preprocess_program(program_path: str, depth: int, node: Optional[str] = None, overwrite: bool = False) -> str:
    cache_dir = root / 'cache' / Path(__file__).stem / 'preprocess_program' / f'{modify_string_for_file(program_path)}_depth_{depth:02d}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    print_vcv_url(cache_dir.as_posix())
    transform_json_path = cache_dir / TRANSFORM_JSON_FN
    geometry_json_path = cache_dir / GEOMETRY_JSON_FN
    normalize_json_path = cache_dir / NORMALIZE_JSON_FN

    if not overwrite and transform_json_path.exists() and geometry_json_path.exists() and normalize_json_path.exists():
        print(f'[INFO] cached info found: {cache_dir}')
        return cache_dir.as_posix()

    library, library_equiv, _ = parse_program([program_path], roots=None if node is None else [node])
    if node is None:
        if ENGINE_MODE == 'mi_from_minecraft':
            node = next(iter(library.keys()))  # FIXME make it unified? also need to fix `impl_utils.py`
        else:
            node = next(reversed(library.keys()))
        # node = get_root(library_equiv)
    prompt = library[node]['docstring']  # use root function docstring as global prompt

    with open((cache_dir / 'prompt.txt').as_posix(), 'w') as f:
        f.write(prompt)

    node_depths = calculate_node_depths(library_equiv, node)
    if depth == -1:
        depth = max(node_depths.values())
    if depth > max(node_depths.values()):
        raise ValueError(f'Invalid depth {depth}, max depth is {max(node_depths.values())}')

    new_library = make_new_library(
        library=library,
        library_equiv=library_equiv,
        tree_depth=depth,
        engine_mode='box',
        root=node,
    )
    shape = new_library[node]['__target__']()

    box = compute_bbox(shape)
    with open(normalize_json_path.as_posix(), 'w') as f:
        json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in box._asdict().items()}, f, indent=4)

    _ = execute_from_preset(shape, save_dir=(cache_dir / 'renderings').as_posix(),
                            preset_id='indoors_no_window', timestep=(0, 1))
    individual_info = load_shape(shape)
    with open(transform_json_path.as_posix(), 'w') as f:
        json.dump(individual_info, f, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else obj, indent=4)

    geometry_info: dict[str, str] = {}
    for ind_prompt in individual_info.keys():
        geometry_info[ind_prompt] = compute_geometry_convert_from(ind_prompt)
        print(f'[INFO] {ind_prompt=} {geometry_info[ind_prompt]}')

    with open(geometry_json_path.as_posix(), 'w') as f:
        json.dump(geometry_info, f, indent=4)
        # json.dump({pt: f'shap-e:{pt}' for pt in individual_info.keys()}, f)

    return cache_dir.as_posix()


def compute_geometry_convert_from(prompt: str, init: Literal['empty', 'shap-e', 'lrm'] = 'shap-e') -> str:
    # return ""
    prompt = prompt.lower()
    cache_dir = root / 'cache' / Path(__file__).stem / 'compute_geometry_convert_from' / modify_string_for_file(prompt)
    cache_dir.mkdir(parents=True, exist_ok=True)
    exp_tag = f"init_{init}"
    exp_root_dir = cache_dir / 'threestudio'
    exp_dir = exp_root_dir / 'gs-sds-generation' / exp_tag
    print_vcv_url(exp_dir.as_posix())
    checkpoint_path = exp_dir / 'ckpts/epoch=0-step=1200.ckpt'
    if checkpoint_path.exists():
        return checkpoint_path.as_posix()

    # https://github.com/cxh0519/threestudio-gaussiandreamer/blob/master/configs/gaussiandreamer.yaml
    command = [
        # 'python', (Path(PROJ_DIR) / 'engine/launch_threestudio.py').as_posix(),
        'python', '/viscam/projects/concepts/third_party/threestudio/launch.py',
        '--config', '/viscam/projects/concepts/third_party/threestudio/custom/threestudio-gaussiandreamer/configs/gaussiandreamer.yaml',
        '--train', '--gpu', '0',
        f'system.prompt_processor.prompt="{prompt}"',
        f'tag="{exp_tag}"',
        f'exp_root_dir={exp_root_dir.as_posix()}',
        'use_timestamp=False',
    ]
    if init != 'empty':
        command.append(f'system.geometry.geometry_convert_from="{init}:{prompt}"')

    success = execute_command(
        ' '.join(command),
        save_dir=cache_dir.as_posix(),
        print_stdout=False,
        print_stderr=False,
        cwd='/viscam/projects/concepts/third_party/threestudio',
    )
    if success != 0:
        import ipdb; ipdb.set_trace()
    if not checkpoint_path.exists():
        import ipdb; ipdb.set_trace()
    return checkpoint_path.as_posix()


def main():
    args = get_parser().parse_args()
    log_dir = setup_save_dir(args.log_dir, args.log_unique)
    cache_dir = Path(preprocess_program(args.program_path, args.depth, args.root, overwrite=args.overwrite))
    print_vcv_url(cache_dir.as_posix())
    transform_json_path = cache_dir / TRANSFORM_JSON_FN
    geometry_json_path = cache_dir / GEOMETRY_JSON_FN
    normalize_json_path = cache_dir / NORMALIZE_JSON_FN
    with open(cache_dir / 'prompt.txt', 'r') as f:
        prompt = f.read().strip()
    print(f'[INFO] {prompt=}')

    command = [
        'python', (Path(PROJ_DIR) / 'engine/launch_threestudio.py').as_posix(),
        # '/viscam/projects/concepts/third_party/threestudio/launch.py',
        '--config', (Path(PROJ_DIR) / 'engine/threestudio_related/config/multigaussiandreamer.yaml').as_posix(),
        '--train', '--gpu', '0',
        f'system.prompt_processor.prompt="{prompt}"',
        f'system.pre_transformations_path="{transform_json_path.as_posix()}"',
        f'system.pre_geometry_convert_from_path="{geometry_json_path.as_posix()}"',
        f'system.pre_normalization_path="{normalize_json_path.as_posix()}"',
        f'exp_root_dir={(root / "outputs/threestudio").as_posix()}'
    ]
    execute_command(' '.join(command), save_dir=log_dir.as_posix(), print_stdout=True, print_stderr=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        ipdb.post_mortem(tb)
