import os
import argparse
from pathlib import Path
from engine.constants import ENGINE_MODE, PROMPT_MODE
import time

root = Path(__file__).parent.parent.absolute()

header_in_dir = root / 'prompts'
header_out_dir_base = root / 'outputs' / 'stubgen'


def read_header(path: str) -> str:
    with open(path, 'r') as f:
        s = []
        for ss in f.readlines():
            s.append(ss)
            if ss.startswith('__all__'):
                s = []
    s = ''.join(s)
    return s


def read_file(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

def read_import() -> str:
    # TODO hard-coded
    return """from typing import NamedTuple, Any, Callable, Literal
import math
import numpy as np
"""
    with open(header_in_dir / 'helper.py', 'r') as f:
        s = []
        for ss in f.readlines():
            if 'import *' in ss:
                continue
            s.append(ss)
    s = ''.join(s)
    return s


def collect_header(header_out_dir: str) -> str:
    header_out_dir = Path(header_out_dir)

    header = [
        '''"""This module contains a Domain-Specific Language (DSL) designed 
with built-in support for loops and functions for shape construction and transformation.
"""\n''',
        read_import(),
        "# type aliases and DSL syntax sugar",
        # read_header((header_in_dir / 'type_utils.py').as_posix()),
        # read_header((header_out_dir / 'type_utils.pyi').as_posix()),
        # read_header((header_out_dir / 'type_utils.pyi').as_posix()). \
        #     replace('P: Incomplete', 'P = Any  # 3D vector, e.g., a point or direction'). \
        #     replace('T: Incomplete', 'T = Any  # 4x4 transformation matrix'),
        """\
P = Any  # 3D vector, e.g., a point or direction
T = Any  # 4x4 transformation matrix
Shape = list[dict[str, Any]]  # a shape is a list of primitive shapes
""",
        # "# flow controls",
        # read_header((header_out_dir / 'dsl_utils.pyi').as_posix()). \
        #     replace('RR: Incomplete', 'RR = Callable[["RR"], Callable[[int], Shape]]'),
        "# shape function library utils",
        read_header((header_out_dir / 'dsl_utils.pyi').as_posix()),
        read_header((header_out_dir / f'_engine_utils_{ENGINE_MODE}.pyi').as_posix()),
        "# control flows",
        read_header((header_out_dir / 'flow_utils.pyi').as_posix()),
        "# shape manipulation",
        read_header((header_out_dir / 'shape_utils.pyi').as_posix()),
    ]
    if PROMPT_MODE not in ['sketch']:
        header.extend([
            "# pose transformation",
            read_header((header_out_dir / 'math_utils_minecraft.pyi').as_posix()) if ENGINE_MODE == 'minecraft' else read_header((header_out_dir / 'math_utils.pyi').as_posix()),
        ])
    if PROMPT_MODE in ['calc']:
        header.extend([
            "# calculate locations and sizes of shape bounding boxes",
            read_header((header_out_dir / 'calc_utils.pyi').as_posix()),
        ])
    if PROMPT_MODE in ['assert']:
        header.extend([
            "# shape constraints",
            read_header((header_out_dir / 'assert_utils.pyi').as_posix()),
        ])

    # Attach the minecraft type utils .pyi
    # if ENGINE_MODE == 'minecraft':
    #     header.extend([
    #         "# Valid minecraft types for blocks and entities. You can ONLY use values from these sets when invoking a primitive call.",
    #         '# We will be testing to make sure you are using values from this list, so pay very careful attention when selecting block types that it is valid.',
    #         read_file((header_in_dir / 'minecraft_types.py').as_posix())
    #     ])
    
    header = '\n'.join(header)

    with open((header_out_dir / 'header.pyi').as_posix(), 'w') as f:
        f.write(header)

    return header


def read_examples() -> str:
    with open((header_in_dir / 'examples.py').as_posix(), 'r') as f:
        s = f.read()
        ind = s.index('if __name__ == "__main__":')
        s = s[:ind]
    return s


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--engine-mode',
                        type=str, choices=['neural', 'mi'], required=True)
    return parser


def main():
    # parser = get_parser()
    # args = parser.parse_args()

    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    header_out_dir = header_out_dir_base / f'{timestamp}-{ENGINE_MODE}-{PROMPT_MODE}'
    assert not header_out_dir.exists(), header_out_dir

    files = []
    files.extend(header_in_dir.glob('[!_]*_utils.py'))  # Globbing, not regex
    files.extend(header_in_dir.glob(f"_engine_utils_{ENGINE_MODE}.py"))  # Specific engine mode file
    files.extend(header_in_dir.glob(f"math_utils_minecraft.py"))

    command = [
        'stubgen', '--include-docstrings',  # '--export-less',
        '-o', header_out_dir.as_posix(),
        *[file.as_posix() for file in files],
    ]
    os.system(' '.join(command))

    collect_header(header_out_dir.as_posix())


if __name__ == "__main__":
    main()
