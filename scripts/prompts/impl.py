from pathlib import Path
import numpy as np
from helper import *
import mitsuba as mi
from example_postprocess import parse_program
from engine.utils.graph_utils import strongly_connected_components, get_root
from engine.constants import PROJ_DIR, ENGINE_MODE, ONLY_RENDER_ROOT
import traceback
import ipdb

import random
import math
import sys


from tu.train_setup import set_seed
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    from dsl_utils import library, set_lock_enabled
    from shape_utils import create_hole

    if ENGINE_MODE == 'mi':
        from mi_helper import execute
    elif ENGINE_MODE == 'minecraft':
        from minedojo_helper import execute
    else:
        raise TypeError(f'Engine mode does not exist: {ENGINE_MODE}')

    set_seed(0)

    save_dir = Path(__file__).parent / 'renderings'
    save_dir.mkdir(exist_ok=True)

    if not ONLY_RENDER_ROOT:
        nodes = []
        while len(nodes) < len(library):
            for name in list(library.keys()):
                if name in [node.name for node in nodes]:
                    continue
                node: 'Hole' = create_hole(name=name, docstring=library[name]['docstring'],
                                        check=library[name]['check'])
                node.implement(lambda: library[name]['__target__'])
                nodes.append(node)

                try:
                    execute(node(), save_dir=(save_dir / name).as_posix())
                except Exception as e:
                    print('first attempt')
                    print(e)
                    print(traceback.format_exc())
                    traceback.print_exc()
                    try:
                        args, kwargs = library[name]['last_call']
                        with set_lock_enabled(True):
                            _ = execute(node(*args, **kwargs), save_dir=(save_dir / name).as_posix())
                    except Exception as e:
                        print('second attempt')
                        print(e)
                        print(traceback.format_exc())
    else:
        exp_program_path = Path(__file__).parent / 'program.py'
        _, library_equiv = parse_program(exp_program_path)
        scc = strongly_connected_components(library_equiv)
        print(f'{scc=}')
        root_name = get_root(library_equiv)
        root = library_equiv[root_name]

        execute(root(), save_dir=(save_dir / root.name).as_posix())