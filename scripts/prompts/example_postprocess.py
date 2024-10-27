import importlib.util
import traceback
import inspect
from engine.utils.graph_utils import strongly_connected_components, get_root

# import from `impl.py`
import sys
from pathlib import Path
sys.path.append((Path(__file__).parent.parent / 'prompts').as_posix())

from helper import *
import numpy as np
import random
import math

from dsl_utils import library, set_lock_enabled, set_fake_call_enabled
from _shape_utils import Hole  # don't use the library here


def parse_program(path: str) -> tuple[dict[str, dict], dict[str, Hole]]:
    library.clear()  # FIXME hack, this is bad practice

    tmp_path = Path(__file__).parent.parent / 'outputs/tmp/parse_program.py'
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = ['from helper import *\n'] + lines
    with open(tmp_path, 'w') as f:
        f.writelines(lines)

    spec = importlib.util.spec_from_file_location("input_program", tmp_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # register all functions including local ones

    library_equiv: dict[str, Hole] = {}
    while len(library_equiv) < len(library):
        for name in list(library.keys()):
            if name in library_equiv:
                continue
            node = Hole(name=name, docstring=library[name]['docstring'], check=library[name]['check'], normalize=False)
            # manually implement the function
            node.fn = library[name]['__target__']
            library_equiv[name] = node

            try:
                _ = node()
            except TypeError:
                try:
                    args, kwargs = library[name]['last_call']
                    with set_lock_enabled(True):
                        _ = node(*args, **kwargs)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
            except Exception as e:
                print(e)
                traceback.print_exc()
    print(library_equiv)
    for name, node in library_equiv.items():
        with set_fake_call_enabled(True) as _children:
            _children.clear()
            try:
                _ = node()
            except TypeError:
                try:
                    args, kwargs = library[node.name]['last_call']
                    with set_lock_enabled(True):
                        _ = node(*args, **kwargs)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
            child_names = _children.copy()
            _children.clear()
        # manually record the dependency as the program won't call `create_hole`
        node.children = set()
        for child_name in child_names:
            child_node = library_equiv[child_name]
            node.children.add(child_node)
            child_node.add_parent(node)

    return library, library_equiv


def process(path: str):
    library, library_equiv = parse_program(path)
    scc = strongly_connected_components(library_equiv)
    print(f'{scc=}')
    try:
        root = get_root(library_equiv)
        print(f'{root=}')
    except Exception as e:
        # sometimes a functino is implemented but never used, so there is no shared ancestor
        print(e)

    library_source: dict[str, str] = {}
    for name in library.keys():
        try:
            library_source[name] = inspect.getsource(library[name]['__target__'])
        except TypeError as e:
            print(e)

    return library, library_equiv, library_source
