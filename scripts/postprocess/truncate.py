import argparse
import sys
import traceback
import pdb
import hashlib
import json
import os
import re

import astor
import ast
import shutil

from engine.utils.graph_utils import strongly_connected_components, get_root, calculate_node_depths
from engine.utils.argparse_utils import setup_save_dir
from engine.utils.execute_utils import execute_command
from engine.constants import ENGINE_MODE, DEBUG, PROJ_DIR
from pathlib import Path


prompts_root = Path(PROJ_DIR) / 'scripts' / 'prompts'
sys.path.insert(0, prompts_root.as_posix())

from scripts.prompts.sketch_helper import parse_program
from scripts.prompts.impl_helper import make_new_library, generate_prompt_key
from scripts.prompts.impl_preset import core

def yes_or_no(prompt) -> bool:
    while True:
        response = input(f'{prompt} (y/n): ').lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")


def input_with_confirm(prompt, eval_fn=lambda s: s):
    while True:
        response = input(prompt)
        try:
            response = eval_fn(response)
        except:
            traceback.print_exc()
            print("Input cannot be evaluated.")
        confirm = input(f"You entered: '{response}'. Is this correct? ([y]/n): ").lower()
        if confirm in ['y', 'yes', ""]:
            return response
        else:
            print("Let's try again.")


class FunctionReplacer(ast.NodeTransformer):
    def __init__(self, name, new_node):
        self.name = name
        self.new_node = new_node

    def visit_FunctionDef(self, node):
        if node.name == self.name:
            return self.new_node
        return node


def process_program(path: Path, save_dir: Path, overwrite: bool = False, skip_prompt: bool = False):
    save_path = save_dir / 'program.py'
    if save_path.exists() and not overwrite:
        print(f"File already exists at {save_path}. Skipping.")
        # load_program(save_path)
        return
    print(f'[INFO] Loading from to:')

    library, library_equiv, library_source = parse_program([path], roots=None)

    sccs, scc_edges = strongly_connected_components(defined_fns=library_equiv)
    print(f'Dependency graph: \n{sccs}')

    try:
        root_name = get_root(library_equiv)
    except Exception:
        print(f'Failed to find root in dependency graph.')
        root_name = None
        while root_name not in library.keys():
            root_name = input_with_confirm('Please type root function name: \n')
    depths = calculate_node_depths(library_equiv, root=root_name)
    print(f'Dependency graph: \n{depths}')
    root_scc_ind = None
    for scc_ind, scc in enumerate(sccs):
        if root_name in scc:
            root_scc_ind = scc_ind
            break
    if root_scc_ind is None:
        raise RuntimeError(f'Root {root_name} not found in SCCs {sccs}')

    node_to_scc_ind = {name: scc_ind for scc_ind, scc in enumerate(sccs) for name in scc}
    decorator_info: dict[str, dict] = {name: {'is_leaf': False, 'is_parent': False} for name in library_equiv}
    queue = [root_scc_ind]
    visited = []
    while queue:
        scc_ind = queue.pop(0)
        for name in sccs[scc_ind]:
            func_source = library_source[name]
            is_leaf = yes_or_no(
                '####\n'
                f"Found {sccs[scc_ind]} with children {set().union(*[sccs[dep_ind] for dep_ind in scc_edges[scc_ind]])}\n"
                f"Function implementation: \n{func_source}"
                f"Should this function be leaf?")
            decorator_info[name].update({'is_leaf': is_leaf, 'is_parent': not is_leaf})
            visited.append(name)
            if not is_leaf:
                for child in library_equiv[name].children:
                    if child.name in visited:
                        continue
                    child_scc_ind = node_to_scc_ind[child.name]
                    if child_scc_ind in queue:
                        continue
                    queue.append(child_scc_ind)

    print(f'Found parent and leaf nodes: {visited}')
    has_exterior = False # yes_or_no("Are there any exterior functions?")

    class MinecraftFunctionModifier(ast.NodeTransformer):
        def __init__(self, func_name):
            self.target_function_name = func_name
            self.inside_target_function = False

        def visit_FunctionDef(self, node):
            if node.name == self.target_function_name:
                self.inside_target_function = True
                node = self.generic_visit(node)
                self.inside_target_function = False
                return node
            return self.generic_visit(node)

        def visit_Call(self, node):
            if self.inside_target_function and isinstance(node.func, ast.Name) and node.func.id == 'primitive_call':
                context = f"Function: {node.func.id}, {ast.unparse(node)}"
                key = generate_prompt_key()
                prompt = input_with_confirm(
                    f'Found primitive (leaf): {context}\nPlease provide a prompt for the leaf primitive: \n') if not skip_prompt else node.func.id
                is_exterior = has_exterior and yes_or_no(f'Is this an exterior function?')
                assert next((kw for kw in node.keywords if kw.arg == key), None) is None, node.keywords
                node.keywords.append(ast.keyword(
                    arg=key,
                    value=ast.Dict(keys=[ast.Constant('prompt'), ast.Constant('is_exterior'), ast.Constant('yaw'), ast.Constant('negative_prompt')],
                                   values=[ast.Constant(value=prompt), ast.Constant(value=is_exterior), ast.Constant(value=90), ast.Constant(value='')])
                ))
            return self.generic_visit(node)

    with open(path, 'r') as f:
        program = f.read()
    tree = ast.parse(program)
    for name in visited:
        func_source = library_source[name]
        print(f'Function implementation:\n{func_source}')
        is_parent = decorator_info[name]['is_parent']
        is_leaf = decorator_info[name]['is_leaf']
        print(f'Function info: {is_parent=}, {is_leaf=}')
        # leaf_prompts = [] if not (is_parent and not is_leaf) else list(
        #     filter(None, map(lambda s: s.strip(), input_with_confirm("").split(';'))))
        # leaf_prompts = [] if not (is_parent and not is_leaf) else input_with_confirm(
        #     "Please provide leaf prompts as a python list: ", eval_fn=eval,
        # )
        if is_parent and not is_leaf:
            _ = MinecraftFunctionModifier(name).visit(tree)

        if is_leaf:
            prompt = input_with_confirm("Please provide a prompt for this leaf function: ") if not skip_prompt else name
        elif name == root_name:
            prompt = input_with_confirm("Please provide a prompt for the scene: ") if not skip_prompt else name
        else:
            prompt = library[name]['docstring']
        is_exterior = has_exterior and is_leaf and yes_or_no("Is this function an exterior?")
        decorator_info[name].update({
            'prompt': prompt,
            'is_exterior': is_exterior,
            'yaw': 90,
            'negative_prompt': '',
        })

    class DecoratorTransformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name not in decorator_info:
                print(f'Found {node.name} not in library')
                return node
            node.decorator_list = [ast.Call(
                func=ast.Name(id='register', ctx=ast.Load()),
                args=[],
                keywords=[ast.keyword(
                    arg='docstring',
                    value=ast.Str(s=json.dumps(decorator_info[node.name]))
                )]
            )]
            return node

    _ = DecoratorTransformer().visit(tree)
    new_program = astor.to_source(tree)

    with open(save_path, 'w') as f:
        f.write(new_program)
    print(f'[INFO] Saved to: {save_path}')


def render_program(save_dir: Path, overwrite: bool):
    load_program(save_dir / 'program.py')
    core(engine_modes=['mesh'], overwrite=overwrite, save_dir=save_dir.as_posix())


def load_program(path: str):
    print(f'[INFO] Loading from: {path}')
    library, library_equiv, library_source = parse_program([path], roots=None)
    root_name = get_root(library_equiv)
    new_library = make_new_library(library, library_equiv, tree_depth=-1,
                                   root=root_name, engine_mode='box')
    shape = new_library[root_name]['__target__']()
    # for s in shape:
    #     print(s['info']['docstring'])
    print("[INFO] Load program test passed!")


def main():
    args = get_parser().parse_args()
    exp_dir = Path(args.exp_dir)
    exp_subdir_matched = sum([
        list(exp_dir.glob(exp_pattern) if not Path(exp_pattern).is_absolute() else Path(exp_pattern).glob("**"))
        for exp_pattern in args.exp_patterns
    ], [])
    out_dir = Path(PROJ_DIR) / 'logs' / Path(__file__).stem
    out_dir = setup_save_dir(out_dir.as_posix(), args.log_unique)

    for program_path in exp_dir.rglob('program.py'):
        if program_path.parent not in exp_subdir_matched:
            continue
        print(f'[INFO] processing {program_path.as_posix()}')
        save_dir = out_dir / program_path.relative_to(exp_dir).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        process_program(program_path, save_dir, overwrite=args.overwrite, skip_prompt=args.skip_prompt)
        render_program(save_dir, overwrite=args.overwrite)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', default=(Path(PROJ_DIR) / 'scripts' / 'outputs').as_posix(), type=str)
    parser.add_argument('--exp-patterns', nargs='+', type=str, required=True)
    parser.add_argument('--log-unique', action='store_true', help='append timestamp to logging dir')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing renderings')
    parser.add_argument('--skip-prompt', action='store_true', help='use function name as asset description')
    return parser


if __name__ == "__main__":
    try:
        main()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
