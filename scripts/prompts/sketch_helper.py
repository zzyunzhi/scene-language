import importlib.util
import numpy as np
import re
import time
import copy
import os
import argparse
import ast
import astor
import uuid
from engine.utils.lm_utils import unwrap_results
from impl_parse_dependency import parse_dependency
from engine.constants import ENGINE_MODE, DEBUG
import inspect
from engine.utils.graph_utils import strongly_connected_components, get_root
import traceback
from typing import Literal, Optional
from pathlib import Path
from impl_utils import create_nodes
from dsl_utils import library
from _shape_utils import Hole
from prompt_helper import load_pyi, execute, load_program, create_lm


root = Path(__file__).parent.parent


def transfer_dependency_to_library(library_equiv_alt: dict[str, Hole]) -> dict[str, Hole]:
    # WARNING: This function catastrophically mutates (global) library.
    #  This is not true? It only creates a `library_equiv` that borrows `__target__` from `library` and relations from `library_equiv_alt`
    print(f'[WARNING] Calling `transfer_dependency_to_library` which mutates the library.')
    if set(library.keys()) != set(library_equiv_alt.keys()):
        print(f"[ERROR] {len(library)=}, {len(library_equiv_alt)=}")
        raise RuntimeError(f"Implemented dependency does not match input dependency: \n"
                           f"{list(sorted(library.keys()))}\n"
                           f"{list(sorted(library_equiv_alt.keys()))}")
    library_equiv: dict[str, Hole] = {}
    for name in library.keys():
        node = Hole(name=name, docstring=library[name]['docstring'], check=library[name]['check'], normalize=False)
        node.fn = library[name]['__target__']
        library_equiv[name] = node

    for name in library.keys():
        node = library_equiv[name]
        node_alt = library_equiv_alt[name]
        print(f'[INFO] {name=}, children: {node_alt.children}, parents: {node_alt.parents}')
        node.children = set([library_equiv[n.name] for n in node_alt.children])
        node.parents = set([library_equiv[n.name] for n in node_alt.parents])
    return library_equiv


def parse_sketch(path: str, library_equiv_alt: dict[str, Hole]):
    return parse_program(paths=[path], library_equiv_alt=library_equiv_alt,
                         # parse_helper_path=(root / 'prompts/prepare_sketch.py').as_posix()  # deprecated; now sketch function has no body
                         )


def parse_program(paths: list[str], roots: Optional[list[str]] = None, library_equiv_alt: Optional[dict[str, Hole]] = None,
                  parse_helper_path: str = (root / 'prompts/prepare_program.py').as_posix()):
    print(f'[WARNING] CLEARING UP LIBRARY!!!')
    library.clear()  # FIXME !!!!! THIS MAY CAUSE PROBLEMS IF WE HAVE MULTIPLE SKETCHES

    tmp_path = root / 'outputs/tmp/parse_program.py'
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parse_helper_path, 'r') as f:
        parse_helper = f.read()

    parse_program = ''
    for path in paths:
        # later paths will overwrite earlier ones
        with open(path, 'r') as f:
            lines = f.readlines()
            if 'from helper import *\n' in lines:
                lines.remove('from helper import *\n')
            parse_program += ''.join(lines)
    with open(tmp_path, 'w') as f:
        f.write(parse_helper + '\n' + parse_program)

    spec = importlib.util.spec_from_file_location("input_program", tmp_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    try:
        library_equiv_ref = create_nodes(roots=roots)
        print('[INFO] if parsing from sketch..')
        print(strongly_connected_components(defined_fns=library_equiv_ref))
    except Exception:
        print(traceback.format_exc())
        print('[INFO] cannot parse from scratch')
        library_equiv_ref = None

    if library_equiv_alt is None:
        # raise RuntimeError("library_equiv_alt must be provided")
        # print(f'[ERROR] {library_equiv_alt=} not provided')
        library_equiv = library_equiv_ref
    else:
        print(f'[ERROR] {library_equiv_alt=} is a deprecated argument; use `roots` instead')
        library_equiv = transfer_dependency_to_library(library_equiv_alt)

    library_source: dict[str, str] = {}
    for name in library.keys():
        try:
            library_source[name] = inspect.getsource(library[name]['__target__'])
        except TypeError as e:
            print(e)

    return library, library_equiv, library_source


def parse_sketch_from_dependency(sketch_path: str, dependency_path: str):
    dependency = load_program(dependency_path)
    _, library_equiv_alt = parse_dependency(dependency)
    library, library_equiv, library_source = parse_sketch(sketch_path, library_equiv_alt=library_equiv_alt)
    # # transfer the dependency from library_equiv_alt to library_equiv
    # if len(library_equiv) != len(library_equiv_alt):
    #     print(f"[ERROR] {len(library_equiv)=}, {len(library_equiv_alt)=}")
    #     print(library_equiv)
    #     print(library_equiv_alt)
    #     raise RuntimeError(f"Dependency does not match sketch: {sketch_path=}, {dependency_path=}")
    # for name in library_equiv:
    #     library_equiv[name].children = library_equiv_alt[name].children
    #     library_equiv[name].parents = library_equiv_alt[name].parents
    return library, library_equiv, library_source


# def prepend_hash(lines: str) -> str:
#     suffix = '\n' if lines.endswith('\n') else ''
#     return "\n".join(["# " + line for line in lines.split("\n")]) + suffix


def get_implement_scc_order(sccs, scc_edges, defined_fns, implemented_sccs) -> list[str]:
    print('[INFO] Starting dry run of `implement_scc`...')
    implemented_sccs = copy.deepcopy(implemented_sccs)  # must copy! will be mutated by `implement_scc`
    impl_order = []
    for scc_idx, _ in enumerate(sccs):
        _ = implement_scc(scc_idx, sccs, implemented_sccs, scc_edges, defined_fns,
                          codegen=lambda node, skip_cache_completions: impl_order.append(node.name),
                          save_dir=None, debug=True, backtrack=False, max_attempts=1)
    print('[INFO] Finished dry run of `implement_scc`.')
    return impl_order


def implement_scc(scc_idx, sccs, implemented_sccs, scc_edges, defined_fns, codegen, save_dir,
                  allow_autofill=False, should_expand=False, debug=False, seed=42, backtrack=False,
                  max_attempts=2):
    if scc_idx in implemented_sccs:
        print("[INFO] Found implemented SCC", scc_idx, sccs[scc_idx])
        return implemented_sccs[scc_idx]
    print("[INFO] Implementing SCC recursively", scc_idx, sccs[scc_idx])
    dependencies_str = ""
    for edge in scc_edges[scc_idx]:
        dependencies_str += implement_scc(edge, sccs, implemented_sccs, scc_edges, defined_fns, codegen, save_dir,
                                          allow_autofill, should_expand, debug)  # can only backtrack by one node degree

    # print("[INFO] Implementing SCC itself", scc_idx, sccs[scc_idx])
    error = None
    # We exponentially increase the number of completions until we reach the max, "num_completions"
    for i in range(max_attempts):  # FIXME it doesn't make sense if temperature is 0
        # print(f"Trying {num_completions} completions")
        try:
            for fn_name in sccs[scc_idx]:
                fn = defined_fns[fn_name]
                codegen(fn, skip_cache_completions=i)

            new_str = dependencies_str + eval_scc(
                sccs[scc_idx], dependencies_str, defined_fns, codegen, save_dir, allow_autofill, should_expand, debug, seed=seed, backtrack=False)
            implemented_sccs[scc_idx] = new_str
            return new_str
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            error = e
            print(f"[INFO] attempt # {i} for {sccs[scc_idx]}, error: {e}")
            print(traceback.format_exc())
    if backtrack and len(scc_edges[scc_idx]) > 0:
        # Backtracking allows us to try new implementations
        # of the dependencies if we fail to implement the SCC
        print("[INFO] Backtracking due to error", error)
        clear_scc(scc_idx, sccs, implemented_sccs, scc_edges, defined_fns, codegen, save_dir, allow_autofill, should_expand, debug)
        for implemented_scc in list(implemented_sccs.keys()):
            del implemented_sccs[implemented_scc]
        new_str = implement_scc(
            scc_idx, sccs, implemented_sccs, scc_edges, defined_fns, codegen, save_dir, allow_autofill, should_expand, debug, seed=seed + 1, backtrack=False)  # can only backtrack once
        implemented_sccs[scc_idx] = new_str
        return new_str
    print(f"[ERROR] failed after {backtrack=} for {sccs[scc_idx]}")
    # raise error
    new_str = f'\n###\n{sccs[scc_idx]} failed to implement: \n{error}\n###\n' + dependencies_str
    implemented_sccs[scc_idx] = new_str
    for fn_name in sccs[scc_idx]:
        fn = defined_fns[fn_name]
        fn.implementation = -2  # hack: use -2 to indicate failure after backtracking
    return new_str  # still keep going


def clear_scc(scc_idx, sccs, implemented_sccs, scc_edges, defined_fns, codegen, save_dir, allow_autofill=False, should_expand=False, debug=False):
    for edge in scc_edges[scc_idx]:
        clear_scc(edge, sccs, implemented_sccs, scc_edges, defined_fns, codegen, save_dir, allow_autofill, should_expand, debug)
    for fn_name in sccs[scc_idx]:
        fn = defined_fns[fn_name]
        fn.implementation = -1 #None  # hack: use -1 to track if we should use num_completions = 2
    print(f'[INFO] in-place shuffle scc edges')
    np.random.shuffle(sccs[scc_idx])  # shuffle implementation order; otherwise the same cache will be hit even with backtrack=True


def eval_scc(scc, dependencies_str, defined_fns, codegen, save_dir, allow_autofill=False, should_expand=False, debug=False, seed=42, backtrack=False):
    if debug:  # skip evaluation
        return ','.join(scc) + ','
    all_implementations = {n.name: n.implementation for n in defined_fns.values() if n.implementation is not None}
    for n in defined_fns.values():
        if n.name in scc and n.implementation is None:
            print(f'[ERROR] {n.name} is not implemented')

    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    attempt_save_dir = Path(save_dir) / '-'.join(scc) / timestamp
    if attempt_save_dir.exists():
        attempt_save_dir = Path(save_dir) / '-'.join(scc) / f'{timestamp}-{uuid.uuid4()}'
    attempt_save_dir.mkdir(exist_ok=True, parents=True)

    program = '\n'.join(all_implementations.values())
    program_path = attempt_save_dir / 'program.py'
    with open(program_path, 'w') as f:
        f.write(program)

    with open(attempt_save_dir / 'dependencies_str.txt', 'w') as f:
        f.write(dependencies_str)

    if execute(program_path=program_path.as_posix(), save_dir=attempt_save_dir.as_posix(), mode='eval', roots=scc) != 0:
        raise RuntimeError(f"[ERROR] evaluation failed: {attempt_save_dir}")

    return ','.join(scc) + ','


# HELPER_HEADER = load_pyi()


def load_example_sketch():
    text = load_program((root / 'prompts' / 'oracle_v21_02_sketch_to_program_input.py').as_posix())
    text = re.sub(r'^\s*(import .*|from .* import .*)\n', '', text, flags=re.MULTILINE)
    return text


def load_example_program():
    text = load_program((root / 'prompts' / 'oracle_v21_02_sketch_to_program_output_partial.py').as_posix())
    if 'if __name__ == ' in text:
        ind = text.index('if __name__ == ')
        text = text[:ind]
    return text


# EXAMPLE_SKETCH = load_example_sketch()
# EXAMPLE_PROGRAM = load_example_program()

SYSTEM_PROMPT = "You are a code completion model and an excellent 3D designer. You can only write python functions wrapped within ```python```."


def sketch_to_program(sketch_path: str, dependency_path: str, save_dir: str, dry_run: bool = False):
    model = create_lm()
    save_dir = Path(save_dir)
    with open(save_dir / 'system_prompt.md', 'w') as f:
        f.write(SYSTEM_PROMPT)

    def get_header(func_name: str, include_register_line: bool = True, include_def_line: bool = True) -> str:
        text = library_source[func_name]
        register_match = re.search(r'@register\(.*?\)', text)
        if register_match is None:
            print(f'[ERROR]: {func_name=} does not have register line')
            import ipdb; ipdb.set_trace()
        register_line = register_match.group()
        def_match = re.search(r'def\s+\w+\s*\((?:.|\n)*?\)(?:\s*->\s*[^\s:]+)?\s*:', text, re.DOTALL)
        if def_match is None:
            print(f'[ERROR]: {func_name=} does not have def line')
            import ipdb; ipdb.set_trace()
        def_line = def_match.group()
        return (f"{register_line}\n" if include_register_line else "") + (f"{def_line}\n" if include_def_line else "")

    def get_sketch(func_name: str):
        return library_source[func_name]

    def get_implementation(func_name: str):
        if library_equiv[func_name].implementation is None:
            # would only happen if there are circles, i.e. |scc| > 1
            print(func_name)
            import ipdb; ipdb.set_trace()
        if library_equiv[func_name].implementation == -1:
            print(func_name)
            import ipdb; ipdb.set_trace()
        return library_equiv[func_name].implementation

    def get_codex_input(self):
        other_descendants = [n for n in self.get_descendants().values() if n.name != self.name]
        siblings = set().union(*[n.children for n in self.parents])
        siblings.discard(self)
        implemented_siblings = [n for n in siblings if n.implementation is not None]
        return """Your task is to convert a function sketch to an actual implementation that compiles and outputs a shape that is physically accurate.
You are provided with `helper.py`:
```python
{helper}
```
Strictly follow these rules:
1. Only use functions and imported libraries in `helper.py`.
2. Camera coordinate system: +x is right, +y is up, +z is {z_direction}. 
3. Follow the input sketch closely. 
    - Keep its signature, default keyword arguments, and skeleton.
    - Make sure constraints specified in the sketch are satisfied; no need to copy over the comments.

Example input sketch:
```python
{example_sketch}
```
Example output:
```python
{example_program}
```

Your input sketch:
```python
{sketch}
```
Now continue the following script. \
Dependent shape functions, if any, are already implemented below and can only be called via `library_call`. \
Your implementation must be consistent with their signatures and output scales, positions, and orientations.
```python
from helper import *
{prepend_program}
{header}
```
""".format(helper=HELPER_HEADER,
           example_sketch=EXAMPLE_SKETCH,
           example_program=EXAMPLE_PROGRAM,
           z_direction='forward' if ENGINE_MODE == 'minecraft' else 'backward',
           sketch=get_sketch(self.name),
           prepend_program='\n'.join([get_implementation(n.name) for n in other_descendants + implemented_siblings]),
           header=get_header(self.name, include_def_line=False))

    query_save_dir = save_dir / 'queries'

    def implement(node: 'Hole'):
        print(f'[INFO] calling gpt for node {node.name}')
        user_prompt = get_codex_input(library_equiv[node.name])

        timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        query_node_save_dir = query_save_dir / node.name / timestamp
        if query_node_save_dir.exists():
            query_node_save_dir = query_save_dir / node.name / f'{timestamp}-{uuid.uuid4()}'
        query_node_save_dir.mkdir(exist_ok=True, parents=True)

        with open(query_node_save_dir / 'user_prompt.md', 'w') as f:
            f.write(user_prompt)

        if dry_run:
            program = get_header(node.name) + "    return []\n"  # dummy implementation
            node.implementation = program
            return
        if node.implementation == -1:
            print(f'[INFO] updating implementation for {node.name=}')
            num_completions = 2  # FIXME we support max 2 completions as we hard code max_attempts = 1 and backtrack can only be done once
        else:
            num_completions = 1

        _, results = model.generate(user_prompt=user_prompt, system_prompt=SYSTEM_PROMPT, num_completions=num_completions)
        result = results[-1]
        with open(query_node_save_dir / 'raw.txt', 'w') as f:
            f.write('\n'.join(result))
        try:
            matches = re.findall(r'```python\s*(.*?)\s*```', '\n'.join(result), re.DOTALL)
            program = '\n'.join([match.strip() for match in matches])
            # lines = unwrap_results(result)
            # if lines is None:
            #     raise RuntimeError("No lines")
        except Exception as _:
            with open((query_node_save_dir / 'error.txt').as_posix(), 'w') as f:
                f.write(traceback.format_exc())
            # program = get_header(node.name) + "    return []\n"  # dummy implementation
            # impl_library_source[node.name] = program
            raise RuntimeError(query_node_save_dir.as_posix())

        program = re.sub(r'from children import \*\n|from grandchildren import \*\n|from helper import \*\n', '', program)
        if "@register" not in program:
            if not program.startswith(f'def {node.name}'):
                print(f'[ERROR] {node.name=}')
                print(program)
                import ipdb; ipdb.set_trace()
            program = get_header(node.name, include_def_line=False) + program
        else:
            tree = ast.parse(program)
            tree.body = [n for n in tree.body if not (
                    isinstance(n, ast.FunctionDef) and
                    any(isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'register' for d in n.decorator_list) and
                    n.name != node.name
            )]
            program = astor.to_source(tree)

        with open((query_node_save_dir / 'program.py').as_posix(), 'w') as f:
            f.write(program)

        node.implementation = program

    library, library_equiv, library_source = parse_sketch_from_dependency(sketch_path, dependency_path)

    sccs, scc_edges = strongly_connected_components(defined_fns=library_equiv)
    print(f'[INFO] {sccs=}, {scc_edges}')
    implemented_sccs = {}
    for scc_idx, _ in enumerate(sccs):
        implement_scc(scc_idx, sccs, implemented_sccs, scc_edges,
                      defined_fns=library_equiv, codegen=implement, save_dir=query_save_dir,
                      allow_autofill=False, should_expand=False, debug=False, backtrack=True)
    for n in library_equiv.values():
        if n.implementation is None:
            print(f'[ERROR] {n.name=}')
    program = '\n'.join([n.implementation for n in library_equiv.values() if n.implementation is not None])
    with open((save_dir / 'program.py').as_posix(), 'w') as f:
        f.write(program)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketch-path', required=True, help='path to sketch program')
    parser.add_argument('--dependency-path', required=True, help='path to dependency')
    parser.add_argument('--log-dir', type=str, default=(root / 'outputs' / Path(__file__).stem).as_posix())
    return parser


def main():
    # save_dir = root / 'outputs' / Path(__file__).stem
    # save_dir.mkdir(exist_ok=True, parents=True)
    # sketch_path = root / 'prompts' / 'oracle_01_dependency_to_sketch.py'
    parser = get_parser()
    args = parser.parse_args()
    save_dir = Path(args.log_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    sketch_to_program(sketch_path=args.sketch_path, dependency_path=args.dependency_path, save_dir=save_dir.as_posix())
    # execute((save_dir / 'program.py').as_posix(), save_dir.as_posix(), mode='preset')


if __name__ == "__main__":
    main()
