import importlib.util
import json
import re
import time
import os
import argparse
import ast
import astor
import uuid
from engine.utils.lm_utils import unwrap_results
from impl_parse_dependency import parse_dependency
from engine.constants import ENGINE_MODE, DEBUG
import inspect
from engine.utils.graph_utils import strongly_connected_components, overwrite_dependency, check_dependency_match
from engine.utils.parse_utils import add_function_prefix, parse_dependency_to_str
import traceback
from pathlib import Path
from _shape_utils import Hole
from prompt_helper import load_pyi, execute, load_program, create_lm
from sketch_helper import implement_scc, HELPER_HEADER, get_implement_scc_order, parse_program, transfer_dependency_to_library
from scripts.exp.icl_0512.run_two_rounds import DEPENDENCY_TO_PROGRAM_SYSTEM_PROMPT as SYSTEM_PROMPT


root = Path(__file__).parent.parent


DESC_MAX_DEPTH = 1  # 1: direct children; 2: descendants up to 2 levels; ...
SIBL_MAX_DEPTH = 0  # -1: no siblings; 0: only provide implemented siblings; 1: siblings and their direct children; ...
print(f'[INFO] {DESC_MAX_DEPTH=} {SIBL_MAX_DEPTH=}')


def dependency_to_program(dependency_path: str, save_dir: str, library_paths: list[str]):
    model = create_lm()
    save_dir = Path(save_dir)
    with open(save_dir / 'system_prompt.md', 'w') as f:
        f.write(SYSTEM_PROMPT)

    def get_implementation(func_name: str):
        if library_equiv[func_name].implementation is None:
            # would only happen if there are circles, i.e. |scc| > 1
            print(func_name)
            import ipdb; ipdb.set_trace()
        if library_equiv[func_name].implementation == -1:
            print(func_name)
            import ipdb; ipdb.set_trace()
        return library_equiv[func_name].implementation

    def sort_nodes_by_impl_order(nodes: list[Hole]) -> list[Hole]:
        return list(sorted(nodes, key=lambda x: library_impl_order.index(x.name)))

    def get_codex_input(self: Hole):
        other_descendants = [n for n in self.get_descendants_by_depth(max_depth=DESC_MAX_DEPTH).values() if n.name != self.name]
        siblings = set().union(*[n.children for n in self.parents])
        siblings.discard(self)
        implemented_siblings = sum([list(n.get_descendants_by_depth(max_depth=SIBL_MAX_DEPTH).values())
                                    for n in siblings if n.implementation is not None and n.implementation != -1], [])
        parents = list(self.parents)
        # remove duplicates
        other_descendants_names = [n.name for n in other_descendants]
        implemented_siblings = [n for n in implemented_siblings if n.name not in other_descendants_names]

        other_descendants = sort_nodes_by_impl_order(other_descendants)
        implemented_siblings = sort_nodes_by_impl_order(implemented_siblings)
        parents = sort_nodes_by_impl_order(parents)
        print(f'[INFO] implementing {self.name}, providing desc {other_descendants}, siblings {implemented_siblings}')

        if len(other_descendants) == 0:
            descendant_desc = ''
            descendent_program = ''
        else:
            descendant_desc = (f"This function directly or indirectly depends on `{','.join([n.name for n in other_descendants])}`. "
                               f"It must call its direct children `{','.join([n.name for n in self.children])}` via `library_call`.\n")
            descendent_program = '# Descendant functions.\n' + '\n'.join([get_implementation(n.name) for n in other_descendants]) + '\n'
        if len(implemented_siblings) == 0:
            siblings_desc = ''
            sibling_program = ''
        else:
            siblings_desc = 'Some sibling functions, as specified in the input dependency graph, are provided.\n'
            sibling_program = '# Sibling functions, for reference only.\n' + '\n'.join([get_implementation(n.name) for n in implemented_siblings]) + '\n'
        if len(other_descendants) + len(implemented_siblings) == 0:
            extra_desc = '\n'
            input_desc = 'Your input:\n'
        else:
            extra_desc = ("Your implementation must be consistent with the signatures and output scales, positions, and orientations of the functions already implemented, "
                          "on top of adhering to the input dependency graph.\n")
            input_desc = 'Your input (graph for functions already implemented may be truncated):\n'
        if len(parents) == 0:
            parents_desc = ''
        else:
            parents_desc = ('In the future (NOT in your implementation), this function will be called '
                            'by its parent functions `{parents}`.\n').format(parents=parents)
        return '''
{input_disc}
```python
{test_input}
```

Now continue the following script. \
IMPORTANT NOTE: You need to implement only one function, `{func_name}` IN THE CONTEXT OF THE DEPENDENCY GRAPH ABOVE, 
and specify default values for ALL its function arguments. {parents_desc}
{siblings_desc}{descendant_desc}{extra_desc}
Your output:
```python
from helper import *

{sibling_program}{descendant_program}

# Your implementation for `{func_name}` starts here.
```
'''.format(helper=HELPER_HEADER, input_disc=input_desc, test_input=dependency, func_name=self.name,
           parents_desc=parents_desc,
           descendant_desc=descendant_desc, siblings_desc=siblings_desc, extra_desc=extra_desc,
           descendant_program=descendent_program, sibling_program=sibling_program)

    query_save_dir = save_dir / 'queries'

    def implement(node: Hole, skip_cache_completions: int):
        check_health_and_save()

        print(f'[INFO] calling gpt for node {node.name}')
        user_prompt = get_codex_input(library_equiv[node.name])

        timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        query_node_save_dir = query_save_dir / node.name / f'{timestamp}-{skip_cache_completions:02d}'
        if query_node_save_dir.exists():
            query_node_save_dir = query_node_save_dir.with_stem(f'{query_node_save_dir.stem}-{uuid.uuid4()}')
        query_node_save_dir.mkdir(exist_ok=True, parents=True)

        with open(query_node_save_dir / 'user_prompt.md', 'w') as f:
            f.write(user_prompt)

        _, results = model.generate(user_prompt=user_prompt, system_prompt=SYSTEM_PROMPT,
                                    num_completions=1,
                                    skip_cache_completions=skip_cache_completions)
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

        program = re.sub(r'from helper import \*\n', '', program)

        tree = ast.parse(program)
        tree.body = [n for n in tree.body if not (
                isinstance(n, ast.FunctionDef) and
                any(isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'register' for d in n.decorator_list) and
                n.name != node.name
        )]
        program = astor.to_source(tree)

        program_path = query_node_save_dir / 'program.py'
        with open(program_path.as_posix(), 'w') as f:
            f.write(program)

        node.implementation = program
        library_info['content'].append({'roots': [node.name],
                                        'program_path': program_path.resolve().as_posix(),
                                        'dependency_path': Path(dependency_path).resolve().as_posix()})

    library_info = {'content': [], 'loaded': {'library_paths': [Path(p).resolve().as_posix() for p in library_paths]}}

    def check_health_and_save():
        for n in library_equiv.values():
            if n.implementation == -2:
                print(f'[ERROR] implementation == -2: {n.name}')
                exit()

        program = '\n'.join([n.implementation for n in sort_nodes_by_impl_order(list(library_equiv.values()))
                             if n.implementation is not None and n.implementation != -1])
        with open((save_dir / 'program.py').as_posix(), 'w') as f:
            f.write(program)
        with open((save_dir / 'library.json').as_posix(), 'w') as f:
            json.dump(library_info, f)

    _, loaded_library_equiv, _, loaded_dependency = load_library_single_root(library_paths, tmp_dir=(save_dir / 'tmp').as_posix())
    # _, loaded_library_equiv, _, loaded_dependency = load_library(library_paths)

    dependency = load_program(dependency_path)
    overwrite_scope = set(loaded_library_equiv.keys())
    _, library_equiv = parse_dependency(dependency, return_roots=True, overwrite_scope=overwrite_scope)
    dependency = parse_dependency_to_str(dependency, overwrite_scope=overwrite_scope)

    overwrite_dependency(library_equiv, loaded_library_equiv)

    with open((save_dir / 'dependency.txt').as_posix(), 'w') as f:  # CONCATENATED
        f.write(loaded_dependency + dependency)  # FIXME this will get very long and requires pruning

    sccs, scc_edges = strongly_connected_components(defined_fns=library_equiv)
    print(f'[INFO] {sccs=}, {scc_edges}')
    library_impl_order: list[str] = get_implement_scc_order(sccs, scc_edges, library_equiv, {})
    # TODO shouldn't use froze order; it will be shuffled in `clear_scc`

    implemented_sccs = {}
    for scc_idx, scc in enumerate(sccs):
        fns_implemented = [fn_name in loaded_library_equiv and loaded_library_equiv[fn_name].implementation is not None
                           for fn_name in scc]
        if all(fns_implemented):
            print(f'[INFO] {scc=} is implemented in loaded library')
            # for fn_name in scc:
            #     library_equiv[fn_name].implementation = loaded_library_equiv[fn_name].implementation
            #     library_equiv[fn_name].docstring = loaded_library_equiv[fn_name].docstring
            implemented_sccs[scc_idx] = 'loaded'
        elif any(fns_implemented):
            print(f'[ERROR] {scc=} only some functions are implemented: {fns_implemented}; ignored')
            # this should never happen
            # import ipdb; ipdb.set_trace()
        else:
            pass

    for scc_idx, _ in enumerate(sccs):
        implement_scc(scc_idx, sccs, implemented_sccs, scc_edges,
                      defined_fns=library_equiv, codegen=implement, save_dir=query_save_dir,
                      allow_autofill=False, should_expand=False, debug=False, backtrack=True)

    check_health_and_save()  # must save again **after** all implementations are done


def remove_duplicates(lst: list[str]) -> list[str]:
    visited = set()
    ret = []
    for x in lst:
        if x in visited:
            continue
        ret.append(x)
        visited.add(x)
    return ret


def load_library_content_recursive(library_paths: list[str], loaded=None, visited=None) -> list[str]:
    # dependent libraries will precede the current library in `loaded`
    if visited is None:
        visited = set()
    if loaded is None:
        loaded = []

    for library_path in library_paths:
        if library_path in visited:
            continue
        # there may be duplicated entries; later ones will overwrite earlier ones according to `register` from `dsl_utils.py`
        library_path = Path(library_path)
        if not library_path.exists():
            raise RuntimeError(f'Library path {library_path} does not exist')
        with open(library_path.as_posix(), 'r') as f:
            library_info = json.load(f)
        visited.add(library_path)
        load_library_content_recursive(library_info['loaded']['library_paths'], loaded=loaded, visited=visited)
        loaded.append(library_path)
    return loaded

    # to_load = remove_duplicates(to_load)


def load_library_single_root(library_paths: list[str], tmp_dir: str):
    to_load = load_library_content_recursive(library_paths)
    library_content: list[dict[str, str]] = []
    print(f'[INFO] after expansion, loading libraries from {to_load}')
    for library_path in to_load:
        with open(library_path, 'r') as f:
            library_info = json.load(f)
            library_content.extend(library_info['content'])

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    loaded_roots = []
    loaded_program_paths = []
    loaded_dependency_paths = []
    for item in library_content:
        if item['program_path'] in loaded_program_paths:
            continue
        if len(item['roots']) > 1:
            raise NotImplementedError(item)
        # rewrite such that non-root functions have root function name as prefixes
        name, = item['roots']
        if name in loaded_roots:
            raise RuntimeError(f'Root {name} is defined in multiple libraries')
        tmp_path = tmp_dir / f'{name}.py'
        renamed_functions = add_function_prefix(item['program_path'], name, tmp_path.as_posix())
        loaded_roots.append(name)
        loaded_program_paths.append(tmp_path.as_posix())

        # dependency = load_program(item['dependency_path'])
        # for fn_name in renamed_functions:
        #     dependency = re.sub(rf'\b{fn_name}\b', renamed_functions[fn_name], dependency)
        # tmp_path = tmp_dir / f'{root}.txt'
        # with open(tmp_path.as_posix(), 'w') as f:
        #     f.write(dependency)
        # loaded_dependency_paths.append(tmp_path.as_posix())

    print(f'[INFO] parse programs from {loaded_program_paths}')
    loaded_library, loaded_library_equiv, loaded_library_source = parse_program(paths=loaded_program_paths, roots=loaded_roots)
    if loaded_library_equiv is None:
        raise NotImplementedError(f'[ERROR] cannot extract dependency from implementations')
    for node in loaded_library_equiv.values():
        node.implementation = loaded_library_source[node.name]
    print(f'[INFO] loaded library with keys: {loaded_library_equiv.keys()}')
    loaded_dependency = '\n'.join([load_program(p) for p in loaded_dependency_paths])
    return loaded_library, loaded_library_equiv, loaded_library_source, loaded_dependency


def load_library(library_paths: list[str]):
    to_load = load_library_content_recursive(library_paths)
    library_content: list[dict[str, str]] = []
    print(f'[INFO] after expansion, loading libraries from {to_load}')
    for library_path in to_load:
        with open(library_path, 'r') as f:
            library_info = json.load(f)
            library_content.extend(library_info['content'])

    loaded_roots: list[str] = remove_duplicates(sum([item['roots'] for item in library_content], []))
    loaded_program_paths: list[str] = remove_duplicates([item['program_path'] for item in library_content])
    # TODO for each root, if testing args/kwargs are stored, call root(*args, **kwargs) with TRACK_HISTORY on
    print(f'[INFO] parse programs from {loaded_program_paths}')
    loaded_library, loaded_library_equiv, loaded_library_source = parse_program(paths=loaded_program_paths, roots=loaded_roots)

    loaded_dependency_paths: list[str] = remove_duplicates([item['dependency_path'] for item in library_content])
    print(f'[INFO] read dependency from {loaded_dependency_paths}...')
    loaded_dependency = '\n'.join([load_program(p) for p in loaded_dependency_paths])
    loaded_roots_alt, loaded_library_equiv_alt = parse_dependency(loaded_dependency, return_roots=True)

    if loaded_library_equiv is None:
        print(f'[ERROR] cannot extract dependency from implementations; try parsing text dependency graph instead')
        loaded_library_equiv = transfer_dependency_to_library(loaded_library_equiv_alt)  # this step is in theory unnecessary if dependency from python implementations perfectly match the text
    else:
        _ = check_dependency_match(loaded_library_equiv, loaded_library_equiv_alt)

    for node in loaded_library_equiv.values():
        node.implementation = loaded_library_source[node.name]
    print(f'[INFO] loaded library with keys: {loaded_library_equiv.keys()}')
    return loaded_library, loaded_library_equiv, loaded_library_source, loaded_dependency


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library-paths', nargs='*', default=[], help='paths to a json file containing library information')
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
    dependency_to_program(dependency_path=args.dependency_path, save_dir=save_dir.as_posix(),
                          library_paths=args.library_paths)
    # execute((save_dir / 'program.py').as_posix(), save_dir.as_posix(), mode='preset')


if __name__ == "__main__":
    main()
