import argparse
from typing import Union, Optional
from _shape_utils import Hole
from engine.utils.graph_utils import strongly_connected_components, get_root
from engine.utils.parse_utils import preprocess_dependency


def parse_to_fn(line, parent, defined_fns):
    fn_name = line.strip()
    # fn_name, fn_args, fn_ret, desc = parse_line(line.strip())
    # print(f"Parsing {fn_name}({fn_args}) -> {fn_ret}")
    # print("Line:", line)
    if parent is None:
        raise RuntimeError('[ERROR] Parent is should never be None')
    if fn_name in defined_fns:
        new_fn = defined_fns[fn_name]
    else:
        new_fn = Hole(fn_name, docstring='', check=None, normalize=False)
        new_fn.children = set()
        defined_fns[fn_name] = new_fn
    new_fn.parents.add(parent)
    parent.children.add(new_fn)
    return new_fn


def initial_node(line, cur_node):
    new_node = {
        'name': line.strip(),
        'line': line,
        'children': [],
        'parent': cur_node,
        # 'asserts': [],
    }
    if cur_node is not None:
        cur_node['children'].append(new_node)

    return new_node


def fill_graph(node, node_equiv, defined_fns):
    child_equivs = []
    for child in node['children']:
        # asserts = child['asserts']
        child_node = parse_to_fn(child['line'], node_equiv, defined_fns)
        # defined_fns[child_node.name].asserts += asserts
        child_equivs.append(child_node)
    for child, child_equiv in zip(node['children'], child_equivs):
        fill_graph(child, child_equiv, defined_fns)
    return defined_fns


# Inspired by https://stackoverflow.com/questions/45964731/how-to-parse-hierarchy-based-on-indents-with-python
def parse_dependency(dependency: str, return_roots: bool = False, overwrite_scope: Optional[set[str]] = None) -> tuple[Union[Hole, list[Hole]], dict[str, Hole]]:
    dependency = preprocess_dependency(dependency, overwrite_scope=overwrite_scope)
    print('[INFO] preprocessed dependency:')
    print(dependency)
    lines = dependency.strip().split('\n')
    root = initial_node("_root", None)
    cur_node = root
    indentation = [-1]
    depth = -1
    buffer_line = ""
    for cur_line in lines:
        # Handle line continuations
        # if cur_line[-1] == "\\":
        #     buffer_line += cur_line[:-1] + "\n"
        #     continue
        line = buffer_line + cur_line
        buffer_line = ""

        indent = len(line) - len(line.lstrip())
        if not line.strip():
            continue

        if indent > indentation[-1]:
            new_node = initial_node(line, cur_node)
            cur_node = new_node
            depth += 1
            indentation.append(indent)
            continue

        if indent < indentation[-1]:
            while indent < indentation[-1]:
                depth -= 1
                indentation.pop()
                cur_node = cur_node['parent']

            if indent != indentation[-1]:
                raise RuntimeError("Bad formatting")

        if indent == indentation[-1]:
            if False:  #CONSTS['assert_check'](line):
                cur_node['asserts'].append(line.strip())
            else:
                new_node = initial_node(line, cur_node['parent'])
                cur_node = new_node

    # temp_root = Function(name="root", args=[], desc="Main function", ret=[], parent=None, asserts=[])
    temp_root = Hole('_root', docstring='', check=None, normalize=False)
    temp_root.children = set()
    defined_fns = {'_root': temp_root}
    fill_graph(root, temp_root, defined_fns=defined_fns)
    del defined_fns['_root']
    if return_roots:
        for node in temp_root.children:
            node.parents.remove(temp_root)
        roots = list(temp_root.children)
        return roots, defined_fns

    assert len(temp_root.children) == 1, "There should only be one root function"
    root_fn_graph = next(iter(temp_root.children))
    root_fn_graph.parents.remove(temp_root)
    return root_fn_graph, defined_fns


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dependency-path', required=True, help='path to sketch program')
    parser.add_argument('--allow-multiple-roots', action='store_true', help='allow multiple roots')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.dependency_path) as f:
        dependency = f.read()

    print('[INFO] parsing dependency..')
    print('--->')
    print(dependency)
    roots, defined_fns = parse_dependency(dependency, return_roots=args.allow_multiple_roots)
    print('--->')
    print(roots)
    print(defined_fns)
    sccs, scc_edges = strongly_connected_components(defined_fns=defined_fns)
    print('--->')
    print(sccs, scc_edges)


def test():
    text1 = """
    desk_with_papers
        desk
            cube
        paper_pile  # black
            loop ?
                paper
                    cube
        paper_pile  # white
    """
    text2 = """
    checkerboard
        loop ?


            loop ?
                square
        loop ?
            chess
    """
    text3 = """
    piano_keyboard
        keyboard_body
            cube
        keys
            loop 88
                key
                    cube  
    """

    text2 = """
matryoshka_doll
    outer_layer
        sphere
    inner_layer
        loop ?
            matryoshka_doll
                this part will be skipped
        loop ?
            decoration

windmill_tower
    tower
        sphere
    blades
        loop 4
            blade
                cube
"""

    # Parse and display the dependency graph
    for text in [text2]:
        print(text)
        roots, defined_fns = parse_dependency(text, return_roots=True)
        print(roots)
        print(defined_fns)
        sccs, scc_edges = strongly_connected_components(defined_fns=defined_fns)
        print(sccs, scc_edges)
        for node in defined_fns.values():
            print(node.name, node.parents, node.children)
        print('root is:')
        print(get_root(defined_fns))


if __name__ == "__main__":
    main()
    # test()
