import re
import difflib
import ast
import astor
from typing import Optional
from pathlib import Path


PRIMITIVES = {'cube', 'sphere'}  # {'cube', 'sphere', 'cylinder', 'cone'}


def parse_dependency_to_str(text, overwrite_scope: Optional[set[str]] = None):
    ret_lines = []
    lines = text.strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        ret_lines.append(line)

        line = re.split(r'#', line)[0].rstrip()  # remove comments
        current_indent = compute_indent(line)
        node = line.strip()
        i += 1
        if node in overwrite_scope:
            print(f'[INFO] skipping {node} in `parse_dependency_to_str`')
            while i < len(lines) and compute_indent(lines[i]) > current_indent:
                i += 1
    return '\n'.join(ret_lines)


def preprocess_dependency(text, overwrite_scope: Optional[set[str]] = None) -> str:
    if '---->' in text:
        import ipdb; ipdb.set_trace()
        text = text.split('---->')[-1]
    lines = text.strip().split('\n')
    lines = remove_loops(lines, scope=set(), overwrite_scope=overwrite_scope if overwrite_scope is not None else {})
    lines = remove_primitives(lines)
    return '\n'.join(lines)


def remove_primitives(lines):
    # check that all leaves are primitives
    result = []
    for line in lines:
        is_primitive = any(line.strip() == p for p in PRIMITIVES)
        if not is_primitive:
            result.append(line)
    return result


def compute_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def remove_loops(lines, scope: set[str], overwrite_scope: set[str]):
    i = 0
    result = []
    while i < len(lines):
        line = lines[i]
        line = re.split(r'#', line)[0].rstrip()  # remove comments
        node = line.strip()

        current_indent = compute_indent(line)
        is_primitive = any(node == p for p in PRIMITIVES)
        is_loop = re.match(r' *loop (\d+|\?)', line) is not None
        is_empty_line = node == ''
        has_children = i + 1 < len(lines) and compute_indent(lines[i + 1]) > current_indent
        assert not (is_primitive and is_loop), f"[ERROR] {i=}, {line=}, {current_indent} is primitive and a loop"

        def collect_body():
            # `i` is current line index
            body = []
            j = i + 1
            while j < len(lines) and compute_indent(lines[j]) > current_indent:
                body.append(lines[j])
                j += 1
            return body

        if is_primitive:
            # Check if the next line is a child
            if has_children:
                raise ValueError(
                    f"[ERROR] Primitive '{i=}, {line=}, {current_indent}, {lines[i + 1]}, {compute_indent(lines[i + 1])}' should not have children.")
            # # Skip adding this line to the result
            # i += 1
            # continue
        # Check if the line is a loop line
        elif is_loop:
            # loop_body = []
            # i += 1  # Move to the next line, which should be the start of the loop body
            #
            # # Collect all lines that are part of the loop body
            # while i < len(lines) and compute_indent(lines[i]) > current_indent:
            #     loop_body.append(lines[i])
            #     i += 1
            loop_body = collect_body()
            if len(loop_body) == 0:
                print(f'[ERROR] Loop body is empty {i=} {line=} {current_indent}')
                raise RuntimeError('Loop body is empty')
            i += len(loop_body) + 1  # Skip the loop body

            # Recursively process the loop body to handle nested loops
            processed_body = remove_loops(loop_body, scope, overwrite_scope)
            if len(processed_body) > 0:  # processed_body would be zero if loop body is overwritten by overwrite_scope

                shrink_indent = compute_indent(processed_body[0]) - current_indent

                # Adjust the indentation of the processed body
                for line in processed_body:
                    # Reduce one level of indentation
                    adjusted_line = ' ' * max(0, (len(line) - len(line.lstrip()) - shrink_indent)) + line.lstrip()
                    result.append(adjusted_line)

            continue  # Skip further processing in the main loop since the recursion handles it
        elif is_empty_line:
            if i + 1 < len(lines) and compute_indent(lines[i + 1]) > 0:
                print(f'[ERROR] Got empty line {i=} {line=} {current_indent}')
            i += 1
            continue
        else:
            if node in overwrite_scope:
                print(f'[INFO] {node} will be overwritten by given scope')
                children_body = collect_body()
                i += len(children_body) + 1
                continue

            if node in scope:
                if has_children:
                    children_body = collect_body()
                    print(f"[ERROR] Node '{i=}, {line=}, {current_indent}' has been defined but has children \n{children_body}")
                    # we don't allow name collision or function re-definition
                    i += len(children_body)  # skip children
                    # do NOT continue here as we need to process the current line
            else:
                if not has_children:
                    print(f"[ERROR] Non-primitive '{i=}, {line=}, {current_indent}' should have children.")
                scope.add(node)

        # For non-loop lines, just append to result
        result.append(line)
        i += 1

    return result


def diff_program(program1, program2) -> str:
    lines1 = program1.splitlines()
    lines2 = program2.splitlines()
    differ = difflib.Differ()
    diff = list(differ.compare(lines1, lines2))
    return '\n'.join(diff)


def remove_repeated_functions(program: str) -> str:
    tree = ast.parse(program)

    # Dictionary to store the last occurrence of each function definition
    func_defs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            func_defs[func_name] = node

    # Collect the lines to keep
    lines_to_keep = set()
    for func in func_defs.values():
        # Include lines for decorators
        if func.decorator_list:
            for decorator in func.decorator_list:
                start_line = decorator.lineno - 1
                end_line = decorator.end_lineno if hasattr(decorator, 'end_lineno') else start_line + 1
                lines_to_keep.update(range(start_line, end_line))
        # Include lines for the function itself
        lines_to_keep.update(range(func.lineno - 1, func.end_lineno))

    # Create a list to hold the lines of the cleaned script
    cleaned_lines = []
    for i, line in enumerate(program.splitlines()):
        if i in lines_to_keep:
            cleaned_lines.append(line)

    cleaned_script = '\n'.join(cleaned_lines)
    return cleaned_script


def add_function_prefix(path: str, root: str, save_path: str):
    with open(path, 'r') as file:
        source = file.read()
        tree = ast.parse(source)

    class FunctionRenamer(ast.NodeTransformer):
        def __init__(self):
            self.functions = {}

        def visit_FunctionDef(self, node):
            is_registered = False
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'register':
                    is_registered = True
                    break
            if is_registered:
                new_name = f"{root}_{node.name}" if node.name != root else node.name
                self.functions[node.name] = new_name
                node.name = new_name

            self.generic_visit(node)
            return node

    transformer = FunctionRenamer()
    tree = transformer.visit(tree)
    renamed_functions = transformer.functions
    if len([f for f in renamed_functions.keys() if f == root]) != 1:
        raise ValueError(f"Root function '{root}' is not defined exactly once in the script: {renamed_functions}")

    class FunctionCallRenamer(ast.NodeTransformer):
        def visit_Call(self, node):
            # Rename function calls
            if isinstance(node.func, ast.Name):
                if node.func.id in renamed_functions.keys():
                    node.func.id = renamed_functions[node.func.id]

            if isinstance(node.func, ast.Name) and node.func.id == "library_call":
                if isinstance(node.args[0], ast.Str) and node.args[0].s in renamed_functions.keys():
                    node.args[0].s = renamed_functions[node.args[0].s]
            self.generic_visit(node)
            return node

    tree = FunctionCallRenamer().visit(tree)

    with open(save_path, 'w') as file:
        file.write(astor.to_source(tree))

    return renamed_functions


def preprocess_code(content):
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)
    content = re.sub(r'#.*', '', content)
    # Remove empty lines
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    return '\n'.join(lines)

def create_diff(original_file, modified_file, diff_file):
    with open(original_file, 'r') as f1, open(modified_file, 'r') as f2:
        original_content = preprocess_code(f1.read())
        modified_content = preprocess_code(f2.read())

        diff = difflib.unified_diff(
            original_content.splitlines(keepends=True),
            modified_content.splitlines(keepends=True),
            fromfile=str(original_file),
            tofile=str(modified_file)
        )

    with open(diff_file, 'w') as f:
        f.writelines(diff)


def create_diff2(original_file, modified_file, diff_file):
    prog1 = Path(original_file).read_text()
    prog2 = Path(modified_file).read_text()
    diff = diff_program(prog1, prog2)
    Path(diff_file).write_text(diff)
