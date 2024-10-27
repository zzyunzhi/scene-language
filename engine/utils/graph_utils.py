
def strongly_connected_components(defined_fns):
    # Identify the nodes reachable from each node
    reachable = {fn_name: {fn_name} for fn_name in defined_fns}
    changed = True
    while changed:
        changed = False
        # Loop through all the pairs of fn_name and the functions reachable from it
        for fn_name, fns_reachable in reachable.items():
            # Loop through all the functions reachable from fn_name
            for fn_reachable_name in fns_reachable.copy():
                fn = defined_fns[fn_reachable_name]
                # Loop through all the children of the functions reachable from fn_name
                for child in fn.children:
                    initial_len = len(reachable[fn_name])
                    # Try to add the child to the set of functions reachable from fn_name
                    reachable[fn_name].add(child.name)
                    # # If the child has no asserts, it also depends on the parent
                    # if not child.asserts and not CONSTS['implicit_assert'] and consider_asserts:
                    #     initial_len_2 = len(reachable[child.name])
                    #     reachable[child.name].add(fn_reachable_name)
                    #     if len(reachable[child.name]) > initial_len_2:
                    #         changed = True
                    if len(reachable[fn_name]) > initial_len:
                        changed = True
                # Reachability is transitive, so add everything reachable from anything reachable from fn_name
                for fn_reachable_name_2 in fns_reachable.copy():
                    initial_len = len(reachable[fn_name])
                    reachable[fn_name].update(reachable[fn_reachable_name_2])
                    if len(reachable[fn_name]) > initial_len:
                        changed = True

    # Identify the strongly connected components
    sccs = []
    remaining_nodes = set(defined_fns)
    for fn_name in defined_fns.keys():
        if fn_name not in remaining_nodes:
            continue
        remaining_nodes.remove(fn_name)
        scc = {fn_name}
        for child_name in reachable[fn_name]:
            if fn_name in reachable[child_name]:
                if child_name in remaining_nodes:
                    scc.add(child_name)
                    remaining_nodes.remove(child_name)
        sccs.append(scc)

    # Identify the relationships between the strongly connected components
    scc_edges = []
    for scc_1_idx, scc_1 in enumerate(sccs):
        scc_1_edges = []
        for scc_2_idx, scc_2 in enumerate(sccs):
            if scc_1_idx == scc_2_idx:
                continue
            if list(scc_2)[0] in reachable[list(scc_1)[0]]:
                scc_1_edges += [scc_2_idx]
        scc_edges.append(scc_1_edges)
    return sccs, scc_edges


def get_ancestors(node, visited=None):
    if visited is None:
        visited = {node.name: node}
    for parent in node.parents:
        if parent.name not in visited:
            visited[parent.name] = parent
            get_ancestors(parent, visited)
    return visited


def get_root(defined_fns) -> str:
    # Identify a function which is the parent of all other functions
    # We allow for cycles, so we can't use just parents
    shared_ancestors = None
    for fn in defined_fns.values():
        anc = get_ancestors(fn)
        # print(f'{fn.name=}: {anc=}')
        if shared_ancestors is None:
            shared_ancestors = set(anc) | {fn.name}
        else:
            shared_ancestors.intersection_update(anc)
    shared_defined = shared_ancestors & set(defined_fns.keys())
    return shared_defined.pop()


def calculate_node_depths(defined_fns, root):
    depths = {fn_name: -1 for fn_name in defined_fns}  # Start with -1 (undefined depth)
    sccs, _ = strongly_connected_components(defined_fns)
    scc_map = {node: idx for idx, scc in enumerate(sccs) for node in scc}
    reduced_graph = {i: set() for i in range(len(sccs))}

    for node, fn in defined_fns.items():
        current_scc = scc_map[node]
        for child in fn.children:
            child_scc = scc_map[child.name]
            if current_scc != child_scc:
                reduced_graph[current_scc].add(child_scc)
    depths = [-1] * len(sccs)

    def dfs(_scc_index, current_depth):
        if depths[_scc_index] < current_depth:
            depths[_scc_index] = current_depth
            for neighbor in reduced_graph[_scc_index]:
                dfs(neighbor, current_depth + 1)

    for scc_index in reduced_graph:
        if root in sccs[scc_index]:
            dfs(scc_index, 0)
            break

    # # Find root nodes in the reduced graph
    # all_children = set(c for neighbors in reduced_graph.values() for c in neighbors)
    # root_sccs = [scc_index for scc_index in reduced_graph if scc_index not in all_children]
    # root = get_root(defined_fns)
    # for root_scc in root_sccs:
    #     if root not in sccs[root_scc]:
    #         print(f'Warning: root {root} not in root sccs {root_sccs}')
    #     # raise RuntimeError()
    #
    # # Calculate depths starting from each root SCC
    # for root_scc in root_sccs:
    #     dfs(root_scc, 0)

    # Map SCC depths back to individual nodes
    node_depths = {}
    for node, scc_index in scc_map.items():
        node_depths[node] = depths[scc_index]
    return node_depths


def overwrite_dependency(defined_fns, defined_fns_transfer_from):
    nodes_to_update = set(defined_fns_transfer_from.keys()).intersection(defined_fns.keys())
    for fn_name in nodes_to_update:
        node = defined_fns[fn_name]
        node_alt = defined_fns_transfer_from[fn_name]
        for other_node in defined_fns.values():
            if node in other_node.parents:
                other_node.parents.remove(node)
                other_node.parents.add(node_alt)
            if node in other_node.children:
                other_node.children.remove(node)
                other_node.children.add(node_alt)
    defined_fns.update(defined_fns_transfer_from)


def transfer_dependency(defined_fns, defined_fns_transfer_from,
                        transfer_docstring=True, transfer_implementation=True):
    """
    Mutates `defined_fns` in place to include the dependency graph from `defined_fns_transfer_from`.
    """
    for fn_name in defined_fns.keys():
        if fn_name not in defined_fns_transfer_from.keys():
            continue
        defined_fns[fn_name].children.update(defined_fns_transfer_from[fn_name].children)
        defined_fns[fn_name].parents.update(defined_fns_transfer_from[fn_name].parents)
        if transfer_docstring:
            defined_fns[fn_name].docstring = defined_fns_transfer_from[fn_name].docstring
        if transfer_implementation:
            defined_fns[fn_name].implementation = defined_fns_transfer_from[fn_name].implementation


def check_dependency_match(defined_fns, defined_fns_ref, check_key: bool = True,
                           check_docstring: bool = False, check_implementation: bool = False) -> bool:
    # only check for functions defined in `defined_fns`
    ret = True
    if check_key:
        if set(defined_fns.keys()) != set(defined_fns_ref.keys()):
            print('[ERROR] keys not matched', set(defined_fns.keys()), set(defined_fns_ref.keys()))
            ret = False
    for fn_name, fn in defined_fns.items():
        fn_ref = defined_fns_ref[fn_name]
        if set([n.name for n in fn.children]) != set([n.name for n in fn_ref.children]):
            print('[ERROR] children not matched', fn.name, fn.children, fn_ref.children)
            ret = False
        if set([n.name for n in fn.parents]) != set([n.name for n in fn_ref.parents]):
            print('[ERROR] parents not matched', fn.name, fn.parents, fn_ref.parents)
            ret = False
        if check_docstring:
            if fn.docstring != fn_ref.docstring:
                print('[ERROR] docstring not matched', fn.name, fn.docstring, fn_ref.docstring)
                ret = False
        if check_implementation:
            if fn.implementation != fn_ref.implementation:
                print('[ERROR] implementation not matched', fn.implementation, fn_ref.implementation)
                ret = False
    print(f'[INFO] check passed: dependency from implementations matches text dependency graph')
    return ret
