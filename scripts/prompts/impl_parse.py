import argparse
from sketch_helper import parse_sketch_from_dependency
from engine.utils.graph_utils import strongly_connected_components, get_root


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketch-path', required=True, help='path to sketch program')
    parser.add_argument('--dependency-path', required=True, help='path to dependency')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    # parse_sketch catastrophically clears the library
    library, library_equiv, library_source = parse_sketch_from_dependency(sketch_path=args.sketch_path, dependency_path=args.dependency_path)
    sccs, scc_edges = strongly_connected_components(defined_fns=library_equiv)
    print(f'[INFO] {sccs=}, {scc_edges}')

    root = get_root(library_equiv)
    print(f'[INFO] {root=}')


if __name__ == "__main__":
    main()
