from pathlib import Path
import numpy as np

from helper import *
import mitsuba as mi
import traceback
import ipdb

import random
import math
import sys
import os

import mi_helper  # such that primitive call will be implemented

import argparse
from impl_utils import create_nodes, run
from engine.utils.graph_utils import strongly_connected_components
from tu.loggers.utils import print_vcv_url


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default=(Path(__file__).parent / 'renderings').as_posix(), help='log directory')
    parser.add_argument('--roots', type=str, nargs='+', help='names of root functions')
    return parser


def main():
    args = get_parser().parse_args()
    core(save_dir=args.log_dir, roots=args.roots)


def core(save_dir: str, roots: list[str]):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    print_vcv_url(save_dir.as_posix())

    library_equiv = create_nodes(roots=roots)
    scc = strongly_connected_components(library_equiv)
    print(f'{scc=}')
    if len(roots) > 1:
        print(f'[ERROR] more than one roots: {roots}')
    for root in roots:
        print(f'[INFO] executing `{root}`...')
        _ = run(root, save_dir=(save_dir / root).as_posix(), preset_id='table', num_views=1)
        print(f'[INFO] executing `{root}` done!')
