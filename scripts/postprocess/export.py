import argparse
import sys
import traceback
import pdb

from engine.utils.argparse_utils import setup_save_dir
from engine.constants import PROJ_DIR
from pathlib import Path


prompts_root = Path(PROJ_DIR) / 'scripts' / 'prompts'
sys.path.insert(0, prompts_root.as_posix())

from scripts.prompts.sketch_helper import parse_program
from scripts.prompts.impl_preset import core
from scripts.prompts.impl_helper import make_new_library
from engine.utils.graph_utils import get_root

def main():
    args = get_parser().parse_args()
    exp_dir = Path(args.exp_dir).absolute()
    exp_subdir_matched = sum([
        list(exp_dir.glob(exp_pattern) if not Path(exp_pattern).is_absolute() else Path(exp_pattern).glob("**"))
        for exp_pattern in args.exp_patterns
    ], [])
    if len(exp_subdir_matched) == 0:
        raise ValueError(f'No matching subdirectories found for {args.exp_patterns} in {exp_dir}')

    out_dir = Path(PROJ_DIR) / 'logs' / Path(__file__).stem
    out_dir = setup_save_dir(out_dir.as_posix(), args.log_unique)

    for program_path in exp_dir.rglob('program.py'):
        # import ipdb; ipdb.set_trace()   
        if program_path.parent not in exp_subdir_matched:
            continue
        print(f'[INFO] processing {program_path.as_posix()}')
        save_dir = out_dir / program_path.relative_to(exp_dir).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        parse_program([program_path], roots=None)
        core(engine_modes=['all', 'mesh'], overwrite=args.overwrite, save_dir=save_dir.as_posix())


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', default=(Path(PROJ_DIR) / 'scripts' / 'outputs').as_posix(), type=str)
    parser.add_argument('--exp-patterns', nargs='+', type=str, required=True)
    parser.add_argument('--log-unique', action='store_true', help='append timestamp to logging dir')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing renderings')
    return parser


if __name__ == "__main__":
    main()