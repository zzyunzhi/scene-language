import argparse
import shutil

from engine.utils.argparse_utils import setup_save_dir
from engine.utils.execute_utils import execute_command
from engine.constants import ENGINE_MODE, DEBUG
from pathlib import Path


root = Path(__file__).parent.parent.resolve()


def read_impl() -> str:
    impl_file = 'impl_minecraft.py' if ENGINE_MODE == 'minecraft' else 'impl_preset.py'
    with open(root / 'prompts' / impl_file, 'r') as f:
        s = f.read()
    return s


IMPL_HEADER = read_impl()

# EXP_NAME = 'run_single_round_20240629-143841_b0d48339-0122-4185-865e-c86b34515063'
# EXP_NAME = 'run_single_round_20240629-201456_46178cae-3c28-422b-81c0-f1c40543d24c'
#
# EXP_NAME = 'run_two_rounds'
# EXP_NAME = '20240703-4'  # http://vcv.stanford.edu/cgi-bin/file-explorer/?dir=/viscam/projects/concepts/engine/scripts/exp/icl_0512/outputs/20240703-4&patterns_show=*&patterns_highlight=&w=375&n=4&autoplay=1&showmedia=1
# # EXP_NAME = '20240608-2'
# EXP_NAME = '20240608-2-two-rounds'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--input-pattern', type=str, default=['**'], nargs='+', help='pattern to match subdirectories')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--engine-modes', nargs='*', default=[''],
                        choices=['omost', 'lmd', 'loosecontrol', 'neural', 'box', 'migc', 'densediffusion', 'gala3d', 'exterior', 'interior', 'mesh', 'all'])
    parser.add_argument('--log-unique', action='store_true', help='append timestamp to logging dir')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing renderings')
    parser.add_argument('--dry-run', action='store_true', help='print commands without executing')
    return parser


def main():
    args = get_parser().parse_args()
    exp_dir = Path(args.input_dir)
    if args.output_dir is None:
        out_dir = root / 'outputs' / Path(__file__).stem / exp_dir.name
    else:
        out_dir = args.output_dir
    out_dir = setup_save_dir(out_dir, args.log_unique)
    exp_subsubdir_matched = sum([list(exp_dir.glob(pattern)) for pattern in args.input_pattern], [])
    print(f'[INFO] Found {len(exp_subsubdir_matched)} subdirectories matching pattern {args.input_pattern}')
    for program_path in sorted(exp_dir.rglob('**/program.py')):
        if program_path.parent in exp_subsubdir_matched:
            print(f'[INFO] Processing {program_path}...')
    # for exp_subdir in sorted(exp_dir.iterdir()):
    #     if not exp_subdir.is_dir():
    #         continue
    #     task: str = exp_subdir.name
    #
    #     dependency_path = exp_subdir / 'task_to_dependency_0/0/program.py'
    #     if not dependency_path.exists():
    #         dependency_path = None
    #
    #     for exp_subsubdir in sorted(exp_subdir.iterdir()):
    #         if not exp_subsubdir.is_dir():
    #             continue
    #         if exp_subsubdir not in exp_subsubdir_matched:
    #             # print(f'[INFO] Skipping {exp_subsubdir} due to pattern mismatch with {args.input_pattern}')
    #             continue
    #         print(f'[INFO] Processing {exp_subsubdir}...')
    #
    #         exp_completion_index: str = exp_subsubdir.name
    #         if exp_subsubdir.name.startswith('task_to_dependency'):
    #             continue
    #         if exp_subsubdir.name.startswith('dependency_to_program'):
    #             exp_subsubdir = exp_subsubdir / '0'
    #         if exp_subsubdir.name.startswith('dependency_to_program_iterative'):
    #             pass
    #
    #         program_path = exp_subsubdir / 'program.py'
    #         if not program_path.exists():
    #             continue
    #         out_subdir = out_dir / f'{task}_{exp_completion_index}'
            out_subdir = out_dir / program_path.parent.relative_to(exp_dir)
            with open(program_path.as_posix(), 'r') as f:
                program = f.read()

            impl = """\n
{header}

{program}

if __name__ == "__main__":
    main()
            """.format(header=IMPL_HEADER, program=program)

            out_subdir.mkdir(exist_ok=True, parents=True)
            shutil.copy(program_path.as_posix(), (out_subdir / f'program.py').as_posix())
            impl_path = (out_subdir / 'impl.py').as_posix()
            with open(impl_path, "w") as f:
                f.write(impl)

            rendering_out_dir = out_subdir / 'renderings'
            command = [
                f'ENGINE_MODE={ENGINE_MODE} DEBUG={"1" if DEBUG else "0"} PYTHONPATH={root / "prompts"}:$PYTHONPATH',
                'python', impl_path,
                '--engine-modes', ' '.join(args.engine_modes),
                '--log-dir', rendering_out_dir.as_posix(),
                '--program-path', program_path.as_posix(),
            ]
            # if dependency_path is not None:
            #     command.extend(['--dependency-path', dependency_path.as_posix()])
            if args.overwrite:
                command.append('--overwrite')
            command = ' '.join(command)

            success = execute_command(command, out_subdir.as_posix(), dry_run=args.dry_run)

            # symlink_path = out_dir / f'{task}_{exp_completion_index}.html'
            # if symlink_path.exists():
            #     symlink_path.unlink()
            # symlink_path.symlink_to(rendering_out_dir / 'index.html')
            # vi_helper.dump_table(vi, [[(rendering_out_dir / 'index.html').as_posix()]])

    print(f'[INFO] All done! Renderings are saved in {out_dir}')


if __name__ == "__main__":
    main()
