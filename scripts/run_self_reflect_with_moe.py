from pathlib import Path
import os
from engine.utils.argparse_utils import setup_save_dir, modify_string_for_file
from engine.constants import ENGINE_MODE
from run_utils import read_tasks, read_example, run_self_reflect_and_moe
import argparse

root = Path(__file__).parent


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', help='scene description')
    parser.add_argument('--cond', type=str, default='text', choices=['text'])
    parser.add_argument('--log-dir', type=str, default=(root / 'outputs' / Path(__file__).stem).as_posix())
    parser.add_argument('--temperature', type=float, default=.2, help='LM inference temperature')
    parser.add_argument('--num-reflections', type=int, default=5, help='Number of self-reflection rounds for the LM')
    parser.add_argument('--num-experts', type=int, default=4, help='Number of experts contributing proposals')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    tasks = args.tasks if args.tasks is not None else read_tasks()

    save_dir = setup_save_dir(args.log_dir, log_unique=True)

    for task in tasks:
        name = modify_string_for_file(task)
        save_subdir = save_dir / name
        save_subdir.mkdir(exist_ok=True)
        # save_prompts(save_subdir.as_posix(), SYSTEM_PROMPT, user_prompt)

        run_self_reflect_and_moe(save_dir=save_subdir.as_posix(), 
                                 task=task, 
                                 animate=args.cond == 'animate', 
                                 num_reflections=args.num_reflections, 
                                 num_experts=args.num_experts, 
                                 extra_info={'task': task},
                                 lm_config={'num_completions': 1, 'temperature': args.temperature}
                                 )


if __name__ == "__main__":
    main()
