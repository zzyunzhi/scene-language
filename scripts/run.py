from pathlib import Path
import os
from engine.utils.argparse_utils import setup_save_dir, modify_string_for_file
from engine.constants import ENGINE_MODE
from run_utils import SYSTEM_HEADER, run, read_tasks, SYSTEM_RULES, read_example, save_prompts
from engine.utils.parse_utils import create_diff, create_diff2
import argparse


root = Path(__file__).parent

SYSTEM_PROMPT = """\
You are a code completion model and can only write python functions wrapped within ```python```.

You are provided with the following `helper.py` which defines the given functions and definitions:
```python
{header}
```

{rules}

You should be precise and creative.
""".format(
    header=SYSTEM_HEADER, rules=SYSTEM_RULES
)


def get_user_prompt(task: str, animate: bool):
    animation_blurb = (
        "\nIMPORTANT: Since you selected the animation option, the task involves creating a realistic animation in Minecraft. "
        "Focus on smooth transitions, appropriate timing, and accurate rendering to achieve the desired effect."
        if animate and ENGINE_MODE == "minecraft"
        else ""
    )

    return '''Here are some examples of how to use `helper.py`:
```python
{example}
```
IMPORTANT: THE FUNCTIONS ABOVE ARE JUST EXAMPLES, YOU CANNOT USE THEM IN YOUR PROGRAM! {animation_blurb}

Now, write a similar program for the given task:    
```python
from helper import *

"""
{task}
"""
```
'''.format(
        task=task,
        example=read_example(animate=animate),
        animation_blurb=animation_blurb,
    )


def get_user_prompt_reconstruction(path: str):
    assert os.path.exists(path), path
    return [
        {
            "type": "text",
            "text": get_user_prompt("Reconstruct the input scene", animate=False),
        },
        {"type": "image_url", "image_url": os.path.abspath(path)},
    ]


def get_user_prompt_edit(task: str):
    with open(task, "r") as f:
        path = f.readline().strip()
        task = f.readline().strip()
    assert os.path.exists(path), path
    with open(path, "r") as f:
        example = f.read()
    return '''Here is a program using `helper.py`: 
```python
{example}
```
Now, do minimal edit to the program such that the scene function, when called, will follow the instruction: {task}.
Your code starts here.
```python
from helper import *

"""
{task}
"""
```
'''.format(
        task=task, example=example
    )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="scene description")
    parser.add_argument(
        "--cond", type=str, default="text", choices=["text", "image", "animate", "edit"]
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=(root / "outputs" / Path(__file__).stem).as_posix(),
    )
    parser.add_argument(
        "--num-completions", type=int, default=4, help="number of samples"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="LM inference temperature"
    )
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
        if args.cond == "text":
            assert not Path(task).exists(), f"use --cond image for {task}"
            user_prompt = get_user_prompt(task, animate=False)
        elif args.cond == "animate":
            assert not Path(task).exists(), f"use --cond image for {task}"
            user_prompt = get_user_prompt(task, animate=True)
        elif args.cond == "image":
            user_prompt = get_user_prompt_reconstruction(task)
        elif args.cond == "edit":
            user_prompt = get_user_prompt_edit(task)
        else:
            raise NotImplementedError(args.cond)

        save_prompts(save_subdir.as_posix(), SYSTEM_PROMPT, user_prompt)

        run(
            save_dir=save_subdir.as_posix(),
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            extra_info={"task": task},
            lm_config={
                "num_completions": args.num_completions,
                "temperature": args.temperature,
            },
        )

        if args.cond == 'edit':
            with open(task, 'r') as f:
                orig_prog = f.readline().strip()
            for p in save_subdir.glob('*/program.py'):
                create_diff(orig_prog, p.as_posix(), p.with_name('diff.txt').as_posix())
                create_diff2(orig_prog, p.as_posix(), p.with_name('diff2.txt').as_posix())


if __name__ == "__main__":
    main()
