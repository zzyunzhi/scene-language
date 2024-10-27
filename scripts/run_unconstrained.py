from pathlib import Path
import uuid
from run_utils import SYSTEM_HEADER, run, read_tasks, SYSTEM_RULES, read_example
from engine.utils.argparse_utils import setup_save_dir, modify_string_for_file
from engine.constants import ENGINE_MODE, PROMPT_MODE, LLM_PROVIDER
try:
    from tu.loggers.utils import print_vcv_url
except:
    print("tu not available for import")
from typing import List, Union
import argparse
import time
import os


root = Path(__file__).parent

SYSTEM_PROMPT = '''
You are a code completion model and can only write python functions wrapped within ```python```.

You are provided with the following `helper.py` which defines the given functions and definitions:
```python
{header}
```

{rules}
You should be precise and creative.
'''.format(header=SYSTEM_HEADER, rules=SYSTEM_RULES)


def get_user_prompt(scene_description, latest_program: str):
    # This means it is the first iteration, it will start by producing some basic objects
    if not latest_program:
        return '''Here are some examples of how to use `helper.py`:
```python
{example}
```
IMPORTANT: THE FUNCTIONS ABOVE ARE JUST EXAMPLES, YOU CANNOT USE THEM IN YOUR PROGRAM!

Now, write a similar program for the given description of a scene:    
```python
from helper import *

"""
{scene_description}
"""
```
'''.format(scene_description=scene_description, example=read_example())
    else:
        return '''In previous rounds, you have already implemented a scene using `helper.py`:
```python
{latest_program}
```

Your task is to read over the above, and WITHOUT changing or modifying the existing program, either:

1. Compose the existing functions into new object groupings or new scenes.
2. Write brand new functions that DON'T EXIST yet in the library. 

We will then add your new functions to the previous program and iteratively build the scene. Remember that the end goal is to make the following scene: {scene_description}

```python
# TODO: Add your new functions here!
```
'''.format(scene_description=scene_description, latest_program=latest_program)

    


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', default=2, type=int, help='depth')
    parser.add_argument('--task', required=True, type=str, help='scene description')
    parser.add_argument('--log-dir', type=str, default=(root / 'outputs' / Path(__file__).stem).as_posix())
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    save_dir = setup_save_dir(args.log_dir, log_unique=True)

    # Process task for directory name
    task = args.task
    name = task.split(' - ')[0]
    save_subdir = save_dir / modify_string_for_file(name.replace(' ', '_'))
    save_subdir.mkdir(exist_ok=True)

    depth = args.depth
    print(f"[INFO] Running unconstrained generation with depth={depth}")

    # print_vcv_url(save_dir.as_posix())

    programs = []
    for idx in range(depth):
        print(f"[INFO] depth={idx}")
        depth_dir_name = f"depth_{idx}"
        depth_save_subdir = save_subdir / depth_dir_name
        depth_save_subdir.mkdir(exist_ok=True)
        latest_program = None if not programs else '\n\n'.join(programs)

        user_prompt = get_user_prompt(task, latest_program)
        with open(depth_save_subdir / 'system_prompt.md', 'w') as f:
            f.write(SYSTEM_PROMPT)
        with open(depth_save_subdir / 'user_prompt.md', 'w') as f:
            f.write(user_prompt)

        # Because num_completions is 1, we only want to extract the first one
        new_programs = run(save_dir=depth_save_subdir.as_posix(),
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            extra_info={'task': task},
            lm_config={'num_completions': 1, 'temperature': .5},
            code_only=True,
            prepend_program=latest_program
            )[0]

        programs.append(new_programs)


if __name__ == "__main__":
    main()
