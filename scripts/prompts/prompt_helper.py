import os
import subprocess
import argparse
from typing import Literal
from pathlib import Path
from engine.constants import ENGINE_MODE, PROMPT_MODE, DEBUG, LLM_PROVIDER
from engine.utils.parsel_utils import setup_gpt
from engine.utils.claude_client import setup_claude
from engine.utils.execute_utils import execute_command_retries, execute_command
import time

root = Path(__file__).parent.parent.absolute()


def create_lm():
    return {'gpt': setup_gpt, 'claude': setup_claude}[LLM_PROVIDER]()


def load_program(path: str):
    with open(path, 'r') as f:
        s = f.read()
    return s


def load_pyi(engine_mode: str = ENGINE_MODE, prompt_mode: str = PROMPT_MODE) -> str:
    if engine_mode == 'mi_from_minecraft':
        return ''  # we assume this is for rendering only

    # FIXME must manually check this .pyi file
    header_out_dir_base = root / 'outputs' / 'stubgen'
    header_dir = sorted(header_out_dir_base.glob(f'*-{engine_mode}-{prompt_mode}'), key=os.path.getmtime)[-1]
    header_file = header_dir / 'header.pyi'
    print(f'[INFO] loading header from {header_file=}')
    return load_program(header_file.as_posix())


def load_impl_header() -> str:
    impl_file = 'impl_minecraft.py' if ENGINE_MODE == 'minecraft' else 'impl_preset.py'
    return load_program((root / 'prompts' / impl_file).as_posix())


IMPL_HEADER = load_impl_header()
IMPL_EVAL_HEADER = load_program((root / 'prompts' / 'impl_eval.py').as_posix())


def get_impl(program: str):
    return """\n
{header}
{program}

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
""".format(header=IMPL_HEADER, program=program)


def get_impl_eval(program: str):
    return """\n
{header}
{program}

if __name__ == "__main__":
    main()
""".format(header=IMPL_EVAL_HEADER, program=program)


def execute(program_path: str, save_dir: str, mode: Literal['eval', 'preset'], roots: list[str]) -> int:
    save_dir = Path(save_dir)
    program = load_program(program_path)

    impl = {'eval': get_impl_eval}[mode](program)
    impl_path = save_dir / 'impl.py'
    with open(impl_path.as_posix(), "w") as f:
        f.write(impl)

    command = (f'DEBUG={"1" if DEBUG else "0"} '
               f'PYTHONPATH={Path(__file__).parent}:$PYTHONPATH python {impl_path.as_posix()} --roots {" ".join(roots)}')

    return execute_command(command, save_dir.as_posix())
    # return execute_command_retries(command, save_dir.as_posix(), retries=3, timeout=60)
