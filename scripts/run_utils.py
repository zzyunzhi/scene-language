import traceback
from pathlib import Path
import json
import os
from enum import Enum
from engine.utils.parsel_utils import setup_gpt
from engine.utils.claude_client import setup_claude

try:
    from engine.utils.code_llama_client import setup_llama
except:
    print("Unable to import Llama modules. Are you running on cluster?")
from engine.utils.lm_utils import unwrap_results
from engine.utils.execute_utils import execute_command
from engine.constants import (
    ENGINE_MODE,
    PROMPT_MODE,
    DEBUG,
    LLM_PROVIDER,
    TEMPERATURE,
    NUM_COMPLETIONS,
    MAX_TOKENS,
    DRY_RUN,
)
from typing import List, Union, Optional

root = Path(__file__).parent


def load_program(path: str):
    with open(path, "r") as f:
        s = f.read()
    return s


def read_impl() -> str:
    impl_file = "impl_minecraft.py" if ENGINE_MODE == "minecraft" else "impl_preset.py"
    with open(root / "prompts" / impl_file, "r") as f:
        s = f.read()
    return s


IMPL_HEADER = read_impl()


def read_header(engine_mode: str = ENGINE_MODE, prompt_mode: str = PROMPT_MODE) -> str:
    if engine_mode == "mi_from_minecraft":
        return ""  # we assume this is for rendering only
    if engine_mode == "mi":
        print(f"[INFO] engine_mode=mi, but use engine_mode=exposed for prompting")
        engine_mode = "exposed"
    # FIXME must manually check this .pyi file
    header_dir = sorted(
        (root / "outputs/stubgen").glob(f"*-{engine_mode}-{prompt_mode}"),
        key=os.path.getmtime,
    )[-1]
    # files = sorted((root / 'outputs' / 'stubgen').iterdir(), key=os.path.getmtime)[-1]
    header_file = header_dir / "header.pyi"
    print(header_file)
    with open(header_file.as_posix(), "r") as f:
        s = f.read()
    return s


SYSTEM_HEADER = read_header()

# TODO unify rule 2
SYSTEM_RULES = f"""STRICTLY follow these rules:
1. Only use the functions and imported libraries in `helper.py`.
2. You can only write functions. Follow a modular approach and use the `register` decorator to define semantic shapes or shape groups. Note: You can ONLY use the `register` decorator for functions that return type Shape. Any helper functions that you attempt to register will cause an error.
3. Camera coordinate system: +x is right, +y is up, +z is {'forward' if ENGINE_MODE == 'minecraft' else 'backward'}. 
"""
# TODO: Need to find a better way to do this
if PROMPT_MODE == "assert":
    raise NotImplementedError(PROMPT_MODE)
    SYSTEM_RULES += """4. List all constraints using `assert_*` with a separate `test` function. 
For `assert_*`, the name of a shape is either the name of the registered function that returns it, or `primitive` if it is returned from `primitive_call`.
Different shape instances may have the same name which leads to ambiguity, but try your best to be comprehensive. 
Assertions should only be applied to shapes that are DIRECT outputs of functions called within the function under test.
"""
if ENGINE_MODE == "minecraft":
    SYSTEM_RULES += """4. Make sure to only pass in values into block_type that are supported by standard Minecraft engines. Pass in block_kwargs for blocks that need additional properties to define a block's state fully, such as stair blocks."""
    SYSTEM_RULES += """\n5. Pay attention that the objects are not too large so it can't be rendered."""
# if ENGINE_MODE == 'mi_material':
#     SYSTEM_RULES += "4. Specify materials to be as realistic as possible; see documentation for `primitive_call`.\n"
if ENGINE_MODE in ["mi", "mi_material", "exposed"]:
    #     SYSTEM_RULES += "4. You can use shape primitives to approximate shape components that are too complex. You must make sure shape have correct poses. \
    # Be careful about `set_mode` and `set_to` from `primitive_call`.\n"
    SYSTEM_RULES += "4. You must use `library_call` to call registered functions.\n"
    #     SYSTEM_RULES += "6. Be very careful about transformation orders and the `point` argument for `rotation_matrix` which specifies the rotation centers."
    if PROMPT_MODE == "calc":
        SYSTEM_RULES += "5. Use `compute_shape_*` from `helper.py` if possible to compute transformations.\n"
        # SYSTEM_RULES += ('IMPORTANT: Explicitly list constraints as inline comments'
        # and use `align_with_*` and `attach` whenever possible, '
        #                  'otherwise there are easily mistakes in transformations!! \n')  # claude model just doesn't use these functions


def save_prompts(
    save_dir: str, system_prompt: str, user_prompt: Union[str, list[dict[str, str]]]
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    with open((save_dir / "system_prompt.md").as_posix(), "w") as f:
        f.write(system_prompt)

    if isinstance(user_prompt, str):
        with open((save_dir / "user_prompt.md").as_posix(), "w") as f:
            f.write(user_prompt)
    else:
        for ind, prompt in enumerate(user_prompt):
            if prompt["type"] == "text":
                with open((save_dir / f"user_prompt_{ind}.md").as_posix(), "w") as f:
                    f.write(prompt["text"])
            elif prompt["type"] == "image_url":
                pass
            else:
                raise NotImplementedError(f"{prompt['type']=}")


def read_tasks() -> List[str]:
    with open(root / "test_0608.txt", "r") as f:
        return [line.strip() for line in f]


def read_example(path: Optional[str] = None, animate: bool = False) -> str:
    if path is None:
        path = (
            root
            / "prompts"
            / {
                #     ('mi', 'assert'): 'oracle_assert.py',  # outdated
                # ('mi', 'default'): 'oracle_rotate.py',
                # ('mi', 'calc'): 'oracle_rotate_v3.py',
                ("exposed", "calc", True): "oracle_0831_animation.py",
                ("exposed", "calc", False): "oracle_0807.py",
                ("mi_material", "calc", False): "oracle_material.py",
                ("minecraft", "default", True): "oracle_minecraft_animation.py",
                ("minecraft", "default", False): "oracle_minecraft.py",
            }[ENGINE_MODE, PROMPT_MODE, animate]
        ).as_posix()
    # if not Path(path).name in [
    #     "oracle_0807.py",
    #     "oracle_material.py",
    #     "oracle_v21_02_sketch_to_program.py",
    #     "oracle_v2_00_task_to_dependency.txt",
    # ]:
    #     print(f"[ERROR] this example may be outdated: {path=}")
    # print(f"[INFO] using example: {path}")
    with open(path, "r") as f:
        s = f.read()
    if Path(path).suffix == ".py":
        ind = s.index('if __name__ == "__main__":')
        s = s[:ind]
    return s


def get_assert_impl(program: str):
    return """\n
{header}

{program}

if __name__ == "__main__":
    try:
        test()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        
    print('test done!')
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
""".format(
        header=IMPL_HEADER, program=program
    )


def get_default_impl(program: str):
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
""".format(
        header=IMPL_HEADER, program=program
    )


get_impl = {
    "assert": get_assert_impl,
    "default": get_default_impl,
    "calc": get_default_impl,
}[PROMPT_MODE]


def generate(
    user_prompt: Union[str, list[dict[str, str]], None],
    system_prompt: str,
    prepend_messages: Optional[list] = None,
    lm_config: Optional[dict] = None,
    skip_cache: bool = False,
):
    if LLM_PROVIDER == "gpt":
        model = setup_gpt()
        _, results = model.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            prepend_messages=prepend_messages,
            **lm_config,
        )
        return results
    elif LLM_PROVIDER == "claude":
        model = setup_claude()
        if prepend_messages is not None:
            raise NotImplementedError(prepend_messages)
        _, results = model.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            skip_cache=skip_cache,
            **lm_config,
        )
        return results
    elif LLM_PROVIDER == "llama":
        model = setup_llama()
        _, results = model.generate(
            user_prompt=user_prompt, system_prompt=system_prompt, **lm_config
        )
        return results
    else:
        raise NotImplementedError(f"{LLM_PROVIDER=}")


def run(
    save_dir: str,
    user_prompt: Union[str, list[dict[str, str]], None],
    system_prompt: str,
    extra_info: Optional[dict] = None,
    prepend_messages: Optional[list] = None,
    prepend_program: Optional[str] = None,
    execute: bool = True,
    lm_config: Optional[dict] = None,
    code_only: bool = False,
    dry_run: bool = False,
):
    save_dir = Path(save_dir)

    lm_config = lm_config if lm_config is not None else {}
    info = {"lm_config": lm_config, **({} if extra_info is None else extra_info)}
    with open((save_dir / "info.json").as_posix(), "w") as f:
        json.dump(info, f)

    # Generate using GPT
    results = generate(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        prepend_messages=prepend_messages,
        lm_config=lm_config,
    )

    programs = []
    for ind, result in enumerate(results):
        trial_save_dir = save_dir / str(ind)
        trial_save_dir.mkdir(exist_ok=True)
        with open((trial_save_dir / "raw.txt").as_posix(), "w") as f:
            f.write("\n".join(result))

        try:
            lines = unwrap_results(result, code_only)
        except Exception as _:
            with open((trial_save_dir / "error.txt").as_posix(), "w") as f:
                f.write(traceback.format_exc())
            continue
        if lines is None:
            # with open((trial_save_dir / 'response.txt').as_posix(), 'w') as f:
            #     f.write('\n'.join(result))
            continue
        program = "\n".join(lines)
        programs.append(program)
        with open((trial_save_dir / "raw.py").as_posix(), "w") as f:
            f.write(program)
        full_program = (
            program if prepend_program is None else (prepend_program + "\n" + program)
        )
        with open((trial_save_dir / "program.py").as_posix(), "w") as f:
            f.write(full_program)
        if not execute:
            continue
        impl = get_impl(full_program)

        save_to = (trial_save_dir / "impl.py").as_posix()
        with open(save_to, "w") as f:
            f.write(impl)

        command = (
            f'ENGINE_MODE={ENGINE_MODE} DEBUG={"1" if DEBUG else "0"} '
            f'PYTHONPATH={Path(__file__).parent / "prompts"}:$PYTHONPATH python {save_to}'
        )

        # command_file = (trial_save_dir / "command.txt").as_posix()
        # with open(command_file, "w") as f:
        #     f.write(command)

        execute_command(command, trial_save_dir.as_posix(), dry_run=dry_run)

    return programs


###################################################################################################
########################################## REFLECT + MOE ##########################################
###################################################################################################


class Role(Enum):
    CRITIC = 1
    WRITER = 2
    JUDGE = 3


def switch_reflection_role(role: Role) -> Role:
    if role == Role.WRITER:
        return Role.CRITIC
    elif role == Role.CRITIC:
        return Role.WRITER
    else:
        raise ValueError("Invalid role provided.")


def get_system_prompt(
    role: Role, header: str = SYSTEM_HEADER, rules: str = SYSTEM_RULES
) -> str:
    if role == Role.WRITER:
        return f"""\
You are a code completion model and can only write Python functions wrapped within ```python```.

You are provided with the following `helper.py` which defines the given functions and definitions:
```python
{header}
```

{rules}

You should be precise and creative.
"""
    elif role == Role.CRITIC:
        return f"""\
You are a code critic. Your task is to review and provide detailed feedback on the provided Python program. 

You are provided with the following `helper.py` which defines the given functions and definitions:
```python
{header}
```

{rules}

Your feedback should include feedback on the logical consistency between the code and the intended 3D scene. Does the code faithfully represent the described scene or object?
"""
    elif role == Role.JUDGE:
        return f"""\
You are an objective judge. Your task is to evaluate three different Python code proposals and select the best one based on the following criteria:

- Code correctness and functionality, particularly the usage of the provided DSL.
- Whether the generated 3D scene matches the described task.
- Code readability and maintainability.

You are provided with the following `helper.py` which defines the given functions and definitions:
```python
{header}
```

{rules}

You will be presented with three code snippets, and you must select the best one, providing a rationale for your choice. Be impartial and thorough in your evaluation.
"""
    else:
        raise ValueError("Invalid role provided.")


def get_writer_prompt_initial(task: str, animate: bool):
    return '''Here are some examples of how to use `helper.py`:
```python
{example}
```
IMPORTANT: THE FUNCTIONS ABOVE ARE JUST EXAMPLES, YOU CANNOT USE THEM IN YOUR PROGRAM!

Now, write a similar program for the given task:    
```python
from helper import *

"""
{task}
"""
```
'''.format(
        task=task, example=read_example(animate=animate)
    )


def get_writer_prompt_with_critiques(task: str, draft: str, critique: str):
    if critique is None:
        import ipdb

        ipdb.set_trace()

    return '''Here was your previous attempt at writing a program in the given DSL:
```python
{draft}
```

The following is a review for the previous attempt:

"""
{critique}
"""

Now, addressing the points brought up in the review, please re-write the program for the given task:
```python
from helper import *

"""
{task}
"""
```
'''.format(
        task=task, draft=draft, critique=critique
    )


def get_critic_prompt(
    task: str, writer_code: str, image_path: Optional[str]
) -> Union[str, list]:
    compilation_blurb = (
        "The current proposal cannot be properly executed and rendered! Analyze code errors in your review."
        if image_path is None
        else ""
    )
    feedback_blurb = (
        ""
        if image_path is None
        else (
            "Carefully examine the provided image rendered from the current proposal. "
            "For EACH function output, check if the object is in the right position and orientation. "
            "A typical failure mode is translation missing by half of the object size!! "
            "Include error analysis in your review."
        )
    )
    text = f"""Your task is to review the following Python code and provide detailed feedback on (ordered by importance):
- Code correctness and functionality, particularly the usage of the provided DSL. {compilation_blurb}
- Whether the generated 3D scene matches the described task and common sense. {feedback_blurb}
- Code readability and maintainability.
- Only if everything else is correct, improve on scene details and aesthetics. 

Task description:
{task}

Here is the current code proposal from the writer:
```python
{writer_code}
```

Provide your critiques and suggestions for improvement below in a formatted list.
"""
    if image_path is None:
        return text
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": image_path},
    ]


def get_judge_prompt(task: str, code_proposals: list[str]) -> str:
    # Dynamically generate code proposals in the prompt
    proposal_section = ""
    for i, code in enumerate(code_proposals, 1):
        proposal_section += f"Code Proposal {i}:\n```python\n{code}\n```\n\n"

    return f"""Your task is to evaluate the following code proposals for the task described below and select the best one.

Task description:
{task}

You will be presented with the following code proposals. Please evaluate each based on (ordered by importance):
- Code correctness and functionality, particularly the usage of the provided DSL.
- Whether the generated 3D scene matches the described task.
- Code readability and maintainability.

{proposal_section}

Your response will be used directly to execute in a Python interpreter, so make sure to output the best code proposal only:
"""


def get_user_prompt_reparam(program: str):
    return """Your task is to refactor an input program following these rules:
1. Update all argument values for `primitive_call` in the input program to follow the documentation for `helper.py`.
2. The updated argument values must accurately describe the texture and material properties of shapes.
3. Your output must be a full, compilable program. No need to copy comments. 

Now, refactor the following program. 
```python
{test_input}
```
""".format(
        test_input=program
    )


def compile_raw_gpt_response_to_program(result: str):
    lines = unwrap_results(result)
    return "\n".join(lines)


def find_rendering(save_dir: Path) -> Optional[str]:
    rendering_path = list(save_dir.glob("renderings/*/rendering_traj_000.png"))
    if len(rendering_path) == 0:
        print(f"[ERROR] no renderings found")
        return None
    else:
        return rendering_path[0].as_posix()


def run_self_reflect_and_moe(
    save_dir: str,
    task: str,
    animate: bool,
    num_reflections: int,
    num_experts: int,
    extra_info: Optional[dict] = None,
    lm_config: Optional[dict] = None,
):
    assert (
        LLM_PROVIDER == "claude"
    ), "self-reflect and MOE only works with Claude for now - need to update the other generate functions to skip the cache"

    save_dir = Path(save_dir)
    if lm_config.get("num_completions") > 1:
        print("[INFO] Setting num_completions to 1 for self-reflect and MOE")
    lm_config["num_completions"] = 1
    lm_config = lm_config if lm_config is not None else {}
    info = {"lm_config": lm_config, **({} if extra_info is None else extra_info)}
    with open((save_dir / "info.json").as_posix(), "w") as f:
        json.dump(info, f)

    code_proposals = []

    user_prompt = get_writer_prompt_initial(task, animate)
    system_prompt = get_system_prompt(role=Role.WRITER)
    expert_role_save_dir = save_dir / f"prompts/expert_draft"
    save_prompts(expert_role_save_dir.as_posix(), system_prompt, user_prompt)
    experts = generate(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        lm_config={**lm_config, "num_completions": num_experts},
        skip_cache=False,
    )

    for expert in range(num_experts):
        role = Role.WRITER
        critique = None
        draft = experts[expert]
        program = compile_raw_gpt_response_to_program(draft)
        save_and_execute_trial(expert_role_save_dir / str(expert), program)
        rendering_path = find_rendering(expert_role_save_dir / str(expert))

        role = switch_reflection_role(role)

        num_reflections = (num_reflections - 1) // 2 * 2 + 1
        for i in range(1, num_reflections):
            print(
                f"[INFO] Running self-reflection round {i + 1}/{num_reflections} for expert {expert + 1}/{num_experts}"
            )
            if role == Role.WRITER:
                user_prompt = get_writer_prompt_with_critiques(task, program, critique)
                system_prompt = get_system_prompt(role=role)
                draft = generate(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    lm_config=lm_config,
                    skip_cache=True,
                )[0]
                role_save_dir = (
                    save_dir / f"prompts/expert_{expert:02d}_refl_{i:02d}_writer"
                )
                save_prompts(role_save_dir.as_posix(), system_prompt, user_prompt)
                program = compile_raw_gpt_response_to_program(draft)
                save_and_execute_trial(role_save_dir / "0", program)
                rendering_path = find_rendering(role_save_dir / "0")
            elif role == Role.CRITIC:
                role_save_dir = (
                    save_dir / f"prompts/expert_{expert:02d}_refl_{i:02d}_critic"
                )
                user_prompt = get_critic_prompt(task, program, rendering_path)
                system_prompt = get_system_prompt(role=role)
                critique = "\n".join(
                    generate(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        lm_config=lm_config,
                        skip_cache=True,
                    )[0]
                )
                save_prompts(role_save_dir.as_posix(), system_prompt, user_prompt)

                (role_save_dir / "0").mkdir(exist_ok=True)
                with open((role_save_dir / "0/raw.txt").as_posix(), "w") as f:
                    f.write(critique)

            else:
                raise ValueError("Invalid role provided: " + role)
            role = switch_reflection_role(role)

        # role_save_dir = save_dir / f'prompts/expert_{expert:02d}_reparam'
        # system_prompt = get_system_prompt(Role.WRITER, header=read_header(engine_mode='mi_material'))
        # user_prompt = get_user_prompt_reparam(program)
        # save_prompts(role_save_dir.as_posix(), system_prompt, user_prompt)
        # draft = generate(user_prompt=user_prompt, system_prompt=system_prompt, lm_config=lm_config, skip_cache=True)[0]
        # program = compile_raw_gpt_response_to_program(draft)
        # save_and_execute_trial(role_save_dir / '0', program, engine_mode='mi_material')

        code_proposals.append(program)

    user_prompt = get_judge_prompt(task, code_proposals=code_proposals)
    system_prompt = get_system_prompt(role=Role.JUDGE)
    role_save_dir = save_dir / f'prompts/judge'
    save_prompts(role_save_dir.as_posix(), system_prompt, user_prompt)
    draft = generate(user_prompt=user_prompt, system_prompt=system_prompt, lm_config=lm_config, skip_cache=True)[0]
    program = compile_raw_gpt_response_to_program(draft)
    save_and_execute_trial(role_save_dir / '0', program)


def save_and_execute_trial(trial_save_dir: Path, program, engine_mode=ENGINE_MODE):
    trial_save_dir.mkdir(exist_ok=True)
    with open((trial_save_dir / "program.py").as_posix(), "w") as f:
        f.write(program)
    impl = get_impl(program)

    save_to = (trial_save_dir / "impl.py").as_posix()
    with open(save_to, "w") as f:
        f.write(impl)

    command = (
        f'ENGINE_MODE={engine_mode} DEBUG={"1" if DEBUG else "0"} '
        f'PYTHONPATH={Path(__file__).parent / "prompts"}:$PYTHONPATH python {save_to}'
    )

    command_file = (trial_save_dir / "command.txt").as_posix()
    with open(command_file, "w") as f:
        f.write(command)

    execute_command(command, trial_save_dir.as_posix())
