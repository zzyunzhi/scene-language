from engine.third_party.codellama.llama import Llama
import os
from typing import Callable
from engine.third_party.codellama.llama.tokenizer import Tokenizer
from engine.third_party.codellama.llama.generation import B_INST, E_INST, B_SYS, E_SYS, Dialog
from transformers import CodeLlamaTokenizer
import itertools
from engine.constants import IGNORE_INDEX
import logging
logger = logging.getLogger(__name__)


def tokenize_dialog(dialog: Dialog, tokenizer: CodeLlamaTokenizer | Tokenizer, debug: bool = False) -> dict:
    # this function does not support turn-based dialogs, https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf#chat-prompt
    encode: Callable[[str, bool, bool], list[int]]
    if isinstance(tokenizer, CodeLlamaTokenizer):
        def encode(text: str, bos: bool, eos: bool):
            t = tokenizer.encode(text, add_special_tokens=False)
            if bos:
                t = [tokenizer.bos_token_id] + t
            if eos:
                t = t + [tokenizer.eos_token_id]
            return t

        if debug:
            def encode(text: str, bos: bool, eos: bool):
                t = tokenizer.tokenize(text)
                # t = [text]  # dummy tokenizer
                if bos:
                    t = [tokenizer.bos_token] + t
                if eos:
                    t = t + [tokenizer.eos_token]
                return t
    elif isinstance(tokenizer, Tokenizer):
        assert tokenizer.step_id is None, tokenizer
        assert not debug, "debug mode is not supported for Tokenizer"
        logger.warning(f'Should not happen during training time: {tokenizer}')
        encode = tokenizer.encode
    else:
        raise NotImplementedError(tokenizer)

    if dialog[0]["role"] == "system":
        dialog = [  # type: ignore
                     {
                         "role": dialog[1]["role"],
                         "content": B_SYS
                                    + dialog[0]["content"]
                                    + E_SYS
                                    + dialog[1]["content"],
                     }
                 ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    assert (
            dialog[-1]["role"] == "assistant"
    ), f"Last message must be from user, got {dialog[-1]['role']}"

    # following engine.third_party.llama_recipes.recipes.finetuning.datasets.custom_dataset.tokenize_dialog
    prompt_tokens = [encode(f"{B_INST} {prompt['content'].strip()} {E_INST}", bos=True, eos=False)
                     for prompt, _ in zip(dialog[::2], dialog[1::2])]
    answer_tokens = [encode(f"{answer['content'].strip()} ", bos=False, eos=True)
                     for _, answer in zip(dialog[::2], dialog[1::2])]

    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    # Add labels, convert prompt token to -100 in order to ignore in loss function
    labels_tokens = [len(c) * [IGNORE_INDEX,] if i % 2 == 0 else c for i, c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1] * len(combined_tokens["input_ids"]))
