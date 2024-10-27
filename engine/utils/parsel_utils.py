from engine.constants import PROJ_DIR, OPENAI_API_KEY, MAX_TOKENS, TEMPERATURE, NUM_COMPLETIONS
try:
    from openai import OpenAI
    import openai
except ModuleNotFoundError:
    print("[ERROR] OpenAI package not installed. Please ignore this error if you intend to use other language models.")
from typing import Optional
from PIL import Image
import io
import base64
import json
import os
import time
import random


# MODEL_NAME = 'gpt-3.5-turbo-0125'
# MODEL_NAME = 'gpt-4-turbo-2024-04-09'
MODEL_NAME = 'gpt-4o-2024-08-06'
MAX_TOKENS = 16000


# https://github.com/openai/openai-python
# https://platform.openai.com/docs/api-reference/chat/create

class CodeGen:

    def __init__(self, model_name=MODEL_NAME, cache="cache.json"):

        self.cache_file = cache
        self.model_name = model_name
        self.exponential_backoff = 1
        # Load the cache JSON file, if cache file exists. Else, cache is {}
        if os.path.exists(self.cache_file):
            cache = None
            print(f'[INFO] Loading cache file from {self.cache_file}...')
            while cache is None:
                while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
                    time.sleep(1)
                try:
                    with open(self.cache_file, "r") as f:
                        cache = json.load(f)
                except Exception as e:
                    print(e)
                    time.sleep(1)
            print('[INFO] Loading cache done!')
            self.cache = cache
        else:
            self.cache = {}

        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate(self,
                 user_prompt: Optional[str],
                 system_prompt: Optional[str],
                 prepend_messages: list[dict[str, str]] = None,
                 num_completions=NUM_COMPLETIONS, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, presence_penalty=0.0,
                 stop=None, indented=False,
                 indented_after_first_line=False, require=None, cache_key=None,
                 rate_limit_tokens=MAX_TOKENS, verbose=False
                 ):
        if verbose:
            print(user_prompt)
            print("-----")

        assert 0 <= temperature <= 1, temperature
        if num_completions > 1:
            if temperature == 0:
                print(f'[ERROR] temperature must be > 0 for num_completions > 1, but got {temperature=}, {num_completions=}')
                num_completions = 1

        messages = []
        if system_prompt is not None:
            messages = messages + [{"role": "system", "content": system_prompt}]
        if prepend_messages is not None:
            messages = messages + prepend_messages
        if user_prompt is not None:
            messages = messages + [{"role": "user", "content": user_prompt}]
        assert messages[0]['role'] == 'system', messages[0]
        assert all(messages[i]['role'] == 'user' for i in range(1, len(messages), 2)), messages
        assert all(messages[i]['role'] == 'assistant' for i in range(2, len(messages), 2)), messages
        assert messages[-1]['role'] == 'user', messages[-1]

        for message in messages:
            if isinstance(message['content'], list):
                for content_item in message['content']:
                    if content_item['type'] == 'image_url' and os.path.exists(content_item['image_url']):
                        assert message['role'] == 'user', message
                        image = content_item['image_url']
                        base64_image = encode_image(image)
                        content_item['image_url'] = {
                            "url": f"data:image/png;base64,{base64_image}",
                            'detail': 'low',
                        }
            else:
                assert isinstance(message['content'], str)

        if cache_key is not None:
            cache_key_base = cache_key
        else:
            cache_key_base = tuple((s['role'], s['content']) for s in messages)
        # cache_key_base = codex_in if cache_key is None else cache_key
        cache_key_list = (cache_key_base, max_tokens, temperature, stop, indented, indented_after_first_line, require)
        if presence_penalty != 0.0:
            cache_key_list = cache_key_list + (presence_penalty,)
        cache_key = str(cache_key_list)
        if cache_key in self.cache:
            if len(self.cache[cache_key]) < num_completions:
                num_completions -= len(self.cache[cache_key])
                results = self.cache[cache_key]
            else:
                cur_implementations = self.cache[cache_key].copy()
                # if "shuffle_implementations" in CONSTS and CONSTS["shuffle_implementations"]:
                #     random.shuffle(cur_implementations)
                return None, cur_implementations[:num_completions]
        else:
            results = []

        print(f"Calling {self.model_name} for {num_completions=}!")
        # raise Exception("Codex is not available")
        total_tokens = num_completions * max_tokens
        completions_per_call = rate_limit_tokens // max_tokens
        while total_tokens > 0:
            num_completions = min(total_tokens // max_tokens, completions_per_call)
            print(num_completions, "completions", max_tokens, "tokens each")
            while True:
                try:
                    time.sleep(8)
                    assert stop is None, stop
                    completions = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        presence_penalty=presence_penalty,
                        # stop=stop,  # otherwise it errors with vision inputs for unknown reasons
                        n=num_completions,
                    ).choices
                    self.exponential_backoff = 1
                    break
                except openai.RateLimitError:
                    import traceback; traceback.print_exc()
                    import ipdb; ipdb.set_trace()
                    print("Rate limit reached. Waiting before retrying...")
                    time.sleep(16 * self.exponential_backoff)
                    self.exponential_backoff *= 2
            for completion in completions:
                result = []
                for line_idx, line in enumerate(completion.message.content.split("\n")):
                    if (indented or (indented_after_first_line and line_idx > 0)) and line.lstrip() == line and line.strip() != "":
                        break
                    if require is not None and line.strip() != "" and require not in line:
                        break
                    result += [line]
                results.append(result)

            # Save updated cache - reopen in case multiple processes running
            # Save to a temp file first, then rename
            # Check if a temp file exists, and if so, wait for it to be deleted
            while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
                time.sleep(0.1)
            # create an empty file to indicate that we are writing to the cache
            with open(self.cache_file + ".lock", "w") as f:
                pass
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            self.cache[cache_key] = results
            with open(self.cache_file + ".tmp", "w") as f:
                json.dump(self.cache, f)
            os.rename(self.cache_file + ".tmp", self.cache_file)
            os.remove(self.cache_file + ".lock")
            total_tokens -= num_completions * max_tokens
        return None, results


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def setup_gpt():
    try:
        username = os.getlogin()
    except OSError:
        username = os.environ.get('USER') or os.environ.get('LOGNAME')
    model = CodeGen(MODEL_NAME, 'cache.json' if not os.path.exists('/viscam/') else f'cache_{username}.json')
    return model
