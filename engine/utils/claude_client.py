from engine.constants import PROJ_DIR, ANTHROPIC_API_KEY, MAX_TOKENS, TEMPERATURE, NUM_COMPLETIONS
import copy
from pathlib import Path
import base64
import json
import os
import time
import random
from PIL import Image
import io
import anthropic


CLAUDE_MODEL_NAME = 'claude-3-5-sonnet-20240620'  # this the model used throughout the paper
CLAUDE_MODEL_NAME = 'claude-3-5-sonnet-20241022'
CLAUDE_MODEL_NAME = 'claude-3-7-sonnet-20250219'

class ClaudeClient:
    def __init__(self, model_name=CLAUDE_MODEL_NAME, cache="cache.json"):
        self.cache_file = cache
        self.model_name = model_name
        self.exponential_backoff = 1
        # Load the cache JSON file, if cache file exists. Else, cache is {}
        if os.path.exists(cache):
            while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
                time.sleep(0.1)
            with open(cache, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Tip, if you want to extend MAX_TOKENS to 8000, attach default_headers after the api_key arg above.
        # default_headers={
        #    "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"
         #}

    def generate(self, user_prompt, system_prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop_sequences=None, verbose=False,
                 num_completions=NUM_COMPLETIONS, skip_cache_completions=0, skip_cache=False):

        print(f'[INFO] Claude: querying for {num_completions=}, {skip_cache_completions=}')
        if skip_cache:
            print(f'[INFO] Claude: Skipping cache')
        if verbose:
            print(user_prompt)
            print("-----")

        # Prepare messages for the API request
        if isinstance(user_prompt, str):
            content = [{"type": "text", "text": user_prompt}]
        elif isinstance(user_prompt, list):
            content = []
            for content_item in user_prompt:
                if content_item['type'] == 'image_url' and os.path.exists(content_item['image_url']):
                    with open(content_item['image_url'], "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    content_item = {
                        'type': 'image',
                        "source": {
                            "type": "base64",
                            "media_type": f'image/{Path(content_item["image_url"]).suffix.removeprefix(".")}',
                            "data": image_data,
                        },
                    }
                content.append(content_item)
        else:
            raise RuntimeError(user_prompt)
        messages = [{"role": "user", "content": content}]

        cache_key = None
        results = []
        if not skip_cache:
            cache_key = str((user_prompt, system_prompt, max_tokens, temperature, stop_sequences, 'claude'))

            num_completions = skip_cache_completions + num_completions
            if cache_key in self.cache:
                print(f'[INFO] Claude: cache hit {len(self.cache[cache_key])}')
                if len(self.cache[cache_key]) < num_completions:
                    num_completions -= len(self.cache[cache_key])
                    results = self.cache[cache_key]
                else:
                    return cache_key, self.cache[cache_key][skip_cache_completions:num_completions]

        while num_completions > 0:
            with self.client.messages.stream(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences
            ) as stream:
                content = ''
                for text in stream.text_stream:
                    content += text
            indented = content.split('\n')
            results.append(indented)
            
            print(f'[INFO] Claude usage', stream.get_final_message().usage)

            num_completions -= 1

        if not skip_cache:
            self.update_cache(cache_key, results)

        return cache_key, results[skip_cache_completions:]

    def update_cache(self, cache_key, results):
        while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
            time.sleep(0.1)
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


def setup_claude():
    try:
        username = os.getlogin()
    except OSError:
        username = os.environ.get('USER') or os.environ.get('LOGNAME')

    model = ClaudeClient(cache='cache.json' if not os.path.exists('/viscam/') else f'cache_{username}.json')
    return model


def test_claude():
    cache_dir = Path(PROJ_DIR) / 'cache'
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / 'test_claude_client.json'
    # cache_file.unlink(missing_ok=True)
    model = ClaudeClient(cache=cache_file.as_posix())
    # cache_key, response = model.generate("Generate a random number.", "You are a helpful code assistant.",
    #                                      num_completions=2, temperature=1,
    #                                      skip_cache_completions=1)
    cache_key, response = model.generate(
        [{'type': 'text', 'text': 'what is in the image?'},
         {'type': 'image_url', 'image_url': '/Users/yzzhang/projects/engine/scripts/exp/icl_0512/outputs/run_two_rounds_20240712-161451_43fa6127-9d7b-4140-a2a7-a061340e1f78/Golden_Gate_Bridge/dependency_to_program_0_0/0/renderings/mi_golden_gate_bridge_indoors_no_window_frame_00/rendering_traj_000.png'}
         ], 'You are a helpful assistant.', num_completions=2, temperature=0.2,
    )
    print(cache_key)
    print(response)
    print(len(response))
    print(len(model.cache[cache_key]))


if __name__ == "__main__":
    test_claude()
