import os
import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from engine.constants import MAX_TOKENS, TEMPERATURE, NUM_COMPLETIONS


class LlamaClient:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", cache="llama_cache.json"):
        self.cache_file = cache
        self.model_name = model_name

        # Load cache
        if os.path.exists(cache):
            while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
                time.sleep(0.1)
            with open(cache, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        # Load model and tokenizer
        print("Loading tokenizer and model...")
        load_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        load_end = time.time()
        print(f"Model loading time: {load_end - load_start:.2f} seconds")
        print("Model device:", self.model.device)

    def generate(self, user_prompt, system_prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop_sequences=None, verbose=False,
                 num_completions=NUM_COMPLETIONS, skip_cache_completions=0):
        
        print(f'[INFO] Llama3: querying for {num_completions=}, {skip_cache_completions=} before searching cache')
        if verbose:
            print(user_prompt)
            print("-----")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        cache_key = str((user_prompt, system_prompt, max_tokens, temperature, stop_sequences, 'llama'))

        num_completions = skip_cache_completions + num_completions
        if cache_key in self.cache:
            print(f'[INFO] Llama: cache hit {len(self.cache[cache_key])}')
            if len(self.cache[cache_key]) < num_completions:
                num_completions -= len(self.cache[cache_key])
                results = self.cache[cache_key]
            else:
                return cache_key, self.cache[cache_key][skip_cache_completions:num_completions]
        else:
            results = []

        print(f'[INFO] Llama: querying for {num_completions=}')

        while num_completions > 0:
            response, _ = self.generate_response(messages, max_tokens, temperature)
            results.append(response.split('\n'))
            num_completions -= 1

        self.update_cache(cache_key, results)
        return cache_key, results[skip_cache_completions:]

    def generate_response(self, messages, max_tokens, temperature):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        input_ids = inputs.to(self.model.device)
        
        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        start_time = time.time()
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        end_time = time.time()
        
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True), end_time - start_time

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

def setup_llama():
    try:
        username = os.getlogin()
    except OSError:
        username = os.environ.get('USER') or os.environ.get('LOGNAME')

    model = LlamaClient(cache='llama_cache.json' if not os.path.exists('/viscam/') else f'llama_cache_{username}.json')
    return model
