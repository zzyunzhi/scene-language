import unittest
from engine.utils.code_llama_client import LlamaClient  
import os
import json
import threading
import torch


class TestLlamaClient(unittest.TestCase):
    current_script = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_script)
    _test_cache_path = os.path.join(parent_dir, 'test_llama_cache.json')
    print(f"Test cache path: {_test_cache_path}")

    @classmethod
    def setUpClass(cls):
        """Set up a LlamaClient with a temporary cache file."""
        cls.generator = LlamaClient(cache=cls._test_cache_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up by deleting the cache file after all tests."""
        if os.path.exists(cls._test_cache_path):
            os.remove(cls._test_cache_path)
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_generate_basic_query(self):
        """Test a basic generation query to the model."""
        cache_key, response = self.generator.generate(
            "What is the capital of France?", 
            "You are a helpful geography expert.",
            num_completions=1
        )
        self.assertIsNotNone(response)
        self.assertTrue(any("Paris" in resp for resp in response[0]), "Expected 'Paris' in the response")

        self.assertTrue(os.path.exists(self.generator.cache_file))

        # Confirm that it is in the cache
        with open(self.generator.cache_file, "r") as f:
            cache = json.load(f)
            self.assertTrue(cache_key in cache)
            self.assertTrue(any("Paris" in resp for resp in cache[cache_key][0]), "Expected 'Paris' in the cached response")

    def test_multiple_completions(self):
        """Test generating multiple completions."""
        cache_key, responses = self.generator.generate(
            "List three colors.",
            "You are a helpful assistant.",
            num_completions=3,
            temperature=0.8  
        )
        self.assertEqual(len(responses), 3, "Expected 3 responses")
        for response in responses:
            self.assertTrue(any(color in ' '.join(response).lower() for color in ['red', 'blue', 'green', 'yellow', 'purple', 'orange']),
                            "Expected color names in responses")

    def test_cache_retrieval(self):
        """Test retrieving from cache and generating new responses."""
        prompt = "What's a famous landmark in Paris?"
        system = "You are a travel guide."
        
        # First call should generate new response
        cache_key1, response1 = self.generator.generate(prompt, system, num_completions=1)
        
        # Second call should retrieve from cache
        cache_key2, response2 = self.generator.generate(prompt, system, num_completions=1)
        
        self.assertEqual(cache_key1, cache_key2, "Cache keys should be identical for the same query")
        self.assertEqual(response1, response2, "Responses should be identical when retrieved from cache")
        
        # Third call with skip_cache_completions should generate new response
        _, response3 = self.generator.generate(prompt, system, num_completions=1, skip_cache_completions=1)
        
        self.assertNotEqual(response1, response3, "New response should be generated when skipping cache")

if __name__ == '__main__':
    unittest.main()