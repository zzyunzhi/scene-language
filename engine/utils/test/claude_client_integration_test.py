import unittest
from engine.utils.claude_client import ClaudeClient  
import os
import tempfile
import shutil
import time
import json
import threading


class TestClaudeClient(unittest.TestCase):
    current_script = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_script)
    _test_cache_path = parent_dir + '/test_cache.json' 

    def setUp(self):
        """Set up a ClaudeClient with a temporary cache file."""
        self.generator = ClaudeClient(cache=TestClaudeClient._test_cache_path)

    def tearDown(self):
        """Clean up by deleting the cache file after each test."""
        if os.path.exists(self._test_cache_path):
            os.remove(self._test_cache_path)
    
    def test_generate_basic_query(self):
        """Test a basic generation query to the API."""
        cache_key, response = self.generator.generate("What is the capital of France?", "You are a helpful geography expert.")
        self.assertIsNotNone(response)
        self.assertTrue("Paris" in response[0])

        self.assertTrue(os.path.exists(self.generator.cache_file))

        # Confirm that it is in the cache
        with open(self.generator.cache_file, "r") as f:
            cache = json.load(f)
            self.assertTrue(cache_key in cache)
            self.assertTrue("Paris" in cache[cache_key][0])

    def test_concurrent_cache_updates(self):
        """Test concurrent updates to ensure .tmp and .lock files are managed correctly."""
        def update_cache():
            self.generator.update_cache("test_key", {"data": "value"})

        # Start multiple threads to update the cache simultaneously
        threads = [threading.Thread(target=update_cache) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Check for presence of .lock or .tmp files indicating incomplete or overlapping operations
        self.assertFalse(os.path.exists(self.generator.cache_file + ".lock"), ".lock file should not exist after operations complete.")
        self.assertFalse(os.path.exists(self.generator.cache_file + ".tmp"), ".tmp file should not exist after operations complete.")

if __name__ == '__main__':
    unittest.main()
