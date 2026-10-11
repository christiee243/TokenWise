"""Tests for prompt optimization and token-saving guarantees."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from prompt_optimizer import count_tokens, optimize_prompt


class TestPromptOptimizer(unittest.TestCase):
    def test_short_prompt_removes_filler_without_adding_boilerplate(self):
        result = optimize_prompt("Could you please summarize this clearly?")

        self.assertEqual(result["optimized"], "Summarize this clearly.")
        self.assertNotIn("Could you please", result["optimized"])
        self.assertNotIn("Task:", result["optimized"])
        self.assertEqual(result["optimized_tokens"], count_tokens(result["optimized"]))
        self.assertGreaterEqual(result["tokens_saved"], 0)

    def test_already_concise_prompt_is_preserved_cleanly(self):
        prompt = "Summarize this clearly."
        result = optimize_prompt(prompt)

        self.assertEqual(result["optimized"], "Summarize this clearly.")
        self.assertNotIn("Task:", result["optimized"])
        self.assertEqual(result["optimized_tokens"], count_tokens(result["optimized"]))
        self.assertIn("already concise", result["changes"][0].lower())

    def test_question_is_rewritten_as_a_direct_task(self):
        result = optimize_prompt("What is tokenization?")

        self.assertEqual(result["optimized"], "Explain tokenization.")
        self.assertNotIn("Task:", result["optimized"])

    def test_verbose_filler_and_constraints_are_structured(self):
        prompt = (
            "Hello AI, can you please kindly act as an expert in Python and tell me how "
            "in order to optimize tokens due to the fact that they cost money? "
            "Make sure that you give a clear table."
        )
        result = optimize_prompt(prompt)

        self.assertNotIn("Hello AI", result["optimized"])
        self.assertNotIn("can you please", result["optimized"])
        self.assertNotIn("due to the fact that", result["optimized"])
        self.assertNotIn("Task:", result["optimized"])
        self.assertIn("Role: Expert in Python", result["optimized"])
        self.assertIn("Requirements:", result["optimized"])
        self.assertGreater(result["tokens_saved"], 0)
        self.assertGreater(result["reduction_percent"], 20.0)


if __name__ == "__main__":
    unittest.main()