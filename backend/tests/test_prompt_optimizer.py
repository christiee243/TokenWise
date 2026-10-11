"""Tests for prompt optimization and token-saving guarantees."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from prompt_optimizer import _clean_constraint, count_tokens, optimize_prompt


CORPUS = [
    "AI safety is important.",
    "Model the data as a graph.",
    "Aim for 3 bullets.",
    "There are 3 options to compare.",
    "Write a note to say thank you.",
    "Hello AI, summarize this report.",
    "Could you please explain vector databases?",
    "I have a CSV with 1M rows. How should I process it?",
    "Explain how to avoid memory leaks in Python. Use examples.",
    "Summarize the main argument in 5 bullets.",
    "Do not include implementation details.",
    "Never reveal the private key.",
    "Compare PostgreSQL and SQLite for a small app.",
    "In order to deploy this, review the configuration first.",
    "Please list the three largest risks.",
    "Tell me why the migration failed.",
    "Write a note to say thanks a lot.",
    "Use e.g. real-world examples when explaining this.",
    "Do not change `foo()` or the URL https://example.com/a?b=c.",
    'Keep the literal "do not modify" exactly as written.',
    "Give an explanation of the retry policy.",
    "Make sure that you give a clear table.",
    "Thanks in advance for reviewing the code.",
    "What is tokenization?",
    "Is Python faster than Go?",
    "Output as JSON with no extra commentary.",
    "Limit the answer to 200 words.",
    "At this point in time, the service is unavailable.",
    "Prior to release, run the complete test suite.",
    "Write a short email to the project team.",
    "Summarize:\n1. alpha\n2. beta\n3. gamma",
    "Actually, to be honest, it needs another review.",
]


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

    def test_question_rewrite_must_be_strictly_shorter(self):
        result = optimize_prompt("What is tokenization?")

        self.assertLessEqual(result["optimized_tokens"], result["original_tokens"])
        if result["optimized"] != "What is tokenization?":
            self.assertEqual(result["optimized"], "Explain tokenization.")
            self.assertLess(result["optimized_tokens"], result["original_tokens"])
        self.assertNotIn("Task:", result["optimized"])

    def test_verbose_filler_and_constraints_are_cleaned_without_boilerplate(self):
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
        self.assertIn("clear table", result["optimized"].lower())
        self.assertGreater(result["tokens_saved"], 0)

    def test_greetings_do_not_strip_prefixes_from_regular_words(self):
        prompt = "History of Rome is a classic topic."
        result = optimize_prompt(prompt)

        self.assertIn("History of Rome", result["optimized"])
        self.assertNotIn("story of Rome", result["optimized"].lower())

    def test_must_not_constraint_keeps_negation(self):
        result = optimize_prompt("Must not include jargon in the answer.")

        self.assertIn("Must not include jargon", result["optimized"])
        self.assertNotIn("Not include jargon", result["optimized"])

    def test_main_task_not_moved_into_requirements(self):
        result = optimize_prompt(
            "Explain how to avoid memory leaks in Python. Use examples."
        )

        self.assertIn("Explain how to avoid memory leaks in Python", result["optimized"])
        self.assertEqual(result["optimized"].splitlines()[0], "Explain how to avoid memory leaks in Python. Use examples.")

    def test_ai_subject_not_stripped(self):
        self.assertTrue(optimize_prompt("AI safety is important.")["optimized"].startswith("AI safety"))

    def test_model_verb_not_stripped(self):
        self.assertTrue(optimize_prompt("Model the data as a graph.")["optimized"].startswith("Model"))

    def test_question_not_demoted_to_requirement(self):
        output = optimize_prompt("I have a CSV with 1M rows. How should I process it?")["optimized"]
        self.assertNotIn("Requirements:", output)
        self.assertIn("How should I process it?", output)

    def test_thank_you_inside_task_kept(self):
        self.assertIn("thank you", optimize_prompt("Write a note to say thank you.")["optimized"].lower())

    def test_multiline_list_preserved(self):
        prompt = "Summarize:\n1. alpha\n2. beta\n3. gamma"
        self.assertEqual(optimize_prompt(prompt)["optimized"].count("\n"), 3)

    def test_constraint_cleanup_removes_subject_pronoun(self):
        self.assertEqual(_clean_constraint("Make sure that you give a clear table."), "Give a clear table")

    def test_abbreviation_does_not_create_follow_up_sentence(self):
        output = optimize_prompt("Summarize the cases, e.g. use cases from production.")["optimized"]
        self.assertIn("e.g. use cases", output)

    def test_protected_contractions_and_literals_are_preserved(self):
        prompt = "Don't rewrite this: `it's literal`; preserve 'user's value' and the URL https://example.com/a."
        result = optimize_prompt(prompt)
        self.assertIn("Don't", result["optimized"])
        self.assertIn("`it's literal`", result["optimized"])
        self.assertIn("https://example.com/a", result["optimized"])

    def test_result_schema_is_consistent_for_empty_and_fallback(self):
        expected = {
            "optimized", "original_tokens", "optimized_tokens", "tokens_saved",
            "reduction_percent", "changes", "method", "semantic_warning",
        }
        self.assertEqual(set(optimize_prompt("").keys()), expected)
        self.assertEqual(set(optimize_prompt("Hello!").keys()), expected)

    def test_never_longer_than_input_for_realistic_corpus(self):
        for prompt in CORPUS:
            with self.subTest(prompt=prompt):
                result = optimize_prompt(prompt)
                self.assertLessEqual(result["optimized_tokens"], result["original_tokens"])

    def test_yes_no_questions_stay_questions(self):
        result = optimize_prompt("Is Python faster than Go?")

        self.assertTrue(result["optimized"].endswith("?"))
        self.assertIn("Is Python faster than Go", result["optimized"])

    def test_optimizer_returns_method_and_falls_back_when_no_gain(self):
        result = optimize_prompt("Hello!")

        self.assertEqual(result["optimized"], "Hello!")
        self.assertEqual(result["tokens_saved"], 0)
        self.assertIn("method", result)
        self.assertIn(result["method"], {"tiktoken", "approx"})


if __name__ == "__main__":
    unittest.main()