import unittest

from eval.metrics import accuracy, exact_match, f1_score, normalize_answer


class MetricsTests(unittest.TestCase):
    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("The United States!"), "united states")

    def test_exact_match_case_and_punctuation_insensitive(self):
        self.assertEqual(exact_match("Richard Nixon", "richard nixon."), 1.0)

    def test_f1_partial_overlap(self):
        score = f1_score("Richard Milhous Nixon", "Richard Nixon")
        self.assertAlmostEqual(score, 0.8, places=2)

    def test_accuracy_label_match(self):
        self.assertEqual(accuracy("supports", "SUPPORTS"), 1.0)


if __name__ == "__main__":
    unittest.main()
