import unittest

from react_agent.parsing import parse_action_line, parse_thought_action


class ParsingTests(unittest.TestCase):
    def test_parse_action_line_strict(self):
        action = parse_action_line("Action 2: Search[Bhutan]")
        self.assertEqual(action, ("Search", "Bhutan"))

    def test_parse_action_line_relaxed(self):
        action = parse_action_line("Search[Cheli La pass]")
        self.assertEqual(action, ("Search", "Cheli La pass"))

    def test_parse_thought_action_happy_path(self):
        text = (
            "Thought 1: I should find the country.\n"
            "Action 1: Search[Cheli La pass]"
        )
        thought, action, action_input = parse_thought_action(text, step=1)
        self.assertEqual(thought, "I should find the country.")
        self.assertEqual(action, "Search")
        self.assertEqual(action_input, "Cheli La pass")

    def test_parse_thought_action_fallback(self):
        thought, action, action_input = parse_thought_action("No action emitted", step=1)
        self.assertEqual(thought, "No action emitted")
        self.assertEqual(action, "Search")
        self.assertEqual(action_input, "No action emitted")


if __name__ == "__main__":
    unittest.main()
