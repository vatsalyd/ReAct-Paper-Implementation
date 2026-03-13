"""Parsing helpers for ReAct-style and action-only agent outputs."""

from __future__ import annotations

import re


def parse_action_line(text: str) -> tuple[str, str] | None:
    """
    Extract an action and input from model text.

    Accepted forms:
    - Action 2: Search[Bhutan]
    - Search[Bhutan]
    """
    strict = re.search(
        r"Action\s*\d*\s*:\s*(\w+)\s*\[(.*?)\]",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if strict:
        return strict.group(1).strip(), strict.group(2).strip()

    relaxed = re.search(
        r"\b(\w+)\s*\[(.*?)\]",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if relaxed:
        return relaxed.group(1).strip(), relaxed.group(2).strip()

    return None


def parse_thought_action(response: str, step: int) -> tuple[str, str, str]:
    """
    Parse a ReAct response into (thought, action, action_input).
    """
    thought_match = re.search(
        rf"Thought\s*(?:{step})?\s*:\s*(.*?)(?=Action\s*(?:{step})?\s*:|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    if thought_match:
        thought = thought_match.group(1).strip()
    else:
        action_start = re.search(r"Action\s*\d*\s*:", response, re.IGNORECASE)
        if action_start:
            thought = response[:action_start.start()].strip()
        else:
            thought = response.strip()

    parsed_action = parse_action_line(response)
    if parsed_action:
        action, action_input = parsed_action
    else:
        action = "Search"
        action_input = thought[:50] if thought else "unknown"

    return thought, action, action_input
