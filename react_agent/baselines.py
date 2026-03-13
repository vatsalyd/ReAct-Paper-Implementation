"""Baseline agents used for ReAct-vs-CoT-vs-Act-only comparisons."""

from __future__ import annotations

import re

from .llm import LLMClient
from .tools import WikipediaEnv
from .parsing import parse_action_line


FEVER_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def _extract_final_answer(text: str) -> str:
    match = re.search(r"Final\s*Answer\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _normalize_fever_label(text: str) -> str:
    upper = text.upper()
    for label in FEVER_LABELS:
        if label in upper:
            return label
    return text.strip()


class CoTAgent:
    """
    Chain-of-thought baseline.

    The model reasons in text only and cannot call tools.
    """

    def __init__(self, task: str = "hotpotqa", llm: LLMClient | None = None):
        self.task = task
        self.llm = llm or LLMClient()
        self.trajectory = []

    def _build_prompt(self, question: str) -> str:
        if self.task == "hotpotqa":
            return (
                "Answer the question using reasoning only (no tools).\n"
                "Think step by step, then output exactly one final line in this format:\n"
                "Final Answer: <answer>\n\n"
                f"Question: {question}"
            )

        if self.task == "fever":
            return (
                "Classify the claim using reasoning only (no tools).\n"
                "Return one of: SUPPORTS, REFUTES, NOT ENOUGH INFO.\n"
                "Output exactly one final line in this format:\n"
                "Final Answer: <label>\n\n"
                f"Claim: {question}"
            )

        raise ValueError(f"Unknown task: {self.task}. Choose 'hotpotqa' or 'fever'.")

    def run(self, question: str) -> tuple[str, list[dict]]:
        self.trajectory = []
        prompt = self._build_prompt(question)
        response = self.llm.generate(prompt)
        answer = _extract_final_answer(response)

        if self.task == "fever":
            answer = _normalize_fever_label(answer)

        self.trajectory.append({"step": 1, "response": response, "answer": answer})
        return answer, self.trajectory


class ActOnlyAgent:
    """
    Action-only baseline.

    The model can call tools but is instructed not to emit explicit thought text.
    """

    def __init__(
        self,
        task: str = "hotpotqa",
        llm: LLMClient | None = None,
        max_steps: int = 7,
    ):
        self.task = task
        self.llm = llm or LLMClient()
        self.max_steps = max_steps
        self.env = WikipediaEnv()
        self.trajectory = []

    def _build_prompt(self, question: str, step: int) -> str:
        history_lines = []
        for item in self.trajectory:
            n = item["step"]
            history_lines.append(f"Action {n}: {item['action']}[{item['action_input']}]")
            history_lines.append(f"Observation {n}: {item['observation']}")

        history = "\n".join(history_lines) if history_lines else "(none yet)"

        task_header = "Question" if self.task == "hotpotqa" else "Claim"
        return (
            "You are an action-only agent.\n"
            "Do NOT output Thought lines. Output exactly one action line.\n"
            "Valid actions:\n"
            "- Search[entity]\n"
            "- Lookup[keyword]\n"
            "- Finish[answer]\n\n"
            f"{task_header}: {question}\n"
            f"History:\n{history}\n\n"
            f"Output format: Action {step}: <Action>[<input>]"
        )

    def _fallback_action(self, question: str, step: int) -> tuple[str, str]:
        if step == 1:
            return "Search", question
        if self.task == "fever" and step >= self.max_steps:
            return "Finish", "NOT ENOUGH INFO"
        return "Lookup", question.split()[0] if question.split() else "answer"

    def run(self, question: str) -> tuple[str, list[dict]]:
        self.trajectory = []
        self.env.reset()
        answer = "I could not determine the answer within the allowed steps."

        for step in range(1, self.max_steps + 1):
            prompt = self._build_prompt(question, step)
            response = self.llm.generate(prompt, stop=["Observation:"])
            parsed = parse_action_line(response)

            if parsed is None:
                action, action_input = self._fallback_action(question, step)
            else:
                action, action_input = parsed

            if action.lower() == "finish":
                observation = action_input
                answer = action_input
                self.trajectory.append(
                    {
                        "step": step,
                        "action": action,
                        "action_input": action_input,
                        "observation": observation,
                        "response": response,
                    }
                )
                break

            observation = self.env.step(action, action_input)
            self.trajectory.append(
                {
                    "step": step,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                    "response": response,
                }
            )

        if self.task == "fever":
            answer = _normalize_fever_label(answer)

        return answer, self.trajectory
