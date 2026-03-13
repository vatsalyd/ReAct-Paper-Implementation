"""
The ReAct Agent — the core contribution of the paper.

From the paper (Section 1):
    "We present ReAct, a general paradigm to synergize reasoning and acting
     in language models. ReAct prompts LLMs to generate both verbal reasoning
     traces and actions pertaining to a task in an interleaved manner, which
     allows the model to perform dynamic reasoning to create, maintain, and
     adjust high-level plans for acting, while also interact with the external
     environments to incorporate additional information into reasoning."

How the loop works:
    1. Build prompt = system instructions + few-shot examples + question + trajectory-so-far
    2. Send to LLM, get back text containing Thought and Action
    3. Parse out the action (Search/Lookup/Finish) and its input
    4. Execute the action in the environment → get Observation
    5. Append Thought + Action + Observation to trajectory
    6. Repeat until Finish or max steps

Why this works (the paper's key insight):
    - CoT alone can hallucinate mid-chain because there's no external verification
    - Act alone (just calling tools) lacks strategic planning
    - ReAct interleaves both: think about what to do → do it → observe → think about what to do next
    - The thoughts create a "reasoning scaffold" AND the actions provide "grounded verification"
"""

from .llm import LLMClient
from .tools import WikipediaEnv
from .prompts import build_prompt
from .parsing import parse_thought_action


class ReactAgent:
    """
    A ReAct agent that interleaves reasoning and acting.

    Usage:
        agent = ReactAgent()
        answer, trace = agent.run("Who was the first president of the US?")
        print(answer)
        agent.print_trace()
    """

    def __init__(
        self,
        task: str = "hotpotqa",
        llm: LLMClient = None,
        max_steps: int = 7,
    ):
        """
        Args:
            task: "hotpotqa" or "fever" — determines which few-shot prompt to use
            llm: LLM client (defaults to Groq free tier)
            max_steps: Maximum reasoning steps before forced termination.
                       The paper uses ~7 steps. Too few = can't solve multi-hop.
                       Too many = wasted tokens on hopeless queries.
        """
        self.task = task
        self.llm = llm or LLMClient()
        self.max_steps = max_steps
        self.env = WikipediaEnv()

        # Trajectory — the growing list of (thought, action, action_input, observation) tuples
        # This IS the agent's "memory" — it gets appended to the prompt each iteration
        self.trajectory = []
        self.answer = None
        self.finished = False

    def run(self, question: str) -> tuple[str, list[dict]]:
        """
        Run the full ReAct loop on a question.

        Returns:
            (answer, trajectory) where trajectory is a list of step dicts
        """
        self.trajectory = []
        self.answer = None
        self.finished = False
        self.env.reset()

        for step in range(1, self.max_steps + 1):
            # ── Step 1: Build the prompt with trajectory so far ──
            trajectory_text = self._format_trajectory()
            prompt = build_prompt(self.task, question, trajectory_text)

            # ── Step 2: Ask the LLM to generate Thought + Action ──
            # Stop at "Observation:" — the LLM must NOT generate observations
            # because observations come from the real environment (tools).
            # This is critical: it prevents the model from hallucinating what
            # it THINKS search results would say.
            response = self.llm.generate(prompt, stop=["Observation:"])

            # ── Step 3: Parse the response ──
            thought, action, action_input = self._parse_response(response, step)

            # ── Step 4: Execute the action in the environment ──
            if action.lower() == "finish":
                observation = action_input
                self.answer = action_input
                self.finished = True
            else:
                observation = self.env.step(action, action_input)

            # ── Step 5: Record the step ──
            step_record = {
                "step": step,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            }
            self.trajectory.append(step_record)

            # ── Step 6: Check if done ──
            if self.finished:
                break

        # If we hit max steps without finishing, take the last thought as answer
        if not self.finished:
            self.answer = "I could not determine the answer within the allowed steps."

        return self.answer, self.trajectory

    def _parse_response(self, response: str, step: int) -> tuple[str, str, str]:
        """
        Parse the LLM's response to extract Thought, Action, and Action Input.

        The LLM is prompted to output in this format:
            Thought N: <reasoning>
            Action N: <ActionType>[<input>]

        We need to robustly extract these, handling various formatting quirks.
        """
        return parse_thought_action(response, step)

    def _format_trajectory(self) -> str:
        """
        Format the trajectory so far as text to append to the prompt.

        Each step becomes:
            Thought N: <thought>
            Action N: <action>[<input>]
            Observation N: <observation>

        This growing context is what gives the agent "memory" of its
        previous reasoning and findings.
        """
        lines = []
        for step in self.trajectory:
            n = step["step"]
            lines.append(f"Thought {n}: {step['thought']}")
            lines.append(f"Action {n}: {step['action']}[{step['action_input']}]")
            lines.append(f"Observation {n}: {step['observation']}")
        return "\n".join(lines)

    def print_trace(self):
        """
        Pretty-print the agent's reasoning trace.

        This visualization is useful for understanding HOW the agent
        arrived at its answer — which is the whole point of ReAct's
        interpretability advantage over pure action-based agents.
        """
        print("=" * 70)
        print("REACT TRACE")
        print("=" * 70)

        for step in self.trajectory:
            n = step["step"]
            print(f"\n{'─' * 50}")
            print(f"  💭 Thought {n}: {step['thought']}")
            print(f"  ⚡ Action  {n}: {step['action']}[{step['action_input']}]")

            obs = step['observation']
            if len(obs) > 200:
                obs = obs[:200] + "..."
            print(f"  👁️ Observe {n}: {obs}")

        print(f"\n{'─' * 50}")
        print(f"  ✅ Answer: {self.answer}")
        print(f"  📊 Steps: {len(self.trajectory)} | Tokens used: {self.llm.total_tokens_used}")
        print("=" * 70)
