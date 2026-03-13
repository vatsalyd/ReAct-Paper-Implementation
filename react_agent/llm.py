"""
Thin LLM wrapper — works with any OpenAI-compatible API (Groq, OpenAI, Together, etc.)

The paper used GPT-3 (text-davinci-002). We use Groq's free tier with Llama 3.3 70B,
which is a capable open-source alternative at zero cost.

Key insight from the paper: The prompting strategy matters MORE than the model.
ReAct's few-shot prompts work across different LLMs because the reasoning pattern
(Thought→Action→Observation) is model-agnostic.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Minimal LLM client. The paper's contribution isn't about model architecture —
    it's about the prompting framework. So this wrapper is intentionally thin.
    """

    def __init__(
        self,
        provider: str = "groq",
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        if provider == "groq":
            self.client = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )
            self.model = model or "llama-3.3-70b-versatile"
        elif provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o-mini"
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_tokens_used = 0

    def generate(self, prompt: str, stop: list[str] = None) -> str:
        """
        Generate a completion from the LLM.

        The stop sequences are crucial for ReAct — we stop at "Observation:"
        because the observation comes from the environment (tool output),
        not from the LLM. This prevents the model from hallucinating observations.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
        )

        # Track token usage for analysis
        if response.usage:
            self.total_tokens_used += response.usage.total_tokens

        return response.choices[0].message.content.strip()
