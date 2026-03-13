"""
HotpotQA Evaluation — run ReAct on a subset of multi-hop questions.

HotpotQA (Yang et al., 2018) is a multi-hop QA dataset where answering
requires reasoning over multiple Wikipedia paragraphs.

Example: "Were the directors of 'Jaws' and 'Schindler's List' both American?"
    → Requires finding the director of Jaws (Spielberg), the director of
      Schindler's List (also Spielberg), and then answering "yes."

The paper uses this to show that ReAct outperforms:
    - CoT alone (which hallucinates facts)
    - Act alone (which lacks strategic planning)

We use a small curated subset since running on the full dev set would
cost too many API calls for a learning project.
"""

import json
import os
import sys
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from react_agent.agent import ReactAgent
from react_agent.llm import LLMClient
from eval.metrics import exact_match, f1_score


# ── Curated HotpotQA subset ──
# These are real HotpotQA questions, chosen to represent different
# reasoning patterns: bridge questions, comparison questions, yes/no questions
HOTPOTQA_SAMPLES = [
    {
        "question": "Were Pavel Urysohn and Leonid Levin known for the same type of work?",
        "answer": "yes",
        "type": "comparison",
    },
    {
        "question": "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        "answer": "approximately 1,800 to 7,000 ft",
        "type": "bridge",
    },
    {
        "question": "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who was named after who?",
        "answer": "Richard Nixon",
        "type": "bridge",
    },
    {
        "question": "Which documentary is about Finnish rock groups, 'The Saimaa Gesture' or 'Global Metal'?",
        "answer": "The Saimaa Gesture",
        "type": "comparison",
    },
    {
        "question": "What nationality is the creator of the TV series 'Lost'?",
        "answer": "American",
        "type": "bridge",
    },
    {
        "question": "Which magazine was started first, 'Arthur's Magazine' or 'First for Women'?",
        "answer": "Arthur's Magazine",
        "type": "comparison",
    },
    {
        "question": "What profession does Nicholas combatively and Gene Siskel have in common?",
        "answer": "film critic",
        "type": "comparison",
    },
    {
        "question": "Which film has the director born later, 'El Dorado' or 'The Man from Laramie'?",
        "answer": "El Dorado",
        "type": "comparison",
    },
    {
        "question": "The creator of 'Wallace and Gromit' also created what other clay animation?",
        "answer": "Shaun the Sheep",
        "type": "bridge",
    },
    {
        "question": "What is the capital of the country where the Cheli La pass is located?",
        "answer": "Thimphu",
        "type": "bridge",
    },
    {
        "question": "In which city was the composer of 'Finlandia' born?",
        "answer": "Hämeenlinna",
        "type": "bridge",
    },
    {
        "question": "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas?",
        "answer": "I'm a Jayhawk",
        "type": "bridge",
    },
]


def run_evaluation(n: int = 5, verbose: bool = True):
    """Run ReAct on a subset of HotpotQA and compute metrics."""

    llm = LLMClient()
    samples = HOTPOTQA_SAMPLES[:n]

    results = []
    total_em = 0
    total_f1 = 0

    print(f"\n{'='*70}")
    print(f"  HOTPOTQA EVALUATION — {n} questions")
    print(f"  Model: {llm.model}")
    print(f"{'='*70}\n")

    for i, sample in enumerate(samples):
        question = sample["question"]
        gold_answer = sample["answer"]

        print(f"\n[{i+1}/{n}] {question}")
        print(f"  Gold: {gold_answer}")

        agent = ReactAgent(task="hotpotqa", llm=llm)

        try:
            pred_answer, trajectory = agent.run(question)
        except Exception as e:
            pred_answer = f"ERROR: {e}"
            trajectory = []

        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)

        total_em += em
        total_f1 += f1

        results.append({
            "question": question,
            "gold": gold_answer,
            "predicted": pred_answer,
            "em": em,
            "f1": f1,
            "steps": len(trajectory),
        })

        status = "✅" if em > 0 else ("🔶" if f1 > 0.5 else "❌")
        print(f"  Pred: {pred_answer}")
        print(f"  {status} EM={em:.0f}  F1={f1:.2f}  Steps={len(trajectory)}")

        if verbose and trajectory:
            agent.print_trace()

        # Small delay to respect rate limits
        time.sleep(1)

    # ── Summary ──
    avg_em = total_em / n
    avg_f1 = total_f1 / n
    avg_steps = sum(r["steps"] for r in results) / n

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Exact Match:  {avg_em:.2%}")
    print(f"  F1 Score:     {avg_f1:.2%}")
    print(f"  Avg Steps:    {avg_steps:.1f}")
    print(f"  Total Tokens: {llm.total_tokens_used}")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReAct on HotpotQA")
    parser.add_argument("--n", type=int, default=5, help="Number of questions to evaluate")
    parser.add_argument("--quiet", action="store_true", help="Suppress trace output")
    args = parser.parse_args()

    run_evaluation(n=args.n, verbose=not args.quiet)
