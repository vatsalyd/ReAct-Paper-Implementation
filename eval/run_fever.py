"""
FEVER Evaluation — run ReAct on a subset of fact verification claims.

FEVER (Fact Extraction and VERification, Thorne et al., 2018) asks:
    Given a claim, classify it as:
    - SUPPORTS (claim is true based on evidence)
    - REFUTES  (claim is false based on evidence)
    - NOT ENOUGH INFO (can't determine from available evidence)

This tests a DIFFERENT skill than QA — the agent must:
    1. Search for relevant evidence
    2. Read and interpret the evidence
    3. Make a JUDGMENT (not just extract an answer)

The paper shows ReAct excels here because the reasoning traces
make the judgment process transparent and verifiable.
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from react_agent.agent import ReactAgent
from react_agent.llm import LLMClient
from eval.metrics import accuracy


# ── Curated FEVER subset ──
FEVER_SAMPLES = [
    {
        "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
        "label": "SUPPORTS",
    },
    {
        "claim": "Stranger Things is set in Bloomington, Indiana.",
        "label": "REFUTES",
    },
    {
        "claim": "The Godfather was directed by Francis Ford Coppola.",
        "label": "SUPPORTS",
    },
    {
        "claim": "Python was created by Guido van Rossum.",
        "label": "SUPPORTS",
    },
    {
        "claim": "The Great Wall of China was built in the 20th century.",
        "label": "REFUTES",
    },
    {
        "claim": "Marie Curie won the Nobel Prize in Physics.",
        "label": "SUPPORTS",
    },
    {
        "claim": "The Amazon River flows through Africa.",
        "label": "REFUTES",
    },
    {
        "claim": "Tesla Motors was founded by Elon Musk alone.",
        "label": "REFUTES",
    },
    {
        "claim": "The Beatles were a British band.",
        "label": "SUPPORTS",
    },
    {
        "claim": "Mount Everest is located in Japan.",
        "label": "REFUTES",
    },
    {
        "claim": "Linux was created by Linus Torvalds.",
        "label": "SUPPORTS",
    },
    {
        "claim": "The human body has 206 bones.",
        "label": "SUPPORTS",
    },
]


def run_evaluation(n: int = 5, verbose: bool = True):
    """Run ReAct on a subset of FEVER and compute accuracy."""

    llm = LLMClient()
    samples = FEVER_SAMPLES[:n]

    results = []
    total_correct = 0

    print(f"\n{'='*70}")
    print(f"  FEVER EVALUATION — {n} claims")
    print(f"  Model: {llm.model}")
    print(f"{'='*70}\n")

    for i, sample in enumerate(samples):
        claim = sample["claim"]
        gold_label = sample["label"]

        print(f"\n[{i+1}/{n}] {claim}")
        print(f"  Gold: {gold_label}")

        agent = ReactAgent(task="fever", llm=llm)

        try:
            pred_label, trajectory = agent.run(claim)
        except Exception as e:
            pred_label = f"ERROR: {e}"
            trajectory = []

        acc = accuracy(pred_label, gold_label)
        total_correct += acc

        results.append({
            "claim": claim,
            "gold": gold_label,
            "predicted": pred_label,
            "correct": acc,
            "steps": len(trajectory),
        })

        status = "✅" if acc > 0 else "❌"
        print(f"  Pred: {pred_label}")
        print(f"  {status} Steps={len(trajectory)}")

        if verbose and trajectory:
            agent.print_trace()

        time.sleep(1)

    # ── Summary ──
    avg_acc = total_correct / n

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy:     {avg_acc:.2%}")
    print(f"  Correct:      {int(total_correct)}/{n}")
    print(f"  Total Tokens: {llm.total_tokens_used}")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReAct on FEVER")
    parser.add_argument("--n", type=int, default=5, help="Number of claims to evaluate")
    parser.add_argument("--quiet", action="store_true", help="Suppress trace output")
    args = parser.parse_args()

    run_evaluation(n=args.n, verbose=not args.quiet)
