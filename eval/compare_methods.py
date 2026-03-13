"""
Compare ReAct, CoT, and Act-only on the same subset.

This script is designed to make the core paper claim visible:
interleaving reasoning and acting typically outperforms reasoning-only
or acting-only baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import exact_match, f1_score, accuracy
from eval.run_hotpotqa import HOTPOTQA_SAMPLES
from eval.run_fever import FEVER_SAMPLES
from react_agent.agent import ReactAgent
from react_agent.baselines import CoTAgent, ActOnlyAgent
from react_agent.llm import LLMClient


def _get_samples(task: str, n: int) -> list[dict]:
    if task == "hotpotqa":
        return HOTPOTQA_SAMPLES[:n]
    if task == "fever":
        return FEVER_SAMPLES[:n]
    raise ValueError(f"Unknown task: {task}")


def _score(task: str, pred: str, gold: str) -> dict[str, float]:
    if task == "hotpotqa":
        return {"em": exact_match(pred, gold), "f1": f1_score(pred, gold)}
    return {"accuracy": accuracy(pred, gold)}


def evaluate_task(task: str, n: int = 3, quiet: bool = False) -> dict:
    methods = {
        "react": ReactAgent,
        "cot": CoTAgent,
        "act_only": ActOnlyAgent,
    }
    samples = _get_samples(task, n)

    method_llms = {name: LLMClient() for name in methods}
    aggregates = {
        name: {"count": 0, "em": 0.0, "f1": 0.0, "accuracy": 0.0, "steps": 0}
        for name in methods
    }
    rows = []

    print(f"\n{'=' * 70}")
    print(f"  COMPARISON ({task.upper()}) — n={n}")
    print("  Methods: ReAct vs CoT vs Act-only")
    print(f"{'=' * 70}\n")

    for idx, sample in enumerate(samples, start=1):
        if task == "hotpotqa":
            prompt = sample["question"]
            gold = sample["answer"]
        else:
            prompt = sample["claim"]
            gold = sample["label"]

        print(f"[{idx}/{n}] {prompt}")
        print(f"  Gold: {gold}")

        for name, cls in methods.items():
            llm = method_llms[name]
            agent = cls(task=task, llm=llm)

            try:
                pred, trace = agent.run(prompt)
            except Exception as exc:
                pred, trace = f"ERROR: {exc}", []

            metrics = _score(task, pred, gold)
            row = {
                "task": task,
                "method": name,
                "input": prompt,
                "gold": gold,
                "predicted": pred,
                "steps": len(trace),
                **metrics,
            }
            rows.append(row)

            aggregates[name]["count"] += 1
            aggregates[name]["steps"] += len(trace)
            for metric, value in metrics.items():
                aggregates[name][metric] += value

            if not quiet:
                if task == "hotpotqa":
                    print(
                        f"  - {name:8} | EM={metrics['em']:.0f} "
                        f"F1={metrics['f1']:.2f} Steps={len(trace)} | Pred={pred}"
                    )
                else:
                    print(
                        f"  - {name:8} | Acc={metrics['accuracy']:.0f} "
                        f"Steps={len(trace)} | Pred={pred}"
                    )
        print("")

    summary = {}
    for name, agg in aggregates.items():
        count = agg["count"] or 1
        method_summary = {
            "avg_steps": agg["steps"] / count,
            "total_tokens": method_llms[name].total_tokens_used,
        }
        if task == "hotpotqa":
            method_summary["em"] = agg["em"] / count
            method_summary["f1"] = agg["f1"] / count
        else:
            method_summary["accuracy"] = agg["accuracy"] / count
        summary[name] = method_summary

    print(f"{'-' * 70}")
    print("SUMMARY")
    for name, stats in summary.items():
        if task == "hotpotqa":
            print(
                f"  {name:8} | EM={stats['em']:.2%} "
                f"F1={stats['f1']:.2%} Steps={stats['avg_steps']:.1f} "
                f"Tokens={stats['total_tokens']}"
            )
        else:
            print(
                f"  {name:8} | Acc={stats['accuracy']:.2%} "
                f"Steps={stats['avg_steps']:.1f} Tokens={stats['total_tokens']}"
            )
    print(f"{'-' * 70}\n")

    return {"task": task, "n": n, "rows": rows, "summary": summary}


def save_results(payload: dict) -> str:
    os.makedirs("eval/results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("eval", "results", f"comparison_{payload['task']}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare ReAct vs CoT vs Act-only on curated subsets."
    )
    parser.add_argument(
        "--task",
        choices=["hotpotqa", "fever", "both"],
        default="both",
        help="Task to evaluate.",
    )
    parser.add_argument("--n", type=int, default=3, help="Samples per task.")
    parser.add_argument("--quiet", action="store_true", help="Compact output.")
    parser.add_argument("--save", action="store_true", help="Save JSON report.")
    args = parser.parse_args()

    tasks = ["hotpotqa", "fever"] if args.task == "both" else [args.task]
    for selected_task in tasks:
        result = evaluate_task(selected_task, n=args.n, quiet=args.quiet)
        if args.save:
            out_path = save_results(result)
            print(f"Saved: {out_path}")
