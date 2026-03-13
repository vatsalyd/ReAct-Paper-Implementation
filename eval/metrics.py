"""
Evaluation metrics — standard NLP metrics used in the paper.

The paper evaluates ReAct using:
    - Exact Match (EM): Does the predicted answer exactly match the gold answer?
    - F1 Score: Token-level overlap between prediction and gold answer
    - Accuracy: For classification tasks like FEVER (SUPPORTS/REFUTES/NOT ENOUGH INFO)

These are the SAME metrics used across most QA research (SQuAD, HotpotQA, etc.),
so implementing them from scratch teaches you how NLP evaluation actually works.

Key subtlety: EM is harsh — "Richard Nixon" vs "richard nixon" would fail without
normalization. The normalize_answer function handles this.
"""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """
    Normalize answer for fair comparison.

    This is the standard normalization from the SQuAD evaluation script,
    also used by HotpotQA. It:
        1. Lowercases
        2. Removes articles (a, an, the)
        3. Removes punctuation
        4. Removes extra whitespace

    Without this, "The United States" vs "United States" would count as wrong.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact Match — the strictest metric.

    Returns 1.0 if normalized prediction == normalized ground truth, else 0.0.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score — more forgiving than EM.

    F1 = 2 * (precision * recall) / (precision + recall)

    For example:
        pred = "Richard Milhous Nixon"
        gold = "Richard Nixon"
        Shared tokens: {"richard", "nixon"} → precision=2/3, recall=2/2, F1=0.8

    This captures partial credit for answers that are close but not exact.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def accuracy(prediction: str, ground_truth: str) -> float:
    """
    Simple accuracy for classification tasks (FEVER).

    Returns 1.0 if the prediction label matches the ground truth label.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))
