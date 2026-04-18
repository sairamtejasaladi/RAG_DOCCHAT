"""
Agent-specific evaluation metrics.

Evaluates individual agents in the DocChat pipeline:
1. **RelevanceChecker** — classification accuracy (CAN_ANSWER / PARTIAL / NO_MATCH)
2. **VerificationAgent** — hallucination detection accuracy
3. **Self-Correction Loop** — convergence and improvement tracking
"""

from typing import List, Dict, Tuple
from collections import Counter

from docchat.evaluation.models import EvalEntry, EvalResult
from docchat.utils.logging import logger


# ---------------------------------------------------------------------------
# Relevance Checker Evaluation
# ---------------------------------------------------------------------------

def evaluate_relevance_checker(
    entries: List[EvalEntry],
    results: List[EvalResult],
) -> Dict[str, float]:
    """
    Compare the RelevanceChecker's classification against ground-truth labels.

    Each ``EvalEntry`` has a ``relevance_label`` (annotated by humans) and each
    ``EvalResult`` has a ``relevance_classification`` (produced by the agent).

    Returns a dictionary with accuracy, false-positive rate, and false-negative rate.
    """
    if not entries or not results:
        return {"accuracy": 0.0, "false_positive_rate": 0.0, "false_negative_rate": 0.0}

    # Build lookup: entry_id → ground truth label
    gt_labels = {e.id: e.relevance_label for e in entries}

    correct = 0
    total = 0
    false_positives = 0  # classified as answerable but actually NO_MATCH
    false_negatives = 0  # classified as NO_MATCH but actually answerable
    positive_gt = 0      # ground truth says answerable
    negative_gt = 0      # ground truth says NO_MATCH

    for r in results:
        gt = gt_labels.get(r.entry_id)
        if gt is None or not r.relevance_classification:
            continue

        pred = r.relevance_classification.upper().strip()
        gt_upper = gt.upper().strip()
        total += 1

        # Binary: CAN_ANSWER or PARTIAL → "answerable"; NO_MATCH → "not answerable"
        gt_answerable = gt_upper in ("CAN_ANSWER", "PARTIAL")
        pred_answerable = pred in ("CAN_ANSWER", "PARTIAL")

        if gt_answerable:
            positive_gt += 1
        else:
            negative_gt += 1

        if gt_answerable == pred_answerable:
            correct += 1
        elif pred_answerable and not gt_answerable:
            false_positives += 1
        elif not pred_answerable and gt_answerable:
            false_negatives += 1

    accuracy = correct / total if total > 0 else 0.0
    fpr = false_positives / negative_gt if negative_gt > 0 else 0.0
    fnr = false_negatives / positive_gt if positive_gt > 0 else 0.0

    # Multi-class breakdown
    confusion: Dict[str, Counter] = {}
    for r in results:
        gt = gt_labels.get(r.entry_id)
        if gt and r.relevance_classification:
            confusion.setdefault(gt, Counter())[r.relevance_classification] += 1

    logger.info(
        f"RelevanceChecker eval — accuracy={accuracy:.3f}, "
        f"FPR={fpr:.3f}, FNR={fnr:.3f}, total={total}"
    )

    return {
        "accuracy": accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "total_evaluated": total,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


# ---------------------------------------------------------------------------
# Verification Agent Evaluation
# ---------------------------------------------------------------------------

def evaluate_verification_agent(
    results: List[EvalResult],
    faithfulness_threshold: float = 0.70,
) -> Dict[str, float]:
    """
    Evaluate how well the VerificationAgent's verdicts align with measured
    faithfulness scores.

    Compares:
    - Verification says "Supported: YES" vs measured faithfulness ≥ threshold
    - Verification says "Supported: NO" vs measured faithfulness < threshold

    Requires Phase 2 generation metrics (faithfulness) to be populated.
    """
    correct = 0
    total = 0
    false_rejections = 0  # says NO but answer is actually faithful
    missed_hallucinations = 0  # says YES but answer has low faithfulness

    for r in results:
        if not r.generated_answer or r.generation.faithfulness == 0.0:
            continue  # no generation metrics available

        # Parse the verification report for "Supported: YES/NO"
        verification = r.e2e  # We don't directly have verification text here
        # Instead, we check if the pipeline returned a verification that
        # contains "Supported: YES" or "Supported: NO"
        # For now, we'll use faithfulness as proxy comparison
        total += 1
        is_faithful = r.generation.faithfulness >= faithfulness_threshold

        # The verification agent's decision is reflected in iteration_count:
        # iteration_count > 1 means verification failed at least once
        verification_passed = r.iteration_count <= 1

        if verification_passed == is_faithful:
            correct += 1
        elif verification_passed and not is_faithful:
            missed_hallucinations += 1
        elif not verification_passed and is_faithful:
            false_rejections += 1

    accuracy = correct / total if total > 0 else 0.0
    false_rej_rate = false_rejections / total if total > 0 else 0.0
    miss_rate = missed_hallucinations / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "false_rejection_rate": false_rej_rate,
        "missed_hallucination_rate": miss_rate,
        "total_evaluated": total,
    }


# ---------------------------------------------------------------------------
# Self-Correction Loop Evaluation
# ---------------------------------------------------------------------------

def evaluate_self_correction(
    results: List[EvalResult],
) -> Dict[str, float]:
    """
    Analyze the self-correction loop behavior across all evaluated queries.

    Metrics:
    - Average iterations per query
    - Fraction of queries that triggered re-research
    - Convergence rate (passed verification within max iterations)
    """
    iterations = []
    triggered_loop = 0
    converged = 0

    for r in results:
        if r.error or not r.generated_answer:
            continue
        count = r.iteration_count
        iterations.append(count)
        if count > 1:
            triggered_loop += 1
        # Converged = has a generated answer (did not get stuck)
        if r.generated_answer and r.generated_answer.strip():
            converged += 1

    total = len(iterations)
    avg_iter = sum(iterations) / total if total > 0 else 0.0
    loop_rate = triggered_loop / total if total > 0 else 0.0
    conv_rate = converged / total if total > 0 else 0.0

    return {
        "average_iterations": avg_iter,
        "loop_trigger_rate": loop_rate,
        "convergence_rate": conv_rate,
        "max_iterations_seen": max(iterations) if iterations else 0,
        "total_evaluated": total,
    }
