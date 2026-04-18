"""
LLM-as-Judge — evaluates answer correctness by comparing generated answers
against ground-truth references using a structured rubric.

Uses the local Ollama model (same as DocChat's agents) to score answers on a
1–5 scale, then normalises to [0, 1].
"""

import re
from typing import Optional

from docchat.utils.logging import logger


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_CORRECTNESS_PROMPT = """You are an evaluation judge. Compare the GENERATED ANSWER against the REFERENCE ANSWER for the given QUESTION.

Scoring rubric (1-5):
5 = Fully correct — covers all key points from the reference
4 = Mostly correct — minor omissions that don't affect understanding
3 = Partially correct — captures main idea but misses important details
2 = Mostly incorrect — contains some relevant info but misleading overall
1 = Completely incorrect or irrelevant

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

GENERATED ANSWER:
{generated}

Respond with ONLY two lines:
Score: [1-5]
Reason: [one sentence explaining the score]"""


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _parse_score(response_text: str) -> float:
    """Extract an integer score 1-5 and normalise to [0, 1]."""
    match = re.search(r"Score:\s*(\d)", response_text)
    if match:
        raw = int(match.group(1))
        return max(0.0, min((raw - 1) / 4.0, 1.0))  # 1→0.0, 5→1.0
    return 0.0


def _parse_reason(response_text: str) -> str:
    """Extract the reason line from the judge response."""
    match = re.search(r"Reason:\s*(.+)", response_text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _call_ollama(prompt: str) -> str:
    """Call the configured LLM as the evaluation judge."""
    from docchat.utils.llm_factory import get_llm

    llm = get_llm(temperature=0)
    response = llm.invoke(prompt)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_correctness(
    question: str,
    generated_answer: str,
    reference_answer: str,
) -> dict:
    """
    Score a generated answer on a 1–5 correctness rubric.

    Returns
    -------
    dict
        {"score": float [0, 1], "raw_score": int [1-5], "reason": str}
    """
    if not generated_answer or not reference_answer:
        return {"score": 0.0, "raw_score": 1, "reason": "Empty answer or reference"}

    prompt = _CORRECTNESS_PROMPT.format(
        question=question,
        reference=reference_answer,
        generated=generated_answer,
    )

    try:
        response = _call_ollama(prompt)
        norm_score = _parse_score(response)
        reason = _parse_reason(response)

        # Recover raw 1-5 from normalised score
        raw = round(norm_score * 4) + 1

        logger.info(
            f"LLM judge correctness: {raw}/5 (normalised={norm_score:.2f}) — {reason}"
        )
        return {"score": norm_score, "raw_score": raw, "reason": reason}

    except Exception as exc:
        logger.error(f"LLM judge error: {exc}")
        return {"score": 0.0, "raw_score": 1, "reason": f"Judge error: {exc}"}
