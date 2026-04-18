"""
Generation evaluation metrics — faithfulness, relevance, and completeness.

Provides two paths:
1. **Ragas integration** (preferred) — uses the ragas library when available.
2. **Standalone LLM-based fallback** — uses the local Ollama model directly
   when ragas is not installed.

Both paths return a ``GenerationScores`` dataclass.
"""

from typing import List, Optional

from docchat.evaluation.models import GenerationScores
from docchat.utils.logging import logger


# ---------------------------------------------------------------------------
# Ragas-based evaluation (preferred)
# ---------------------------------------------------------------------------

def _try_ragas(
    question: str,
    generated_answer: str,
    ground_truth: Optional[str],
    contexts: List[str],
) -> Optional[GenerationScores]:
    """
    Attempt to evaluate using the ``ragas`` library.

    Returns ``None`` if ragas is not installed or fails, so the caller
    can fall back to the standalone implementation.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        eval_data = {
            "question": [question],
            "answer": [generated_answer],
            "contexts": [contexts],
        }
        if ground_truth:
            eval_data["ground_truth"] = [ground_truth]

        metrics = [faithfulness, answer_relevancy]

        dataset = Dataset.from_dict(eval_data)
        result = ragas_evaluate(dataset=dataset, metrics=metrics)

        scores = result.to_pandas().iloc[0]

        faith_score = float(scores.get("faithfulness", 0.0))
        rel_score = float(scores.get("answer_relevancy", 0.0))

        logger.info(f"Ragas scores — faithfulness={faith_score:.3f}, relevance={rel_score:.3f}")

        return GenerationScores(
            faithfulness=faith_score,
            relevance=rel_score,
            # correctness and completeness are handled separately
            correctness=0.0,
            completeness=0.0,
        )

    except ImportError:
        logger.debug("ragas not installed — falling back to standalone evaluation")
        return None
    except Exception as exc:
        logger.warning(f"Ragas evaluation failed ({exc}) — falling back to standalone")
        return None


# ---------------------------------------------------------------------------
# Standalone LLM-based evaluation (fallback)
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """You are a strict fact-checking judge. Your job is to determine what fraction of claims in the ANSWER are supported by the CONTEXT.

Instructions:
1. List every distinct factual claim in the ANSWER (one per line).
2. For each claim, write SUPPORTED or UNSUPPORTED based on the CONTEXT.
3. At the end, output a single line: Score: <number of supported>/<total claims>

CONTEXT:
{context}

ANSWER:
{answer}

Respond with the list and then the Score line. Nothing else."""

_RELEVANCE_PROMPT = """You are an evaluation judge. Rate how well the ANSWER addresses the QUESTION.

Score from 1 to 5:
5 = Directly and completely addresses the question
4 = Addresses the question with minor tangential content
3 = Partially addresses the question
2 = Loosely related but does not answer
1 = Completely irrelevant

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY:
Score: [1-5]"""

_COMPLETENESS_PROMPT = """You are an evaluation judge. Determine what fraction of the key information points in the REFERENCE ANSWER are also present in the GENERATED ANSWER.

Instructions:
1. List the key information points in the REFERENCE ANSWER.
2. For each point, mark COVERED or MISSING based on the GENERATED ANSWER.
3. Output a single line: Score: <covered>/<total>

REFERENCE ANSWER:
{reference}

GENERATED ANSWER:
{generated}

Respond with the list and then the Score line. Nothing else."""


def _parse_fraction(text: str) -> float:
    """Extract a fraction like '3/5' from text and return as float."""
    import re
    match = re.search(r"Score:\s*(\d+)\s*/\s*(\d+)", text, re.IGNORECASE)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if denominator == 0:
            return 1.0
        return min(numerator / denominator, 1.0)
    return 0.0


def _parse_score_1_5(text: str) -> float:
    """Extract a 1-5 score and normalise to [0, 1]."""
    import re
    match = re.search(r"Score:\s*(\d)", text)
    if match:
        score = int(match.group(1))
        return max(0.0, min((score - 1) / 4.0, 1.0))  # map 1→0, 5→1
    return 0.0


def _call_ollama(prompt: str) -> str:
    """Call the configured LLM for evaluation."""
    from docchat.utils.llm_factory import get_llm

    llm = get_llm(temperature=0)
    response = llm.invoke(prompt)
    return response.content.strip()


def _standalone_faithfulness(answer: str, contexts: List[str]) -> float:
    """Evaluate faithfulness using claim decomposition with local LLM."""
    context = "\n\n".join(contexts[:5])  # Limit context to avoid token overflow
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    try:
        response = _call_ollama(prompt)
        return _parse_fraction(response)
    except Exception as exc:
        logger.error(f"Faithfulness evaluation error: {exc}")
        return 0.0


def _standalone_relevance(question: str, answer: str) -> float:
    """Evaluate answer relevance using 1-5 rubric with local LLM."""
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    try:
        response = _call_ollama(prompt)
        return _parse_score_1_5(response)
    except Exception as exc:
        logger.error(f"Relevance evaluation error: {exc}")
        return 0.0


def _standalone_completeness(
    generated: str,
    reference: str,
) -> float:
    """Evaluate completeness by comparing info points with local LLM."""
    prompt = _COMPLETENESS_PROMPT.format(reference=reference, generated=generated)
    try:
        response = _call_ollama(prompt)
        return _parse_fraction(response)
    except Exception as exc:
        logger.error(f"Completeness evaluation error: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_generation(
    question: str,
    generated_answer: str,
    ground_truth: Optional[str],
    contexts: List[str],
) -> GenerationScores:
    """
    Evaluate generation quality for a single query.

    Tries Ragas first; falls back to standalone LLM-based evaluation.
    Correctness is handled by ``llm_judge.py`` separately.
    """
    if not generated_answer or not generated_answer.strip():
        return GenerationScores()

    # Try Ragas path
    scores = _try_ragas(question, generated_answer, ground_truth, contexts)

    if scores is not None:
        # Ragas doesn't provide completeness — compute separately
        if ground_truth:
            scores.completeness = _standalone_completeness(generated_answer, ground_truth)
        return scores

    # Standalone path
    faith = _standalone_faithfulness(generated_answer, contexts)
    rel = _standalone_relevance(question, generated_answer)
    compl = 0.0
    if ground_truth:
        compl = _standalone_completeness(generated_answer, ground_truth)

    logger.info(
        f"Standalone gen scores — faithfulness={faith:.3f}, "
        f"relevance={rel:.3f}, completeness={compl:.3f}"
    )

    return GenerationScores(
        faithfulness=faith,
        relevance=rel,
        correctness=0.0,  # populated by llm_judge.py
        completeness=compl,
    )
