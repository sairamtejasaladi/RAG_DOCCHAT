"""
Similarity metrics — cosine similarity, exact match, satisfaction proxy.

Uses DocChat's existing ChromaDefaultEmbeddings (all-MiniLM-L6-v2 ONNX) so
no additional embedding dependencies are introduced.
"""

import re
import math
from typing import List


# ---------------------------------------------------------------------------
# Vector similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two vectors (pure-Python, no numpy needed)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_similarity(generated: str, reference: str, embeddings) -> float:
    """
    Embed both texts with the given embeddings model and return cosine sim.

    Parameters
    ----------
    generated : str
        System-generated answer.
    reference : str
        Ground-truth answer.
    embeddings :
        Any object with an ``embed_query(text) -> List[float]`` method.
        In practice this is ``ChromaDefaultEmbeddings`` from the retriever module.
    """
    if not generated or not reference:
        return 0.0
    gen_emb = embeddings.embed_query(generated)
    ref_emb = embeddings.embed_query(reference)
    return cosine_similarity(gen_emb, ref_emb)


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(generated: str, reference: str) -> float:
    """1.0 if normalized generated == normalized reference, else 0.0."""
    if not generated or not reference:
        return 0.0
    return 1.0 if _normalize(generated) == _normalize(reference) else 0.0


# ---------------------------------------------------------------------------
# Satisfaction proxy (composite)
# ---------------------------------------------------------------------------

def satisfaction_proxy(
    correctness: float,
    faithfulness: float,
    completeness: float,
    relevance: float,
    *,
    w_correctness: float = 0.35,
    w_faithfulness: float = 0.30,
    w_completeness: float = 0.20,
    w_relevance: float = 0.15,
) -> float:
    """
    Weighted composite score estimating user satisfaction.

    All inputs should be on [0, 1].  If correctness and relevance are on a
    1–5 rubric, normalise them to [0, 1] before calling.
    """
    return (
        w_correctness * correctness
        + w_faithfulness * faithfulness
        + w_completeness * completeness
        + w_relevance * relevance
    )
