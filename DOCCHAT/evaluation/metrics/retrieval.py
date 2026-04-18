"""
Retrieval evaluation metrics — Recall@K, Precision@K, MRR, Hit@K.

All functions accept ordered lists of chunk identifiers so they are
independent of the underlying retriever implementation.  Two matching
strategies are supported:

1. **ID-based** — fast, exact; requires stable chunk IDs in the dataset.
2. **Content-based** — compares retrieved chunk texts against annotated
   relevant texts using Jaccard word-overlap.  Robust to re-chunking.
"""

from typing import List, Set
from docchat.evaluation.models import RetrievalScores


# ---------------------------------------------------------------------------
# Content-based chunk matching (fallback when IDs are unavailable)
# ---------------------------------------------------------------------------

def _jaccard_word_overlap(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two text strings."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _substring_containment(chunk_text: str, snippet: str) -> bool:
    """Check if the snippet (or most of its words) appear in the chunk text."""
    chunk_lower = chunk_text.lower()
    snippet_lower = snippet.lower().strip()

    # Direct substring match
    if snippet_lower in chunk_lower:
        return True

    # Word-level containment: if ≥80% of snippet words appear in the chunk
    snippet_words = snippet_lower.split()
    if not snippet_words:
        return False
    chunk_words_set = set(chunk_lower.split())
    hits = sum(1 for w in snippet_words if w in chunk_words_set)
    return hits / len(snippet_words) >= 0.80


def match_chunks_by_content(
    retrieved_texts: List[str],
    relevant_texts: List[str],
    threshold: float = 0.30,
) -> List[bool]:
    """
    For each retrieved chunk text, decide if it matches any relevant text.

    Uses a three-tier matching strategy:
    1. Substring containment (handles short snippets)
    2. Word-level containment (≥80% of snippet words found in chunk)
    3. Jaccard word-overlap (fuzzy match for longer excerpts)

    Returns a boolean mask the same length as *retrieved_texts*.
    """
    matches = []
    for ret_text in retrieved_texts:
        matched = any(
            _substring_containment(ret_text, rel_text)
            or _jaccard_word_overlap(ret_text, rel_text) >= threshold
            for rel_text in relevant_texts
        )
        matches.append(matched)
    return matches


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Fraction of relevant docs that appear in the top-K retrieved docs."""
    if not relevant_ids:
        return 1.0  # vacuously true
    top_k: Set[str] = set(retrieved_ids[:k])
    relevant: Set[str] = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Fraction of top-K retrieved docs that are relevant."""
    if k == 0:
        return 0.0
    top_k: Set[str] = set(retrieved_ids[:k])
    relevant: Set[str] = set(relevant_ids)
    return len(top_k & relevant) / k


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Reciprocal rank of the first relevant document in the list."""
    relevant: Set[str] = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> bool:
    """True if at least one relevant doc appears in the top-K."""
    top_k: Set[str] = set(retrieved_ids[:k])
    relevant: Set[str] = set(relevant_ids)
    return bool(top_k & relevant)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> RetrievalScores:
    """Compute all retrieval metrics for a single query."""
    return RetrievalScores(
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_10=recall_at_k(retrieved_ids, relevant_ids, 10),
        precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
        precision_at_10=precision_at_k(retrieved_ids, relevant_ids, 10),
        mrr=mrr(retrieved_ids, relevant_ids),
        hit_at_5=hit_at_k(retrieved_ids, relevant_ids, 5),
        hit_at_10=hit_at_k(retrieved_ids, relevant_ids, 10),
        retrieved_chunk_ids=retrieved_ids,
    )


def evaluate_retrieval_by_content(
    retrieved_texts: List[str],
    relevant_texts: List[str],
    threshold: float = 0.30,
) -> RetrievalScores:
    """
    Compute retrieval metrics using content-based matching.

    Useful when stable chunk IDs are not yet annotated in the dataset.
    Converts the content match into synthetic IDs, then delegates to
    the standard metric functions.
    """
    mask = match_chunks_by_content(retrieved_texts, relevant_texts, threshold)

    # Build synthetic IDs: relevant ones get a "rel_" prefix
    retrieved_ids: List[str] = []
    synthetic_relevant: List[str] = []
    rel_counter = 0
    for i, is_match in enumerate(mask):
        if is_match:
            rid = f"rel_{rel_counter}"
            synthetic_relevant.append(rid)
            rel_counter += 1
        else:
            rid = f"irr_{i}"
        retrieved_ids.append(rid)

    # If the dataset says there should be N relevant texts, but we matched
    # fewer, pad the relevant set so recall denominator is correct.
    total_relevant_expected = len(relevant_texts)
    while len(synthetic_relevant) < total_relevant_expected:
        synthetic_relevant.append(f"missing_{len(synthetic_relevant)}")

    scores = evaluate_retrieval(retrieved_ids, synthetic_relevant)
    scores.retrieved_chunk_ids = retrieved_ids
    return scores
