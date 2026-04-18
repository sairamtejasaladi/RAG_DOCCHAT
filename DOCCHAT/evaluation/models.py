"""
Data models for the evaluation framework.

Defines structured types for evaluation configuration, dataset entries,
per-query results, and aggregate reports.  All models are plain dataclasses
so they serialize cleanly to/from JSON without extra dependencies.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Controls what the evaluation pipeline runs and where results go."""

    dataset_path: str = "docchat/evaluation/datasets/v1_manual_100.json"
    documents_path: str = "docchat/examples/"
    baseline_path: Optional[str] = "docchat/evaluation/results/baseline.json"
    output_dir: str = "docchat/evaluation/results/"

    # Evaluation mode: "retrieval" (Phase 1) or "full" (Phase 2+)
    mode: str = "retrieval"

    # Metric families to compute
    retrieval_metrics: List[str] = field(default_factory=lambda: [
        "recall@5", "recall@10", "precision@5", "precision@10", "mrr",
        "hit@5", "hit@10",
    ])
    generation_metrics: List[str] = field(default_factory=lambda: [
        "faithfulness", "relevance", "correctness", "completeness",
    ])
    e2e_metrics: List[str] = field(default_factory=lambda: [
        "exact_match", "semantic_similarity", "satisfaction_proxy",
    ])

    # Regression detection
    regression_threshold: float = 0.05

    # LLM-as-judge settings (Phase 2)
    judge_model: str = "llama3"
    judge_temperature: float = 0.0

    # Text-matching threshold for content-based chunk relevance
    chunk_match_threshold: float = 0.30


# ---------------------------------------------------------------------------
# Dataset entry
# ---------------------------------------------------------------------------

@dataclass
class EvalEntry:
    """A single row in the evaluation dataset."""

    id: str
    question: str
    ground_truth_answer: Optional[str]
    relevant_chunk_ids: List[str]
    relevant_chunk_texts: List[str]
    source_documents: List[str]
    difficulty: str          # easy | medium | hard
    question_type: str       # factoid | multi-hop | reasoning | comparison | negative | partial | adversarial
    relevance_label: str     # CAN_ANSWER | PARTIAL | NO_MATCH
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalEntry":
        return cls(
            id=d["id"],
            question=d["question"],
            ground_truth_answer=d.get("ground_truth_answer"),
            relevant_chunk_ids=d.get("relevant_chunk_ids", []),
            relevant_chunk_texts=d.get("relevant_chunk_texts", []),
            source_documents=d.get("source_documents", []),
            difficulty=d.get("difficulty", "medium"),
            question_type=d.get("question_type", "factoid"),
            relevance_label=d.get("relevance_label", "CAN_ANSWER"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Per-query scores
# ---------------------------------------------------------------------------

@dataclass
class RetrievalScores:
    """Retrieval-layer scores for one query."""

    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    hit_at_5: bool = False
    hit_at_10: bool = False
    retrieved_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class GenerationScores:
    """Generation-layer scores for one query."""

    faithfulness: float = 0.0
    relevance: float = 0.0
    correctness: float = 0.0
    completeness: float = 0.0


@dataclass
class E2EScores:
    """End-to-end scores for one query."""

    exact_match: float = 0.0
    semantic_similarity: float = 0.0
    satisfaction_proxy: float = 0.0


@dataclass
class EvalResult:
    """Complete evaluation result for a single query."""

    entry_id: str
    question: str
    generated_answer: str = ""
    ground_truth_answer: Optional[str] = None
    retrieval: RetrievalScores = field(default_factory=RetrievalScores)
    generation: GenerationScores = field(default_factory=GenerationScores)
    e2e: E2EScores = field(default_factory=E2EScores)
    relevance_classification: str = ""
    iteration_count: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Aggregate evaluation report across all queries."""

    retrieval: Dict[str, float] = field(default_factory=dict)
    generation: Dict[str, float] = field(default_factory=dict)
    e2e: Dict[str, float] = field(default_factory=dict)
    breakdown: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    per_query: List[Dict[str, Any]] = field(default_factory=list)
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> List[EvalEntry]:
    """Load an evaluation dataset from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalEntry.from_dict(e) for e in data["entries"]]
