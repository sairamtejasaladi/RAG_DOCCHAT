"""
DocChat Evaluation CLI — run retrieval and end-to-end evaluation against a
golden dataset.

Usage (from project root):

    # Phase 1 — retrieval metrics + similarity only (no LLM calls)
    python -m docchat.evaluation.run_eval --mode retrieval

    # Phase 2 — full evaluation including LLM-based generation metrics
    python -m docchat.evaluation.run_eval --mode full

    # Custom paths
    python -m docchat.evaluation.run_eval \\
        --dataset docchat/evaluation/datasets/v1_manual_50.json \\
        --documents docchat/examples/ \\
        --output docchat/evaluation/results/
"""

import argparse
import json
import sys
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any

# Ensure project root is on sys.path when invoked with `python -m`
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from docchat.evaluation.models import (
    EvalConfig,
    EvalEntry,
    EvalResult,
    EvalReport,
    RetrievalScores,
    GenerationScores,
    E2EScores,
    load_dataset,
)
from docchat.evaluation.metrics.retrieval import (
    evaluate_retrieval,
    evaluate_retrieval_by_content,
)
from docchat.evaluation.metrics.similarity import (
    semantic_similarity,
    exact_match,
    satisfaction_proxy,
)
from docchat.utils.logging import logger


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _build_retriever(documents_path: str):
    """Process documents and return (retriever, chunks)."""
    from docchat.document_processor.file_handler import DocumentProcessor
    from docchat.retriever.builder import RetrieverBuilder

    processor = DocumentProcessor()
    doc_dir = Path(documents_path)

    # Collect all supported files
    supported = {".pdf", ".docx", ".txt", ".md"}
    files = [str(p) for p in doc_dir.iterdir() if p.suffix.lower() in supported]
    if not files:
        raise FileNotFoundError(f"No supported documents found in {documents_path}")

    logger.info(f"Processing {len(files)} document(s) from {documents_path}")
    chunks = processor.process(files)
    logger.info(f"Produced {len(chunks)} chunks")

    builder = RetrieverBuilder()
    retriever = builder.build_hybrid_retriever(chunks)
    return retriever, chunks


def _chunk_index(chunks) -> Dict[str, str]:
    """Map chunk_id → page_content for quick lookup."""
    idx: Dict[str, str] = {}
    for i, c in enumerate(chunks):
        cid = c.metadata.get("chunk_id", f"unk_{i}")
        idx[cid] = c.page_content
    return idx


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_retrieval_for_entry(
    entry: EvalEntry,
    retriever,
    chunk_idx: Dict[str, str],
    config: EvalConfig,
) -> RetrievalScores:
    """Run retrieval for a single eval entry and score it."""
    if entry.relevance_label == "NO_MATCH" and not entry.relevant_chunk_texts:
        # Negative sample — skip retrieval scoring
        return RetrievalScores()

    docs = retriever.invoke(entry.question)
    retrieved_ids = [
        d.metadata.get("chunk_id", f"unk_{i}")
        for i, d in enumerate(docs)
    ]
    retrieved_texts = [d.page_content for d in docs]

    # Prefer ID-based matching when chunk IDs are annotated
    if entry.relevant_chunk_ids:
        scores = evaluate_retrieval(retrieved_ids, entry.relevant_chunk_ids)
        scores.retrieved_chunk_ids = retrieved_ids
        return scores

    # Fall back to content-based matching
    if entry.relevant_chunk_texts:
        return evaluate_retrieval_by_content(
            retrieved_texts,
            entry.relevant_chunk_texts,
            threshold=config.chunk_match_threshold,
        )

    return RetrievalScores(retrieved_chunk_ids=retrieved_ids)


def evaluate_similarity_for_entry(
    entry: EvalEntry,
    generated_answer: str,
    embeddings,
) -> E2EScores:
    """Compute similarity-based end-to-end metrics."""
    if not entry.ground_truth_answer:
        return E2EScores()

    em = exact_match(generated_answer, entry.ground_truth_answer)
    sim = semantic_similarity(generated_answer, entry.ground_truth_answer, embeddings)

    return E2EScores(
        exact_match=em,
        semantic_similarity=sim,
        satisfaction_proxy=sim,  # placeholder until generation metrics exist
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _safe_mean(values: list) -> float:
    return mean(values) if values else 0.0


def aggregate(
    results: List[EvalResult],
    config: EvalConfig,
    extra_metadata: Dict[str, Any] | None = None,
) -> EvalReport:
    """Roll up per-query results into an EvalReport."""

    # Filter out entries with no retrieval scoring (negatives)
    scored = [r for r in results if r.retrieval.retrieved_chunk_ids or r.error is None]

    retrieval_agg = {
        "recall@5": _safe_mean([r.retrieval.recall_at_5 for r in scored]),
        "recall@10": _safe_mean([r.retrieval.recall_at_10 for r in scored]),
        "precision@5": _safe_mean([r.retrieval.precision_at_5 for r in scored]),
        "precision@10": _safe_mean([r.retrieval.precision_at_10 for r in scored]),
        "mrr": _safe_mean([r.retrieval.mrr for r in scored]),
        "hit@5": _safe_mean([float(r.retrieval.hit_at_5) for r in scored]),
        "hit@10": _safe_mean([float(r.retrieval.hit_at_10) for r in scored]),
    }

    gen_agg = {
        "faithfulness": _safe_mean([r.generation.faithfulness for r in results if r.generation.faithfulness > 0]),
        "relevance": _safe_mean([r.generation.relevance for r in results if r.generation.relevance > 0]),
        "correctness": _safe_mean([r.generation.correctness for r in results if r.generation.correctness > 0]),
        "completeness": _safe_mean([r.generation.completeness for r in results if r.generation.completeness > 0]),
    }

    e2e_agg = {
        "exact_match": _safe_mean([r.e2e.exact_match for r in results if r.ground_truth_answer]),
        "semantic_similarity": _safe_mean([r.e2e.semantic_similarity for r in results if r.ground_truth_answer]),
        "satisfaction_proxy": _safe_mean([r.e2e.satisfaction_proxy for r in results if r.ground_truth_answer]),
    }

    # Breakdown by question type
    by_type: Dict[str, list] = {}
    for r in results:
        # question_type is not stored on EvalResult; pull from the original entry
        pass  # populated in the full pipeline when we carry metadata

    metadata = {
        "mode": config.mode,
        "dataset": config.dataset_path,
        "total_queries": len(results),
        "scored_queries": len(scored),
        "timestamp": datetime.now().isoformat(),
        **(extra_metadata or {}),
    }

    per_query = [r.to_dict() for r in results]

    return EvalReport(
        retrieval=retrieval_agg,
        generation=gen_agg,
        e2e=e2e_agg,
        breakdown={},
        metadata=metadata,
        per_query=per_query,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config: EvalConfig) -> EvalReport:
    """Execute the evaluation pipeline end-to-end."""
    logger.info(f"=== DocChat Evaluation — mode={config.mode} ===")

    # 1. Load dataset
    dataset = load_dataset(config.dataset_path)
    logger.info(f"Loaded {len(dataset)} evaluation entries from {config.dataset_path}")

    # 2. Build retriever
    retriever, chunks = _build_retriever(config.documents_path)
    chunk_idx = _chunk_index(chunks)

    # 3. Set up embeddings for similarity (reuse the retriever's embeddings)
    from docchat.retriever.builder import ChromaDefaultEmbeddings
    embeddings = ChromaDefaultEmbeddings()

    results: List[EvalResult] = []

    for i, entry in enumerate(dataset):
        logger.info(f"[{i+1}/{len(dataset)}] Evaluating: {entry.id}")
        t0 = time.time()

        try:
            # --- Retrieval evaluation ---
            ret_scores = evaluate_retrieval_for_entry(
                entry, retriever, chunk_idx, config,
            )

            # --- End-to-end similarity (Phase 1: no pipeline run, use ground truth as placeholder) ---
            # In retrieval-only mode we skip the full pipeline
            generated_answer = ""
            gen_scores = GenerationScores()
            rel_class = ""
            iter_count = 0

            if config.mode == "full":
                # Phase 2 path — run the full pipeline
                from docchat.agents.workflow import AgentWorkflow
                workflow = AgentWorkflow()
                pipeline_out = workflow.full_pipeline(
                    question=entry.question,
                    retriever=retriever,
                )
                generated_answer = pipeline_out["draft_answer"]
                rel_class = pipeline_out.get("relevance_classification", "")
                iter_count = pipeline_out.get("iteration_count", 0)

                # Generation metrics
                from docchat.evaluation.metrics.generation import evaluate_generation
                contexts = [d.page_content for d in pipeline_out.get("documents", [])]
                gen_scores = evaluate_generation(
                    question=entry.question,
                    generated_answer=generated_answer,
                    ground_truth=entry.ground_truth_answer,
                    contexts=contexts,
                )

                # LLM-as-judge correctness
                if entry.ground_truth_answer:
                    from docchat.evaluation.metrics.llm_judge import judge_correctness
                    correctness_result = judge_correctness(
                        question=entry.question,
                        generated_answer=generated_answer,
                        reference_answer=entry.ground_truth_answer,
                    )
                    gen_scores.correctness = correctness_result["score"]

            e2e_scores = evaluate_similarity_for_entry(
                entry,
                generated_answer if generated_answer else (entry.ground_truth_answer or ""),
                embeddings,
            )

            elapsed = (time.time() - t0) * 1000

            result = EvalResult(
                entry_id=entry.id,
                question=entry.question,
                generated_answer=generated_answer,
                ground_truth_answer=entry.ground_truth_answer,
                retrieval=ret_scores,
                generation=gen_scores,
                e2e=e2e_scores,
                relevance_classification=rel_class if config.mode == "full" else "",
                iteration_count=iter_count if config.mode == "full" else 0,
                latency_ms=elapsed,
            )
            results.append(result)

        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            logger.error(f"Error evaluating {entry.id}: {exc}")
            results.append(EvalResult(
                entry_id=entry.id,
                question=entry.question,
                error=str(exc),
                latency_ms=elapsed,
            ))

    # 4. Aggregate
    report = aggregate(results, config, extra_metadata={
        "num_chunks": len(chunks),
        "num_documents": len(set(c.metadata.get("source", "") for c in chunks)),
    })

    # 5. Agent-specific evaluation (Phase 2)
    if config.mode == "full":
        from docchat.evaluation.metrics.agent_eval import (
            evaluate_relevance_checker,
            evaluate_self_correction,
        )
        rel_checker_eval = evaluate_relevance_checker(dataset, results)
        loop_eval = evaluate_self_correction(results)
        report.breakdown["relevance_checker"] = rel_checker_eval
        report.breakdown["self_correction_loop"] = loop_eval

    # 6. Generate reports (JSON + Markdown)
    from docchat.evaluation.reporting import generate_full_report
    md_path = generate_full_report(
        report,
        output_dir=config.output_dir,
        baseline_path=config.baseline_path,
    )
    logger.info(f"Markdown report: {md_path}")

    # Print summary
    _print_summary(report)

    return report


def _print_summary(report: EvalReport) -> None:
    """Pretty-print a compact summary to stdout."""
    print("\n" + "=" * 60)
    print("  DOCCHAT EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Mode:    {report.metadata.get('mode', '?')}")
    print(f"  Queries: {report.metadata.get('total_queries', 0)}")
    print(f"  Time:    {report.metadata.get('timestamp', '')}")
    print("-" * 60)

    if report.retrieval:
        print("\n  RETRIEVAL METRICS")
        for k, v in report.retrieval.items():
            print(f"    {k:>15s}: {v:.4f}")

    if any(v > 0 for v in report.generation.values()):
        print("\n  GENERATION METRICS")
        for k, v in report.generation.items():
            print(f"    {k:>15s}: {v:.4f}")

    if report.e2e:
        print("\n  END-TO-END METRICS")
        for k, v in report.e2e.items():
            print(f"    {k:>15s}: {v:.4f}")

    if report.failures:
        print(f"\n  FAILURES: {len(report.failures)}")
        for f in report.failures[:5]:
            print(f"    {f['entry_id']}: {f.get('error', 'unknown')}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DocChat RAG Evaluation Pipeline",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "full"],
        default="retrieval",
        help="retrieval = Phase 1 (no LLM calls); full = Phase 2+ (runs pipeline)",
    )
    parser.add_argument(
        "--dataset",
        default="docchat/evaluation/datasets/v1_manual_100.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--documents",
        default="docchat/examples/",
        help="Path to source documents directory",
    )
    parser.add_argument(
        "--output",
        default="docchat/evaluation/results/",
        help="Directory for evaluation output",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Jaccard overlap threshold for content-based chunk matching",
    )
    args = parser.parse_args()

    config = EvalConfig(
        dataset_path=args.dataset,
        documents_path=args.documents,
        output_dir=args.output,
        mode=args.mode,
        chunk_match_threshold=args.threshold,
    )

    run(config)


if __name__ == "__main__":
    main()
