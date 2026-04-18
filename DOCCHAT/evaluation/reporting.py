"""
Evaluation reporting — generates JSON and Markdown reports from EvalReport.

Also handles baseline comparison and regression detection.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from docchat.evaluation.models import EvalReport
from docchat.utils.logging import logger


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

def detect_regressions(
    current: EvalReport,
    baseline: EvalReport,
    threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Compare current evaluation results against a baseline.

    Returns a list of regressions where a metric dropped by more than
    ``threshold`` (absolute).
    """
    regressions: List[Dict[str, Any]] = []

    for layer in ("retrieval", "generation", "e2e"):
        current_scores = getattr(current, layer, {})
        baseline_scores = getattr(baseline, layer, {})

        for metric, cur_val in current_scores.items():
            base_val = baseline_scores.get(metric)
            if base_val is None or base_val == 0:
                continue

            delta = cur_val - base_val
            if delta < -threshold:
                severity = "critical" if delta < -2 * threshold else "warning"
                regressions.append({
                    "metric": f"{layer}.{metric}",
                    "current": round(cur_val, 4),
                    "baseline": round(base_val, 4),
                    "delta": round(delta, 4),
                    "severity": severity,
                })

    return regressions


def detect_failures(
    report: EvalReport,
    retrieval_hit_threshold: float = 0.0,
    faithfulness_threshold: float = 0.70,
    correctness_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """Identify per-query failures that fall below critical thresholds."""
    failures: List[Dict[str, Any]] = []

    for q in report.per_query:
        entry_id = q.get("entry_id", "?")
        reasons = []

        ret = q.get("retrieval", {})
        if ret.get("hit_at_10") is False and ret.get("retrieved_chunk_ids"):
            reasons.append("retrieval: zero hits in top-10")

        gen = q.get("generation", {})
        if gen.get("faithfulness", 1.0) < faithfulness_threshold and gen.get("faithfulness", 0) > 0:
            reasons.append(f"faithfulness={gen['faithfulness']:.2f}")
        if gen.get("correctness", 1.0) < correctness_threshold and gen.get("correctness", 0) > 0:
            reasons.append(f"correctness={gen['correctness']:.2f}")

        if reasons:
            failures.append({"entry_id": entry_id, "reasons": reasons})

    return failures


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------

def load_baseline(path: str) -> Optional[EvalReport]:
    """Load a baseline report from JSON.  Returns None if not found."""
    p = Path(path)
    if not p.exists():
        logger.info(f"No baseline found at {path}")
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EvalReport(
        retrieval=data.get("retrieval", {}),
        generation=data.get("generation", {}),
        e2e=data.get("e2e", {}),
        breakdown=data.get("breakdown", {}),
        metadata=data.get("metadata", {}),
        per_query=data.get("per_query", []),
    )


def save_baseline(report: EvalReport, path: str) -> None:
    """Save the current report as the new baseline."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    logger.info(f"Baseline saved to {path}")


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _fmt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


def _status_icon(delta: float, threshold: float = 0.0) -> str:
    if delta > threshold:
        return "✅"
    elif delta < -threshold:
        return "❌"
    return "➖"


def generate_markdown_report(
    report: EvalReport,
    baseline: Optional[EvalReport] = None,
    regressions: Optional[List[Dict]] = None,
    failures: Optional[List[Dict]] = None,
) -> str:
    """Generate a human-readable Markdown evaluation report."""
    ts = report.metadata.get("timestamp", datetime.now().isoformat())
    mode = report.metadata.get("mode", "unknown")
    total = report.metadata.get("total_queries", 0)
    dataset = report.metadata.get("dataset", "?")

    lines = [
        f"# DocChat Evaluation Report — {ts[:10]}",
        "",
        "## Configuration",
        f"- **Mode:** {mode}",
        f"- **Dataset:** {dataset}",
        f"- **Total queries:** {total}",
    ]

    # Add any extra metadata
    for k, v in report.metadata.items():
        if k not in ("timestamp", "mode", "total_queries", "dataset", "scored_queries"):
            lines.append(f"- **{k}:** {v}")

    lines.append("")

    # --- Retrieval ---
    if report.retrieval:
        lines.append("## Retrieval Metrics")
        lines.append("")
        if baseline:
            lines.append("| Metric | Score | Baseline | Delta | Status |")
            lines.append("|--------|-------|----------|-------|--------|")
            for k, v in report.retrieval.items():
                bv = baseline.retrieval.get(k, 0)
                delta = v - bv
                icon = _status_icon(delta, 0.01)
                lines.append(
                    f"| {k} | {_fmt(v)} | {_fmt(bv)} | {delta:+.4f} | {icon} |"
                )
        else:
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            for k, v in report.retrieval.items():
                lines.append(f"| {k} | {_fmt(v)} |")
        lines.append("")

    # --- Generation ---
    if any(v > 0 for v in report.generation.values()):
        lines.append("## Generation Metrics")
        lines.append("")
        if baseline:
            lines.append("| Metric | Score | Baseline | Delta | Status |")
            lines.append("|--------|-------|----------|-------|--------|")
            for k, v in report.generation.items():
                if v > 0:
                    bv = baseline.generation.get(k, 0)
                    delta = v - bv
                    icon = _status_icon(delta, 0.01)
                    lines.append(f"| {k} | {_fmt(v)} | {_fmt(bv)} | {delta:+.4f} | {icon} |")
        else:
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            for k, v in report.generation.items():
                if v > 0:
                    lines.append(f"| {k} | {_fmt(v)} |")
        lines.append("")

    # --- End-to-End ---
    if report.e2e:
        lines.append("## End-to-End Metrics")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for k, v in report.e2e.items():
            lines.append(f"| {k} | {_fmt(v)} |")
        lines.append("")

    # --- Regressions ---
    if regressions:
        lines.append("## Regressions Detected")
        lines.append("")
        for reg in regressions:
            sev = reg["severity"].upper()
            lines.append(
                f"- **[{sev}]** `{reg['metric']}`: "
                f"{reg['current']} (baseline {reg['baseline']}, "
                f"delta {reg['delta']:+.4f})"
            )
        lines.append("")

    # --- Failures ---
    if failures:
        lines.append("## Per-Query Failures")
        lines.append("")
        for f in failures[:20]:  # cap at 20 for readability
            reasons_str = "; ".join(f["reasons"])
            lines.append(f"- `{f['entry_id']}`: {reasons_str}")
        if len(failures) > 20:
            lines.append(f"- ... and {len(failures) - 20} more")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by DocChat Evaluation Framework at {ts}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full report workflow
# ---------------------------------------------------------------------------

def generate_full_report(
    report: EvalReport,
    output_dir: str,
    baseline_path: Optional[str] = None,
) -> str:
    """
    Generate and save both JSON and Markdown reports.

    Returns the path to the Markdown report.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load baseline if available
    baseline = load_baseline(baseline_path) if baseline_path else None

    # Detect regressions and failures
    regressions = []
    if baseline:
        regressions = detect_regressions(report, baseline)
        report.regressions = regressions

    failures = detect_failures(report)
    report.failures = failures

    # Save JSON
    json_path = out / f"run_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    logger.info(f"JSON report saved to {json_path}")

    # Save Markdown
    md_content = generate_markdown_report(report, baseline, regressions, failures)
    md_path = out / f"summary_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Markdown report saved to {md_path}")

    return str(md_path)
