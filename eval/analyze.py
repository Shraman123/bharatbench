"""
BharatBench Results Analyzer
=============================
Takes a results JSON from runner.py and outputs:
  - Per-model aggregate scores
  - Per-language breakdown
  - Per-category breakdown
  - Language gap analysis (English vs Indic)
  - Failure pattern analysis
  - LaTeX table for paper
  - JSON for dashboard

Usage:
    python eval/analyze.py results/eval_20250101_120000.json
    python eval/analyze.py results/eval_20250101_120000.json --latex
    python eval/analyze.py results/eval_20250101_120000.json --dashboard
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional


def load_results(path: str) -> tuple[dict, list]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["metadata"], data["results"]


def aggregate(results: list, group_by: list) -> dict:
    """Group results and compute mean scores."""
    groups = defaultdict(list)
    for r in results:
        key = tuple(r[g] for g in group_by)
        groups[key].append(r["scores"]["overall"])

    return {
        k: {
            "mean":   round(sum(v) / len(v), 4),
            "count":  len(v),
            "min":    round(min(v), 4),
            "max":    round(max(v), 4),
        }
        for k, v in groups.items()
    }


def compute_language_gap(results: list) -> dict:
    """Compute the performance gap between English and Indic languages."""
    by_lang_model = defaultdict(list)
    for r in results:
        key = (r["model"], r["language"])
        by_lang_model[key].append(r["scores"]["overall"])

    gaps = {}
    models = set(r["model"] for r in results)
    for model in models:
        en_scores  = by_lang_model.get((model, "english"),  [])
        bn_scores  = by_lang_model.get((model, "bengali"),  [])
        hi_scores  = by_lang_model.get((model, "hindi"),    [])

        en_mean = sum(en_scores) / len(en_scores) if en_scores else 0
        bn_mean = sum(bn_scores) / len(bn_scores) if bn_scores else 0
        hi_mean = sum(hi_scores) / len(hi_scores) if hi_scores else 0

        gaps[model] = {
            "english":  round(en_mean, 4),
            "bengali":  round(bn_mean, 4),
            "hindi":    round(hi_mean, 4),
            "en_bn_gap": round(en_mean - bn_mean, 4),
            "en_hi_gap": round(en_mean - hi_mean, 4),
            "avg_indic_gap": round(en_mean - (bn_mean + hi_mean) / 2, 4),
        }
    return gaps


def find_failures(results: list, threshold: float = 0.4) -> list:
    """Find questions where ALL models scored below threshold — hard cases."""
    by_question = defaultdict(list)
    for r in results:
        by_question[r["question_id"]].append(r["scores"]["overall"])

    failures = []
    for qid, scores in by_question.items():
        if all(s < threshold for s in scores):
            r = next(x for x in results if x["question_id"] == qid)
            failures.append({
                "question_id": qid,
                "language":    r["language"],
                "category":    r["category"],
                "difficulty":  r["difficulty"],
                "question":    r["question"][:200],
                "max_score":   round(max(scores), 4),
                "avg_score":   round(sum(scores) / len(scores), 4),
            })
    return sorted(failures, key=lambda x: x["avg_score"])


def generate_latex_table(gaps: dict) -> str:
    """Generate a LaTeX table for the paper."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{BharatBench: Model Performance Across Languages}",
        r"\label{tab:results}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{English} & \textbf{Bengali} & \textbf{Hindi} & \textbf{EN-BN Gap} & \textbf{EN-HI Gap} \\",
        r"\hline",
    ]
    for model, g in gaps.items():
        lines.append(
            f"{model} & {g['english']:.3f} & {g['bengali']:.3f} & "
            f"{g['hindi']:.3f} & {g['en_bn_gap']:+.3f} & {g['en_hi_gap']:+.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_dashboard_json(metadata: dict, results: list, gaps: dict) -> dict:
    """Generate JSON for the React dashboard."""
    by_model = aggregate(results, ["model"])
    by_lang  = aggregate(results, ["language"])
    by_cat   = aggregate(results, ["category"])
    by_diff  = aggregate(results, ["difficulty"])
    by_model_lang = aggregate(results, ["model", "language"])

    return {
        "metadata":          metadata,
        "summary": {
            "by_model":      {str(k): v for k, v in by_model.items()},
            "by_language":   {str(k): v for k, v in by_lang.items()},
            "by_category":   {str(k): v for k, v in by_cat.items()},
            "by_difficulty": {str(k): v for k, v in by_diff.items()},
        },
        "by_model_language": {str(k): v for k, v in by_model_lang.items()},
        "language_gaps":     gaps,
        "failures":          find_failures(results),
        "total_evaluations": len(results),
    }


def print_report(metadata: dict, results: list) -> None:
    """Print a human-readable report to stdout."""
    print("\n" + "="*60)
    print("BHARATBENCH EVALUATION REPORT")
    print("="*60)
    print(f"Run ID:     {metadata.get('run_id', 'unknown')}")
    print(f"Models:     {', '.join(metadata.get('models', []))}")
    print(f"Languages:  {', '.join(metadata.get('languages', []))}")
    print(f"Questions:  {metadata.get('total_questions', 0)}")
    print(f"Timestamp:  {metadata.get('timestamp', '')}")
    print()

    gaps = compute_language_gap(results)

    print("── LANGUAGE GAP ANALYSIS ──────────────────────────────")
    print(f"{'Model':<20} {'English':>8} {'Bengali':>8} {'Hindi':>8} {'BN Gap':>8} {'HI Gap':>8}")
    print("-" * 62)
    for model, g in gaps.items():
        print(
            f"{model:<20} {g['english']:>8.3f} {g['bengali']:>8.3f} "
            f"{g['hindi']:>8.3f} {g['en_bn_gap']:>+8.3f} {g['en_hi_gap']:>+8.3f}"
        )

    print()
    by_cat = aggregate(results, ["category", "language"])
    print("── CATEGORY × LANGUAGE ────────────────────────────────")
    langs = ["english", "bengali", "hindi"]
    cats  = ["math", "reasoning", "knowledge", "instruction", "code"]
    print(f"{'Category':<15} {'English':>9} {'Bengali':>9} {'Hindi':>9}")
    print("-" * 44)
    for cat in cats:
        row = f"{cat:<15}"
        for lang in langs:
            key = str((cat, lang))
            val = by_cat.get((cat, lang), {}).get("mean", "-")
            row += f" {val:>9.3f}" if isinstance(val, float) else f" {'N/A':>9}"
        print(row)

    failures = find_failures(results)
    if failures:
        print(f"\n── TOP FAILURE CASES (score < 0.4) ──────────────────")
        for f in failures[:5]:
            print(f"  [{f['language'].upper()}] [{f['category']}] {f['question_id']} — avg {f['avg_score']:.3f}")
            print(f"    Q: {f['question'][:120]}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BharatBench Results Analyzer")
    parser.add_argument("results_file", help="Path to results JSON")
    parser.add_argument("--latex",     action="store_true", help="Print LaTeX table")
    parser.add_argument("--dashboard", action="store_true", help="Save dashboard JSON")
    args = parser.parse_args()

    metadata, results = load_results(args.results_file)
    print_report(metadata, results)

    gaps = compute_language_gap(results)

    if args.latex:
        print("\n── LaTeX TABLE ────────────────────────────────────────")
        print(generate_latex_table(gaps))

    if args.dashboard:
        dashboard = generate_dashboard_json(metadata, results, gaps)
        out_path = Path(args.results_file).parent / "dashboard_data.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, ensure_ascii=False, indent=2)
        print(f"\nDashboard JSON saved to: {out_path}")
