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

from stats import bootstrap_ci, bootstrap_mean_diff_test, reliability_caveat

# Windows consoles default to cp1252, which can't encode the box-drawing
# characters used below and crashes with UnicodeEncodeError. Force UTF-8 on
# stdout so this runs the same on Windows, macOS, and Linux.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_results(path: str) -> tuple[dict, list]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["metadata"], data["results"]


def usable(results: list) -> list:
    """Results with a valid (non-degraded) judge score."""
    return [r for r in results if not r["scores"].get("judge_parse_failed", False)]


def degraded(results: list) -> list:
    """Results where the judge output couldn't be parsed — excluded from aggregates."""
    return [r for r in results if r["scores"].get("judge_parse_failed", False)]


def aggregate(results: list, group_by: list) -> dict:
    """Group results and compute mean scores + a 95% bootstrap CI around the
    mean. Callers must pass usable(results). CI is None when count == 0;
    a single-point group has ci_low == ci_high == mean (see stats.py)."""
    groups = defaultdict(list)
    for r in results:
        key = tuple(r[g] for g in group_by)
        groups[key].append(r["scores"]["overall"])

    out = {}
    for k, v in groups.items():
        ci = bootstrap_ci(v)
        out[k] = {
            "mean":    round(sum(v) / len(v), 4),
            "count":   len(v),
            "min":     round(min(v), 4),
            "max":     round(max(v), 4),
            "ci_low":  ci["ci_low"],
            "ci_high": ci["ci_high"],
            "reliability_caveat": reliability_caveat(len(v)),
        }
    return out


def pairwise_significance(results: list, group_field: str = "model") -> list:
    """Bootstrap comparison of every pair of `group_field` values (e.g. every
    pair of models) on "overall" score. Pairs by question_id when both sides
    share question IDs -- a paired bootstrap has more statistical power and
    is the right comparison when every model answered the same questions --
    and falls back to an unpaired two-sample bootstrap otherwise (e.g. runs
    with different --limit values per model). See stats.py for the
    significance floor (MIN_RELIABLE_N) that keeps small-sample comparisons
    from being over-stated."""
    scores_by_group = defaultdict(list)
    ids_by_group = defaultdict(list)
    for r in results:
        key = r[group_field]
        scores_by_group[key].append(r["scores"]["overall"])
        ids_by_group[key].append(r["question_id"])

    names = sorted(scores_by_group.keys())
    comparisons = []
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            result = bootstrap_mean_diff_test(
                scores_by_group[a_name], scores_by_group[b_name],
                ids_a=ids_by_group[a_name], ids_b=ids_by_group[b_name],
            )
            n_for_caveat = result.get("n") if result["paired"] else min(result.get("n_a", 0), result.get("n_b", 0))
            comparisons.append({
                group_field + "_a": a_name,
                group_field + "_b": b_name,
                **result,
                "reliability_caveat": reliability_caveat(n_for_caveat),
            })
    return comparisons


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
    good = usable(results)
    bad  = degraded(results)
    by_model = aggregate(good, ["model"])
    by_lang  = aggregate(good, ["language"])
    by_cat   = aggregate(good, ["category"])
    by_diff  = aggregate(good, ["difficulty"])
    by_model_lang = aggregate(good, ["model", "language"])

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
        "pairwise_model_comparisons": pairwise_significance(good, "model"),
        "failures":          find_failures(good),
        "total_evaluations": len(results),
        "usable_evaluations": len(good),
        "degraded_evaluations": len(bad),
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

    bad = degraded(results)
    good = usable(results)
    print(f"Evaluations: {len(results)} total, {len(bad)} degraded (judge parse failed, excluded below)")
    print()

    gaps = compute_language_gap(good)

    by_model = aggregate(good, ["model"])
    print("── MODEL COMPARISON (mean, 95% bootstrap CI, n) ────────")
    print(f"{'Model':<20} {'Mean':>8} {'95% CI':>18} {'n':>5}")
    print("-" * 55)
    for (model,), g in sorted(by_model.items()):
        ci_str = f"[{g['ci_low']:.3f}, {g['ci_high']:.3f}]"
        print(f"{model:<20} {g['mean']:>8.3f} {ci_str:>18} {g['count']:>5}")
        if g["reliability_caveat"]:
            print(f"    ⚠ {g['reliability_caveat']}")

    print()
    comparisons = pairwise_significance(good, "model")
    if comparisons:
        print("── PAIRWISE MODEL COMPARISON (bootstrap, * = p<0.05 and n>=10) ─")
        print(f"{'A':<16} {'B':<16} {'A-B diff':>10} {'95% CI':>18} {'p':>7}")
        print("-" * 72)
        for c in comparisons:
            flag = " *" if c["significant"] else ""
            ci_str = f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
            print(f"{c['model_a']:<16} {c['model_b']:<16} {c['mean_diff']:>+10.3f} {ci_str:>18} {c['p_value']:>7.3f}{flag}")
            if c["reliability_caveat"]:
                print(f"    ⚠ {c['reliability_caveat']}")
        print()

    print("── LANGUAGE GAP ANALYSIS ──────────────────────────────")
    print(f"{'Model':<20} {'English':>8} {'Bengali':>8} {'Hindi':>8} {'BN Gap':>8} {'HI Gap':>8}")
    print("-" * 62)
    for model, g in gaps.items():
        print(
            f"{model:<20} {g['english']:>8.3f} {g['bengali']:>8.3f} "
            f"{g['hindi']:>8.3f} {g['en_bn_gap']:>+8.3f} {g['en_hi_gap']:>+8.3f}"
        )

    print()
    by_cat = aggregate(good, ["category", "language"])
    print("── CATEGORY × LANGUAGE ────────────────────────────────")
    langs = ["english", "bengali", "hindi"]
    cats  = ["math", "reasoning", "knowledge", "instruction", "code"]
    print(f"{'Category':<15} {'English':>9} {'Bengali':>9} {'Hindi':>9}")
    print("-" * 44)
    for cat in cats:
        row = f"{cat:<15}"
        for lang in langs:
            val = by_cat.get((cat, lang), {}).get("mean", "-")
            row += f" {val:>9.3f}" if isinstance(val, float) else f" {'N/A':>9}"
        print(row)

    if bad:
        print("\n── DEGRADED (judge parse failed, excluded from scores) ─")
        by_model_degraded = defaultdict(int)
        for r in bad:
            by_model_degraded[r["model"]] += 1
        for model, count in sorted(by_model_degraded.items()):
            print(f"  {model:<20} {count} record(s)")

    failures = find_failures(good)
    if failures:
        print("\n── TOP FAILURE CASES (score < 0.4) ──────────────────")
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

    gaps = compute_language_gap(usable(results))

    if args.latex:
        print("\n── LaTeX TABLE ────────────────────────────────────────")
        print(generate_latex_table(gaps))

    if args.dashboard:
        dashboard = generate_dashboard_json(metadata, results, gaps)
        out_path = Path(args.results_file).parent / "dashboard_data.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, ensure_ascii=False, indent=2)
        print(f"\nDashboard JSON saved to: {out_path}")
