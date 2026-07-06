"""
Bootstrap statistics for BharatBench results: confidence intervals around
mean scores, and pairwise significance testing between models.

Pure stdlib (random) rather than scipy/numpy -- these samples are small
(the whole dataset is 67 questions; some category/language cells have as
few as 2), so a distribution-free bootstrap is a more honest choice than a
normal-approximation t-test would be, and it avoids a heavy new dependency
for what is a genuinely small amount of arithmetic.

Bootstrap resampling uses a fixed default seed (BOOTSTRAP_SEED) so that
running analyze.py twice on the same results file produces the same CI/
p-values -- reports should be reproducible, not jitter run to run.
"""

import random

BOOTSTRAP_SEED = 42
DEFAULT_N_RESAMPLES = 2000

# Below this many samples, bootstrap CIs/p-values are treated as too noisy
# to call anything "significant" -- this dataset has category/language cells
# as small as n=2, and a bootstrap CI on 2 points is close to meaningless.
MIN_RELIABLE_N = 10


def reliability_caveat(n: int) -> str:
    """None if n is large enough to trust the CI/p-value at face value,
    otherwise a caveat string explaining why not."""
    if n < MIN_RELIABLE_N:
        return (
            f"n={n} is below the {MIN_RELIABLE_N}-sample floor this project treats as "
            f"minimally reliable for bootstrap CIs/significance testing -- treat this as "
            f"a rough signal, not a confident conclusion."
        )
    return None


def _bootstrap_means(data: list, n_resamples: int, rng: random.Random) -> list:
    n = len(data)
    means = []
    for _ in range(n_resamples):
        resample = [data[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    return means


def _percentile_ci(sorted_values: list, confidence: float) -> tuple:
    n = len(sorted_values)
    alpha = 1 - confidence
    lo_idx = int((alpha / 2) * n)
    hi_idx = min(int((1 - alpha / 2) * n), n - 1)
    return sorted_values[lo_idx], sorted_values[hi_idx]


def bootstrap_ci(
    scores: list,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Percentile bootstrap confidence interval around the mean of `scores`."""
    scores = list(scores)
    n = len(scores)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    if n == 1:
        return {"mean": round(scores[0], 4), "ci_low": round(scores[0], 4),
                 "ci_high": round(scores[0], 4), "n": 1}

    rng = random.Random(seed)
    means = _bootstrap_means(scores, n_resamples, rng)
    ci_low, ci_high = _percentile_ci(means, confidence)
    return {
        "mean": round(sum(scores) / n, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "n": n,
    }


def _two_sided_p_value(boot_diffs: list) -> float:
    n = len(boot_diffs)
    p = 2 * min(
        sum(1 for d in boot_diffs if d <= 0) / n,
        sum(1 for d in boot_diffs if d >= 0) / n,
    )
    return min(p, 1.0)


def _empty_diff_result() -> dict:
    return {
        "mean_diff": None, "ci_low": None, "ci_high": None,
        "p_value": None, "significant": False, "paired": False, "n": 0,
    }


def bootstrap_mean_diff_test(
    a: list,
    b: list,
    ids_a: list = None,
    ids_b: list = None,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap test for whether mean(a) - mean(b) differs from 0.

    If ids_a/ids_b are given and share overlapping IDs (e.g. question_id),
    pairs by ID and bootstraps the paired differences -- more statistical
    power, appropriate when both sides answered the same questions. Falls
    back to an unpaired two-sample bootstrap otherwise.

    "significant" requires both p < 0.05 AND the sample size to be at or
    above MIN_RELIABLE_N -- a low p-value on a handful of points is not
    something this project will label significant.
    """
    rng = random.Random(seed)

    paired = False
    diffs = None
    if ids_a is not None and ids_b is not None:
        map_a = dict(zip(ids_a, a))
        map_b = dict(zip(ids_b, b))
        shared = sorted(set(map_a) & set(map_b))
        if shared:
            paired = True
            diffs = [map_a[i] - map_b[i] for i in shared]

    if paired:
        n = len(diffs)
        if n == 0:
            return _empty_diff_result()
        observed_diff = sum(diffs) / n
        boot_diffs = _bootstrap_means(diffs, n_resamples, rng)
        ci_low, ci_high = _percentile_ci(boot_diffs, confidence)
        p_value = _two_sided_p_value(boot_diffs)
        return {
            "mean_diff": round(observed_diff, 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05 and n >= MIN_RELIABLE_N,
            "paired": True,
            "n": n,
        }

    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return _empty_diff_result()
    observed_diff = (sum(a) / n_a) - (sum(b) / n_b)
    means_a = _bootstrap_means(a, n_resamples, rng)
    means_b = _bootstrap_means(b, n_resamples, rng)
    boot_diffs = sorted(ma - mb for ma, mb in zip(means_a, means_b))
    ci_low, ci_high = _percentile_ci(boot_diffs, confidence)
    p_value = _two_sided_p_value(boot_diffs)
    return {
        "mean_diff": round(observed_diff, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05 and min(n_a, n_b) >= MIN_RELIABLE_N,
        "paired": False,
        "n_a": n_a,
        "n_b": n_b,
    }
