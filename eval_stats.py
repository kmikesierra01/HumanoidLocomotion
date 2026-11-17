"""Fall rate (Wilson 95% CI), velocity tracking error (mean, 95% CI) and same-episode McNemar from eval JSONs.
Same method as LeRobotTest/stats.py. stdlib only.

Usage: python eval_stats.py <isaac.json> [<mujoco.json>]   (no args: self-check)
"""
import json
import math
import statistics
import sys


def wilson(k, n, z=1.96):
    """95% confidence interval of a rate k/n (Wilson score interval)."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar(only_a, only_b):
    """Exact McNemar p: chance of a split at least this lopsided if both were equally likely to fall."""
    n = only_a + only_b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n)


def mean_ci(xs, z=1.96):
    m = statistics.fmean(xs)
    return m, (z * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan"))


def summary(r):
    """'falls k/n = p% (95% CI) |v_xy err| ... |wz err| ...' for one list of episode results."""
    k, n = sum(e["fell"] for e in r), len(r)
    lo, hi = wilson(k, n)
    errs = [[e[key] for e in r if e[key] is not None] for key in ("err_xy", "err_wz")]
    xy, wz = (mean_ci(x) if x else (float("nan"),) * 2 for x in errs)
    return (f"falls {k}/{n} = {k / n:.1%} (95% CI {lo:.1%} - {hi:.1%})  "
            f"|v_xy err| {xy[0]:.3f} +- {xy[1]:.3f} m/s  |wz err| {wz[0]:.3f} +- {wz[1]:.3f} rad/s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        lo, hi = wilson(160, 200)
        assert abs(lo - 0.739) < 1e-3 and abs(hi - 0.850) < 1e-3
        assert abs(mcnemar(16, 34) - 0.015) < 1e-3 and mcnemar(0, 0) == 1.0
        assert mean_ci([1.0, 3.0]) == (2.0, 1.96 * math.sqrt(2) / math.sqrt(2))
        sys.exit(__doc__)

    runs = [json.load(open(f)) for f in sys.argv[1:3]]
    for f, r in zip(sys.argv[1:], runs):
        print(f"{f}: {summary(r)}")
    if len(runs) == 2:
        a, b = runs
        assert [e["cmd"] for e in a] == [e["cmd"] for e in b], "episodes differ (different commands)"
        only_a = sum(x["fell"] and not y["fell"] for x, y in zip(a, b))
        only_b = sum(y["fell"] and not x["fell"] for x, y in zip(a, b))
        print(f"same episodes: fell only in 1st {only_a}, only in 2nd {only_b} -> McNemar p = {mcnemar(only_a, only_b):.3f}")
