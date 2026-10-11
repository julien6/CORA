"""Seed-level descriptive summaries, IQM bootstrap, and paired comparisons."""
import numpy as np
from scipy.stats import ttest_rel


def iqm(values):
    """Mean over the central 50% with fractional boundary weights."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("IQM requires a finite nonempty vector")
    low, high = len(x) * 0.25, len(x) * 0.75
    weights = np.maximum(0, np.minimum(np.arange(len(x)) + 1, high) - np.maximum(np.arange(len(x)), low))
    return float(x @ weights / weights.sum())


def summarize(values, replicates=1000, confidence=0.95, seed=0):
    x = np.asarray(values, dtype=float)
    if not len(x) or replicates < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid summary settings")
    rng = np.random.default_rng(seed)
    bootstrap = np.array([iqm(rng.choice(x, len(x), replace=True)) for _ in range(replicates)])
    tail = (1 - confidence) / 2
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)), "iqm": iqm(x),
            "iqm_ci": np.quantile(bootstrap, [tail, 1 - tail]).tolist() if len(x) > 1 else None,
            "confidence": confidence, "meets_suggested_seed_count": len(x) >= 10}


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1, running)
    return adjusted


def aggregate(rows, settings):
    groups = {}
    metrics = ("simple_regret", "heldout_mae", "heldout_effect_mae", "harmful_reference", "constraint_violation", "abstention", "selected_net_value", "wall_seconds", "diagnostic_coverage")
    for row in rows:
        groups.setdefault((row["benchmark"], row["method"], row["budget"]), []).append(row)
    summaries = []
    for (benchmark, method, budget), group in sorted(groups.items()):
        stats = {}
        executed = [r for r in group if not r.get("abstention", 0)]
        stats["harm_rate_among_executed_nonnoop"] = {
            "executed": len(executed),
            "harmful": sum(r["harmful_reference"] for r in executed),
            "rate": float(np.mean([r["harmful_reference"] for r in executed])) if executed else None,
            "reference": "expected-value oracle estimate; not realized rollout safety",
        }
        for metric in metrics:
            values = [float(r[metric]) for r in group if r.get(metric) is not None]
            if values:
                stats[metric] = summarize(values, int(settings["bootstrap_samples"]), float(settings["confidence"]))
        summaries.append({"benchmark": benchmark, "method": method, "budget": budget, "metrics": stats})
    paired = []
    for benchmark in sorted({r["benchmark"] for r in rows}):
        for budget in sorted({r["budget"] for r in rows}):
            cora = {r["seed"]: r["simple_regret"] for r in rows if r["benchmark"] == benchmark and r["budget"] == budget and r["method"] == "cora"}
            for method in sorted({r["method"] for r in rows} - {"cora", "oracle"}):
                baseline = {r["seed"]: r["simple_regret"] for r in rows if r["benchmark"] == benchmark and r["budget"] == budget and r["method"] == method}
                seeds = sorted(cora.keys() & baseline.keys())
                if len(seeds) < 3:
                    continue
                a, b = np.array([cora[s] for s in seeds]), np.array([baseline[s] for s in seeds])
                difference = a - b
                if np.std(difference) < 1e-12:
                    # A degenerate sample cannot justify a zero p-value.
                    p = 1.0
                else:
                    p = float(ttest_rel(a, b).pvalue)
                paired.append({"benchmark": benchmark, "budget": budget, "baseline": method, "n": len(seeds),
                               "mean_regret_difference_cora_minus_baseline": float(difference.mean()), "p_value": p})
    if paired:
        for row, p in zip(paired, holm([r["p_value"] for r in paired])):
            row["holm_adjusted_p_value"] = float(p)
    return {"summaries": summaries, "paired_comparisons": paired,
            "limitations": "Bootstrap intervals are descriptive; tiny seed counts cannot substantiate research claims. Holm family contains every reported paired test."}
