"""Metric helpers with explicit NaN handling and paired comparisons."""

from __future__ import annotations

import numpy as np
from scipy.stats import gmean, spearmanr


def finite(values):
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def percentile(values, q):
    values = finite(values)
    return float(np.percentile(values, q)) if len(values) else float("nan")


def median(values):
    return percentile(values, 50)


def mean(values):
    values = finite(values)
    return float(np.mean(values)) if len(values) else float("nan")


def safe_spearman(a, b):
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else 0.0


def geometric_mean(values):
    values = finite(values)
    values = values[values > 0]
    return float(gmean(values)) if len(values) else float("nan")


def reduction_percent(baseline, candidate):
    if not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    return 100.0 * (baseline - candidate) / baseline
