"""Shared wandb-fetch + plotting infrastructure for comparing simulation results
across conditions (obfuscation or ablation) — used by both
simulation_analysis_obfuscation.py and simulation_analysis_ablation.py.

Adapted from dimi_analysis.py's fetch/plot pattern (load_run, get_metric_from_run,
plot_metrics_comparison), decoupled from its `settings_config`/personas-filename
matching (which doesn't apply to run_persona_pipeline.py's synthetic personas
settings) in favor of condition labels/colors passed in explicitly.
"""

import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

plt.rcParams.update({
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'medium',
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.titleweight': 'medium',
    # Axes
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.axisbelow': True,
    # Ticks
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    # Lines & patches
    'lines.linewidth': 1.5,
    'patch.edgecolor': 'white',
    'patch.linewidth': 0.5,
    # Grid (used selectively)
    'grid.color': '#333333',
    'grid.alpha': 0.15,
    'grid.linestyle': '-',
    # Figure
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    # PDF/PS export with editable text
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# Same metric set dimi_analysis.py already established for simulation comparisons.
METRICS = {
    'EI_index': 'EI Index',
    'avg_clustering_coefficient': 'Avg. Clustering Coefficient',
    'correlation_retweets_partisan': 'Correlation Retweets - Partisanship',
}


def fig_path(base_name: str, batch_id: str, ext: str = "pdf") -> Path:
    """Figure output path, tagged with the batch id so figures from different
    comparisons don't overwrite each other — same convention as
    persona_interviews' fig_path."""
    return FIGS_DIR / f"{batch_id}_{base_name}.{ext}"


def load_run(run, retries: int = 3, backoff: float = 5.0):
    """Eagerly load a wandb run's full data with retries."""
    last_exc = None
    for attempt in range(retries):
        try:
            run.load_full_data()
            return run
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Failed to load run '{run.id}' after {retries} attempts") from last_exc


def get_metric_from_run(run, metric: str, retries: int = 3, backoff: float = 5.0):
    """Get a run's final value for `metric`: first from `run.summary['final/{metric}']`
    (logged once at the end of the simulation), falling back to the last logged value
    in its step history if the summary doesn't have it. `scan_history` is used instead
    of `history(keys=...)`, which uses a flakier GraphQL endpoint."""
    val = run.summary.get(f"final/{metric}")
    if val is not None:
        return _to_float(val)
    last_exc = None
    for attempt in range(retries):
        try:
            rows = list(run.scan_history(keys=[metric]))
            vals = [r[metric] for r in rows if metric in r and r[metric] is not None]
            return _to_float(vals[-1]) if vals else None
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Failed to fetch history for run '{run.name}', metric '{metric}' after {retries} attempts") from last_exc


def _to_float(val):
    """wandb's summary/history API encodes NaN/Infinity as the strings 'NaN'/
    'Infinity'/'-Infinity' (its JSON-safe representation) rather than native floats
    — coerce to a real float so a run with an undefined metric (e.g. correlation on
    a zero-variance series) is treated as NaN and excluded by nan-aware aggregation,
    instead of poisoning it with a stray string."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def fetch_and_aggregate(
    runs_by_condition: dict[str, list], metrics: dict[str, str] = METRICS
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[float]]]]:
    """Given condition label -> list of wandb runs (one per seed) sharing that
    condition, eagerly load each run and pull each metric's final value. Returns
    (data, raw_data): `data[label][metric]` is the cross-seed mean and
    `data[label][f'{metric}_se']` its standard error; `raw_data[label][metric]` is
    the list of per-seed raw values (used for the Mann-Whitney significance test in
    `plot_metrics_comparison`)."""
    # Pre-initialize every condition label (not just ones that end up with at least
    # one valid value) — a label whose runs never logged a given metric (e.g. no
    # follow links formed, so EI_index/clustering are never computed) would
    # otherwise be silently missing from `data`/`raw_data`, even though the caller
    # still expects every label in `runs_by_condition` to have an entry.
    raw_data: dict[str, dict[str, list[float]]] = {label: defaultdict(list) for label in runs_by_condition}
    for label, runs in runs_by_condition.items():
        for run in runs:
            load_run(run)
            for metric in metrics:
                val = get_metric_from_run(run, metric)
                # NaN is a legitimate "undefined for this run" value (e.g. correlation
                # on a zero-variance series) — excluded here so it's absent from both
                # the nan-mean/SE below and the Mann-Whitney test in
                # plot_metrics_comparison, which isn't nan-aware.
                if val is not None and not math.isnan(val):
                    raw_data[label][metric].append(val)

    data: dict[str, dict[str, float]] = {}
    for label, vals in raw_data.items():
        data[label] = {}
        for metric in metrics:
            v = vals.get(metric, [])
            if v:
                data[label][metric] = np.nanmean(v)
                data[label][f'{metric}_se'] = np.nanstd(v) / np.sqrt(len(v))
            else:
                data[label][metric] = np.nan
                data[label][f'{metric}_se'] = np.nan
    return data, dict(raw_data)


def plot_metrics_comparison(
    ax_row,
    labels: list[str],
    condition_colors: dict[str, str],
    data: dict[str, dict[str, float]],
    raw_data: dict[str, dict[str, list[float]]] | None = None,
    metrics: dict[str, str] = METRICS,
    alpha: float = 0.05,
) -> None:
    """One bar chart per metric (`ax_row` supplies one Axes per metric, in
    `metrics` order), bars = `labels` (conditions) in the given order. The first
    label is treated as the baseline: if `raw_data` is given, a Mann-Whitney U test
    marks each other condition's bar with '*' where it differs significantly from
    baseline (p < alpha) for that metric."""
    bar_colors = [condition_colors[l] for l in labels]
    x = np.arange(len(labels))
    width = 0.65
    baseline = labels[0]

    sig_map = {}
    if raw_data and baseline in raw_data:
        for metric_idx, metric in enumerate(metrics):
            baseline_vals = raw_data.get(baseline, {}).get(metric, [])
            for bar_idx, label in enumerate(labels[1:], start=1):
                cond_vals = raw_data.get(label, {}).get(metric, [])
                if len(baseline_vals) >= 2 and len(cond_vals) >= 2:
                    _, p = mannwhitneyu(baseline_vals, cond_vals, alternative='two-sided')
                    sig_map[(metric_idx, bar_idx)] = p < alpha

    for idx, (metric, metric_label) in enumerate(metrics.items()):
        ax = ax_row[idx]
        values = [data[l][metric] for l in labels]
        errors = [data[l][f'{metric}_se'] for l in labels]

        ax.bar(x, values, width, color=bar_colors, yerr=errors, capsize=4,
               error_kw={'elinewidth': 1.0, 'capthick': 1.0})

        y_range = max((abs(v) + e for v, e in zip(values, errors) if not np.isnan(v)), default=1.0)
        y_pad = y_range * 0.05
        for bar_idx in range(1, len(labels)):
            if sig_map.get((idx, bar_idx), False):
                bar_val = values[bar_idx]
                bar_err = errors[bar_idx]
                if bar_val >= 0:
                    y_pos = bar_val + bar_err + y_pad
                    va = 'bottom'
                else:
                    y_pos = bar_val - bar_err - y_pad
                    va = 'top'
                ax.text(x[bar_idx], y_pos, '*', ha='center', va=va,
                        fontsize=13, fontweight='bold', color='#333333')

        ax.set_title(metric_label, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.grid(True)
