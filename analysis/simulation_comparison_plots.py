"""Shared wandb-fetch + plotting infrastructure for comparing simulation results
across conditions (obfuscation or ablation) — used by both
simulation_analysis_obfuscation.py and simulation_analysis_ablation.py.

Adapted from dimi_analysis.py's fetch/plot pattern (load_run, get_metric_from_run,
plot_metrics_comparison), decoupled from its `settings_config`/personas-filename
matching (which doesn't apply to run_persona_pipeline.py's synthetic personas
settings) in favor of condition labels/colors passed in explicitly.
"""

import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "persona_interviews"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from interview_comparison_plots import PARTY_COLORS  # noqa: E402
import Platform  # noqa: E402,F401  (must be importable under this name to unpickle a platform artifact)

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

# Shared with persona_simulation_analysis_ablation.py's platform-artifact cache,
# so a run downloaded for one comparison is reused by the other.
PLATFORM_CACHE_DIR = Path(__file__).parent / "persona_interviews" / "wandb_cache"

# Same metric set dimi_analysis.py already established for simulation comparisons.
METRICS = {
    'EI_index': 'EI Index',
    'avg_clustering_coefficient': 'Avg. Clustering Coefficient',
    'correlation_retweets_partisan': 'Correlation Retweets - Partisanship',
}

# Network-structure metrics (see compute_metrics in src/main.py) — one chart.
NETWORK_METRICS = {
    'EI_index_dem_rep': 'EI Index (Dem/Rep)',
    'modularity_dem_rep': 'Modularity (Dem/Rep)',
    'network_reciprocity': 'Network Reciprocity',
    'network_density': 'Network Density',
    'avg_clustering_coefficient': 'Avg. Clustering Coefficient',
}

# Correlation/inequality metrics (see compute_metrics in src/main.py) — a
# separate chart from NETWORK_METRICS.
CORRELATION_METRICS = {
    'correlation_retweets_partisan': 'Correlation Retweets - Partisanship',
    'correlation_followers_partisan': 'Correlation Followers - Partisanship',
    'gini_followers': 'Gini Coefficient (Followers)',
    'gini_reposts': 'Gini Coefficient (Reposts)',
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


def download_platform(run, cache_dir: Path = PLATFORM_CACHE_DIR):
    """Download (or reuse a cached copy of) a run's platform artifact and unpickle
    it — one directory per run.id (same caching pattern as
    interview_wandb.download_results_dataframe)."""
    run_dir = cache_dir / run.id
    pkl_files = list(run_dir.glob("*.pkl"))
    if not pkl_files:
        artifacts = [a for a in run.logged_artifacts() if a.type == "platform"]
        if not artifacts:
            raise RuntimeError(f"Run '{run.id}' has no logged platform artifact.")
        artifacts[0].download(root=str(run_dir))
        pkl_files = list(run_dir.glob("*.pkl"))
    try:
        with open(pkl_files[0], "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        # Truncated cache from an interrupted download — drop it and fetch once more
        # (a second failure means the artifact itself is bad, so let it raise).
        if not pkl_files[0].exists():
            raise
        print(f"Cached platform for run '{run.id}' is truncated; re-downloading.")
        pkl_files[0].unlink()
        return download_platform(run, cache_dir)


def modularity_dem_rep(platform) -> float | None:
    """Modularity of the Democrat/Republican partition on the follow network
    restricted to Democrat/Republican nodes — same computation as compute_metrics
    in src/main.py. None if undefined (a party is empty, or no links between
    Democrat/Republican nodes)."""
    G = nx.DiGraph()
    G.add_nodes_from(u.identifier for u in platform.users)
    G.add_edges_from(platform.user_links)
    democrats = {u.identifier for u in platform.users if u.persona['party'] == 'Democrat'}
    republicans = {u.identifier for u in platform.users if u.persona['party'] == 'Republican'}
    G_dem_rep = G.subgraph(democrats | republicans)
    if not democrats or not republicans or G_dem_rep.number_of_edges() == 0:
        return None
    return nx.community.modularity(G_dem_rep, [democrats, republicans])


# Metrics recomputed from each run's final platform artifact instead of read from
# its logged wandb value. modularity_dem_rep: runs logged before compute_metrics
# was restricted to Democrat/Republican nodes treated non-partisans as a third
# community, so the logged value isn't comparable across old and new runs.
RECOMPUTED_METRICS = {
    'modularity_dem_rep': modularity_dem_rep,
}


def plot_networks(labels: list[str], platforms: dict, output_path: str) -> None:
    """One network diagram per condition, nodes colored by party."""
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, label in enumerate(labels):
        platform = platforms[label]
        G = nx.DiGraph()
        G.add_nodes_from(user.identifier for user in platform.users)
        G.add_edges_from(platform.user_links)
        # Isolates (users who never followed/were followed) carry no structure and
        # just pile up in the middle of the layout.
        G.remove_nodes_from(list(nx.isolates(G)))

        party_by_id = {user.identifier: user.persona.get('party', '') for user in platform.users}
        node_colors = [PARTY_COLORS.get(party_by_id[n], '#999999') for n in G]

        # Lay out on the undirected graph: Kamada-Kawai on the DiGraph uses directed
        # shortest paths, so most pairs are "unreachable" and the layout loses the
        # party communities (an EI of -0.8 network rendered as fully mixed).
        pos = nx.kamada_kawai_layout(G.to_undirected())

        ax = axes[idx]
        nx.draw(G, pos, ax=ax, node_color=node_colors, edgecolors='black',
                node_size=100, width=1.0, linewidths=0.5)
        panel_letter = chr(ord('A') + idx)
        ax.set_title(f"({panel_letter}) {label}", pad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(str(output_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")


def fetch_and_aggregate(
    runs_by_condition: dict[str, list], metrics: dict[str, str] = METRICS
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[float]]]]:
    """Given condition label -> list of wandb runs (one per seed) sharing that
    condition, eagerly load each run and pull each metric's final value (metrics in
    RECOMPUTED_METRICS are recomputed from the run's platform artifact instead). Returns
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
            platform = download_platform(run) if any(m in RECOMPUTED_METRICS for m in metrics) else None
            for metric in metrics:
                if metric in RECOMPUTED_METRICS:
                    val = RECOMPUTED_METRICS[metric](platform)
                else:
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
