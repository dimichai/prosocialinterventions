"""Simulation-side ablation comparison, wandb-batch analogue of
persona_interviews_analysis_ablation.py — but for network metrics (EI index,
clustering, etc.), network diagrams, and cross-party follows, instead of
interview yes/no + feeling-thermometer answers.

Context: run_persona_pipeline.py runs the interview and simulation stages of
one seed inside a single shared wandb run (own_wandb_run=False for both), so
this reuses the exact same runs persona_interviews_analysis_ablation.py
fetches for a --batch_id — just reading different data off them: network
metrics from run.summary["final/<metric>"] (logged by src/main.py's
compute_metrics) and the full network from each run's "platform"-type
wandb.Artifact (a pickled src.Platform.Platform instance), rather than the
"interview_results" artifact.

Usage:
    python analysis/persona_simulation_analysis_ablation.py --batch_id <batch_id>
"""

import argparse
import os
import pickle
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "persona_interviews"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import interview_wandb  # noqa: E402  (fetch_runs_by_group is generic, not interview-specific)
from persona_interviews_analysis_ablation import ablation_label  # noqa: E402
from interview_comparison_plots import party_color_map  # noqa: E402  (reused for cross-party-follow lines)
import Platform  # noqa: E402,F401  (must be importable under this name to unpickle the platform artifact)

FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)
PLATFORM_CACHE_DIR = Path(__file__).parent / "persona_interviews" / "wandb_cache"

METRICS = {
    'EI_index': 'EI Index',
    'avg_clustering_coefficient': 'Avg. Clustering Coefficient',
    'correlation_retweets_partisan': 'Correlation Retweets - Partisanship',
}

# Fixed colors for the AP/PID/VB ablation chain (see persona_interviews_analysis_ablation.
# CUSTOM_ABLATION_LABELS, whose labels these keys match) — same palette
# dimi_analysis.py originally used for this progression; other labels
# (e.g. a batch outside this chain) fall back to CONDITION_FALLBACK_PALETTE.
CUSTOM_ABLATION_COLORS = {
    "Full Persona": "#4878A8",
    "No AP": "#2D7D2D",
    "No AP & PID": "#5AAD5A",
    "No AP & PID & VB": "#8DD38D",
}
CONDITION_FALLBACK_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7d5ba6"]


def condition_color_map(labels: list[str]) -> dict[str, str]:
    """Condition label -> bar color, matching CUSTOM_ABLATION_COLORS for the
    recognized AP/PID/VB chain; any other label falls back to a fixed extra
    palette (mirrors interview_comparison_plots.party_color_map's pattern)."""
    colors, i = {}, 0
    for label in labels:
        if label in CUSTOM_ABLATION_COLORS:
            colors[label] = CUSTOM_ABLATION_COLORS[label]
        else:
            colors[label] = CONDITION_FALLBACK_PALETTE[i % len(CONDITION_FALLBACK_PALETTE)]
            i += 1
    return colors


plt.rcParams.update({
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
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.axisbelow': True,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 1.5,
    'patch.edgecolor': 'white',
    'patch.linewidth': 0.5,
    'grid.color': '#333333',
    'grid.alpha': 0.15,
    'grid.linestyle': '-',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def fig_path(base_name: str, batch_id: str) -> str:
    """Figure output path, tagged with the batch id (same convention as
    interview_comparison_plots.fig_path) so figures from different
    comparisons don't overwrite each other."""
    return str(FIGS_DIR / f"{batch_id}_{base_name}.pdf")


def fetch_condition_runs(
    batch_id: str, wandb_project: str = 'persona-simulation'
) -> dict[str, list]:
    """Fetch every wandb run sharing `batch_id` and group them by ablation
    combo label (baseline first, then alphabetically — same ordering
    persona_interviews_analysis_ablation.fetch_condition_dfs uses)."""
    runs = interview_wandb.fetch_runs_by_group(wandb_project, batch_id)
    if not runs:
        raise RuntimeError(f"No wandb runs found in project '{wandb_project}' for group '{batch_id}'.")

    runs_by_ablations: dict[tuple[str, ...], list] = defaultdict(list)
    for run in runs:
        runs_by_ablations[tuple(sorted(run.config["ablations"]))].append(run)

    ordered_combos = sorted(runs_by_ablations, key=lambda c: (c != (), c))
    return {ablation_label(combo): runs_by_ablations[combo] for combo in ordered_combos}


def get_metric_from_run(run, metric: str, retries: int = 3, backoff: float = 5.0):
    """Get the last logged value of a metric from a run's summary or history."""
    val = run.summary.get(f"final/{metric}")
    if val is not None:
        return val
    # scan_history is more reliable than history(keys=...) which uses a flaky GraphQL endpoint
    last_exc = None
    for attempt in range(retries):
        try:
            rows = list(run.scan_history(keys=[metric]))
            vals = [r[metric] for r in rows if metric in r and r[metric] is not None]
            return vals[-1] if vals else None
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Failed to fetch history for run '{run.name}', metric '{metric}' after {retries} attempts") from last_exc


def download_platform(run, cache_dir: Path = PLATFORM_CACHE_DIR):
    """Download (or reuse a cached copy of) a run's platform artifact and
    unpickle it — mirrors interview_wandb.download_results_dataframe's
    caching pattern, one directory per run.id."""
    run_dir = cache_dir / run.id
    pkl_files = list(run_dir.glob("*.pkl"))
    if not pkl_files:
        artifacts = [a for a in run.logged_artifacts() if a.type == "platform"]
        if not artifacts:
            raise RuntimeError(f"Run '{run.id}' has no logged platform artifact.")
        artifacts[0].download(root=str(run_dir))
        pkl_files = list(run_dir.glob("*.pkl"))
    with open(pkl_files[0], "rb") as f:
        return pickle.load(f)


def plot_metrics_comparison(labels: list[str], data: dict, raw_data: dict, output_path: str, alpha: float = 0.05) -> None:
    """Bar chart, one panel per metric, one bar per ablation condition
    (baseline first), with Mann-Whitney significance stars vs baseline."""
    colors = condition_color_map(labels)
    x = np.arange(len(labels))
    width = 0.65
    baseline = labels[0]

    sig_map = {}  # (metric_idx, bar_idx) -> significant?
    for metric_idx, metric in enumerate(METRICS):
        baseline_vals = raw_data.get(baseline, {}).get(metric, [])
        for bar_idx, label in enumerate(labels[1:], start=1):
            vals = raw_data.get(label, {}).get(metric, [])
            if len(baseline_vals) >= 2 and len(vals) >= 2:
                _, p = mannwhitneyu(baseline_vals, vals, alternative='two-sided')
                sig_map[(metric_idx, bar_idx)] = p < alpha

    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.6 * len(METRICS), 4.5))
    if len(METRICS) == 1:
        axes = [axes]

    for idx, (metric, title) in enumerate(METRICS.items()):
        ax = axes[idx]
        values = [data[l][metric] for l in labels]
        errors = [data[l][f'{metric}_se'] for l in labels]
        bar_colors = [colors[l] for l in labels]

        ax.bar(x, values, width, color=bar_colors, yerr=errors, capsize=4,
               error_kw={'elinewidth': 1.0, 'capthick': 1.0})

        y_range = max((abs(v) + e for v, e in zip(values, errors) if not np.isnan(v)), default=1.0)
        y_pad = y_range * 0.05
        for bar_idx in range(1, len(labels)):
            if sig_map.get((idx, bar_idx), False):
                bar_val, bar_err = values[bar_idx], errors[bar_idx]
                y_pos = (bar_val + bar_err + y_pad) if bar_val >= 0 else (bar_val - bar_err - y_pad)
                va = 'bottom' if bar_val >= 0 else 'top'
                ax.text(x[bar_idx], y_pos, '*', ha='center', va=va,
                        fontsize=13, fontweight='bold', color='#333333')

        ax.set_title(title, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.grid(True)

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")


def plot_networks(labels: list[str], platforms: dict, output_path: str) -> None:
    """One network diagram per ablation condition, nodes colored by party."""
    party_colors = {'Democrat': '#03357D', 'Republican': '#D50403', 'Non-partisan': '#888888'}
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, label in enumerate(labels):
        platform = platforms[label]
        G = nx.DiGraph()
        G.add_nodes_from(user.identifier for user in platform.users)
        G.add_edges_from(platform.user_links)

        node_colors = [
            party_colors.get(user.persona.get('party', ''), '#999999')
            for user in sorted(platform.users, key=lambda u: u.identifier)
        ]

        ax = axes[idx]
        nx.draw_kamada_kawai(G, ax=ax, node_color=node_colors, edgecolors='black',
                              node_size=100, width=1.0, linewidths=0.5)
        panel_letter = chr(ord('A') + idx)
        ax.set_title(f"({panel_letter}) {label}", pad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")


def plot_cross_party_follows(labels: list[str], platforms: dict, output_path: str) -> None:
    """Fraction of follows going to each party, one panel per follower party,
    one line per followed party across ablation conditions — same slope-chart
    style as interview_comparison_plots.plot_slope_comparison (party-colored
    lines, error bars, party label at the line's end), just computed directly
    from each condition's platform.user_links rather than aggregated interview
    answers."""
    parties = ['Democrat', 'Republican', 'Non-partisan']
    party_colors = party_color_map(parties)

    rows = []
    for label in labels:
        platform = platforms[label]
        party_by_id = {user.identifier: user.persona.get('party', 'Unknown') for user in platform.users}
        for from_id, to_id in platform.user_links:
            rows.append({
                'label': label,
                'follower_party': party_by_id.get(from_id, 'Unknown'),
                'followed_party': party_by_id.get(to_id, 'Unknown'),
            })
    df = pd.DataFrame(rows)

    x_ticks = list(range(len(labels)))
    fig, axes = plt.subplots(1, len(parties), figsize=(3.3 * len(parties), 4), sharey=True)

    for panel_idx, follower_party in enumerate(parties):
        ax = axes[panel_idx]
        for followed_party in parties:
            vals, errs = [], []
            for label in labels:
                subset = df[(df['label'] == label) & (df['follower_party'] == follower_party)]
                n = len(subset)
                if n == 0:
                    vals.append(float('nan'))
                    errs.append(float('nan'))
                    continue
                p = (subset['followed_party'] == followed_party).mean()
                vals.append(p)
                errs.append(1.96 * (p * (1 - p) / n) ** 0.5)

            color = party_colors.get(followed_party, '#888888')
            ax.errorbar(x_ticks, vals, yerr=errs, marker='o', color=color,
                        linewidth=1.5, markersize=4, solid_capstyle='round',
                        clip_on=False, capsize=2, capthick=0.8, elinewidth=0.8)
            if not pd.isna(vals[-1]):
                ax.text(len(labels) - 1 + 0.12, vals[-1], followed_party,
                        ha='left', va='center', color=color)

        ax.set_title(f"{follower_party} follows", pad=8)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_xlim(-0.4, len(labels) - 1 + 1.6)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.grid(True)
        if panel_idx == 0:
            ax.set_ylabel('Fraction of follows')

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a batch of ablation-comparison persona-simulation runs from wandb "
                     "and plot network-metric, network-diagram, and cross-party-follow results."
    )
    parser.add_argument("--batch_id", type=str, required=True,
                         help="Wandb group/batch id shared by the runs to compare "
                              "(printed by run_persona_pipeline.py after it finishes).")
    parser.add_argument("--wandb_project", type=str, default='persona-simulation',
                         help="Wandb project the runs were logged to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_by_condition = fetch_condition_runs(args.batch_id, wandb_project=args.wandb_project)

    strategies = sorted(set(
        run.config.get("timeline_select_strategy", "unknown")
        for runs in runs_by_condition.values() for run in runs
    ))

    for strategy in strategies:
        print(f"\n{'='*60}\n  Strategy: {strategy}\n{'='*60}")

        strategy_runs_by_condition = {
            label: [r for r in runs if r.config.get("timeline_select_strategy", "unknown") == strategy]
            for label, runs in runs_by_condition.items()
        }
        strategy_runs_by_condition = {l: r for l, r in strategy_runs_by_condition.items() if r}
        labels = list(strategy_runs_by_condition.keys())
        if not labels:
            continue

        # --- Network metrics (bar chart, one panel per metric) ---
        raw_per_condition = defaultdict(lambda: defaultdict(list))
        for label, runs in strategy_runs_by_condition.items():
            for run in runs:
                for metric in METRICS:
                    val = get_metric_from_run(run, metric)
                    if val is not None:
                        raw_per_condition[label][metric].append(val)

        metrics_data = {}
        for label, vals in raw_per_condition.items():
            metrics_data[label] = {}
            for metric in METRICS:
                v = vals.get(metric, [])
                if v:
                    metrics_data[label][metric] = np.nanmean(v)
                    metrics_data[label][f'{metric}_se'] = np.nanstd(v) / np.sqrt(len(v))
                else:
                    metrics_data[label][metric] = np.nan
                    metrics_data[label][f'{metric}_se'] = np.nan

        metric_labels = [l for l in labels if l in metrics_data]
        if metric_labels:
            plot_metrics_comparison(metric_labels, metrics_data, raw_per_condition,
                                     fig_path(f"{strategy}_simulation_metrics_ablation", args.batch_id))
        else:
            print("  No network-metric data found, skipping metrics plot.")

        # --- Network diagrams + cross-party follows (need the platform artifact) ---
        platforms = {}
        for label, runs in strategy_runs_by_condition.items():
            try:
                platforms[label] = download_platform(runs[0])
            except RuntimeError as e:
                print(f"  {label}: {e}")

        network_labels = [l for l in labels if l in platforms]
        if network_labels:
            plot_networks(network_labels, platforms,
                           fig_path(f"{strategy}_networks_ablation", args.batch_id))
            plot_cross_party_follows(network_labels, platforms,
                                      fig_path(f"{strategy}_cross_party_follows_ablation", args.batch_id))
        else:
            print("  No platform artifacts found, skipping network/cross-party-follow plots.")


if __name__ == "__main__":
    main()
