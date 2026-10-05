import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "persona_interviews"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

import interview_wandb  # noqa: E402
from interview_comparison_plots import PARTY_COLORS  # noqa: E402
from simulation_comparison_plots import (  # noqa: E402
    NETWORK_METRICS,
    CORRELATION_METRICS,
    fig_path,
    fetch_and_aggregate,
    plot_metrics_comparison,
    download_platform,
    plot_networks,
)
from follow_decomposition import decompose_runs, plot_decomposition_figure, print_decomposition  # noqa: E402

# Metrics fetched from wandb — the union of both charts' panels below.
ALL_METRICS = {**NETWORK_METRICS, **CORRELATION_METRICS}

# run_persona_pipeline.py's default --wandb_project.
DEFAULT_WANDB_PROJECT = "persona-simulation"

# obfuscation (config value) -> comparison-plot display label, in a fixed display
# order: Real > Neutral > RandomReal > Nonce > RandomNonce. (Duplicated from
# persona_interviews_analysis_obfuscation.py's OBFUSCATION_LABELS — small enough
# that keeping the interview- and simulation-analysis scripts independent of each
# other outweighs sharing a 5-line dict.)
OBFUSCATION_LABELS = {
    "none":         "No Obfuscation",
    "neutral":      "Neutral",
    "randomreal":   "RandomReal",
    "nonce":        "Nonce",
    "randomnonce":  "RandomNonce",
}

CONDITION_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def fetch_runs_by_obfuscation(batch_id: str, wandb_project: str) -> dict[str, list]:
    """Fetch every wandb run sharing `batch_id` (one run per obfuscation x seed —
    run_persona_pipeline.py's --obfuscation is one condition per invocation, so pass
    the same --batch_id across multiple invocations, one per condition, to populate a
    batch spanning more than one), bucketed by display label in OBFUSCATION_LABELS'
    fixed order (only labels actually present in the batch)."""
    runs = interview_wandb.fetch_runs_by_group(wandb_project, batch_id)
    if not runs:
        raise RuntimeError(f"No wandb runs found in project '{wandb_project}' for group '{batch_id}'.")

    runs_by_obfuscation = defaultdict(list)
    for run in runs:
        runs_by_obfuscation[run.config["obfuscation"]].append(run)

    return {
        OBFUSCATION_LABELS[obf]: runs_by_obfuscation[obf]
        for obf in OBFUSCATION_LABELS
        if obf in runs_by_obfuscation
    }


PARTIES = ["Democrat", "Republican"]


def out_party_share(platform) -> dict[str, float]:
    """Per-party decomposition of EI_index_dem_rep (see compute_metrics in
    src/main.py): for each party, the fraction of its members' follows that go to
    the other party, over the same Dem/Rep-only links. The pooled share s over both
    parties gives back the index as EI = 2s - 1. A party with no such follows in
    this run is omitted."""
    party_by_id = {user.identifier: user.persona.get("party") for user in platform.users}
    total = defaultdict(int)
    external = defaultdict(int)
    for from_id, to_id in platform.user_links:
        follower, followed = party_by_id.get(from_id), party_by_id.get(to_id)
        if follower in PARTIES and followed in PARTIES:
            total[follower] += 1
            external[follower] += follower != followed
    return {party: external[party] / total[party] for party in PARTIES if total[party]}


def aggregate_out_party_share(platforms_by_condition: dict[str, list]) -> dict[str, dict[str, list[float]]]:
    """condition label -> party -> per-seed out-party shares."""
    shares = {}
    for label, platforms in platforms_by_condition.items():
        shares[label] = defaultdict(list)
        for platform in platforms:
            for party, share in out_party_share(platform).items():
                shares[label][party].append(share)
    return shares


def plot_out_party_share(labels: list[str], shares: dict[str, dict[str, list[float]]], batch_id: str,
                         alpha: float = 0.05) -> None:
    """Grouped bars: per condition, the share of Democrats' / Republicans' follows
    going to the other party (mean +/- SE across seeds). As in
    plot_metrics_comparison, the first label is the baseline: a Mann-Whitney U test
    marks each other condition's bar with '*' where that party's share differs
    significantly from baseline (p < alpha)."""
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 4.5))
    for i, party in enumerate(PARTIES):
        vals = [shares[l][party] for l in labels]
        means = [np.mean(v) if v else np.nan for v in vals]
        errors = [np.std(v) / np.sqrt(len(v)) if v else np.nan for v in vals]
        bar_x = x + (i - 0.5) * width
        ax.bar(bar_x, means, width, color=PARTY_COLORS[party], label=party,
               yerr=errors, capsize=4, error_kw={'elinewidth': 1.0, 'capthick': 1.0})

        for bar_idx in range(1, len(labels)):
            if len(vals[0]) < 2 or len(vals[bar_idx]) < 2:
                continue
            _, p = mannwhitneyu(vals[0], vals[bar_idx], alternative='two-sided')
            if p < alpha:
                ax.text(bar_x[bar_idx], means[bar_idx] + errors[bar_idx], '*', ha='center', va='bottom',
                        fontsize=13, fontweight='bold', color='#333333')

    ax.set_title("Out-party follows by party", pad=8)
    ax.set_ylabel("Share of follows to the other party")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', rotation_mode='anchor')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.grid(True)
    ax.legend(frameon=False)
    fig.tight_layout()

    base_name = "simulation_results_obfuscation_out_party_share"
    out_path = fig_path(base_name, batch_id)
    fig.savefig(out_path)
    fig.savefig(fig_path(base_name, batch_id, ext="png"))
    plt.close(fig)
    print(f"Saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a batch of obfuscation-comparison simulation runs from wandb "
                     "and plot final simulation metrics by condition."
    )
    parser.add_argument("--batch_id", type=str, required=True,
                         help="Wandb group/batch id shared by the runs to compare "
                              "(printed by run_persona_pipeline.py after it finishes — "
                              "pass the same --batch_id across multiple invocations, "
                              "one per --obfuscation value, to populate this batch).")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT,
                         help="Wandb project the runs were logged to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_by_condition = fetch_runs_by_obfuscation(args.batch_id, args.wandb_project)
    labels = list(runs_by_condition.keys())
    print(f"Found conditions: {labels}")

    data, raw_data = fetch_and_aggregate(runs_by_condition, metrics=ALL_METRICS)
    condition_colors = {l: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, l in enumerate(labels)}

    for base_name, title, metrics in [
        ("simulation_results_obfuscation_network", "Network structure by obfuscation condition", NETWORK_METRICS),
        ("simulation_results_obfuscation_correlations", "Correlations/inequality by obfuscation condition", CORRELATION_METRICS),
    ]:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
        plot_metrics_comparison(axes, labels, condition_colors, data, raw_data=raw_data, metrics=metrics)
        fig.suptitle(title)
        fig.tight_layout()

        out_path = fig_path(base_name, args.batch_id)
        fig.savefig(out_path)
        fig.savefig(fig_path(base_name, args.batch_id, ext="png"))
        plt.close(fig)
        print(f"Saved to {out_path}")

    # --- Two-step decomposition of the EI index (repost step vs follow step),
    # read from each run's console log ---
    decomposition = decompose_runs(runs_by_condition)
    plot_decomposition_figure({"": (labels, decomposition)}, args.batch_id)
    print_decomposition(labels, decomposition)

    # --- Platform artifacts: every seed for the out-party decomposition, the
    # first seed per condition for the network diagrams ---
    platforms_by_condition = {}
    for label in labels:
        platforms_by_condition[label] = []
        for run in runs_by_condition[label]:
            try:
                platforms_by_condition[label].append(download_platform(run))
            except RuntimeError as e:
                print(f"  {label}: {e}")

    network_labels = [l for l in labels if platforms_by_condition[l]]
    if network_labels:
        shares = aggregate_out_party_share(platforms_by_condition)
        plot_out_party_share(network_labels, shares, args.batch_id)
        plot_networks(network_labels, {l: platforms_by_condition[l][0] for l in network_labels},
                       fig_path("simulation_results_obfuscation_networks", args.batch_id))
    else:
        print("  No platform artifacts found, skipping out-party share and network diagrams.")

    print(f"\n{'='*60}")
    print("  Simulation metrics (rows = obfuscation condition)")
    print(f"{'='*60}")
    for label in labels:
        row = "  ".join(f"{m}={data[label][m]:.4f} ± {data[label][f'{m}_se']:.4f}" for m in ALL_METRICS)
        print(f"  {label:<20} {row}")

    if network_labels:
        print(f"\n{'='*60}")
        print("  Out-party follow share by party (mean ± SE across seeds)")
        print(f"{'='*60}")
        for label in network_labels:
            row = "  ".join(
                f"{party}={np.mean(v):.1%} ± {np.std(v) / np.sqrt(len(v)):.1%} (n={len(v)})"
                for party in PARTIES if (v := shares[label][party])
            )
            print(f"  {label:<20} {row}")


if __name__ == "__main__":
    main()
