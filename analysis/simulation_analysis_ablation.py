import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "persona_interviews"))

import matplotlib.pyplot as plt  # noqa: E402

import interview_wandb  # noqa: E402
from simulation_comparison_plots import (  # noqa: E402
    METRICS,
    fig_path,
    fetch_and_aggregate,
    plot_metrics_comparison,
)

# run_persona_pipeline.py's default --wandb_project.
DEFAULT_WANDB_PROJECT = "persona-simulation"

# Colorblind-validated categorical palette; ablation combos are an open-ended set
# (unlike obfuscation's fixed 5), so this is extended by repeating if a batch has
# more conditions than colors.
CONDITION_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7d5ba6", "#4c6663", "#c1666b"]


def ablation_label(ablations: tuple[str, ...]) -> str:
    """Display label for one ablation combo — 'None' for the baseline (no
    ablations), else the sorted ablation names joined with '+'. (Duplicated from
    persona_interviews_analysis_ablation.py — small enough that keeping the
    interview- and simulation-analysis scripts independent of each other outweighs
    sharing this one-line function.)"""
    return "None" if not ablations else "+".join(ablations)


def fetch_runs_by_ablations(batch_id: str, wandb_project: str) -> dict[str, list]:
    """Fetch every wandb run sharing `batch_id` (one run per ablation-combo x seed —
    run_persona_pipeline.py's --ablations is one additive combo per invocation, so
    pass the same --batch_id across multiple invocations, one per combo, to populate
    a batch spanning more than one), bucketed by display label — baseline (no
    ablations) first, then the rest alphabetically by combo."""
    runs = interview_wandb.fetch_runs_by_group(wandb_project, batch_id)
    if not runs:
        raise RuntimeError(f"No wandb runs found in project '{wandb_project}' for group '{batch_id}'.")

    runs_by_ablations = defaultdict(list)
    for run in runs:
        runs_by_ablations[tuple(sorted(run.config["ablations"]))].append(run)

    ordered_combos = sorted(runs_by_ablations, key=lambda c: (c != (), c))
    return {ablation_label(combo): runs_by_ablations[combo] for combo in ordered_combos}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a batch of ablation-comparison simulation runs from wandb "
                     "and plot final simulation metrics by condition."
    )
    parser.add_argument("--batch_id", type=str, required=True,
                         help="Wandb group/batch id shared by the runs to compare "
                              "(printed by run_persona_pipeline.py after it finishes — "
                              "pass the same --batch_id across multiple invocations, "
                              "one per --ablations combo, to populate this batch).")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT,
                         help="Wandb project the runs were logged to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_by_condition = fetch_runs_by_ablations(args.batch_id, args.wandb_project)
    labels = list(runs_by_condition.keys())
    print(f"Found conditions: {labels}")

    data, raw_data = fetch_and_aggregate(runs_by_condition)
    condition_colors = {l: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, l in enumerate(labels)}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.5))
    plot_metrics_comparison(axes, labels, condition_colors, data, raw_data=raw_data)
    fig.suptitle("Simulation metrics by ablation condition")
    fig.tight_layout()

    out_path = fig_path("simulation_results_ablation", args.batch_id)
    fig.savefig(out_path)
    fig.savefig(fig_path("simulation_results_ablation", args.batch_id, ext="png"))
    plt.close(fig)
    print(f"Saved to {out_path}")

    print(f"\n{'='*60}")
    print("  Simulation metrics (rows = ablation condition)")
    print(f"{'='*60}")
    for label in labels:
        row = "  ".join(f"{m}={data[label][m]:.4f} ± {data[label][f'{m}_se']:.4f}" for m in METRICS)
        print(f"  {label:<20} {row}")


if __name__ == "__main__":
    main()
