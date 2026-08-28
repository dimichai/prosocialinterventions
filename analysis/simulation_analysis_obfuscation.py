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

    data, raw_data = fetch_and_aggregate(runs_by_condition)
    condition_colors = {l: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, l in enumerate(labels)}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.5))
    plot_metrics_comparison(axes, labels, condition_colors, data, raw_data=raw_data)
    fig.suptitle("Simulation metrics by obfuscation condition")
    fig.tight_layout()

    out_path = fig_path("simulation_results_obfuscation", args.batch_id)
    fig.savefig(out_path)
    fig.savefig(fig_path("simulation_results_obfuscation", args.batch_id, ext="png"))
    plt.close(fig)
    print(f"Saved to {out_path}")

    print(f"\n{'='*60}")
    print("  Simulation metrics (rows = obfuscation condition)")
    print(f"{'='*60}")
    for label in labels:
        row = "  ".join(f"{m}={data[label][m]:.4f} ± {data[label][f'{m}_se']:.4f}" for m in METRICS)
        print(f"  {label:<20} {row}")


if __name__ == "__main__":
    main()
