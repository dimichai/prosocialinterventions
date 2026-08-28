import argparse
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import interview_wandb  # noqa: E402
import persona_interviews as interview  # noqa: E402
from interview_comparison_plots import (  # noqa: E402
    TRAIT_QUESTIONS,
    QUESTIONS,
    fig_path,
    aggregate_population_metrics,
    print_population_table,
    print_question_tables,
    plot_metric_comparison,
    plot_thermometer_comparison,
)

# Colorblind-validated categorical palette (fixed order, not cycled while it lasts);
# ablation combos are an open-ended set (unlike obfuscation's fixed 5), so this is
# extended by repeating if a batch has more conditions than colors.
CONDITION_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7d5ba6", "#4c6663", "#c1666b"]


def ablation_label(ablations: tuple[str, ...]) -> str:
    """Display label for one ablation combo — 'None' for the baseline (no
    ablations), else the sorted ablation names joined with '+'."""
    return "None" if not ablations else "+".join(ablations)


def fetch_condition_dfs(
    batch_id: str, wandb_project: str = interview_wandb.WANDB_PROJECT
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, tuple[float, float]]]]:
    """Fetch every wandb run sharing `batch_id` (one run per ablation-combo x seed —
    run_persona_pipeline.py's --ablations is one additive combo per invocation, so
    pass the same --batch_id across multiple invocations, one per combo, to populate
    a batch spanning more than one). Download each run's raw per-persona results, and
    aggregate across seeds within each ablation combo.

    Returns (dfs, population), where `dfs` maps condition label -> aggregated
    question/thermometer results and `population` maps condition label -> aggregated
    persona-population attribute stats (see `aggregate_population_metrics`)."""
    runs = interview_wandb.fetch_runs_by_group(wandb_project, batch_id)
    if not runs:
        raise RuntimeError(f"No wandb runs found in project '{wandb_project}' for group '{batch_id}'.")

    runs_by_ablations: dict[tuple[str, ...], list] = defaultdict(list)
    for run in runs:
        runs_by_ablations[tuple(sorted(run.config["ablations"]))].append(run)

    # Baseline (no ablations) first, then the rest alphabetically by combo.
    ordered_combos = sorted(runs_by_ablations, key=lambda c: (c != (), c))

    dfs = {}
    population = {}
    for combo in ordered_combos:
        condition_runs = runs_by_ablations[combo]
        label = ablation_label(combo)
        raw_dfs = [interview_wandb.download_results_dataframe(run) for run in condition_runs]
        cfg = condition_runs[0].config
        questions = interview.build_questions(
            cfg["trump_label"], cfg["biden_label"], cfg["democrats_label"], cfg["republicans_label"]
        )
        thermometer_targets = interview.build_thermometer_targets(
            cfg["trump_label"], cfg["biden_label"], cfg["democrats_label"], cfg["republicans_label"]
        )
        dfs[label] = interview.aggregate_interview_runs(raw_dfs, questions, thermometer_targets)
        population[label] = aggregate_population_metrics(raw_dfs)

    return dfs, population


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a batch of ablation-comparison persona-interview runs from wandb "
                     "and plot yes/no + feeling-thermometer results by party."
    )
    parser.add_argument("--batch_id", type=str, required=True,
                         help="Wandb group/batch id shared by the runs to compare "
                              "(printed by run_persona_pipeline.py after it finishes — "
                              "pass the same --batch_id across multiple invocations, "
                              "one per --ablations combo, to populate this batch).")
    parser.add_argument("--wandb_project", type=str, default=interview_wandb.WANDB_PROJECT,
                         help="Wandb project the runs were logged to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dfs, population = fetch_condition_dfs(args.batch_id, wandb_project=args.wandb_project)

    print_population_table(population)

    # Row 1: Democrats / loves Biden / hates Biden. Row 2: Republicans / loves
    # Trump / hates Trump — each party grouped with its own leader's questions.
    follow_keys = ['q1_dem', 'q4', 'q5', 'q1_rep', 'q2', 'q3']
    # Interleaved (dem, rep) per trait — not all-Democrat-traits-then-all-Republican
    # — so the trait battery plot below can put each trait's two panels side by
    # side (e.g. "Democrats: intelligent?" right next to "Republicans: intelligent?").
    trait_keys  = [k for key, _ in TRAIT_QUESTIONS for k in (f"dem_{key}", f"rep_{key}")]

    # Real labels only — ablation batches are always the real-label (obfuscation
    # "none") condition, so unlike the obfuscation comparison there's nothing to
    # annotate as "(obfuscated per condition)".
    question_labels = {
        'q1_dem': "Would you follow\na member of Democrats?",
        'q1_rep': "Would you follow\na member of Republicans?",
        'q2': "Would you follow\nsomeone who loves Trump?",
        'q3': "Would you follow\nsomeone who hates Trump?",
        'q4': "Would you follow\nsomeone who loves Biden?",
        'q5': "Would you follow\nsomeone who hates Biden?",
    }
    for key, trait in TRAIT_QUESTIONS:
        question_labels[f"dem_{key}"] = f"Democrats:\nare they {trait}?"
        question_labels[f"rep_{key}"] = f"Republicans:\nare they {trait}?"
    question_texts = {k: t for k, t in QUESTIONS}

    def keys_with_titles(keys: list[str]) -> list[tuple[str, str]]:
        # Only plot keys present (as a "question" row) in every comparison file,
        # so bars aren't silently missing for whichever condition lacks the data.
        present = [k for k in keys if all(((df["metric"] == "question") & (df["key"] == k)).any() for df in dfs.values())]
        return [(k, question_labels.get(k, question_texts.get(k, k))) for k in present]

    follow_present = keys_with_titles(follow_keys)
    trait_present  = keys_with_titles(trait_keys)
    # For printing, use every question key (not just ones present in every
    # comparison file) so conditions with extra questions still get reported.
    all_keys = follow_keys + trait_keys

    all_parties = sorted(set().union(*[set(df["party"].dropna().unique()) for df in dfs.values()]))
    labels      = list(dfs.keys())

    condition_colors = {l: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, l in enumerate(labels)}

    # Ablation combos aren't a strict nested/cumulative chain in general (any subset
    # can be combined), so — same reasoning as the obfuscation comparison — a
    # grouped bar chart per condition is the honest comparison rather than a line
    # chart implying an ordering that isn't guaranteed to exist.
    plot_metric_comparison(dfs, follow_present, all_parties, condition_colors,
                            fig_path("interview_results_ablation", args.batch_id), "question", "pct_yes_mean", "pct_yes_std",
                            ncols=3)
    plot_metric_comparison(dfs, trait_present, all_parties, condition_colors,
                            fig_path("interview_results_ablation_traits", args.batch_id), "question", "pct_yes_mean", "pct_yes_std",
                            ncols=2, show_dont_know=True)

    print(f"\n{'='*60}")
    print("  Question answers (rows = ablation condition, columns = party)")
    print(f"{'='*60}")
    print_question_tables(dfs, all_keys, all_parties, question_labels, question_texts, trait_keys)

    plot_thermometer_comparison(dfs, all_parties, condition_colors,
                                 fig_path("interview_results_ablation_thermometer", args.batch_id))


if __name__ == "__main__":
    main()
