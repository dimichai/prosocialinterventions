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

# obfuscation (config value) -> comparison-plot display label, in a fixed display
# order: Real > Neutral > RandomReal > Nonce > RandomNonce.
OBFUSCATION_LABELS = {
    "none":         "No Obfuscation",
    "neutral":      "Neutral",
    "randomreal":   "RandomReal",
    "nonce":        "Nonce",
    "randomnonce":  "RandomNonce",
}


def fetch_condition_dfs(
    batch_id: str, wandb_project: str = interview_wandb.WANDB_PROJECT
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, tuple[float, float]]]]:
    """Fetch every wandb run sharing `batch_id` (one run per obfuscation x seed —
    run_persona_pipeline.py's --obfuscation is one condition per invocation, so pass
    the same --batch_id across multiple invocations, one per condition, to populate a
    batch spanning more than one). Download each run's raw per-persona results, and
    aggregate across seeds within each obfuscation condition — i.e. the same shape
    `load_and_prepare` used to return from a pre-aggregated local CSV, just sourced
    from wandb instead.

    Returns (dfs, population), where `dfs` maps condition label -> aggregated
    question/thermometer results (as before) and `population` maps condition
    label -> aggregated persona-population attribute stats (see
    `aggregate_population_metrics`)."""
    runs = interview_wandb.fetch_runs_by_group(wandb_project, batch_id)
    if not runs:
        raise RuntimeError(f"No wandb runs found in project '{wandb_project}' for group '{batch_id}'.")

    runs_by_obfuscation: dict[str, list] = defaultdict(list)
    for run in runs:
        runs_by_obfuscation[run.config["obfuscation"]].append(run)

    dfs = {}
    population = {}
    for obfuscation, label in OBFUSCATION_LABELS.items():
        condition_runs = runs_by_obfuscation.get(obfuscation)
        if not condition_runs:
            continue
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
        description="Load a batch of obfuscation-comparison persona-interview runs from wandb "
                     "and plot yes/no + feeling-thermometer results by party."
    )
    parser.add_argument("--batch_id", type=str, required=True,
                         help="Wandb group/batch id shared by the runs to compare "
                              "(printed by run_persona_pipeline.py after it finishes — "
                              "pass the same --batch_id across multiple invocations, "
                              "one per --obfuscation value, to populate this batch).")
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

    question_labels = {
        'q1_dem': "Would you follow\na member of Democrats\n(obfuscated per condition)?",
        'q1_rep': "Would you follow\na member of Republicans\n(obfuscated per condition)?",
        'q2': "Would you follow\nsomeone who loves Trump\n(obfuscated per condition)?",
        'q3': "Would you follow\nsomeone who hates Trump\n(obfuscated per condition)?",
        'q4': "Would you follow\nsomeone who loves Biden\n(obfuscated per condition)?",
        'q5': "Would you follow\nsomeone who hates Biden\n(obfuscated per condition)?",
    }
    for key, trait in TRAIT_QUESTIONS:
        question_labels[f"dem_{key}"] = f"Democrats:\nare they {trait}?\n(obfuscated per condition)"
        question_labels[f"rep_{key}"] = f"Republicans:\nare they {trait}?\n(obfuscated per condition)"
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

    # Colorblind-validated categorical palette (fixed order, not cycled):
    # blue, orange, aqua, yellow, magenta.
    condition_colors = {l: c for l, c in zip(labels, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"])}

    # Obfuscation conditions are distinct schemes, not a progressive/additive series
    # (unlike the ablation comparison), so a grouped bar chart per condition
    # is the honest comparison — a connecting line would imply an ordering that isn't there.
    plot_metric_comparison(dfs, follow_present, all_parties, condition_colors,
                            fig_path("interview_results_obfuscation", args.batch_id), "question", "pct_yes_mean", "pct_yes_std",
                            ncols=3)
    # Trait battery: one row per trait, Democrats and Republicans side by side
    # (trait_keys is already interleaved dem/rep, so a 2-column grid lines up
    # each trait's pair of panels instead of splitting all-dem/all-rep onto
    # separate rows). Each cell also carries its don't-know rate (small "dk NN%"
    # label, top-right of the bar) — makes forced-choice artifacts (non-partisans
    # with no basis to judge an obfuscated group) visible in this same chart
    # instead of a separate one.
    plot_metric_comparison(dfs, trait_present, all_parties, condition_colors,
                            fig_path("interview_results_obfuscation_traits", args.batch_id), "question", "pct_yes_mean", "pct_yes_std",
                            ncols=2, show_dont_know=True)

    print(f"\n{'='*60}")
    print("  Question answers (rows = obfuscation condition, columns = party)")
    print(f"{'='*60}")
    print_question_tables(dfs, all_keys, all_parties, question_labels, question_texts, trait_keys)

    plot_thermometer_comparison(dfs, all_parties, condition_colors,
                                 fig_path("interview_results_obfuscation_thermometer", args.batch_id),
                                 therm_title_suffix="\n(obfuscated per condition)")


if __name__ == "__main__":
    main()
