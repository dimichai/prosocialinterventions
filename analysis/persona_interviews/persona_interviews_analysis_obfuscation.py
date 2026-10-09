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
    POSITIVE_TRAIT_QUESTIONS,
    NEGATIVE_TRAIT_QUESTIONS,
    QUESTIONS,
    GAP_ROLE_MAPS,
    FOLLOW_GAP_ROLE_MAPS,
    TRAIT_GAP_ROLE_MAPS,
    TRAIT_DONT_KNOW_ROLES,
    fig_path,
    aggregate_population_metrics,
    aggregate_ground_truth_thermometer,
    print_population_table,
    print_question_tables,
    print_thermometer_table,
    print_ground_truth_thermometer_table,
    print_ground_truth_gap_table,
    print_gap_comparison_table,
    plot_metric_grouped_bar_comparison,
    plot_gap_grouped_bar_comparison,
    plot_role_average_grouped_bar_comparison,
    plot_trait_gap_heatmap,
    party_color_map,
)

# obfuscation (config value) -> comparison-plot display label, in a fixed display
# order: Real > Neutral > Unrelated > Nonce. "nonce" and "randomnonce" are two
# independent sets of nonce strings, so their runs are pooled under one "Nonce"
# label to average out accidental associations of either set.
OBFUSCATION_LABELS = {
    "none":         "No Obfuscation",
    "neutral":      "Neutral",
    "randomreal":   "Unrelated",
    "nonce":        "Nonce",
    "randomnonce":  "Nonce",
}


def fetch_condition_dfs(
    batch_id: str, wandb_project: str = interview_wandb.WANDB_PROJECT
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, tuple[float, float]]], dict[str, dict[str, tuple[float, float]]]]:
    """Fetch every wandb run sharing `batch_id` (one run per obfuscation x seed —
    run_persona_pipeline.py's --obfuscation is one condition per invocation, so pass
    the same --batch_id across multiple invocations, one per condition, to populate a
    batch spanning more than one). Download each run's raw per-persona results, and
    aggregate across seeds within each obfuscation condition — i.e. the same shape
    `load_and_prepare` used to return from a pre-aggregated local CSV, just sourced
    from wandb instead.

    Returns (dfs, population, ground_truth), where `dfs` maps condition label ->
    aggregated question/thermometer results (as before), `population` maps condition
    label -> aggregated persona-population attribute stats (see
    `aggregate_population_metrics`), and `ground_truth` maps thermometer role ->
    party -> (value, 95%-CI half-width) of real ANES respondents' own rating (see
    `aggregate_ground_truth_thermometer`) — computed once, from the "No Obfuscation"
    condition, since it's identical across obfuscation conditions sharing a seed
    (obfuscating a persona's own description doesn't change the real respondent
    behind it)."""
    runs = interview_wandb.fetch_runs_by_group(wandb_project, batch_id)
    if not runs:
        raise RuntimeError(f"No wandb runs found in project '{wandb_project}' for group '{batch_id}'.")

    runs_by_obfuscation: dict[str, list] = defaultdict(list)
    for run in runs:
        runs_by_obfuscation[run.config["obfuscation"]].append(run)

    dfs = {}
    population = {}
    ground_truth = None
    runs_by_label: dict[str, list] = defaultdict(list)
    for obfuscation, label in OBFUSCATION_LABELS.items():
        runs_by_label[label] += runs_by_obfuscation.get(obfuscation, [])

    for label, condition_runs in runs_by_label.items():
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
        if label == OBFUSCATION_LABELS["none"]:
            ground_truth = aggregate_ground_truth_thermometer(raw_dfs)

    return dfs, population, ground_truth


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
    parser.add_argument("--wandb_project", type=str, default='persona-simulation',
                         help="Wandb project the runs were logged to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dfs, population, ground_truth = fetch_condition_dfs(args.batch_id, wandb_project=args.wandb_project)

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

    party_colors = party_color_map(all_parties)

    # Obfuscation conditions are distinct schemes, not a progressive/additive series
    # (unlike the ablation comparison), so each chart below is a grouped bar chart —
    # one cluster of bars per condition, colored by party — rather than a slope
    # chart connecting conditions with a line, which would imply an ordering that
    # isn't there. Coloring by party (matching the ablation script's line colors)
    # still lets you compare Democrats vs. Republicans at a glance within a
    # condition, and compare a party across conditions by its color, same as the
    # two comparisons a slope chart offers.
    plot_metric_grouped_bar_comparison(dfs, follow_present, all_parties, party_colors,
                                        fig_path("interview_results_obfuscation", args.batch_id),
                                        "question", "pct_yes_mean", "pct_yes_std", ncols=3)

    # Follow gap: Pr(follow in-party) - Pr(follow out-party), and the
    # analogous gap between agents who love vs. hate each candidate — see
    # FOLLOW_GAP_ROLE_MAPS.
    plot_gap_grouped_bar_comparison(dfs, FOLLOW_GAP_ROLE_MAPS, all_parties, party_colors,
                                     fig_path("interview_results_obfuscation_follow_gap", args.batch_id),
                                     ncols=3, value_label="Follow gap",
                                     ylim=(-1, 1), metric="question", value_col="pct_yes_mean", std_col="pct_yes_std")
    print_gap_comparison_table(dfs, FOLLOW_GAP_ROLE_MAPS["Party"], all_parties,
                                "Follow gap (party): Pr(follow in-party) - Pr(follow out-party)",
                                metric="question", value_col="pct_yes_mean", std_col="pct_yes_std")
    print_gap_comparison_table(dfs, FOLLOW_GAP_ROLE_MAPS["Biden"], all_parties,
                                "Follow gap (Biden): Pr(follow lover) - Pr(follow hater)",
                                metric="question", value_col="pct_yes_mean", std_col="pct_yes_std")
    print_gap_comparison_table(dfs, FOLLOW_GAP_ROLE_MAPS["Trump"], all_parties,
                                "Follow gap (Trump): Pr(follow lover) - Pr(follow hater)",
                                metric="question", value_col="pct_yes_mean", std_col="pct_yes_std")

    plot_metric_grouped_bar_comparison(dfs, trait_present, all_parties, party_colors,
                                        fig_path("interview_results_obfuscation_traits", args.batch_id),
                                        "question", "pct_yes_mean", "pct_yes_std", ncols=2)
    # Don't-know rate as its own chart, rather than annotated on the yes-rate
    # panels, so forced-choice artifacts (respondents with no basis to judge a
    # party) are visible instead of defaulting into the "No" bar.
    plot_metric_grouped_bar_comparison(dfs, trait_present, all_parties, party_colors,
                                        fig_path("interview_results_obfuscation_traits_dont_know", args.batch_id),
                                        "question", "pct_dont_know_mean", "pct_dont_know_std", ncols=2,
                                        value_label='Fraction answering "don\'t know"')

    print(f"\n{'='*60}")
    print("  Question answers (rows = obfuscation condition, columns = party)")
    print(f"{'='*60}")
    print_question_tables(dfs, all_keys, all_parties, question_labels, question_texts, trait_keys)

    therm_role_labels = [
        ("democrats", "Democrats"),
        ("biden", "Biden"),
        ("republicans", "Republicans"),
        ("trump", "Trump"),
    ]
    therm_present = [
        (role, f"Feeling thermometer:\n{label}\n(obfuscated per condition)") for role, label in therm_role_labels
        if all(((df["metric"] == "thermometer") & (df["key"] == role)).any() for df in dfs.values())
    ]
    plot_metric_grouped_bar_comparison(dfs, therm_present, all_parties, party_colors,
                                        fig_path("interview_results_obfuscation_thermometer", args.batch_id),
                                        "thermometer", "rating_mean", "rating_std", ncols=4,
                                        value_label="Mean rating (0-100)", ylim=(0, 100),
                                        ground_truth=ground_truth)
    print_thermometer_table(dfs, therm_role_labels, all_parties)
    if ground_truth:
        print_ground_truth_thermometer_table(ground_truth, therm_role_labels, all_parties)

    # Thermometer gap T_in - T_out (own-side minus opposing-side rating), the
    # standard affective-polarization measure, reported for parties and
    # candidates separately plus their combined average — see GAP_ROLE_MAPS.
    plot_gap_grouped_bar_comparison(dfs, GAP_ROLE_MAPS, all_parties, party_colors,
                                     fig_path("interview_results_obfuscation_thermometer_gap", args.batch_id),
                                     ncols=3, ylim=(-20, 100), ground_truth=ground_truth)
    for title in GAP_ROLE_MAPS:
        print_gap_comparison_table(dfs, GAP_ROLE_MAPS[title], all_parties,
                                    "Thermometer gap",
                                    metric="thermometer", value_col="rating_mean", std_col="rating_std")
    if ground_truth:
        print_ground_truth_gap_table(ground_truth, GAP_ROLE_MAPS, all_parties)

    # Trait differential: the share of positive (resp. negative) traits a
    # party's respondents attribute to their own party minus the share they
    # attribute to the opposing party — don't-know responses reported as
    # their own rate rather than recoded into the differential.
    plot_gap_grouped_bar_comparison(dfs, TRAIT_GAP_ROLE_MAPS, all_parties, party_colors,
                                     fig_path("interview_results_obfuscation_trait_differential", args.batch_id),
                                     ncols=2, value_label="Trait gap",
                                     ylim=(-1, 1), metric="question", value_col="pct_yes_mean", std_col="pct_yes_std")
    plot_role_average_grouped_bar_comparison(dfs, TRAIT_DONT_KNOW_ROLES, all_parties, party_colors,
                                              fig_path("interview_results_obfuscation_trait_differential_dont_know", args.batch_id),
                                              ncols=2, value_label='Fraction answering "don\'t know"', ylim=(0, 1))
    # Same trait gap, broken out per individual trait rather than aggregated
    # across the whole positive/negative battery — a compact heatmap (rows =
    # trait, columns = condition) rather than one bar/slope panel per trait,
    # which doesn't scale past a handful of traits. Paper-table-friendly.
    plot_trait_gap_heatmap(dfs, POSITIVE_TRAIT_QUESTIONS + NEGATIVE_TRAIT_QUESTIONS,
                            fig_path("interview_results_obfuscation_trait_gap_heatmap", args.batch_id),
                            parties=("Democrat", "Republican"),
                            separator_after=len(POSITIVE_TRAIT_QUESTIONS) - 1)
    print_gap_comparison_table(dfs, TRAIT_GAP_ROLE_MAPS["Positive traits"], all_parties,
                                "Positive traits: share(in-party) - share(out-party)",
                                metric="question", value_col="pct_yes_mean", std_col="pct_yes_std",
                                dont_know_roles=TRAIT_DONT_KNOW_ROLES["Positive traits"])
    print_gap_comparison_table(dfs, TRAIT_GAP_ROLE_MAPS["Negative traits"], all_parties,
                                "Negative traits: share(in-party) - share(out-party)",
                                metric="question", value_col="pct_yes_mean", std_col="pct_yes_std",
                                dont_know_roles=TRAIT_DONT_KNOW_ROLES["Negative traits"])


if __name__ == "__main__":
    main()
