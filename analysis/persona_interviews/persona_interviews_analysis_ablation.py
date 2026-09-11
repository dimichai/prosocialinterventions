import argparse
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import interview_wandb  # noqa: E402
import persona_interviews as interview  # noqa: E402
from interview_comparison_plots import (  # noqa: E402
    POSITIVE_TRAIT_QUESTIONS,
    NEGATIVE_TRAIT_QUESTIONS,
    trait_gap_role_map,
    trait_dont_know_roles,
    QUESTIONS,
    GAP_ROLE_MAPS,
    FOLLOW_GAP_ROLE_MAPS,
    fig_path,
    aggregate_population_metrics,
    aggregate_ground_truth_thermometer,
    print_population_table,
    print_question_tables,
    print_thermometer_table,
    print_ground_truth_thermometer_table,
    print_ground_truth_gap_table,
    print_gap_comparison_table,
    plot_slope_comparison,
    plot_gap_slope_comparison,
    plot_role_average_comparison,
    party_color_map,
)

# Open-mindedness is excluded from this ablation evaluation — not plotted in
# the trait battery and not used in the trait differential — so this script
# works off this filtered trait list everywhere instead of the full
# TRAIT_QUESTIONS.
POSITIVE_TRAIT_QUESTIONS_ABLATION = [q for q in POSITIVE_TRAIT_QUESTIONS if q[0] != "openminded"]
TRAIT_QUESTIONS_ABLATION = POSITIVE_TRAIT_QUESTIONS_ABLATION + NEGATIVE_TRAIT_QUESTIONS


# Human-readable labels for the additive AP/PID/VB ablation chain (see
# run_persona_pipeline.py's --ablations). Combos not listed here (e.g. an
# ablation batch outside this chain) fall back to a generic "+"-joined label.
CUSTOM_ABLATION_LABELS = {
    ("extend_with_ai",): "Full Persona",
    ("extend_with_ai", "love_hate"): "No AP",
    ("extend_with_ai", "love_hate", "party_identity"): "No AP & PID",
    ("extend_with_ai", "love_hate", "party_identity", "voted2020"): "No AP & PID & VB",
}


def ablation_label(ablations: tuple[str, ...]) -> str:
    """Display label for one ablation combo — a fixed label for recognized
    combos (see CUSTOM_ABLATION_LABELS), else 'None' for the baseline (no
    ablations) or the sorted ablation names joined with '+'."""
    if ablations in CUSTOM_ABLATION_LABELS:
        return CUSTOM_ABLATION_LABELS[ablations]
    return "None" if not ablations else "+".join(ablations)


def fetch_condition_dfs(
    batch_id: str, wandb_project: str = 'persona-simulation'
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, tuple[float, float]]], dict[str, dict[str, tuple[float, float]]]]:
    """Fetch every wandb run sharing `batch_id` (one run per ablation-combo x seed —
    run_persona_pipeline.py's --ablations is one additive combo per invocation, so
    pass the same --batch_id across multiple invocations, one per combo, to populate
    a batch spanning more than one). Download each run's raw per-persona results, and
    aggregate across seeds within each ablation combo.

    Returns (dfs, population, ground_truth), where `dfs` maps condition label ->
    aggregated question/thermometer results, `population` maps condition label ->
    aggregated persona-population attribute stats (see `aggregate_population_metrics`),
    and `ground_truth` maps thermometer role -> party -> (value, 95%-CI half-width)
    of real ANES respondents' own rating (see `aggregate_ground_truth_thermometer`,
    which computes the CI across seeds) — computed once, from the
    baseline combo, since it's identical across ablation conditions sharing a seed."""
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
    ground_truth = None
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
        if ground_truth is None:
            ground_truth = aggregate_ground_truth_thermometer(raw_dfs)

    return dfs, population, ground_truth


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
    trait_keys  = [k for key, _ in TRAIT_QUESTIONS_ABLATION for k in (f"dem_{key}", f"rep_{key}")]

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
    for key, trait in TRAIT_QUESTIONS_ABLATION:
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

    party_colors = party_color_map(all_parties)

    plot_slope_comparison(dfs, follow_present, all_parties, party_colors,
                           fig_path("interview_results_ablation", args.batch_id),
                           "question", "pct_yes_mean", "pct_yes_std", ncols=3)

    # Follow gap: Pr(follow in-party) - Pr(follow out-party), and the
    # analogous gap between agents who love vs. hate each candidate — see
    # FOLLOW_GAP_ROLE_MAPS.
    plot_gap_slope_comparison(dfs, FOLLOW_GAP_ROLE_MAPS, all_parties, party_colors,
                               fig_path("interview_results_ablation_follow_gap", args.batch_id),
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

    plot_slope_comparison(dfs, trait_present, all_parties, party_colors,
                           fig_path("interview_results_ablation_traits", args.batch_id),
                           "question", "pct_yes_mean", "pct_yes_std", ncols=2)
    # Don't-know rate as its own chart, rather than annotated on the yes-rate
    # panels, so forced-choice artifacts (respondents with no basis to judge a
    # party) are visible instead of defaulting into the "No" line.
    plot_slope_comparison(dfs, trait_present, all_parties, party_colors,
                           fig_path("interview_results_ablation_traits_dont_know", args.batch_id),
                           "question", "pct_dont_know_mean", "pct_dont_know_std", ncols=2,
                           value_label='Fraction answering "don\'t know"')

    print(f"\n{'='*60}")
    print("  Question answers (rows = ablation condition, columns = party)")
    print(f"{'='*60}")
    print_question_tables(dfs, all_keys, all_parties, question_labels, question_texts, trait_keys)

    therm_role_labels = [
        ("democrats", "Democrats"),
        ("biden", "Biden"),
        ("republicans", "Republicans"),
        ("trump", "Trump"),
    ]
    therm_present = [
        (role, f"Feeling thermometer:\n{label}") for role, label in therm_role_labels
        if all(((df["metric"] == "thermometer") & (df["key"] == role)).any() for df in dfs.values())
    ]
    plot_slope_comparison(dfs, therm_present, all_parties, party_colors,
                           fig_path("interview_results_ablation_thermometer", args.batch_id),
                           "thermometer", "rating_mean", "rating_std", ncols=4,
                           value_label="Mean rating (0-100)", ylim=(0, 100),
                           ground_truth=ground_truth)
    print_thermometer_table(dfs, therm_role_labels, all_parties)
    print_ground_truth_thermometer_table(ground_truth, therm_role_labels, all_parties)

    # Thermometer gap T_in - T_out (own-side minus opposing-side rating), the
    # standard affective-polarization measure, reported for parties and
    # candidates separately plus their combined average — see GAP_ROLE_MAPS.
    plot_gap_slope_comparison(dfs, GAP_ROLE_MAPS, all_parties, party_colors,
                               fig_path("interview_results_ablation_thermometer_gap", args.batch_id),
                               ncols=3, ylim=(-20, 100), ground_truth=ground_truth)
    for title in GAP_ROLE_MAPS:
        print_gap_comparison_table(dfs, GAP_ROLE_MAPS[title], all_parties,
                                    f"Thermometer gap",
                                    metric="thermometer", value_col="rating_mean", std_col="rating_std")
    print_ground_truth_gap_table(ground_truth, GAP_ROLE_MAPS, all_parties)

    # Trait differential: the share of positive (resp. negative) traits a
    # party's respondents attribute to their own party minus the share they
    # attribute to the opposing party — don't-know responses reported as
    # their own rate rather than recoded into the differential. Open-
    # mindedness is excluded from the positive-trait battery (see
    # POSITIVE_TRAIT_QUESTIONS_ABLATION above).
    trait_gap_role_maps = {
        "Positive traits": trait_gap_role_map(POSITIVE_TRAIT_QUESTIONS_ABLATION),
        "Negative traits": trait_gap_role_map(NEGATIVE_TRAIT_QUESTIONS),
    }
    trait_dont_know_role_sets = {
        "Positive traits": trait_dont_know_roles(POSITIVE_TRAIT_QUESTIONS_ABLATION),
        "Negative traits": trait_dont_know_roles(NEGATIVE_TRAIT_QUESTIONS),
    }
    plot_gap_slope_comparison(dfs, trait_gap_role_maps, all_parties, party_colors,
                               fig_path("interview_results_ablation_trait_differential_no_openminded", args.batch_id),
                               ncols=2, value_label="Trait differential",
                               ylim=(-1, 1), metric="question", value_col="pct_yes_mean", std_col="pct_yes_std")
    plot_role_average_comparison(dfs, trait_dont_know_role_sets, all_parties, party_colors,
                                  fig_path("interview_results_ablation_trait_differential_no_openminded_dont_know", args.batch_id),
                                  ncols=2, value_label='Fraction answering "don\'t know"', ylim=(0, 1))
    print_gap_comparison_table(dfs, trait_gap_role_maps["Positive traits"], all_parties,
                                "Positive traits: share(in-party) - share(out-party)",
                                metric="question", value_col="pct_yes_mean", std_col="pct_yes_std",
                                dont_know_roles=trait_dont_know_role_sets["Positive traits"])
    print_gap_comparison_table(dfs, trait_gap_role_maps["Negative traits"], all_parties,
                                "Negative traits: share(in-party) - share(out-party)",
                                metric="question", value_col="pct_yes_mean", std_col="pct_yes_std",
                                dont_know_roles=trait_dont_know_role_sets["Negative traits"])


if __name__ == "__main__":
    main()
