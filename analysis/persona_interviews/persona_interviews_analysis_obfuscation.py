import argparse
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import interview_wandb  # noqa: E402
import persona_interviews_obfuscation as interview  # noqa: E402

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


# Generic fallback labels — this script compares across conditions, so there's
# no single "current" obfuscation to derive labels from (unlike persona_interviews_obfuscation.py).
# (key_suffix, trait) — asked once for Democrats and once for Republicans
TRAIT_QUESTIONS = [
    ("intelligent",   "intelligent"),
    ("honest",        "honest"),
    ("openminded",    "open-minded"),
    ("generous",      "generous"),
    ("hypocritical",  "hypocritical"),
    ("selfish",       "selfish"),
    ("mean",          "mean"),
]

QUESTIONS = [
    ("q1_dem", "On a social network, would you follow someone who is a member of the Democrats?"),
    ("q1_rep", "On a social network, would you follow someone who is a member of the Republicans?"),
    ("q2", "On a social network, would you follow someone who loves Donald Trump?"),
    ("q3", "On a social network, Would you follow someone who hates Donald Trump?"),
    ("q4", "On a social network, would you follow someone who loves Joe Biden?"),
    ("q5", "On a social network, would you follow someone who hates Joe Biden?"),
    *[(f"dem_{key}", f"Do you think Democrats are {trait}?") for key, trait in TRAIT_QUESTIONS],
    *[(f"rep_{key}", f"Do you think Republicans are {trait}?") for key, trait in TRAIT_QUESTIONS],
]

THERMOMETER_TARGETS = [
    ("biden", "Biden"),
    ("trump", "Trump"),
    ("democrats", "Democrats"),
    ("republicans", "Republicans"),
]

# Own-party / opposing-party thermometer roles (see THERMOMETER_TARGETS keys)
# used to compute affective polarization for each partisan group.
PARTY_THERM_ROLES = {
    "Democrat":   ("democrats", "republicans"),
    "Republican": ("republicans", "democrats"),
}

# obfuscation (config value) -> comparison-plot display label, in a fixed display
# order: Real > Neutral > RandomReal > Nonce > RandomNonce.
OBFUSCATION_LABELS = {
    "none":         "No Obfuscation",
    "neutral":      "Neutral",
    "randomreal":   "RandomReal",
    "nonce":        "Nonce",
    "randomnonce":  "RandomNonce",
}

FIGS_DIR = os.path.join(os.path.dirname(__file__), "figs")


def fig_path(base_name: str, batch_id: str) -> str:
    """Figure output path, tagged with the batch id so figures from different
    comparisons don't overwrite each other."""
    return os.path.join(FIGS_DIR, f"{batch_id}_{base_name}.pdf")


POPULATION_METRIC_INFO = {
    # metric key (see interview_wandb.persona_population_metrics) -> (column header, format spec)
    "personas/n":                       ("N",              "{:.0f}"),
    "personas/pct_democrat":            ("% Democrat",     "{:.1%}"),
    "personas/pct_republican":          ("% Republican",   "{:.1%}"),
    "personas/pct_non_partisan":        ("% Non-partisan", "{:.1%}"),
    "personas/avg_feeling_democratic":  ("Avg feeling(D)", "{:.1f}"),
    "personas/avg_feeling_republican":  ("Avg feeling(R)", "{:.1f}"),
    "personas/avg_partisan_democrats":  ("Avg partisan(D)", "{:.2f}"),
    "personas/avg_partisan_republicans": ("Avg partisan(R)", "{:.2f}"),
    "personas/pct_voted_trump":         ("% Voted Trump",  "{:.1%}"),
    "personas/pct_voted_biden":         ("% Voted Biden",  "{:.1%}"),
}


def aggregate_population_metrics(raw_dfs: list[pd.DataFrame]) -> dict[str, float]:
    """Mean of interview_wandb.persona_population_metrics across the seeds within
    one obfuscation condition — the same persona-attribute distribution that gets
    logged to wandb under the "personas/" prefix at run time, recomputed here from
    the downloaded raw per-persona results so it can be printed alongside the
    question-answer tables."""
    per_run = [interview_wandb.persona_population_metrics(df) for df in raw_dfs]
    keys = set().union(*[m.keys() for m in per_run])
    return {key: pd.Series([m.get(key, float("nan")) for m in per_run], dtype=float).mean() for key in keys}


def fetch_condition_dfs(
    batch_id: str, wandb_project: str = interview_wandb.WANDB_PROJECT
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """Fetch every wandb run sharing `batch_id` (one run per obfuscation x seed,
    written by run_persona_obf_pipeline.py / persona_interviews_obfuscation.py),
    download each run's raw per-persona results, and aggregate across seeds within
    each obfuscation condition — i.e. the same shape `load_and_prepare` used to
    return from a pre-aggregated local CSV, just sourced from wandb instead.

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


def print_population_table(population: dict[str, dict[str, float]]) -> None:
    """Persona population statistics per condition (rows = obfuscation condition),
    i.e. the attributes of the personas being interviewed, before any LLM answers —
    the same stats logged to wandb under the "personas/" prefix."""
    present_metrics = [m for m in POPULATION_METRIC_INFO if any(m in stats for stats in population.values())]
    if not present_metrics:
        print("No persona population stats found — skipping.")
        return

    headers = [POPULATION_METRIC_INFO[m][0] for m in present_metrics]
    rows = {}
    for label, stats in population.items():
        row = {}
        for m in present_metrics:
            header = POPULATION_METRIC_INFO[m][0]
            fmt = POPULATION_METRIC_INFO[m][1]
            v = stats.get(m)
            row[header] = fmt.format(v) if v is not None and pd.notna(v) else "n/a"
        rows[label] = row

    table = pd.DataFrame.from_dict(rows, orient="index", columns=headers)
    print(f"\n{'='*60}")
    print("  Persona population statistics (per condition)")
    print(f"{'='*60}")
    print(table.to_string())


def print_question_tables(
    dfs: dict[str, pd.DataFrame],
    keys: list[str],
    all_parties: list[str],
    question_labels: dict[str, str],
    question_texts: dict[str, str],
    trait_keys: list[str],
) -> None:
    """One table per question, rows = obfuscation condition, columns = party
    identity — lines up every condition's answer to the same question side by
    side, for every party, instead of splitting them across per-condition blocks."""
    for key in keys:
        rows = {}
        for label, df in dfs.items():
            subset = df[(df["metric"] == "question") & (df["key"] == key)]
            if subset.empty:
                continue
            row = {}
            for party in all_parties:
                r = subset[subset["party"] == party]
                if r.empty or pd.isna(r.iloc[0]["pct_yes_mean"]):
                    row[party] = "n/a"
                    continue
                rec = r.iloc[0]
                cell = f"{rec['pct_yes_mean'] * 100:.1f}%"
                if key in trait_keys and pd.notna(rec["pct_dont_know_mean"]):
                    cell += f" (dk {rec['pct_dont_know_mean'] * 100:.1f}%)"
                row[party] = cell
            rows[label] = row

        if not rows:
            continue

        q_text = question_labels.get(key, question_texts.get(key, key)).replace("\n", " ")
        table = pd.DataFrame.from_dict(rows, orient="index", columns=all_parties)
        print(f"\n  {q_text}")
        print(table.to_string().replace("\n", "\n  "))
        print(f"    ({key})")


def _lookup(df: pd.DataFrame, metric: str, key: str, party: str, value_col: str, std_col: str) -> tuple[float, float]:
    """Return (value, 95%-CI half-width) for one (metric, key, party) row, where
    the CI is computed from the cross-seed std already stored in the aggregated CSV
    (`1.96 * std / sqrt(n_runs)`) rather than a per-respondent binomial SE, since
    raw per-respondent answers aren't available in this aggregated format."""
    row = df[(df["metric"] == metric) & (df["key"] == key) & (df["party"] == party)]
    if row.empty:
        return float("nan"), float("nan")
    row = row.iloc[0]
    value, std, n_runs = row[value_col], row[std_col], row["n_runs"]
    if pd.isna(value) or n_runs <= 0:
        return float("nan"), float("nan")
    err = 1.96 * std / math.sqrt(n_runs) if pd.notna(std) else float("nan")
    return float(value), float(err)


def plot_metric_comparison(
    dfs: dict[str, pd.DataFrame],
    keys: list[tuple[str, str]],
    all_parties: list[str],
    condition_colors: dict[str, str],
    output_path: str,
    metric: str,
    value_col: str,
    std_col: str,
    ncols: int | None = None,
    ylabel: str = "Fraction answering Yes",
    ylim: tuple[float, float] = (0, 1),
) -> None:
    """Grouped bar chart of a per-party summary statistic, one panel per question/target.

    Panels wrap onto multiple rows (ncols per row) so this scales from a
    handful of questions up to a full trait battery without one absurdly wide row.
    """
    n_panels = len(keys)
    if n_panels == 0:
        print(f"No columns to plot for {output_path} — skipping.")
        return

    labels     = list(dfs.keys())
    n_datasets = len(labels)
    n_parties  = len(all_parties)

    ncols = ncols or n_panels
    nrows = -(-n_panels // ncols)  # ceil division

    group_width = 0.8
    bar_width   = group_width / n_datasets
    group_centers = list(range(n_parties))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows), sharey=True, squeeze=False)
    flat_axes = list(axes.flat)

    for ax, (key, title) in zip(flat_axes, keys):
        for d_idx, label in enumerate(labels):
            vals, errs = [], []
            for party in all_parties:
                v, e = _lookup(dfs[label], metric, key, party, value_col, std_col)
                vals.append(v)
                errs.append(e)
            color = condition_colors.get(label, "#888888")
            positions = [c - group_width / 2 + bar_width * (d_idx + 0.5) for c in group_centers]
            ax.bar(positions, vals, width=bar_width * 0.9, yerr=errs, color=color,
                   capsize=2, error_kw={"elinewidth": 0.8, "capthick": 0.8})

        ax.set_title(title, fontweight='medium', pad=8, fontsize=10)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(all_parties, rotation=15)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#333333')
        ax.set_axisbelow(True)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")
    for row in range(nrows):
        axes[row, 0].set_ylabel(ylabel)

    handles = [plt.Rectangle((0, 0), 1, 1, color=condition_colors.get(l, "#888888")) for l in labels]
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout(pad=1.2, rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")


def plot_thermometer_comparison(
    dfs: dict[str, pd.DataFrame], all_parties: list[str], condition_colors: dict[str, str], output_path: str
) -> None:
    therm_keys = [(role, label) for role, label in THERMOMETER_TARGETS
                  if all(((df["metric"] == "thermometer") & (df["key"] == role)).any() for df in dfs.values())]
    if not therm_keys:
        print("No feeling-thermometer rows found in the comparison CSVs — skipping thermometer comparison plot.")
        return

    labels     = list(dfs.keys())
    n_questions = len(therm_keys)
    n_datasets  = len(labels)
    n_parties   = len(all_parties)

    group_width = 0.8
    bar_width   = group_width / n_datasets
    group_centers = list(range(n_parties))

    fig, axes = plt.subplots(1, n_questions, figsize=(4 * n_questions, 4.5), sharey=True)
    if n_questions == 1:
        axes = [axes]

    for ax, (role, person_label) in zip(axes, therm_keys):
        for d_idx, label in enumerate(labels):
            vals, errs = [], []
            for party in all_parties:
                v, e = _lookup(dfs[label], "thermometer", role, party, "rating_mean", "rating_std")
                vals.append(v)
                errs.append(e)
            color = condition_colors.get(label, "#888888")
            positions = [c - group_width / 2 + bar_width * (d_idx + 0.5) for c in group_centers]
            ax.bar(positions, vals, width=bar_width * 0.9, yerr=errs, color=color,
                   capsize=2, error_kw={"elinewidth": 0.8, "capthick": 0.8})

        ax.set_title(f"Feeling thermometer:\n{person_label}\n(obfuscated per condition)",
                     fontweight='medium', pad=8)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(all_parties, rotation=15)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#333333')
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean rating (0-100)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=condition_colors.get(l, "#888888")) for l in labels]
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout(pad=1.2, rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")

    for label, df in dfs.items():
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        for role, person_label in therm_keys:
            print(f"\n  Feeling thermometer: {person_label}")
            for party in all_parties:
                row = df[(df["metric"] == "thermometer") & (df["key"] == role) & (df["party"] == party)]
                if row.empty or pd.isna(row.iloc[0]["rating_mean"]):
                    continue
                r = row.iloc[0]
                print(f"    {party:<30} {r['rating_mean']:5.1f}  (avg_n={r['avg_n']:.1f}, n_runs={int(r['n_runs'])})")


def plot_affective_polarization_comparison(
    dfs: dict[str, pd.DataFrame], condition_colors: dict[str, str], output_path: str
) -> None:
    labels  = list(dfs.keys())
    parties = [p for p in PARTY_THERM_ROLES
               if any((df["party"] == p).any() for df in dfs.values())]
    if not parties:
        print("No partisan respondents with party feeling-thermometer ratings found — skipping affective polarization plot.")
        return

    def polarization(df: pd.DataFrame, party: str) -> tuple[float, float]:
        own_role, opp_role = PARTY_THERM_ROLES[party]
        own_val, own_err = _lookup(df, "thermometer", own_role, party, "rating_mean", "rating_std")
        opp_val, opp_err = _lookup(df, "thermometer", opp_role, party, "rating_mean", "rating_std")
        if math.isnan(own_val) or math.isnan(opp_val):
            return float("nan"), float("nan")
        # Own/opposing ratings come from the same respondents, but per-run values
        # aren't available in this aggregated format, so their errors are combined
        # assuming independence (a reasonable approximation, not exact).
        own_err = own_err if not math.isnan(own_err) else 0.0
        opp_err = opp_err if not math.isnan(opp_err) else 0.0
        return own_val - opp_val, math.hypot(own_err, opp_err)

    n_parties  = len(parties)
    n_datasets = len(labels)

    group_width = 0.8
    bar_width   = group_width / n_datasets
    group_centers = list(range(n_parties))

    fig, ax = plt.subplots(figsize=(4 * n_parties, 4.5))

    for d_idx, label in enumerate(labels):
        vals, errs = [], []
        for party in parties:
            v, e = polarization(dfs[label], party)
            vals.append(v)
            errs.append(e)
        color = condition_colors.get(label, "#888888")
        positions = [c - group_width / 2 + bar_width * (d_idx + 0.5) for c in group_centers]
        ax.bar(positions, vals, width=bar_width * 0.9, yerr=errs, color=color,
               capsize=2, error_kw={"elinewidth": 0.8, "capthick": 0.8})

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Affective polarization\n(own-party rating minus opposing-party rating)",
                 fontweight='medium', pad=8)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(parties, rotation=15)
    ax.set_ylabel("Mean rating difference (own − opposing)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=condition_colors.get(l, "#888888")) for l in labels]
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout(pad=1.2, rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")

    for label, df in dfs.items():
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print("\n  Affective polarization (own-party minus opposing-party rating)")
        for party in parties:
            v, _ = polarization(df, party)
            if math.isnan(v):
                continue
            print(f"    {party:<30} {v:6.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a batch of obfuscation-comparison persona-interview runs from wandb "
                     "and plot yes/no + feeling-thermometer results by party."
    )
    parser.add_argument("--batch_id", type=str, required=True,
                         help="Wandb group/batch id shared by the runs to compare "
                              "(printed by run_persona_obf_pipeline.py after it finishes).")
    parser.add_argument("--wandb_project", type=str, default=interview_wandb.WANDB_PROJECT,
                         help="Wandb project the runs were logged to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dfs, population = fetch_condition_dfs(args.batch_id, wandb_project=args.wandb_project)

    print_population_table(population)

    follow_keys = ['q1_dem', 'q1_rep', 'q2', 'q3', 'q4', 'q5']
    trait_keys  = ([f"dem_{key}" for key, _ in TRAIT_QUESTIONS]
                  + [f"rep_{key}" for key, _ in TRAIT_QUESTIONS])

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

    dont_know_present = [
        (k, question_labels.get(k, question_texts.get(k, k)) + "\n(don't know)")
        for k in trait_keys
        if all(((df["metric"] == "question") & (df["key"] == k)).any() for df in dfs.values())
    ]

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
                            ncols=len(follow_present) or 1)
    # Trait battery: one row for Democrats, one for Republicans, columns aligned by trait.
    plot_metric_comparison(dfs, trait_present, all_parties, condition_colors,
                            fig_path("interview_results_obfuscation_traits", args.batch_id), "question", "pct_yes_mean", "pct_yes_std",
                            ncols=len(TRAIT_QUESTIONS))
    # Trait "dont_know" rate: makes forced-choice artifacts (non-partisans with no
    # basis to judge an obfuscated group) visible instead of them defaulting to "No".
    plot_metric_comparison(dfs, dont_know_present, all_parties, condition_colors,
                            fig_path("interview_results_obfuscation_traits_dont_know", args.batch_id), "question", "pct_dont_know_mean", "pct_dont_know_std",
                            ncols=len(TRAIT_QUESTIONS), ylabel="Fraction answering \"don't know\"")

    print(f"\n{'='*60}")
    print("  Question answers (rows = obfuscation condition, columns = party)")
    print(f"{'='*60}")
    print_question_tables(dfs, all_keys, all_parties, question_labels, question_texts, trait_keys)

    plot_thermometer_comparison(dfs, all_parties, condition_colors,
                                 fig_path("interview_results_obfuscation_thermometer", args.batch_id))
    plot_affective_polarization_comparison(dfs, condition_colors,
                                            fig_path("interview_results_obfuscation_affective_polarization", args.batch_id))


if __name__ == "__main__":
    main()
