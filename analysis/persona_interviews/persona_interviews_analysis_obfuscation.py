import os

import matplotlib.pyplot as plt
import pandas as pd

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
    ("q1", "On a social network, would you follow someone who is a member of the opposing political party?"),
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

# Own-party / opposing-party feeling-thermometer columns used to compute
# affective polarization for each partisan group.
PARTY_THERM_COLS = {
    "Democrat":   ("democrats_therm_rating", "republicans_therm_rating"),
    "Republican": ("republicans_therm_rating", "democrats_therm_rating"),
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
COMPARISON_FILES = {
    "No Obfuscation": os.path.join(RESULTS_DIR, "persona_interview_results_20260720_personas_with_bio_2000_noExtendWithAi__sample10.csv"),
    "Neutral":         os.path.join(RESULTS_DIR, "persona_interview_results_20260720_personas_with_bio_2000_noExtendWithAi_obfNeutral__sample10.csv"),
    "Nonce":           os.path.join(RESULTS_DIR, "persona_interview_results_20260720_personas_with_bio_2000_noExtendWithAi_obfNonce__sample10.csv"),
}
COMPARISON_OUTPUT_BARS = os.path.join(os.path.dirname(__file__), "figs", "interview_results_obfuscation.pdf")
COMPARISON_OUTPUT_TRAITS = os.path.join(os.path.dirname(__file__), "figs", "interview_results_obfuscation_traits.pdf")
COMPARISON_OUTPUT_THERM = os.path.join(os.path.dirname(__file__), "figs", "interview_results_obfuscation_thermometer.pdf")
COMPARISON_OUTPUT_POLARIZATION = os.path.join(os.path.dirname(__file__), "figs", "interview_results_obfuscation_affective_polarization.pdf")


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for k, _ in QUESTIONS:
        col = f"{k}_answer"
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().eq("true").astype(int)

    # Affective polarization: rating given to the respondent's own party minus
    # the rating given to the opposing party (only defined for partisans).
    df["affective_polarization"] = float("nan")
    for party, (own_col, opp_col) in PARTY_THERM_COLS.items():
        if own_col in df.columns and opp_col in df.columns:
            mask = df["party"] == party
            df.loc[mask, "affective_polarization"] = df.loc[mask, own_col] - df.loc[mask, opp_col]

    return df


def plot_answer_comparison(
    dfs: dict[str, pd.DataFrame],
    cols: list[tuple[str, str]],
    all_parties: list[str],
    condition_colors: dict[str, str],
    output_path: str,
    ncols: int | None = None,
) -> None:
    """Grouped bar chart of fraction-answering-Yes per party, one panel per question.

    Panels wrap onto multiple rows (ncols per row) so this scales from a
    handful of questions up to a full trait battery without one absurdly wide row.
    """
    n_questions = len(cols)
    if n_questions == 0:
        print(f"No columns to plot for {output_path} — skipping.")
        return

    labels     = list(dfs.keys())
    n_datasets = len(labels)
    n_parties  = len(all_parties)

    ncols = ncols or n_questions
    nrows = -(-n_questions // ncols)  # ceil division

    group_width = 0.8
    bar_width   = group_width / n_datasets
    group_centers = list(range(n_parties))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows), sharey=True, squeeze=False)
    flat_axes = list(axes.flat)

    for ax, (col, title) in zip(flat_axes, cols):
        for d_idx, label in enumerate(labels):
            vals, errs = [], []
            for party in all_parties:
                subset = dfs[label][dfs[label]["party"] == party][col]
                if party in dfs[label]["party"].values and len(subset) > 0:
                    p = subset.mean()
                    n = len(subset)
                    vals.append(p)
                    errs.append(1.96 * (p * (1 - p) / n) ** 0.5)
                else:
                    vals.append(float("nan"))
                    errs.append(float("nan"))
            color = condition_colors.get(label, "#888888")
            positions = [c - group_width / 2 + bar_width * (d_idx + 0.5) for c in group_centers]
            ax.bar(positions, vals, width=bar_width * 0.9, yerr=errs, color=color,
                   capsize=2, error_kw={"elinewidth": 0.8, "capthick": 0.8})

        ax.set_title(title, fontweight='medium', pad=8, fontsize=10)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(all_parties, rotation=15)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#333333')
        ax.set_axisbelow(True)

    for ax in flat_axes[n_questions:]:
        ax.axis("off")
    for row in range(nrows):
        axes[row, 0].set_ylabel("Fraction answering Yes")

    handles = [plt.Rectangle((0, 0), 1, 1, color=condition_colors.get(l, "#888888")) for l in labels]
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout(pad=1.2, rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")


def plot_thermometer_comparison(dfs: dict[str, pd.DataFrame], all_parties: list[str], condition_colors: dict[str, str]) -> None:
    labels     = list(dfs.keys())
    therm_cols = [(f"{role}_therm_rating", label) for role, label in THERMOMETER_TARGETS
                  if all(f"{role}_therm_rating" in df.columns for df in dfs.values())]
    if not therm_cols:
        print("No feeling-thermometer columns found in the comparison CSVs — skipping thermometer comparison plot.")
        return

    n_questions = len(therm_cols)
    n_datasets  = len(labels)
    n_parties   = len(all_parties)

    group_width = 0.8
    bar_width   = group_width / n_datasets
    group_centers = list(range(n_parties))

    fig, axes = plt.subplots(1, n_questions, figsize=(4 * n_questions, 4.5), sharey=True)
    if n_questions == 1:
        axes = [axes]

    for ax, (col, person_label) in zip(axes, therm_cols):
        for d_idx, label in enumerate(labels):
            vals, errs = [], []
            for party in all_parties:
                subset = dfs[label].loc[dfs[label]["party"] == party, col].dropna()
                if len(subset) > 0:
                    vals.append(subset.mean())
                    errs.append(1.96 * subset.std(ddof=1) / len(subset) ** 0.5)
                else:
                    vals.append(float("nan"))
                    errs.append(float("nan"))
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
    fig.savefig(COMPARISON_OUTPUT_THERM, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(COMPARISON_OUTPUT_THERM.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {COMPARISON_OUTPUT_THERM}")

    for label, df in dfs.items():
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        for col, person_label in therm_cols:
            print(f"\n  Feeling thermometer: {person_label}")
            for party in all_parties:
                subset = df.loc[df["party"] == party, col].dropna()
                if len(subset) == 0:
                    continue
                print(f"    {party:<30} {subset.mean():5.1f}  (n={len(subset)})")


def plot_affective_polarization_comparison(dfs: dict[str, pd.DataFrame], condition_colors: dict[str, str]) -> None:
    labels  = list(dfs.keys())
    parties = [p for p in PARTY_THERM_COLS
               if any((df["party"] == p).any() for df in dfs.values())]
    if not parties:
        print("No partisan respondents with party feeling-thermometer ratings found — skipping affective polarization plot.")
        return

    n_parties  = len(parties)
    n_datasets = len(labels)

    group_width = 0.8
    bar_width   = group_width / n_datasets
    group_centers = list(range(n_parties))

    fig, ax = plt.subplots(figsize=(4 * n_parties, 4.5))

    for d_idx, label in enumerate(labels):
        vals, errs = [], []
        for party in parties:
            subset = dfs[label].loc[dfs[label]["party"] == party, "affective_polarization"].dropna()
            if len(subset) > 0:
                vals.append(subset.mean())
                errs.append(1.96 * subset.std(ddof=1) / len(subset) ** 0.5)
            else:
                vals.append(float("nan"))
                errs.append(float("nan"))
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
    fig.savefig(COMPARISON_OUTPUT_POLARIZATION, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(COMPARISON_OUTPUT_POLARIZATION.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {COMPARISON_OUTPUT_POLARIZATION}")

    for label, df in dfs.items():
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print("\n  Affective polarization (own-party minus opposing-party rating)")
        for party in parties:
            subset = df.loc[df["party"] == party, "affective_polarization"].dropna()
            if len(subset) == 0:
                continue
            print(f"    {party:<30} {subset.mean():6.1f}  (n={len(subset)})")


def main() -> None:
    dfs = {label: load_and_prepare(path) for label, path in COMPARISON_FILES.items()}

    follow_cols = ['q1_answer', 'q2_answer', 'q3_answer', 'q4_answer', 'q5_answer']
    trait_cols  = ([f"dem_{key}_answer" for key, _ in TRAIT_QUESTIONS]
                   + [f"rep_{key}_answer" for key, _ in TRAIT_QUESTIONS])

    question_labels = {
        'q1': "Would you follow\nan opposing-party member?",
        'q2': "Would you follow\nsomeone who loves Trump\n(obfuscated per condition)?",
        'q3': "Would you follow\nsomeone who hates Trump\n(obfuscated per condition)?",
        'q4': "Would you follow\nsomeone who loves Biden\n(obfuscated per condition)?",
        'q5': "Would you follow\nsomeone who hates Biden\n(obfuscated per condition)?",
    }
    for key, trait in TRAIT_QUESTIONS:
        question_labels[f"dem_{key}"] = f"Democrats:\nare they {trait}?\n(obfuscated per condition)"
        question_labels[f"rep_{key}"] = f"Republicans:\nare they {trait}?\n(obfuscated per condition)"
    question_texts = {k: t for k, t in QUESTIONS}

    def cols_with_titles(cols: list[str]) -> list[tuple[str, str]]:
        # Only plot columns present in every comparison file, so bars aren't
        # silently missing for whichever condition lacks the data.
        present = [c for c in cols if all(c in df.columns for df in dfs.values())]
        return [
            (c, question_labels.get(c.replace('_answer', ''), question_texts.get(c.replace('_answer', ''), c)))
            for c in present
        ]

    follow_present = cols_with_titles(follow_cols)
    trait_present  = cols_with_titles(trait_cols)
    answer_cols    = [c for c, _ in follow_present + trait_present]

    all_parties = sorted(set().union(*[set(df["party"].dropna().unique()) for df in dfs.values()]))
    labels      = list(dfs.keys())

    condition_colors = {l: c for l, c in zip(labels, ["#03357D", "#888888", "#D50403", "#58508D", "#FFA600"])}

    # Obfuscation conditions are distinct schemes, not a progressive/additive series
    # (unlike the ablation comparison), so a grouped bar chart per condition
    # is the honest comparison — a connecting line would imply an ordering that isn't there.
    plot_answer_comparison(dfs, follow_present, all_parties, condition_colors,
                            COMPARISON_OUTPUT_BARS, ncols=len(follow_present) or 1)
    # Trait battery: one row for Democrats, one for Republicans, columns aligned by trait.
    plot_answer_comparison(dfs, trait_present, all_parties, condition_colors,
                            COMPARISON_OUTPUT_TRAITS, ncols=len(TRAIT_QUESTIONS))

    for label, df in dfs.items():
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        for col in answer_cols:
            key = col.replace("_answer", "")
            q_text = question_labels.get(key, question_texts.get(key, col)).replace("\n", " ")
            print(f"\n  {q_text}")
            for party in all_parties:
                subset = df[df["party"] == party]
                if len(subset) == 0:
                    continue
                pct = subset[col].mean() * 100
                print(f"    {party:<30} {pct:5.1f}%  (n={len(subset)})")

    plot_thermometer_comparison(dfs, all_parties, condition_colors)
    plot_affective_polarization_comparison(dfs, condition_colors)


if __name__ == "__main__":
    main()
