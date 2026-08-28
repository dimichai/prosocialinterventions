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
    ("q1", "Would you follow someone who is a member of the opposing political party?"),
    ("q2", "Would you follow someone who loves Donald Trump?"),
    ("q3", "Would you follow someone who hates Donald Trump?"),
    ("q4", "Would you follow someone who loves Joe Biden?"),
    ("q5", "Would you follow someone who hates Joe Biden?"),
    *[(f"dem_{key}", f"Do you think Democrats are {trait}?") for key, trait in TRAIT_QUESTIONS],
    *[(f"rep_{key}", f"Do you think Republicans are {trait}?") for key, trait in TRAIT_QUESTIONS],
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
COMPARISON_FILES = {
    "Full Persona":    os.path.join(RESULTS_DIR, "persona_interview_results_personas.csv"),
    "No AP":           os.path.join(RESULTS_DIR, "persona_interview_results_20260121_personas_with_bio_2000_noLoveHate_.csv"),
    "No AP & PID":     os.path.join(RESULTS_DIR, "persona_interview_results_20260123_personas_with_bio_2000_noLoveHate_noPartyId_.csv"),
    "No AP, PID, & VB": os.path.join(RESULTS_DIR, "persona_interview_results_20260227_personas_with_bio_2000_noLoveHate_noPartyId_noVoted2020_.csv"),
}
COMPARISON_OUTPUT_SLOPE = os.path.join(os.path.dirname(__file__), "figs", "interview_results.pdf")
COMPARISON_OUTPUT_TRAITS = os.path.join(os.path.dirname(__file__), "figs", "interview_results_traits.pdf")
COMPARISON_OUTPUT_TRAITS_DONTKNOW = os.path.join(os.path.dirname(__file__), "figs", "interview_results_traits_dont_know.pdf")


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for k, _ in QUESTIONS:
        col = f"{k}_answer"
        if col in df.columns:
            normalized = df[col].astype(str).str.strip().str.lower()
            # Trait questions allow a "dont_know" answer (see persona_interviews.py).
            # Map it (and any error rows) to NaN rather than 0, so it doesn't get
            # silently counted as "No" below.
            df[f"{col}_dont_know"] = normalized.eq("dont_know")
            df[col] = normalized.map({"true": 1.0, "false": 0.0})
    return df


def plot_slope_comparison(
    dfs: dict[str, pd.DataFrame],
    cols: list[tuple[str, str]],
    all_parties: list[str],
    party_colors: dict[str, str],
    labels: list[str],
    output_path: str,
    right_nudge: dict[str, dict[str, float]] | None = None,
    ncols: int | None = None,
    ylabel: str = "Fraction answering Yes",
) -> None:
    """Slope chart (one line per party, x-axis = ablation condition), one panel per question.

    Panels wrap onto multiple rows (ncols per row) so this scales from a
    handful of questions up to a full trait battery without one absurdly wide row.

    `col` values may contain NaN (e.g. trait "dont_know" answers, recoded to NaN
    in load_and_prepare) — those rows are excluded from both the fraction and its
    denominator `n`, rather than being counted as "No".
    """
    n_questions = len(cols)
    if n_questions == 0:
        print(f"No columns to plot for {output_path} — skipping.")
        return

    right_nudge = right_nudge or {}
    n_datasets  = len(labels)
    x_ticks     = list(range(n_datasets))
    right_margin = 1.6  # accommodates party labels drawn to the right of the last point

    ncols = ncols or n_questions
    nrows = -(-n_questions // ncols)  # ceil division

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 4 * nrows), sharey=True, squeeze=False)
    flat_axes = list(axes.flat)

    for ax, (col, title) in zip(flat_axes, cols):
        r_nudges = right_nudge.get(col, {})
        for party in all_parties:
            vals, errs = [], []
            for label in labels:
                subset = dfs[label][dfs[label]["party"] == party][col]
                n = int(subset.notna().sum())
                if party in dfs[label]["party"].values and n > 0:
                    p = subset.mean()
                    vals.append(p)
                    errs.append(1.96 * (p * (1 - p) / n) ** 0.5)
                else:
                    vals.append(float("nan"))
                    errs.append(float("nan"))
            color = party_colors.get(party, "#888888")
            ax.errorbar(x_ticks, vals, yerr=errs, marker="o", color=color,
                        linewidth=1.5, markersize=4, solid_capstyle="round",
                        clip_on=False, capsize=2, capthick=0.8, elinewidth=0.8)
            if not pd.isna(vals[-1]):
                ax.text(n_datasets - 1 + 0.12, vals[-1] + r_nudges.get(party, 0), party,
                        ha="left", va="center", color=color)

        ax.set_title(title, fontweight='medium', pad=8, fontsize=10)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_xlim(-0.4, n_datasets - 1 + right_margin)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#333333')
        ax.set_axisbelow(True)

    for ax in flat_axes[n_questions:]:
        ax.axis("off")
    for row in range(nrows):
        axes[row, 0].set_ylabel(ylabel)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")


def main() -> None:
    dfs = {label: load_and_prepare(path) for label, path in COMPARISON_FILES.items()}

    follow_cols = ['q1_answer', 'q2_answer', 'q3_answer', 'q4_answer', 'q5_answer']
    trait_cols  = ([f"dem_{key}_answer" for key, _ in TRAIT_QUESTIONS]
                   + [f"rep_{key}_answer" for key, _ in TRAIT_QUESTIONS])

    question_labels = {
        'q1': "Would you follow\nan opposing-party member?",
        'q2': "Would you follow\nsomeone who loves Trump?",
        'q3': "Would you follow\nsomeone who hates Trump?",
        'q4': "Would you follow\nsomeone who loves Biden?",
        'q5': "Would you follow\nsomeone who hates Biden?",
    }
    for key, trait in TRAIT_QUESTIONS:
        question_labels[f"dem_{key}"] = f"Democrats:\nare they {trait}?"
        question_labels[f"rep_{key}"] = f"Republicans:\nare they {trait}?"
    question_texts = {k: t for k, t in QUESTIONS}

    def cols_with_titles(cols: list[str]) -> list[tuple[str, str]]:
        # Only plot columns present in every comparison file, so lines aren't
        # silently missing for whichever condition lacks the data.
        present = [c for c in cols if all(c in df.columns for df in dfs.values())]
        return [
            (c, question_labels.get(c.replace('_answer', ''), question_texts.get(c.replace('_answer', ''), c)))
            for c in present
        ]

    follow_present = cols_with_titles(follow_cols)
    trait_present  = cols_with_titles(trait_cols)
    answer_cols    = [c for c, _ in follow_present + trait_present]

    dont_know_cols = [f"{c}_dont_know" for c in trait_cols]
    dont_know_present = [
        (c, question_labels.get(c.replace('_answer_dont_know', ''),
                                 question_texts.get(c.replace('_answer_dont_know', ''), c)) + "\n(don't know)")
        for c in dont_know_cols if all(c in df.columns for df in dfs.values())
    ]

    all_parties = sorted(set().union(*[set(df["party"].dropna().unique()) for df in dfs.values()]))
    labels      = list(dfs.keys())

    party_colors = {p: c for p, c in zip(all_parties, ["#03357D", "#888888", "#D50403", "#58508D", "#FFA600"])}

    # Per-panel nudges for rightmost label: {col: {party: y_offset}}
    right_nudge = {"q1_answer": {"Non-partisan": -0.05}}

    plot_slope_comparison(dfs, follow_present, all_parties, party_colors, labels,
                           COMPARISON_OUTPUT_SLOPE, right_nudge=right_nudge, ncols=len(follow_present) or 1)
    # Trait battery: one row for Democrats, one for Republicans, columns aligned by trait.
    plot_slope_comparison(dfs, trait_present, all_parties, party_colors, labels,
                           COMPARISON_OUTPUT_TRAITS, ncols=len(TRAIT_QUESTIONS))
    # Trait "dont_know" rate: makes forced-choice artifacts (respondents with no
    # basis to judge a party) visible instead of them defaulting to "No".
    plot_slope_comparison(dfs, dont_know_present, all_parties, party_colors, labels,
                           COMPARISON_OUTPUT_TRAITS_DONTKNOW, ncols=len(TRAIT_QUESTIONS),
                           ylabel="Fraction answering \"don't know\"")

    for label, df in dfs.items():
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        for col in answer_cols:
            key = col.replace("_answer", "")
            q_text = question_labels.get(key, question_texts.get(key, col)).replace("\n", " ")
            print(f"\n  {q_text}")
            dont_know_col = f"{col}_dont_know"
            for party in all_parties:
                subset = df[df["party"] == party]
                if len(subset) == 0:
                    continue
                n_decided = int(subset[col].notna().sum())
                pct_str = f"{subset[col].mean() * 100:5.1f}%" if n_decided > 0 else "  n/a"
                line = f"    {party:<30} {pct_str}  (n={n_decided}"
                if col in trait_cols and dont_know_col in df.columns:
                    line += f", dont_know={subset[dont_know_col].mean() * 100:4.1f}%"
                print(line + ")")


if __name__ == "__main__":
    main()
