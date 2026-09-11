"""Shared plotting/table infrastructure for comparing aggregated interview results
across conditions (obfuscation or ablation) — everything that's condition-agnostic,
used by both persona_interviews_analysis_obfuscation.py and
persona_interviews_analysis_ablation.py. Each of those scripts supplies only its own
`fetch_condition_dfs` (how conditions are bucketed/labeled) and calls into the
plotting functions here.
"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd

import interview_wandb  # noqa: E402

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


# (key_suffix, trait) — asked once for Democrats and once for Republicans.
# Split by polarity (see TRAIT_GAP_ROLE_MAPS/TRAIT_DONT_KNOW_ROLES below) so a
# trait's positive/negative classification lives in exactly one place.
POSITIVE_TRAIT_QUESTIONS = [
    ("intelligent",   "intelligent"),
    ("honest",        "honest"),
    ("generous",      "generous"),
    ("openminded",    "open-minded"),
]
NEGATIVE_TRAIT_QUESTIONS = [
    ("hypocritical",  "hypocritical"),
    ("selfish",       "selfish"),
    ("mean",          "mean"),
]
TRAIT_QUESTIONS = POSITIVE_TRAIT_QUESTIONS + NEGATIVE_TRAIT_QUESTIONS

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

# Own-party / opposing-party thermometer roles (see the "thermometer" rows'
# `key` column — biden/trump/democrats/republicans) used to compute affective
# polarization for each partisan group.
PARTY_THERM_ROLES = {
    "Democrat":   ("democrats", "republicans"),
    "Republican": ("republicans", "democrats"),
}

# Same idea as PARTY_THERM_ROLES, but keyed on each party's leader instead of
# the party label itself — own-leader rating minus other-leader rating.
LEADER_THERM_ROLES = {
    "Democrat":   ("biden", "trump"),
    "Republican": ("trump", "biden"),
}

# Panel title -> role_map for the thermometer-gap slope chart (see
# plot_gap_slope_comparison / _thermometer_gap_fn). Each role_map value is
# (own_roles, opp_roles), tuples-of-roles rather than a single role so
# "Combined" can average across both the party and candidate thermometers —
# built from PARTY_THERM_ROLES / LEADER_THERM_ROLES so the three panels stay
# in sync with those.
GAP_ROLE_MAPS: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "Party": {p: ((own,), (opp,)) for p, (own, opp) in PARTY_THERM_ROLES.items()},
    "Candidate": {p: ((own,), (opp,)) for p, (own, opp) in LEADER_THERM_ROLES.items()},
    "Combined": {
        p: ((PARTY_THERM_ROLES[p][0], LEADER_THERM_ROLES[p][0]),
            (PARTY_THERM_ROLES[p][1], LEADER_THERM_ROLES[p][1]))
        for p in PARTY_THERM_ROLES
    },
}

# Panel title -> role_map for the follow-gap slope chart (see
# plot_gap_slope_comparison, with metric="question", value_col="pct_yes_mean",
# std_col="pct_yes_std"). "Party" is Pr(follow in-party) - Pr(follow
# out-party), own/opp flipping with the respondent's party like
# PARTY_THERM_ROLES (q1_dem/q1_rep, see QUESTIONS). "Biden"/"Trump" are the
# analogous gap between willingness to follow someone who loves vs. hates
# that candidate (q4/q5, q2/q3) — fixed rather than party-flipped, since
# loving/hating a candidate isn't itself party-dependent.
FOLLOW_GAP_ROLE_MAPS: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "Party": {
        "Democrat":   (("q1_dem",), ("q1_rep",)),
        "Republican": (("q1_rep",), ("q1_dem",)),
    },
    "Biden": {
        "Democrat":   (("q4",), ("q5",)),
        "Republican": (("q4",), ("q5",)),
    },
    "Trump": {
        "Democrat":   (("q2",), ("q3",)),
        "Republican": (("q2",), ("q3",)),
    },
}


def trait_gap_role_map(trait_questions: list[tuple[str, str]]) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Build one TRAIT_GAP_ROLE_MAPS-shaped role map (own/opp trait keys per
    party) from an arbitrary trait-question list — the constructor behind
    TRAIT_GAP_ROLE_MAPS itself, exposed so callers can build a variant over a
    different trait subset (e.g. a polarity's battery with one trait
    excluded, for a robustness check) without duplicating the dem_/rep_
    prefixing logic."""
    keys = [k for k, _ in trait_questions]
    return {
        "Democrat":   (tuple(f"dem_{k}" for k in keys), tuple(f"rep_{k}" for k in keys)),
        "Republican": (tuple(f"rep_{k}" for k in keys), tuple(f"dem_{k}" for k in keys)),
    }


def trait_dont_know_roles(trait_questions: list[tuple[str, str]]) -> tuple[str, ...]:
    """Trait keys (both dem_/rep_ targets) for an arbitrary trait-question
    list — the constructor behind TRAIT_DONT_KNOW_ROLES, exposed for the same
    reason as trait_gap_role_map."""
    return tuple(f"{prefix}_{k}" for k, _ in trait_questions for prefix in ("dem", "rep"))


# Positive/negative trait-differential role maps — same idea as
# GAP_ROLE_MAPS, but the "own"/"opposing" ratings being differenced are LLM
# yes-rates on the trait battery (POSITIVE_TRAIT_QUESTIONS/
# NEGATIVE_TRAIT_QUESTIONS) rather than feeling-thermometer scores: the share
# of traits of that polarity a party's respondents attribute to their own
# party minus the share they attribute to the opposing party.
TRAIT_GAP_ROLE_MAPS: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "Positive traits": trait_gap_role_map(POSITIVE_TRAIT_QUESTIONS),
    "Negative traits": trait_gap_role_map(NEGATIVE_TRAIT_QUESTIONS),
}

# Trait keys (both dem_/rep_ targets) of each polarity — the question set a
# trait-differential panel's don't-know rate is averaged across (see
# _role_average). Party-independent since the same questions are asked of
# every respondent regardless of their own party.
TRAIT_DONT_KNOW_ROLES: dict[str, tuple[str, ...]] = {
    "Positive traits": trait_dont_know_roles(POSITIVE_TRAIT_QUESTIONS),
    "Negative traits": trait_dont_know_roles(NEGATIVE_TRAIT_QUESTIONS),
}

PARTY_COLORS = {"Democrat": "#03357D", "Non-partisan": "#888888", "Republican": "#D50403"}
_PARTY_COLOR_FALLBACK = ["#58508D", "#FFA600", "#2D7D2D", "#eda100"]


def party_color_map(all_parties: list[str]) -> dict[str, str]:
    """Party -> line color for slope charts, matching the old ablation-script
    palette (blue Democrat / red Republican / gray Non-partisan); any other
    party label falls back to a fixed extra palette."""
    colors, i = {}, 0
    for p in all_parties:
        if p in PARTY_COLORS:
            colors[p] = PARTY_COLORS[p]
        else:
            colors[p] = _PARTY_COLOR_FALLBACK[i % len(_PARTY_COLOR_FALLBACK)]
            i += 1
    return colors


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


def aggregate_population_metrics(raw_dfs: list[pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """Mean and 95%-CI half-width (`1.96 * std / sqrt(n_runs)`, across the seeds
    within one condition) of interview_wandb.persona_population_metrics — the same
    persona-attribute distribution that gets logged to wandb under the "personas/"
    prefix at run time, recomputed here from the downloaded raw per-persona results
    so it can be printed alongside the question-answer tables."""
    per_run = [interview_wandb.persona_population_metrics(df) for df in raw_dfs]
    keys = set().union(*[m.keys() for m in per_run])
    result = {}
    for key in keys:
        values = pd.Series([m.get(key, float("nan")) for m in per_run], dtype=float)
        n = int(values.notna().sum())
        mean = values.mean()
        err = 1.96 * values.std() / math.sqrt(n) if n > 1 else float("nan")
        result[key] = (mean, err)
    return result


def aggregate_ground_truth_thermometer(raw_dfs: list[pd.DataFrame]) -> dict[str, dict[str, tuple[float, float]]]:
    """Mean and 95%-CI half-width (`1.96 * std / sqrt(n_seeds)`, same convention
    as aggregate_population_metrics/_lookup) of
    interview_wandb.ground_truth_thermometer_by_party across seeds. Each seed
    draws an independent resample of ANES respondents (see
    anes_generate_personas.py::get_anes_rows, `df1.sample(..., replace=True,
    random_state=seed)`), so the ground-truth mean itself carries seed-to-seed
    sampling variance, same as the LLM-answer lines it's compared against.
    Ground truth is identical across ablation conditions sharing a seed (a
    property of the real respondent, unaffected by what's redacted from their
    LLM persona), so callers only need to compute this once per batch, from
    any one condition's raw_dfs (see fetch_condition_dfs) — as long as every
    condition in the batch was run with the same seed(s)."""
    per_seed = [interview_wandb.ground_truth_thermometer_by_party(df) for df in raw_dfs]
    roles = set().union(*[m.keys() for m in per_seed])
    result = {}
    for role in roles:
        parties = set().union(*[m.get(role, {}).keys() for m in per_seed])
        result[role] = {}
        for party in parties:
            values = pd.Series([m.get(role, {}).get(party, float("nan")) for m in per_seed], dtype=float)
            n = int(values.notna().sum())
            mean = values.mean()
            err = 1.96 * values.std() / math.sqrt(n) if n > 1 else float("nan")
            result[role][party] = (mean, err)
    return result


def print_population_table(population: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Persona population statistics per condition (rows = condition), i.e. the
    attributes of the personas being interviewed, before any LLM answers — the same
    stats logged to wandb under the "personas/" prefix. Each cell is mean ± 95%-CI
    half-width across seeds (see `aggregate_population_metrics`)."""
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
            v, err = stats.get(m, (float("nan"), float("nan")))
            row[header] = _fmt_ci(v, err, fmt)
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
    """One table per question, rows = condition, columns = party identity — lines
    up every condition's answer to the same question side by side, for every party,
    instead of splitting them across per-condition blocks. Cells show mean ± 95%-CI
    half-width (the same CI drawn as error bars in the plots, see `_lookup`)."""
    for key in keys:
        rows = {}
        for label, df in dfs.items():
            subset = df[(df["metric"] == "question") & (df["key"] == key)]
            if subset.empty:
                continue
            row = {}
            for party in all_parties:
                v, err = _lookup(df, "question", key, party, "pct_yes_mean", "pct_yes_std")
                cell = _fmt_ci(v * 100 if pd.notna(v) else v, err * 100 if pd.notna(err) else err, "{:.1f}%")
                if key in trait_keys and pd.notna(v):
                    dk_v, dk_err = _lookup(df, "question", key, party, "pct_dont_know_mean", "pct_dont_know_std")
                    dk_cell = _fmt_ci(dk_v * 100 if pd.notna(dk_v) else dk_v,
                                       dk_err * 100 if pd.notna(dk_err) else dk_err, "{:.1f}%")
                    cell += f" (dk {dk_cell})"
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


def _fmt_ci(value: float, err: float, fmt: str = "{:.1f}") -> str:
    """Format a (value, 95%-CI half-width) pair as `mean ± CI` — the single
    style shared by every printed table in this script (population stats,
    question/trait tables, thermometer, affective polarization). Falls back to
    a bare mean if no CI is available, or "N/A" if the value itself is missing
    — always this exact string, so every missing value in every table/chart
    reads the same way."""
    if value is None or pd.isna(value):
        return "N/A"
    s = fmt.format(value)
    if err is not None and pd.notna(err):
        s += f" ± {fmt.format(err)}"
    return s


def _print_comparison_table(
    title: str, labels: list[str], columns: list[str], value_fn,
    secondary_fn=None, secondary_prefix: str = "dk", secondary_fmt: str = "{:.1%}",
) -> None:
    """Print one table: rows = condition (`labels`), columns = `columns` (e.g.
    party), cell = `_fmt_ci(*value_fn(label, column))`. Mirrors the row/column
    layout `print_question_tables` uses, so every printed result set — questions,
    thermometer, affective polarization — reads the same way instead of some
    being one table and others per-condition blocks.

    `secondary_fn`, if given, appends a "(prefix v ± e)" suffix to every cell —
    the same non-response rate (don't-know / not-recognized) that
    `print_question_tables` folds into its trait cells, kept in this one table
    rather than a separate one. Always appended, even when missing (as "N/A"),
    so every cell in a column is annotated the same way."""
    rows = {}
    for label in labels:
        row = {}
        for col in columns:
            cell = _fmt_ci(*value_fn(label, col))
            if secondary_fn is not None:
                sv, se = secondary_fn(label, col)
                sec_cell = _fmt_ci(sv, se, secondary_fmt)
                cell += f" ({secondary_prefix} {sec_cell})"
            row[col] = cell
        rows[label] = row
    table = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    print(f"\n  {title}")
    print(table.to_string().replace("\n", "\n  "))


def _draw_table_panel(
    fig: plt.Figure,
    outer_spec,
    labels: list[str],
    columns: list[str],
    condition_colors: dict[str, str],
    value_fn,
    xlim: tuple[float, float],
    title: str,
    value_fmt: str = "{:.0%}",
    secondary_fn=None,
    secondary_prefix: str = "dk",
    secondary_fmt: str = "{:.0%}",
) -> None:
    """Draw one "table" panel inside `outer_spec`: rows = condition (`labels`),
    columns = `columns` (e.g. party), each cell a single horizontal bar for
    `value_fn(label, column) -> (value, error)`. Row 0 of the inner grid is the
    panel title, row 1 the column (party) headers — both dedicated rows, rather
    than relying on matplotlib's floating axes-title padding, which overlaps
    neighboring rows once cells get short.

    `secondary_fn`, if given, is a non-response rate — don't-know for yes/no
    trait questions, not-recognized for thermometer targets — drawn as a small
    muted label pinned to each cell's top-right corner (fixed axes-fraction
    position, independent of the main bar's length) so it's always visible in
    this same panel instead of needing a separate comparison chart.
    """
    n_datasets = len(labels)
    n_cols = len(columns)
    title_lines = title.count("\n") + 1
    title_row_h = 0.7 * title_lines + 0.4
    header_row_h = 0.6
    inner = outer_spec.subgridspec(n_datasets + 2, n_cols, hspace=0.15, wspace=0.15,
                                    height_ratios=[title_row_h, header_row_h] + [1] * n_datasets)

    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.text(0.5, 0.05, title, ha="center", va="bottom", fontsize=10,
                  fontweight="medium", transform=title_ax.transAxes)

    for c_idx, column in enumerate(columns):
        header_ax = fig.add_subplot(inner[1, c_idx])
        header_ax.axis("off")
        header_ax.text(0.5, 0.1, str(column), ha="center", va="bottom",
                        fontsize=8.5, fontweight="medium", transform=header_ax.transAxes)

    zero_x = 0 if xlim[0] <= 0 <= xlim[1] else xlim[0]
    span = xlim[1] - xlim[0]

    for d_idx, label in enumerate(labels):
        color = condition_colors.get(label, "#888888")
        for c_idx, column in enumerate(columns):
            ax = fig.add_subplot(inner[d_idx + 2, c_idx])
            v, e = value_fn(label, column)
            if not (isinstance(v, float) and math.isnan(v)):
                err = 0.0 if (e is None or (isinstance(e, float) and math.isnan(e))) else e
                ax.barh(0, v, xerr=err or None, height=0.55, color=color, capsize=2,
                        error_kw={"elinewidth": 0.7, "capthick": 0.7}, zorder=3)
                # Label goes outside the bar/error-cap by default; but if that would
                # overflow this axes' xlim, an outside label bleeds into the next
                # column's (opaque) subplot and gets hidden behind it — so switch to
                # placing it inside the bar instead whenever it's too close to the edge.
                offset = span * 0.03
                if v >= 0:
                    if v + err > xlim[1] - span * 0.12:
                        ax.text(v - offset, 0, value_fmt.format(v), va="center",
                                ha="right", fontsize=6.5, color="white", clip_on=True, zorder=4)
                    else:
                        ax.text(v + err + offset, 0, value_fmt.format(v), va="center", ha="left",
                                fontsize=6.5, color="#333333", clip_on=False)
                else:
                    if v - err < xlim[0] + span * 0.12:
                        ax.text(v + offset, 0, value_fmt.format(v), va="center",
                                ha="left", fontsize=6.5, color="white", clip_on=True, zorder=4)
                    else:
                        ax.text(v - err - offset, 0, value_fmt.format(v), va="center", ha="right",
                                fontsize=6.5, color="#333333", clip_on=False)
            else:
                ax.text((xlim[0] + xlim[1]) / 2, 0, "N/A", va="center", ha="center",
                        fontsize=6.5, color="#999999")
            if secondary_fn is not None:
                sv, se = secondary_fn(label, column)
                sv_text = "N/A" if (sv is None or (isinstance(sv, float) and math.isnan(sv))) else secondary_fmt.format(sv)
                ax.text(0.98, 0.92, f"{secondary_prefix} {sv_text}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=5.5, color="#999999", clip_on=False)
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.7, 0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.axvline(zero_x, color="#dddddd", linewidth=0.6, zorder=0)
            if c_idx == 0:
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, ha="right",
                        va="center", fontsize=7)


MAX_GRID_COLS = 4

# Row labels (condition names, drawn inside each panel by _draw_table_panel)
# already identify color/condition, so no legend is needed.
TOP_PAD = 0.96


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
    value_label: str = "Fraction answering Yes",
    xlim: tuple[float, float] = (0, 1),
    value_fmt: str = "{:.0%}",
    show_dont_know: bool = False,
) -> None:
    """Table of horizontal bars, one panel per question/target: within each panel,
    rows = condition, columns = party, cell = a single horizontal bar for that
    condition/party's value.

    Panels wrap onto multiple rows (ncols per row) so this scales from a
    handful of questions up to a full trait battery without one absurdly wide row.

    `show_dont_know`, when set (trait questions only), draws each cell's
    don't-know rate as a small label inside this same panel (see
    `_draw_table_panel`) instead of plotting it as a separate comparison chart.
    """
    n_panels = len(keys)
    if n_panels == 0:
        print(f"No columns to plot for {output_path} — skipping.")
        return

    labels     = list(dfs.keys())
    n_datasets = len(labels)
    n_parties  = len(all_parties)

    ncols = min(ncols or n_panels, MAX_GRID_COLS)
    nrows = -(-n_panels // ncols)  # ceil division

    panel_w = 1.1 * n_parties + 1.0
    panel_h = 0.4 * (n_datasets + 1) + 0.5
    fig = plt.figure(figsize=(panel_w * ncols, panel_h * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.25, wspace=0.4,
                              left=0.08, right=0.97, top=TOP_PAD, bottom=0.04)

    for idx, (key, title) in enumerate(keys):
        r, c = divmod(idx, ncols)

        def value_fn(label, party, _key=key):
            return _lookup(dfs[label], metric, _key, party, value_col, std_col)

        secondary_fn = None
        if show_dont_know:
            def secondary_fn(label, party, _key=key):
                return _lookup(dfs[label], metric, _key, party, "pct_dont_know_mean", "pct_dont_know_std")

        _draw_table_panel(fig, outer[r, c], labels, all_parties, condition_colors,
                           value_fn, xlim, title, value_fmt,
                           secondary_fn=secondary_fn, secondary_prefix="dk")

    fig.text(0.01, 0.5, value_label, va="center", rotation="vertical", fontsize=10, color="#555555")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")


def _declutter_positions(values: list[float], min_gap: float) -> list[float]:
    """Nudge `values` apart (preserving order) so every adjacent pair ends up
    at least `min_gap` apart, moving each as little as possible — used to
    keep line-end labels (e.g. party names) from overlapping when their
    final values are close together. Each iteration splits every too-close
    adjacent pair's deficit evenly between the two (a small 1D force
    relaxation); repeated until no pair violates min_gap, which also
    resolves longer runs of close values since a pair's push can itself
    create (and then resolve) a violation with its other neighbor. NaN
    entries pass through untouched and aren't counted as neighbors."""
    order = sorted((i for i in range(len(values)) if not math.isnan(values[i])), key=lambda i: values[i])
    positions = list(values)
    if len(order) < 2:
        return positions
    for _ in range(len(order) * 4):
        moved = False
        for a, b in zip(order, order[1:]):
            gap = positions[b] - positions[a]
            if gap < min_gap:
                deficit = (min_gap - gap) / 2
                positions[a] -= deficit
                positions[b] += deficit
                moved = True
        if not moved:
            break
    return positions


def _plot_slope_core(
    labels: list[str],
    keys: list[tuple],
    all_parties: list[str],
    party_colors: dict[str, str],
    output_path: str,
    value_fn,
    gt_fn,
    ncols: int | None,
    value_label: str,
    ylim: tuple[float, float],
) -> None:
    """Shared slope-chart drawing loop behind plot_slope_comparison and
    plot_gap_slope_comparison: one line per party (colored by party), x-axis =
    condition (`labels`), one panel per `keys` entry. `value_fn(label, key,
    party) -> (value, error)` supplies each point; `gt_fn(key, party) -> (value,
    error)` supplies the constant real-world reference line/band (skipped where
    it returns NaN) — the two callers differ only in how these are computed
    (a direct (metric,key,party) df lookup vs. a derived own-minus-opposing
    thermometer gap), everything else about the chart is identical."""
    n_panels = len(keys)
    if n_panels == 0:
        print(f"No columns to plot for {output_path} — skipping.")
        return

    n_datasets = len(labels)
    x_ticks = list(range(n_datasets))
    right_margin = 1.6  # room for the party label drawn past the last point

    ncols = min(ncols or n_panels, MAX_GRID_COLS)
    nrows = -(-n_panels // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 4 * nrows), sharey=True, squeeze=False)
    flat_axes = list(axes.flat)

    for ax, (key, title) in zip(flat_axes, keys):
        party_vals = {}
        for party in all_parties:
            vals, errs = [], []
            for label in labels:
                v, e = value_fn(label, key, party)
                vals.append(v)
                errs.append(e)
            party_vals[party] = (vals, errs)

        # Party labels are drawn past the last point at that point's y-value —
        # decluttered so two parties ending up close together (e.g. gap ~0)
        # don't render as overlapping text (see _declutter_positions).
        final_vals = [party_vals[p][0][-1] for p in all_parties]
        label_ys = _declutter_positions(final_vals, (ylim[1] - ylim[0]) * 0.06)

        for party, label_y in zip(all_parties, label_ys):
            vals, errs = party_vals[party]
            color = party_colors.get(party, "#888888")
            ax.errorbar(x_ticks, vals, yerr=errs, marker="o", color=color,
                        linewidth=1.5, markersize=4, solid_capstyle="round",
                        clip_on=False, capsize=2, capthick=0.8, elinewidth=0.8)
            if not pd.isna(vals[-1]):
                ax.text(n_datasets - 1 + 0.12, label_y, party,
                        ha="left", va="center", color=color)

        any_gt = False
        for party in all_parties:
            gt_val, gt_err = gt_fn(key, party)
            if pd.isna(gt_val):
                continue
            any_gt = True
            color = party_colors.get(party, "#888888")
            ax.axhline(gt_val, color=color, linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
            if pd.notna(gt_err):
                ax.axhspan(gt_val - gt_err, gt_val + gt_err, color=color, alpha=0.12, zorder=0)
        if any_gt:
            ax.text(-0.35, ylim[1] - (ylim[1] - ylim[0]) * 0.03, "Survey",
                    fontsize=8, style="italic", color="#666666", ha="left", va="top")

        ax.set_title(title, fontweight="medium", pad=8, fontsize=10)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.4, n_datasets - 1 + right_margin)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.grid(True, linestyle="-", alpha=0.15, color="#333333")
        ax.set_axisbelow(True)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")
    for row in range(nrows):
        axes[row, 0].set_ylabel(value_label)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")


def plot_slope_comparison(
    dfs: dict[str, pd.DataFrame],
    keys: list[tuple[str, str]],
    all_parties: list[str],
    party_colors: dict[str, str],
    output_path: str,
    metric: str,
    value_col: str,
    std_col: str,
    ncols: int | None = None,
    value_label: str = "Fraction answering Yes",
    ylim: tuple[float, float] = (0, 1),
    ground_truth: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> None:
    """Slope chart: one line per party (colored by party), x-axis = condition,
    one panel per question/target — the pre-91c026c ablation-script chart
    style, ported onto the aggregated (metric, key, party) dfs this pipeline
    now produces. Panels wrap onto multiple rows (ncols per row, capped at
    MAX_GRID_COLS).

    `ground_truth`, if given, maps key -> party -> (value, 95%-CI half-width)
    for a constant real-world reference (e.g. real ANES respondents' own
    ratings) drawn as a dashed party-colored horizontal line with a shaded CI
    band — a reference unaffected by the ablation condition on the x-axis, so
    a flat line/band rather than another slope. Keys absent from
    `ground_truth` are drawn with no reference line."""
    def value_fn(label, key, party):
        return _lookup(dfs[label], metric, key, party, value_col, std_col)

    def gt_fn(key, party):
        return (ground_truth or {}).get(key, {}).get(party, (float("nan"), float("nan")))

    _plot_slope_core(list(dfs.keys()), keys, all_parties, party_colors, output_path,
                      value_fn, gt_fn, ncols, value_label, ylim)


def _role_average(get_rating, roles: tuple[str, ...], party: str) -> tuple[float, float]:
    """Mean value (and combined-independent-error) of `get_rating(role, party)
    -> (value, error)` across `roles` — the "one side" of a _thermometer_gap,
    exposed standalone for panels that want a single averaged rate rather
    than an own-minus-opposing difference (e.g. the don't-know rate averaged
    across a trait-polarity's question set, see plot_role_average_comparison).
    NaN if any role is missing."""
    vals, errs = [], []
    for role in roles:
        v, e = get_rating(role, party)
        if math.isnan(v):
            return float("nan"), float("nan")
        vals.append(v)
        errs.append(e if not math.isnan(e) else 0.0)
    return sum(vals) / len(vals), math.hypot(*errs) / len(errs)


def _thermometer_gap(get_rating, role_map: dict[str, tuple[tuple[str, ...], tuple[str, ...]]], party: str) -> tuple[float, float]:
    """Thermometer gap T_in - T_out for `party`: own-side rating minus
    opposing-side rating, where `get_rating(role, party) -> (value, error)`
    is the source (an aggregated df lookup, or a ground-truth dict) and
    `role_map[party] = (own_roles, opp_roles)` (see GAP_ROLE_MAPS) gives each
    side's thermometer role(s) — averaged via _role_average when a side has
    more than one (the "Combined" panel averages the party and candidate
    thermometers; a trait-differential panel averages a whole trait-polarity
    battery, see TRAIT_GAP_ROLE_MAPS). Errors on both sides are combined
    assuming independence (same approximation as _affective_polarization_fn);
    NaN if any role on either side is missing."""
    if party not in role_map:
        return float("nan"), float("nan")
    own_roles, opp_roles = role_map[party]

    own_val, own_err = _role_average(get_rating, own_roles, party)
    opp_val, opp_err = _role_average(get_rating, opp_roles, party)
    if math.isnan(own_val) or math.isnan(opp_val):
        return float("nan"), float("nan")
    return own_val - opp_val, math.hypot(own_err, opp_err)


def plot_gap_slope_comparison(
    dfs: dict[str, pd.DataFrame],
    panels: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]],
    all_parties: list[str],
    party_colors: dict[str, str],
    output_path: str,
    ncols: int | None = None,
    value_label: str = "Thermometer gap",
    ylim: tuple[float, float] = (-20, 100),
    ground_truth: dict[str, dict[str, tuple[float, float]]] | None = None,
    metric: str = "thermometer",
    value_col: str = "rating_mean",
    std_col: str = "rating_std",
) -> None:
    """Slope chart of an own-minus-opposing gap — the standard
    affective-polarization measure, generalized from the feeling thermometer
    to any (metric, value_col, std_col) aggregated column — one panel per
    `panels` entry (e.g. GAP_ROLE_MAPS' "Party"/"Candidate"/"Combined", or
    TRAIT_GAP_ROLE_MAPS' "Positive traits"/"Negative traits" with
    metric="question", value_col="pct_yes_mean", std_col="pct_yes_std"),
    same layout/reference-line convention as plot_slope_comparison (see
    _plot_slope_core), except each point is a derived gap (via
    _thermometer_gap) rather than a direct (metric,key,party) lookup, since
    which two rows to diff depends on the party itself."""
    def value_fn(label, title, party):
        get_rating = lambda role, p: _lookup(dfs[label], metric, role, p, value_col, std_col)
        return _thermometer_gap(get_rating, panels[title], party)

    def gt_fn(title, party):
        if not ground_truth:
            return float("nan"), float("nan")
        get_rating = lambda role, p: ground_truth.get(role, {}).get(p, (float("nan"), float("nan")))
        return _thermometer_gap(get_rating, panels[title], party)

    keys = [(title, title) for title in panels]
    _plot_slope_core(list(dfs.keys()), keys, all_parties, party_colors, output_path,
                      value_fn, gt_fn, ncols, value_label, ylim)


def plot_role_average_comparison(
    dfs: dict[str, pd.DataFrame],
    panels: dict[str, tuple[str, ...]],
    all_parties: list[str],
    party_colors: dict[str, str],
    output_path: str,
    ncols: int | None = None,
    value_label: str = "Rate",
    ylim: tuple[float, float] = (0, 1),
    metric: str = "question",
    value_col: str = "pct_dont_know_mean",
    std_col: str = "pct_dont_know_std",
) -> None:
    """Slope chart of `value_col` averaged across a fixed question-key set per
    panel (see _role_average) — same layout as plot_gap_slope_comparison, but
    one averaged rate per point instead of an own-minus-opposing gap. Used to
    report a trait-differential panel's don't-know rate on its own rather
    than netting it into the gap (see TRAIT_DONT_KNOW_ROLES); unlike
    plot_gap_slope_comparison's panels, `panels[title]` here is a single
    role tuple (not per-party own/opp roles), since the averaged question set
    doesn't depend on the respondent's own party."""
    def value_fn(label, title, party):
        get_rating = lambda role, p: _lookup(dfs[label], metric, role, p, value_col, std_col)
        return _role_average(get_rating, panels[title], party)

    def gt_fn(title, party):
        return float("nan"), float("nan")

    keys = [(title, title) for title in panels]
    _plot_slope_core(list(dfs.keys()), keys, all_parties, party_colors, output_path,
                      value_fn, gt_fn, ncols, value_label, ylim)


def print_thermometer_table(
    dfs: dict[str, pd.DataFrame],
    roles: list[tuple[str, str]],
    all_parties: list[str],
) -> None:
    """Print the LLM-answered feeling-thermometer rating (rows = condition,
    columns = party) for each target in `roles` (as (role, row_label) pairs)
    — the actual per-condition values drawn as points in
    plot_slope_comparison's thermometer chart (mirrors the printed table
    plot_thermometer_comparison already draws for the obfuscation-style
    combined chart, exposed standalone for callers like the ablation script
    that build the thermometer chart via plot_slope_comparison directly).
    Each cell is annotated with its not-recognized rate (the target wasn't a
    well-known enough figure/party label for the respondent to rate), same
    convention as plot_thermometer_comparison."""
    def value_fn(role):
        return lambda label, party: _lookup(dfs[label], "thermometer", role, party, "rating_mean", "rating_std")

    def not_recognized_fn(role):
        def fn(label, party):
            rec, err = _lookup(dfs[label], "thermometer", role, party, "pct_recognized_mean", "pct_recognized_std")
            if pd.isna(rec):
                return float("nan"), float("nan")
            return 1.0 - rec, err
        return fn

    present = [(role, label) for role, label in roles
               if any(((df["metric"] == "thermometer") & (df["key"] == role)).any() for df in dfs.values())]
    if not present:
        return

    print(f"\n{'='*60}")
    print("  Feeling thermometer (rows = condition, columns = party)")
    print(f"{'='*60}")
    for role, label in present:
        _print_comparison_table(f"Feeling thermometer: {label}", list(dfs.keys()), all_parties, value_fn(role),
                                 secondary_fn=not_recognized_fn(role), secondary_prefix="nr")


def print_ground_truth_thermometer_table(
    ground_truth: dict[str, dict[str, tuple[float, float]]],
    roles: list[tuple[str, str]],
    all_parties: list[str],
) -> None:
    """Print real ANES respondents' own feeling-thermometer rating toward each
    target in `roles` (as (role, row_label) pairs), by their own party — the
    same ground truth drawn as the dashed reference lines in
    plot_slope_comparison's thermometer chart (see
    aggregate_ground_truth_thermometer), as an actual table since a line on a
    plot can't be read off precisely."""
    present = [(role, label) for role, label in roles if role in ground_truth]
    if not present:
        return
    role_by_label = {label: role for role, label in present}

    def value_fn(label, party):
        return ground_truth.get(role_by_label[label], {}).get(party, (float("nan"), float("nan")))

    _print_comparison_table("Feeling thermometer ground truth (real ANES respondents, by own party)",
                             [label for _, label in present], all_parties, value_fn)


def print_ground_truth_gap_table(
    ground_truth: dict[str, dict[str, tuple[float, float]]],
    panels: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]],
    all_parties: list[str],
) -> None:
    """Print the thermometer gap T_in - T_out computed from real ANES
    respondents' own ratings (same `panels` as plot_gap_slope_comparison,
    e.g. GAP_ROLE_MAPS) — the ground truth drawn as the dashed reference
    lines in that chart, as an actual table."""
    def value_fn(title, party):
        get_rating = lambda role, p: ground_truth.get(role, {}).get(p, (float("nan"), float("nan")))
        return _thermometer_gap(get_rating, panels[title], party)

    _print_comparison_table("Thermometer gap ground truth (T_in - T_out, real ANES respondents)",
                             list(panels.keys()), all_parties, value_fn)


def print_gap_comparison_table(
    dfs: dict[str, pd.DataFrame],
    role_map: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    all_parties: list[str],
    title: str,
    metric: str = "thermometer",
    value_col: str = "rating_mean",
    std_col: str = "rating_std",
    dont_know_roles: tuple[str, ...] | None = None,
    dont_know_metric: str = "question",
    dont_know_value_col: str = "pct_dont_know_mean",
    dont_know_std_col: str = "pct_dont_know_std",
) -> None:
    """Print the own-minus-opposing gap (see _thermometer_gap) computed per
    condition (rows) and party (columns) from the aggregated per-condition
    LLM-answer dfs — same shape as print_ground_truth_gap_table, but for the
    LLM answers themselves (any (metric, value_col, std_col), e.g. the trait
    battery's pct_yes) rather than the ANES thermometer ground truth.
    `dont_know_roles`, if given, appends each cell's don't-know rate averaged
    across that (party-independent) question set (see _role_average) — kept
    as its own reported rate rather than folded into the gap, same convention
    as print_question_tables' trait cells."""
    def value_fn(label, party):
        get_rating = lambda role, p: _lookup(dfs[label], metric, role, p, value_col, std_col)
        return _thermometer_gap(get_rating, role_map, party)

    secondary_fn = None
    if dont_know_roles is not None:
        def secondary_fn(label, party):
            get_rating = lambda role, p: _lookup(dfs[label], dont_know_metric, role, p, dont_know_value_col, dont_know_std_col)
            return _role_average(get_rating, dont_know_roles, party)

    _print_comparison_table(title, list(dfs.keys()), all_parties, value_fn,
                             secondary_fn=secondary_fn, secondary_prefix="dk", secondary_fmt="{:.1%}")


def _affective_polarization_fn(dfs: dict[str, pd.DataFrame], role_map: dict[str, tuple[str, str]]):
    """Return a `value_fn(label, party) -> (value, error)` computing own-role
    rating minus opposing-role rating for `party`, per `role_map` (see
    PARTY_THERM_ROLES / LEADER_THERM_ROLES)."""
    def value_fn(label, party):
        if party not in role_map:
            return float("nan"), float("nan")
        own_role, opp_role = role_map[party]
        own_val, own_err = _lookup(dfs[label], "thermometer", own_role, party, "rating_mean", "rating_std")
        opp_val, opp_err = _lookup(dfs[label], "thermometer", opp_role, party, "rating_mean", "rating_std")
        if math.isnan(own_val) or math.isnan(opp_val):
            return float("nan"), float("nan")
        # Own/opposing ratings come from the same respondents, but per-run values
        # aren't available in this aggregated format, so their errors are combined
        # assuming independence (a reasonable approximation, not exact).
        own_err = own_err if not math.isnan(own_err) else 0.0
        opp_err = opp_err if not math.isnan(opp_err) else 0.0
        return own_val - opp_val, math.hypot(own_err, opp_err)
    return value_fn


def plot_thermometer_comparison(
    dfs: dict[str, pd.DataFrame], all_parties: list[str], condition_colors: dict[str, str], output_path: str,
    therm_title_suffix: str = "",
) -> None:
    """Fixed 3-row x 2-column grid: row 0 is the Democratic party and its
    leader (Biden), row 1 the Republican party and its leader (Trump), row 2
    the affective-polarization summaries (party-based, then leader-based) —
    every feeling-thermometer result in one chart, rather than a separate
    affective-polarization figure. `therm_title_suffix` (e.g. "\\n(obfuscated
    per condition)") is appended to the "Feeling thermometer: X" panel titles
    only — callers where nothing is obfuscated (ablation comparisons) can
    leave it blank."""
    def therm_present(role: str) -> bool:
        return all(((df["metric"] == "thermometer") & (df["key"] == role)).any() for df in dfs.values())

    labels     = list(dfs.keys())
    n_datasets = len(labels)
    n_parties  = len(all_parties)

    party_polar_parties  = [p for p in PARTY_THERM_ROLES
                             if any((df["party"] == p).any() for df in dfs.values())]
    leader_polar_parties = [p for p in LEADER_THERM_ROLES
                             if any((df["party"] == p).any() for df in dfs.values())]
    party_polar_fn  = _affective_polarization_fn(dfs, PARTY_THERM_ROLES)
    leader_polar_fn = _affective_polarization_fn(dfs, LEADER_THERM_ROLES)

    def therm_value_fn(role):
        return lambda label, party: _lookup(dfs[label], "thermometer", role, party, "rating_mean", "rating_std")

    def therm_not_recognized_fn(role):
        def fn(label, party):
            rec, err = _lookup(dfs[label], "thermometer", role, party, "pct_recognized_mean", "pct_recognized_std")
            if pd.isna(rec):
                return float("nan"), float("nan")
            return 1.0 - rec, err
        return fn

    # (panel kind, role, title) per grid cell; "role" is a thermometer key for
    # "therm" panels and unused (None) for the polarization panels.
    grid = [
        [("therm", "democrats", "Democrats"), ("therm", "biden", "Biden")],
        [("therm", "republicans", "Republicans"), ("therm", "trump", "Trump")],
        [("polar_party", None, "Affective polarization\n(party)"),
         ("polar_leader", None, "Affective polarization\n(leader)")],
    ]
    nrows, ncols = 3, 2

    panel_w = 1.1 * n_parties + 1.0
    panel_h = 0.4 * (n_datasets + 1) + 0.9
    fig = plt.figure(figsize=(panel_w * ncols, panel_h * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.25, wspace=0.4, left=0.08, right=0.97,
                              top=TOP_PAD, bottom=0.04)

    any_drawn = False
    for r, row in enumerate(grid):
        for c, (kind, role, title) in enumerate(row):
            spec = outer[r, c]
            if kind == "therm":
                if not therm_present(role):
                    continue
                any_drawn = True
                _draw_table_panel(fig, spec, labels, all_parties, condition_colors, therm_value_fn(role),
                                   (0, 100), f"Feeling thermometer: {title}{therm_title_suffix}", "{:.0f}",
                                   secondary_fn=therm_not_recognized_fn(role), secondary_prefix="nr")
            elif kind == "polar_party":
                if not party_polar_parties:
                    continue
                any_drawn = True
                _draw_table_panel(fig, spec, labels, party_polar_parties, condition_colors, party_polar_fn,
                                   (-100, 100), f"{title}\n(own-party rating minus opposing-party rating)", "{:.0f}")
            elif kind == "polar_leader":
                if not leader_polar_parties:
                    continue
                any_drawn = True
                _draw_table_panel(fig, spec, labels, leader_polar_parties, condition_colors, leader_polar_fn,
                                   (-100, 100), f"{title}\n(own party's leader minus other leader rating)", "{:.0f}")

    if not any_drawn:
        plt.close(fig)
        print("No feeling-thermometer rows found in the comparison CSVs — skipping thermometer comparison plot.")
        return

    fig.text(0.01, 0.5, "Rating (0-100) / Polarization (-100 to 100)",
              va="center", rotation="vertical", fontsize=10, color="#555555")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")

    print(f"\n{'='*60}")
    print("  Feeling thermometer (rows = condition, columns = party)")
    print(f"{'='*60}")
    for role, person_label in [("democrats", "Democrats"), ("biden", "Biden"),
                                ("republicans", "Republicans"), ("trump", "Trump")]:
        if not therm_present(role):
            continue
        _print_comparison_table(f"Feeling thermometer: {person_label}", labels, all_parties, therm_value_fn(role),
                                 secondary_fn=therm_not_recognized_fn(role), secondary_prefix="nr")

    if party_polar_parties:
        print(f"\n{'='*60}")
        print("  Affective polarization (rows = condition, columns = party)")
        print(f"{'='*60}")
        _print_comparison_table("Affective polarization (party): own-party minus opposing-party rating",
                                 labels, party_polar_parties, party_polar_fn)
    if leader_polar_parties:
        _print_comparison_table("Affective polarization (leader): own party's leader minus other leader rating",
                                 labels, leader_polar_parties, leader_polar_fn)
