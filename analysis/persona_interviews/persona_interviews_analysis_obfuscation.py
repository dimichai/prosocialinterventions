import argparse
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import interview_wandb  # noqa: E402
import persona_interviews as interview  # noqa: E402

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
# no single "current" obfuscation to derive labels from (unlike persona_interviews.py).
# (key_suffix, trait) — asked once for Democrats and once for Republicans
TRAIT_QUESTIONS = [
    ("intelligent",   "intelligent"),
    ("honest",        "honest"),
    ("generous",      "generous"),
    ("openminded",    "open-minded"),
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


def aggregate_population_metrics(raw_dfs: list[pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """Mean and 95%-CI half-width (`1.96 * std / sqrt(n_runs)`, across the seeds
    within one obfuscation condition) of interview_wandb.persona_population_metrics
    — the same persona-attribute distribution that gets logged to wandb under the
    "personas/" prefix at run time, recomputed here from the downloaded raw
    per-persona results so it can be printed alongside the question-answer
    tables."""
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


def fetch_condition_dfs(
    batch_id: str, wandb_project: str = interview_wandb.WANDB_PROJECT
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, tuple[float, float]]]]:
    """Fetch every wandb run sharing `batch_id` (one run per obfuscation x seed,
    written by run_persona_obf_pipeline.py / persona_interviews.py),
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


def print_population_table(population: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Persona population statistics per condition (rows = obfuscation condition),
    i.e. the attributes of the personas being interviewed, before any LLM answers —
    the same stats logged to wandb under the "personas/" prefix. Each cell is
    mean ± 95%-CI half-width across seeds (see `aggregate_population_metrics`)."""
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
    """One table per question, rows = obfuscation condition, columns = party
    identity — lines up every condition's answer to the same question side by
    side, for every party, instead of splitting them across per-condition blocks.
    Cells show mean ± 95%-CI half-width (the same CI drawn as error bars in the
    plots, see `_lookup`)."""
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
    """Print one table: rows = obfuscation condition (`labels`), columns =
    `columns` (e.g. party), cell = `_fmt_ci(*value_fn(label, column))`. Mirrors
    the row/column layout `print_question_tables` uses, so every printed
    result set — questions, thermometer, affective polarization — reads the
    same way instead of some being one table and others per-condition blocks.

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
    """Draw one "table" panel inside `outer_spec`: rows = obfuscation condition
    (`labels`), columns = `columns` (e.g. party), each cell a single horizontal
    bar for `value_fn(label, column) -> (value, error)`. Row 0 of the inner grid
    is the panel title, row 1 the column (party) headers — both dedicated rows,
    rather than relying on matplotlib's floating axes-title padding, which
    overlaps neighboring rows once cells get short.

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


MAX_GRID_COLS = 3

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
    rows = obfuscation condition, columns = party, cell = a single horizontal bar
    for that condition/party's value.

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
    dfs: dict[str, pd.DataFrame], all_parties: list[str], condition_colors: dict[str, str], output_path: str
) -> None:
    """Fixed 3-row x 2-column grid: row 0 is the Democratic party and its
    leader (Biden), row 1 the Republican party and its leader (Trump), row 2
    the affective-polarization summaries (party-based, then leader-based) —
    every feeling-thermometer result in one chart, rather than a separate
    affective-polarization figure."""
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
                                   (0, 100), f"Feeling thermometer: {title}\n(obfuscated per condition)", "{:.0f}",
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
    print("  Feeling thermometer (rows = obfuscation condition, columns = party)")
    print(f"{'='*60}")
    for role, person_label in [("democrats", "Democrats"), ("biden", "Biden"),
                                ("republicans", "Republicans"), ("trump", "Trump")]:
        if not therm_present(role):
            continue
        _print_comparison_table(f"Feeling thermometer: {person_label}", labels, all_parties, therm_value_fn(role),
                                 secondary_fn=therm_not_recognized_fn(role), secondary_prefix="nr")

    if party_polar_parties:
        print(f"\n{'='*60}")
        print("  Affective polarization (rows = obfuscation condition, columns = party)")
        print(f"{'='*60}")
        _print_comparison_table("Affective polarization (party): own-party minus opposing-party rating",
                                 labels, party_polar_parties, party_polar_fn)
    if leader_polar_parties:
        _print_comparison_table("Affective polarization (leader): own party's leader minus other leader rating",
                                 labels, leader_polar_parties, leader_polar_fn)


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
                                 fig_path("interview_results_obfuscation_thermometer", args.batch_id))


if __name__ == "__main__":
    main()
