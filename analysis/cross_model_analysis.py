import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "persona_interviews"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import interview_wandb  # noqa: E402
from interview_comparison_plots import GAP_ROLE_MAPS, FOLLOW_GAP_ROLE_MAPS, TRAIT_GAP_ROLE_MAPS  # noqa: E402
from simulation_comparison_plots import (  # noqa: E402
    NETWORK_METRICS,
    fig_path,
    fetch_and_aggregate,
    download_platform,
)
from simulation_analysis_obfuscation import (  # noqa: E402
    DEFAULT_WANDB_PROJECT,
    PARTIES,
    fetch_runs_by_obfuscation,
)

# fetch_runs_by_obfuscation's display label -> this figure's legend label, in bar order.
CONDITIONS = {
    "No Obfuscation": "Original",
    "Neutral":        "Neutral",
    "Unrelated":      "Unrelated",
    "Nonce":          "Nonce",
}

# One color per CONDITIONS entry, in the same order.
CONDITION_COLORS = ["#003a7d", "#4ecb8d", "#ff73b6", "#ff9d3a"]
# The three obfuscation colors print as the same grey, so each condition also gets
# its own hatch (bars) and marker fill (scatter).
CONDITION_HATCHES = ["", "//////", "", "......"]
CONDITION_FILLS = ["full", "none", "full", "bottom"]

# AAMAS uses acmart's sigconf layout: 8.5in page, 54pt side margins, 2pc column gap
# -> \textwidth = 504pt, \columnwidth = 240pt. Figures are drawn at exactly these
# widths and included unscaled, so the font sizes below are the printed sizes
# (body text is 9pt, captions 8pt).
TEXT_WIDTH, COLUMN_WIDTH = 504 / 72, 240 / 72
plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'hatch.linewidth': 0.4,
    'grid.linewidth': 0.4,
    # 'tight' would crop the canvas and change the printed width.
    'savefig.bbox': 'standard',
})

# llm_model (config value) -> x-axis label. Unlisted models show their raw config value.
MODEL_NAMES = {
    "openai/gpt-6-luna":             "GPT-6 Luna",
    "deepseek/deepseek-v4.1-flash":  "DeepSeek V4.1 Flash",
    "mistralai/mistral-small-2603":  "Mistral Small 2603",
    "meta-llama/llama-4-maverick":   "Llama 4 Maverick",
    "google/gemini-3.1-flash-lite":  "Gemini 3.1 Flash-Lite",
    "google/gemini-2.5-flash-lite":  "Gemini 2.5 Flash-Lite",
    "gpt-4o-mini":                   "GPT-4o mini",
    "openai/gpt-5.4-mini":           "GPT-5.4 mini",
}


def condition_bars(ax, x, i: int, means, errors=None, n_bars: int = len(CONDITIONS), offset: float = 0):
    """Draw condition i's bar for every model (x = model positions) in a group of
    `n_bars` bars per model."""
    width = 0.8 / n_bars
    ax.bar(x + (i - offset - (n_bars - 1) / 2) * width, means, width, color=CONDITION_COLORS[i],
           hatch=CONDITION_HATCHES[i], edgecolor="#222222", linewidth=0.3,
           label=list(CONDITIONS.values())[i], yerr=errors, capsize=1,
           error_kw={'elinewidth': 0.6, 'capthick': 0.6})


def style_model_axis(ax, x, models: list[str], title: str, labels: bool = True) -> None:
    """Title, grid and one tick per model. Model names are only worth their height
    once per column, so callers pass labels=False for panels with one below them."""
    ax.set_title(title, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(models if labels else [], rotation=35, ha='right', rotation_mode='anchor')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def random_mixing_ei(platform) -> float:
    """EI_index_dem_rep expected if every Democrat/Republican followed other
    Democrats/Republicans uniformly at random (no self-follows): depends only on
    the two party sizes."""
    parties = [u.persona.get("party") for u in platform.users]
    d, r = (parties.count(p) for p in PARTIES)
    n = d + r
    return (2 * d * r - d * (d - 1) - r * (r - 1)) / (n * (n - 1))


def implied_ei(df) -> float:
    """E-I index the interview's stated label preferences would produce on their
    own: every Democrat/Republican meets every other one (so exposure follows
    party sizes, as in random_mixing_ei) and follows at their party's yes-rate to
    "would you follow a member of <party>?". `df` is one run's raw interview
    results. Equals random_mixing_ei when the four yes-rates are equal."""
    is_dem, is_rep = df["party"] == "Democrat", df["party"] == "Republican"
    d, r = is_dem.sum(), is_rep.sum()

    def yes(mask, key):
        return interview_wandb.pct_yes_no_dontknow(df.loc[mask, f"{key}_answer"])[0]

    external = d * r * (yes(is_dem, "q1_rep") + yes(is_rep, "q1_dem"))
    internal = d * (d - 1) * yes(is_dem, "q1_dem") + r * (r - 1) * yes(is_rep, "q1_rep")
    return (external - internal) / (external + internal)


# Panel title -> (role map from interview_comparison_plots, whether its roles are
# feeling thermometers rather than yes/no questions). Every gap is own side minus
# opposing side.
INTERVIEW_GAPS = {
    "Thermometer gap (party)":     (GAP_ROLE_MAPS["Party"], True),
    "Thermometer gap (candidate)": (GAP_ROLE_MAPS["Candidate"], True),
    "Follow gap (party)":          (FOLLOW_GAP_ROLE_MAPS["Party"], False),
    "Positive trait gap":          (TRAIT_GAP_ROLE_MAPS["Positive traits"], False),
    "Negative trait gap":          (TRAIT_GAP_ROLE_MAPS["Negative traits"], False),
}


def interview_gap(df, role_map, thermometer: bool) -> float:
    """One run's own-minus-opposing gap, averaged over Democrats and Republicans:
    yes-rate ("don't know" excluded) for questions, mean rating among those who
    recognized the target for thermometers — the per-run statistics
    interview_wandb logs. `df` is the run's raw interview results."""
    def rate(group, role):
        if thermometer:
            return interview_wandb.pct_recognized_and_rating(
                group[f"{role}_therm_recognized"], group[f"{role}_therm_rating"])[1]
        return interview_wandb.pct_yes_no_dontknow(group[f"{role}_answer"])[0]

    gaps = []
    for party, (own, opp) in role_map.items():
        group = df[df["party"] == party]
        # nanmean: a trait everyone in the party answered "don't know" to has no yes-rate.
        gaps.append(np.nanmean([rate(group, r) for r in own]) - np.nanmean([rate(group, r) for r in opp]))
    return np.mean(gaps)


def plot_interview_gaps(models: list[str], interviews: dict) -> dict:
    """Same layout as the network-metrics figure, for the questionnaire's gaps:
    one panel per gap, models on the x-axis, one bar per condition (mean ± SE
    across seeds). Returns the plotted means as gap title -> condition -> array
    over `models`."""
    gap_means = {title: {} for title in INTERVIEW_GAPS}
    x = np.arange(len(models))
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.3), layout="constrained")
    axes = axes.flatten()
    for idx, (ax, (title, (role_map, thermometer))) in enumerate(zip(axes, INTERVIEW_GAPS.items())):
        print(f"\n{title}")
        rows = {m: [] for m in models}
        for i, (condition, legend_label) in enumerate(CONDITIONS.items()):
            means, errors = [], []
            for m in models:
                v = [interview_gap(df, role_map, thermometer) for df in interviews[m].get(condition, [])]
                v = [g for g in v if not np.isnan(g)]
                # A condition a model has no interviews for is a gap, not a zero bar.
                means.append(np.mean(v) if v else np.nan)
                errors.append(np.std(v) / np.sqrt(len(v)) if v else np.nan)
                if v:
                    rows[m].append(f"{legend_label}={means[-1]:.4f} ± {errors[-1]:.4f}")
            condition_bars(ax, x, i, means, errors)
            gap_means[title][condition] = np.array(means)
        for m in models:
            print(f"  {m:<24} " + "  ".join(rows[m]))
        ax.axhline(0, color="#333333", linewidth=0.6)
        ax.set_ylabel("Rating gap (0-100)" if thermometer else "Yes-share gap")
        style_model_axis(ax, x, models, f"({chr(ord('A') + idx)}) {title}", labels=idx >= 2)

    # Panel C has no panel below it: its model names hang into the empty sixth
    # cell (kept out of the layout so they don't add height to the whole top row),
    # above the legend.
    axes[2].xaxis.set_in_layout(False)
    axes[-1].axis("off")
    axes[-1].legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
                    handlelength=2.2, handleheight=1.2)

    out_path = fig_path("interview_gaps", "cross_model")
    fig.savefig(out_path)
    fig.savefig(fig_path("interview_gaps", "cross_model", ext="png"))
    plt.close(fig)
    print(f"Saved to {out_path}")
    return gap_means


MODEL_MARKERS = "os^DvPX*"
# Marker edge per CONDITION_FILLS style (None = the condition color): a dark edge
# separates the light filled markers from the open ones in greyscale.
MARKER_EDGES = {"full": "#222222", "none": None, "bottom": "#222222"}


def marker_style(marker: str, color: str, fill: str = "full") -> dict:
    return {"marker": marker, "color": color, "fillstyle": fill, "markerfacecoloralt": "white",
            "markeredgecolor": MARKER_EDGES[fill] or color, "markeredgewidth": 0.8, "markersize": 5.5}


def condition_handles(skip: int = 0) -> list:
    """Legend entries for the conditions' marker color + fill, from the skip-th on."""
    return [Line2D([], [], linestyle="", label=l, **marker_style("o", c, f))
            for l, c, f in list(zip(CONDITIONS.values(), CONDITION_COLORS, CONDITION_FILLS))[skip:]]


def model_handles(models: list[str]) -> list:
    """Legend entries for the models' marker shapes."""
    return [Line2D([], [], linestyle="", label=m, **marker_style(mk, "#555555")) for m, mk in zip(models, MODEL_MARKERS)]


def plot_gap_change_vs_retained(models: list[str], gap_means: dict, ei: dict, per_condition: bool,
                                column_gaps: list[str] | None = None) -> None:
    """Appendix: does a model whose questionnaire gap shrinks more under
    obfuscation also keep less of its network segregation? One panel per gap,
    x = change of the gap's size from Original to obfuscated (negative = it shrank
    toward zero, so the negative trait gap reads the same way), y = share of the
    Original E-I Index retained, with the Spearman correlation over the plotted
    points. One point per model x obfuscated condition if `per_condition`, else one
    per model with the obfuscated conditions averaged. By default a full-width
    grid of every gap; `column_gaps` instead stacks just those gaps in a
    single-column figure."""
    obfuscated = list(CONDITIONS)[1:]
    retained = {c: ei[c] / ei["No Obfuscation"] for c in obfuscated}
    name = "conditions" if per_condition else "models"
    print(f"\nGap change under obfuscation vs % of Original E-I Index retained (per {name[:-1]})")

    if column_gaps:
        fig, axes = plt.subplots(len(column_gaps), 1, figsize=(COLUMN_WIDTH, 1.55 * len(column_gaps) + 0.8),
                                 layout="constrained", sharey=True)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.6), layout="constrained", sharey=True)
    axes = axes.flatten()
    for idx, (ax, title) in enumerate(zip(axes, column_gaps or INTERVIEW_GAPS)):
        thermometer = INTERVIEW_GAPS[title][1]
        original = gap_means[title]["No Obfuscation"]
        change = {c: np.sign(original) * (gap_means[title][c] - original) for c in obfuscated}
        if per_condition:
            series = [(change[c], retained[c], CONDITION_COLORS[i], CONDITION_FILLS[i])
                      for i, c in enumerate(obfuscated, start=1)]
        else:
            # nanmean would hide a missing condition; a model missing one is dropped instead.
            series = [(np.mean(list(change.values()), axis=0), np.mean(list(retained.values()), axis=0),
                       "#003a7d", "full")]
        for xs, ys, color, fill in series:
            for x, y, marker in zip(xs, ys, MODEL_MARKERS):
                ax.plot(x, y, linestyle="", **marker_style(marker, color, fill))

        xs, ys = (np.concatenate([s[i] for s in series]) for i in (0, 1))
        valid = ~(np.isnan(xs) | np.isnan(ys))
        rho, p_value = spearmanr(xs[valid], ys[valid]) if valid.sum() > 2 else (np.nan, np.nan)
        stats = f"ρ = {rho:.2f}, p {'< .001' if p_value < 0.001 else f'= {p_value:.3f}'} (n = {valid.sum()})"
        print(f"  {title:<28} {stats}")
        # In the title rather than inside the axes, where it would cover points.
        ax.set_title(f"({chr(ord('A') + idx)}) {title}\n{stats}", pad=4, linespacing=1.3)
        ax.set_xlabel("Change in rating gap (points)" if thermometer else "Change in yes-share gap")
        if column_gaps or idx % 3 == 0:
            ax.set_ylabel("Original E-I Index retained")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(True)
        ax.set_axisbelow(True)

    legend_kwargs = {"frameon": False, "handletextpad": 0.2, "columnspacing": 0.6}
    if column_gaps:
        # No spare cell in one column: conditions above the panels, models below.
        name += "_column"
        if per_condition:
            fig.legend(handles=condition_handles(skip=1), loc="outside upper center", ncol=3,
                       borderaxespad=0, **legend_kwargs)
        fig.legend(handles=model_handles(models), loc="outside lower center", ncol=3,
                   borderaxespad=0, labelspacing=0.3, **legend_kwargs)
    else:
        axes[-1].axis("off")
        handles = (condition_handles(skip=1) if per_condition else []) + model_handles(models)
        axes[-1].legend(handles=handles, loc="center", **legend_kwargs)

    out_path = fig_path(f"gap_change_vs_retained_{name}", "cross_model")
    fig.savefig(out_path)
    fig.savefig(fig_path(f"gap_change_vs_retained_{name}", "cross_model", ext="png"))
    plt.close(fig)
    print(f"Saved to {out_path}")


def plot_implied_vs_network(models: list[str], data: dict, implied: dict) -> None:
    """One point per model x condition: implied E-I (interview) vs network E-I
    (simulation), mean ± SE across seeds. On the diagonal the network is what
    label preference alone predicts; below it there is extra segregation."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6), layout="constrained")
    print("\nImplied E-I (interview) vs network E-I")
    for m, marker in zip(models, MODEL_MARKERS):
        for (condition, legend_label), color, fill in zip(CONDITIONS.items(), CONDITION_COLORS, CONDITION_FILLS):
            v = implied[m].get(condition, [])
            if not v or condition not in data[m]:
                continue
            x, x_se = np.mean(v), np.std(v) / np.sqrt(len(v))
            y, y_se = data[m][condition]["EI_index_dem_rep"], data[m][condition]["EI_index_dem_rep_se"]
            ax.errorbar(x, y, xerr=x_se, yerr=y_se, elinewidth=0.6, **marker_style(marker, color, fill))
            print(f"  {m:<24} {legend_label:<10} implied={x:.4f} ± {x_se:.4f}  network={y:.4f} ± {y_se:.4f}")

    # Not a square panel (too tall for one column), so the y = x line is not at 45°.
    # Limits are pinned to the points first: axline's anchor would stretch them to 0.
    ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
    ax.axline((0, 0), slope=1, color="#333333", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("Implied E-I Index (interview)")
    ax.set_ylabel("Network E-I Index (Dem/Rep)")
    ax.grid(True)
    ax.set_axisbelow(True)

    # Condition (color + fill) above the axes, model (shape) below.
    fig.legend(handles=condition_handles(),
               loc="outside upper center", ncol=4, frameon=False, columnspacing=1.0, handletextpad=0.2,
               borderaxespad=0)
    fig.legend(handles=model_handles(models),
               loc="outside lower center", ncol=3, frameon=False, columnspacing=0.6, handletextpad=0.2,
               borderaxespad=0, labelspacing=0.3)

    out_path = fig_path("implied_vs_network_ei", "cross_model")
    fig.savefig(out_path)
    fig.savefig(fig_path("implied_vs_network_ei", "cross_model", ext="png"))
    plt.close(fig)
    print(f"Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Network metrics per model x obfuscation condition, one panel per metric.")
    parser.add_argument("--batch_ids", nargs="+", required=True,
                        help="One wandb group/batch id per model, in x-axis order.")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT)
    args = parser.parse_args()

    models, data, baselines, interviews = [], {}, [], {}
    for batch_id in args.batch_ids:
        runs_by_condition = fetch_runs_by_obfuscation(batch_id, args.wandb_project)
        # Unfinished runs have no final metrics / platform artifact yet.
        finished = {l: [r for r in runs if r.state == "finished"] for l, runs in runs_by_condition.items()}
        print(f"{batch_id}: " + ", ".join(
            f"{l} {len(finished[l])}/{len(runs)} finished" for l, runs in runs_by_condition.items()))
        runs = [r for rs in finished.values() for r in rs]
        if not runs:
            print(f"  {batch_id}: no finished runs, skipping.")
            continue
        llm_models = {r.config["llm_model"] for r in runs}
        assert len(llm_models) == 1, f"{batch_id} mixes models: {llm_models}"
        model = llm_models.pop()
        models.append(MODEL_NAMES.get(model, model))
        data[models[-1]], _ = fetch_and_aggregate(finished, metrics=NETWORK_METRICS)
        # Platforms are already on disk: fetch_and_aggregate downloads them for modularity.
        baselines += [random_mixing_ei(download_platform(r)) for r in runs]
        # The pipeline logs each seed's interview to the same run as its simulation.
        interviews[models[-1]] = {}
        for label, condition_runs in finished.items():
            interviews[models[-1]][label] = []
            for r in condition_runs:
                try:
                    interviews[models[-1]][label].append(interview_wandb.download_results_dataframe(r))
                except RuntimeError as e:
                    print(f"  {label}: {e}")

    # ponytail: one line at the mean over all runs — the baseline only depends on
    # party sizes, identical if every model ran the same personas. Draw per-model
    # segments if the printed range below is ever non-trivial.
    baseline = np.mean(baselines)
    print(f"Random-mixing EI: {baseline:.4f} (range {min(baselines):.4f} to {max(baselines):.4f})")

    x = np.arange(len(models))
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.5), layout="constrained")
    axes = axes.flatten()
    for idx, (ax, (metric, metric_label)) in enumerate(zip(axes, NETWORK_METRICS.items())):
        for i, condition in enumerate(CONDITIONS):
            # A condition a model has no finished runs for is a gap, not a zero bar.
            means = [data[m].get(condition, {}).get(metric, np.nan) for m in models]
            errors = [data[m].get(condition, {}).get(f"{metric}_se", np.nan) for m in models]
            condition_bars(ax, x, i, means, errors)
        if metric == "EI_index_dem_rep":
            ax.axhline(baseline, color="#333333", linestyle="--", linewidth=0.8, label="Random mixing")
        style_model_axis(ax, x, models, f"({chr(ord('A') + idx)}) {metric_label.replace('EI Index', 'E-I Index')}",
                         labels=idx >= 3)

    # Sixth cell: share of each model's Original EI that survives each obfuscation.
    ax = axes[-1]
    ei = {c: np.array([data[m].get(c, {}).get("EI_index_dem_rep", np.nan) for m in models]) for c in CONDITIONS}
    for i, condition in enumerate(CONDITIONS):
        if i:  # Original is 100% by definition
            # ponytail: ratio of cross-seed means, no error bars. Bootstrap over
            # seeds if the summary needs uncertainty.
            condition_bars(ax, x, i, ei[condition] / ei["No Obfuscation"], n_bars=3, offset=1)
    ax.axhline(1, color="#333333", linewidth=0.6)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style_model_axis(ax, x, models, f"({chr(ord('A') + len(NETWORK_METRICS))}) % of Original E-I Index retained")

    # The EI panel's handles include the dashed random-mixing line.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=5, frameon=False,
               handlelength=2.2, handleheight=1.2)

    out_path = fig_path("analysis", "cross_model")
    fig.savefig(out_path)
    fig.savefig(fig_path("analysis", "cross_model", ext="png"))
    plt.close(fig)
    print(f"Saved to {out_path}")

    # Same numbers as the figure: mean ± SE across seeds, then panel F's ratio.
    for metric, metric_label in NETWORK_METRICS.items():
        print(f"\n{metric_label.replace('EI Index', 'E-I Index')}")
        for m in models:
            row = "  ".join(
                f"{legend_label}={data[m][c][metric]:.4f} ± {data[m][c][f'{metric}_se']:.4f}"
                for c, legend_label in CONDITIONS.items() if c in data[m])
            print(f"  {m:<24} {row}")
    print("\n% of Original E-I Index retained")
    for j, m in enumerate(models):
        row = "  ".join(f"{legend_label}={ei[c][j] / ei['No Obfuscation'][j]:.1%}"
                        for c, legend_label in list(CONDITIONS.items())[1:])
        print(f"  {m:<24} {row}")

    implied = {m: {c: [implied_ei(df) for df in dfs] for c, dfs in by_condition.items()}
               for m, by_condition in interviews.items()}
    plot_implied_vs_network(models, data, implied)
    gap_means = plot_interview_gaps(models, interviews)
    plot_gap_change_vs_retained(models, gap_means, ei, per_condition=False)
    plot_gap_change_vs_retained(models, gap_means, ei, per_condition=True)
    plot_gap_change_vs_retained(models, gap_means, ei, per_condition=True,
                                column_gaps=["Thermometer gap (party)", "Follow gap (party)"])


if __name__ == "__main__":
    # Self-check: equal parties of 2 -> each node has 1 in-party and 2 out-party targets.
    from types import SimpleNamespace as NS
    _p = NS(users=[NS(persona={"party": p}) for p in ["Democrat", "Democrat", "Republican", "Republican", "Independent"]])
    assert abs(random_mixing_ei(_p) - 1 / 3) < 1e-12
    # Self-check: no label preference (everyone says yes) -> the random-mixing E-I;
    # in-party only -> -1.
    import pandas as pd
    _df = pd.DataFrame({"party": ["Democrat"] * 3 + ["Republican"] * 2 + ["Non-partisan"],
                        "q1_dem_answer": True, "q1_rep_answer": True})
    assert abs(implied_ei(_df) - (2 * 3 * 2 - 3 * 2 - 2 * 1) / (5 * 4)) < 1e-12
    _df["q1_dem_answer"], _df["q1_rep_answer"] = _df["party"] == "Democrat", _df["party"] == "Republican"
    assert implied_ei(_df) == -1
    # ...and that in-party-only population has a follow gap of 1 - 0.
    assert interview_gap(_df, FOLLOW_GAP_ROLE_MAPS["Party"], False) == 1
    main()
