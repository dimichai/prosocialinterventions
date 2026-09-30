"""Cheap saliency-artifact checks that reuse the already-run `ablation_soccer`
batch (see run_persona_pipeline.py's --inject_soccer_statement), instead of
requiring a new rich/interview-style persona experiment.

Soccer preference is injected into each persona as a single, apolitical
sentence ("You love soccer." / "You hate soccer."), assigned at random and
balanced within party, independent of every other persona trait (see
PersonaGeneration/anes_generate_personas.py). Because it carries zero
real-world social salience, any downstream effect it produces -- the model
volunteering it unprompted, agents clustering by it in the follow network --
is unambiguously an artifact of the trait having been *stated* in the persona
text, not of it being intrinsically meaningful. That lets it stand in for the
mechanism proposed in review: that survey-derived personas can turn selected
identity markers into disproportionately salient behavioral instructions,
regardless of whether a survey (or, for soccer, nothing at all) suggests that
marker should matter to that person.

Two checks, both pure re-analysis of the existing `ablation_soccer` wandb
runs (no new LLM calls or simulations):

1. Leakage: does "soccer" show up unprompted in the free-text explanations
   the model gives for unrelated political interview questions, and does
   that rate change as political content (love/hate -> party ID -> vote
   history) is progressively stripped from the persona -- i.e. does the
   model reach for whatever stated trait is left once the "real" one is gone?
2. Homophily: in the simulated follow network, does clustering by soccer
   stance (love vs. hate) rival clustering by party -- computed with the
   exact same Krackhardt & Stern E-I index used for party in
   src/main.py::compute_metrics, generalized to an arbitrary node->group
   mapping, on the same runs' pickled Platform objects.

Usage:
    python analysis/soccer_saliency_analysis.py --batch_id ablation_soccer
"""

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "persona_interviews"))
# Needed to unpickle a Platform artifact -- it was pickled from within src/
# (module names "Platform"/"Agent"), same as analysis/dimi_analysis.py's setup.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import interview_wandb  # noqa: E402
from simulation_comparison_plots import fig_path  # noqa: E402  (also applies the shared plot style on import)
from simulation_analysis_ablation import (  # noqa: E402
    CUSTOM_ABLATION_LABELS,  # noqa: F401  (re-exported for convenience; used implicitly via ablation_label)
    CUSTOM_ABLATION_COLORS,
    CONDITION_PALETTE,
    ablation_label,
)

from persona_simulation_analysis_ablation import get_metric_from_run  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

SOCCER_LOVE_RE = re.compile(r"you love soccer", re.IGNORECASE)
SOCCER_HATE_RE = re.compile(r"you hate soccer", re.IGNORECASE)
SOCCER_MENTION_RE = re.compile(r"soccer", re.IGNORECASE)


def fetch_ablation_soccer_runs(
    batch_id: str, wandb_project: str, exclude_run_ids: frozenset[str] = frozenset()
) -> dict[str, list]:
    """Runs sharing `batch_id`, bucketed by display label (baseline first, then
    alphabetical by ablation combo) -- same convention as
    persona_interviews_analysis_ablation.py::fetch_condition_dfs and
    simulation_analysis_ablation.py::fetch_runs_by_ablations. run_persona_pipeline.py
    logs one seed/condition's interview results *and* platform artifact into a single
    wandb run, so this one fetch feeds both checks below.

    `exclude_run_ids` skips specific run ids outright (e.g. a run still in
    progress, whose artifacts aren't fully logged yet) rather than erroring or
    silently treating it as done."""
    runs = interview_wandb.fetch_runs_by_group(wandb_project, batch_id)
    if not runs:
        raise RuntimeError(f"No wandb runs found in project '{wandb_project}' for group '{batch_id}'.")
    if exclude_run_ids:
        skipped = [run.id for run in runs if run.id in exclude_run_ids]
        for run_id in skipped:
            print(f"Skipping run '{run_id}' (excluded).")
        runs = [run for run in runs if run.id not in exclude_run_ids]
        if not runs:
            raise RuntimeError(f"All runs for group '{batch_id}' were excluded via --exclude_run_ids.")

    runs_by_ablations: dict[tuple[str, ...], list] = defaultdict(list)
    for run in runs:
        runs_by_ablations[tuple(sorted(run.config["ablations"]))].append(run)

    ordered_combos = sorted(runs_by_ablations, key=lambda c: (c != (), c))
    return {ablation_label(combo): runs_by_ablations[combo] for combo in ordered_combos}


def condition_color(label: str, index: int) -> str:
    return CUSTOM_ABLATION_COLORS.get(label, CONDITION_PALETTE[index % len(CONDITION_PALETTE)])


def save_fig(fig, base_name: str, batch_id: str) -> None:
    """Save a figure as both .pdf (vector, matches the rest of the analysis
    scripts) and .png (raster, for quick viewing/sharing)."""
    for ext in ("pdf", "png"):
        path = fig_path(base_name, batch_id, ext=ext)
        fig.savefig(path)
        print(f"Saved {path}")


def soccer_stance(persona_text: str) -> str | None:
    """'love'/'hate' from the injected sentence in a persona's raw text, or None
    if this persona wasn't given the injection."""
    if SOCCER_LOVE_RE.search(persona_text):
        return "love"
    if SOCCER_HATE_RE.search(persona_text):
        return "hate"
    return None


# --------------------------------------------------------------------------
# Check 1: does "soccer" leak, unprompted, into explanations for unrelated
# political questions?
# --------------------------------------------------------------------------

def explanation_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_explanation")]


def compute_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-persona: soccer stance (read off persona_text) and whether "soccer" is
    mentioned anywhere across this persona's explanation columns in this run."""
    stance = df["persona_text"].fillna("").map(soccer_stance)
    mentions = pd.Series(False, index=df.index)
    for col in explanation_columns(df):
        mentions = mentions | df[col].fillna("").str.contains(SOCCER_MENTION_RE)
    return pd.DataFrame({
        "persona_index": df["persona_index"],
        "stance": stance,
        "mentions_soccer": mentions,
    })


def leakage_by_condition(runs_by_label: dict[str, list]) -> pd.DataFrame:
    """Per ablation condition: fraction of soccer-injected personas whose
    explanations mention "soccer" at least once, averaged across seeds
    (mean +/- std, same aggregation convention as the rest of the ablation
    analysis scripts)."""
    rows = []
    for label, runs in runs_by_label.items():
        per_seed_rates = []
        n_total = 0
        for run in runs:
            df = interview_wandb.download_results_dataframe(run)
            leak = compute_leakage(df)
            leak = leak[leak["stance"].notna()]
            if leak.empty:
                continue
            per_seed_rates.append(leak["mentions_soccer"].mean())
            n_total += len(leak)
        if not per_seed_rates:
            continue
        rates = pd.Series(per_seed_rates)
        rows.append({
            "condition": label,
            "n_runs": len(per_seed_rates),
            "n_personas": n_total,
            "mention_rate_mean": rates.mean(),
            "mention_rate_std": rates.std(),
        })
    return pd.DataFrame(rows)


def print_leakage_table(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}")
    print("  Soccer leakage: fraction of personas whose explanations mention")
    print('  "soccer" unprompted while answering unrelated political questions')
    print(f"{'=' * 70}")
    if df.empty:
        print("  (no data)")
        return
    for _, row in df.iterrows():
        print(f"  {row['condition']:<20} mention_rate={row['mention_rate_mean']:.3f} "
              f"(+/- {row['mention_rate_std']:.3f}, n_runs={row['n_runs']}, n_personas={row['n_personas']})")


def plot_leakage(df: pd.DataFrame, batch_id: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [condition_color(c, i) for i, c in enumerate(df["condition"])]
    ax.bar(df["condition"], df["mention_rate_mean"], yerr=df["mention_rate_std"], color=colors, capsize=4)
    ax.set_title('Fraction of agents mentioning "soccer"')
    ax.set_ylim(0, max(0.05, float(df["mention_rate_mean"].max()) * 1.3))
    plt.xticks(rotation=20, ha="right")
    save_fig(fig, "soccer_leakage", batch_id)
    plt.close(fig)


# --------------------------------------------------------------------------
# Check 2: does the follow network cluster by soccer stance the way it
# clusters by party?
# --------------------------------------------------------------------------

def ei_index(links: list[tuple[int, int]], group_of: dict[int, str | None]) -> float | None:
    """Krackhardt & Stern E-I index, (external - internal) / (external + internal),
    computed once over the whole network -- same formula/semantics as
    src/main.py::compute_metrics's EI_index, generalized to an arbitrary
    node -> group label instead of hardcoding persona['party']. -1 = fully
    clustered within group, +1 = fully mixed across groups."""
    scored = [(group_of.get(u1), group_of.get(u2)) for u1, u2 in links]
    scored = [(g1, g2) for g1, g2 in scored if g1 is not None and g2 is not None]
    if not scored:
        return None
    internal = sum(1 for g1, g2 in scored if g1 == g2)
    external = len(scored) - internal
    return (external - internal) / (external + internal)


def load_platform(run):
    """Download (or reuse a cached copy of) a run's pickled Platform object --
    same pattern as analysis/dimi_analysis.py's network-plotting loop."""
    run_dir = ARTIFACTS_DIR / run.id
    pkl_files = list(run_dir.glob("*.pkl"))
    if not pkl_files:
        artifacts = list(run.logged_artifacts())
        platform_artifact = next((a for a in artifacts if a.type == "platform"), None)
        if platform_artifact is None:
            return None
        platform_artifact.download(root=str(run_dir))
        pkl_files = list(run_dir.glob("*.pkl"))
    if not pkl_files:
        return None
    with open(pkl_files[0], "rb") as f:
        return pickle.load(f)


def homophily_by_condition(runs_by_label: dict[str, list]) -> pd.DataFrame:
    """Per ablation condition: E-I index by party and by soccer stance, both
    recomputed from the same raw follow-edge lists so they're directly
    comparable, averaged across seeds (mean +/- std). Also keeps the raw
    per-seed value lists (party_ei_vals/soccer_ei_vals) for the Mann-Whitney
    significance test in plot_homophily."""
    rows = []
    for label, runs in runs_by_label.items():
        party_vals, soccer_vals = [], []
        for run in runs:
            platform = load_platform(run)
            if platform is None or not platform.user_links:
                continue
            party_of = {u.identifier: u.persona.get("party") for u in platform.users}
            soccer_of = {u.identifier: u.persona.get("inject_soccer_statement") for u in platform.users}
            party_ei = ei_index(platform.user_links, party_of)
            soccer_ei = ei_index(platform.user_links, soccer_of)
            if party_ei is not None:
                party_vals.append(party_ei)
            if soccer_ei is not None:
                soccer_vals.append(soccer_ei)
        if not party_vals and not soccer_vals:
            continue
        rows.append({
            "condition": label,
            "n_runs": len(runs),
            "party_ei_mean": pd.Series(party_vals).mean() if party_vals else float("nan"),
            "party_ei_std": pd.Series(party_vals).std() if party_vals else float("nan"),
            "party_ei_vals": party_vals,
            "soccer_ei_mean": pd.Series(soccer_vals).mean() if soccer_vals else float("nan"),
            "soccer_ei_std": pd.Series(soccer_vals).std() if soccer_vals else float("nan"),
            "soccer_ei_vals": soccer_vals,
        })
    return pd.DataFrame(rows)


def print_homophily_table(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}")
    print("  Follow-network homophily (E-I index: -1 = fully clustered by")
    print("  group, +1 = fully mixed) -- party vs. an arbitrary stated trait")
    print(f"{'=' * 70}")
    if df.empty:
        print("  (no data)")
        return
    for _, row in df.iterrows():
        print(f"  {row['condition']:<20} party={row['party_ei_mean']:.3f} (+/-{row['party_ei_std']:.3f})   "
              f"soccer={row['soccer_ei_mean']:.3f} (+/-{row['soccer_ei_std']:.3f})   n_runs={row['n_runs']}")


def draw_star(ax, xpos: float, mean: float, std: float) -> None:
    """'*' just past the end of a bar's error bar (above for positive bars,
    below for negative) -- same convention as
    simulation_comparison_plots.py::plot_metrics_comparison."""
    err = 0.0 if pd.isna(std) else std
    if mean >= 0:
        y_pos, va = mean + err + 0.03, "bottom"
    else:
        y_pos, va = mean - err - 0.03, "top"
    ax.text(xpos, y_pos, "*", ha="center", va=va, fontsize=13, fontweight="bold", color="#333333")


def significant(a: list[float], b: list[float], alpha: float) -> bool:
    if len(a) < 2 or len(b) < 2:
        return False
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return p < alpha


def draw_homophily(ax, df: pd.DataFrame, alpha: float = 0.05) -> None:
    """Grouped bars of party- vs. soccer-based E-I index per condition, with
    '*' on bars that differ significantly (Mann-Whitney, p < alpha) from the
    first (baseline) condition, separately for party and soccer."""
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["party_ei_mean"], width, yerr=df["party_ei_std"], label="Party-based", color="#4878A8", capsize=4)
    ax.bar(x + width / 2, df["soccer_ei_mean"], width, yerr=df["soccer_ei_std"], label="Soccer-based", color="#eb6834", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)

    for i in range(1, len(df)):
        for offset, key in ((-width / 2, "party_ei"), (width / 2, "soccer_ei")):
            if significant(df[f"{key}_vals"].iloc[0], df[f"{key}_vals"].iloc[i], alpha):
                draw_star(ax, x[i] + offset, df[f"{key}_mean"].iloc[i], df[f"{key}_std"].iloc[i])

    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], rotation=20, ha="right")
    ax.legend()


def plot_homophily(df: pd.DataFrame, batch_id: str, alpha: float = 0.05) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_homophily(ax, df, alpha)
    ax.set_title("EI Index")
    save_fig(fig, "soccer_homophily", batch_id)
    plt.close(fig)


# --------------------------------------------------------------------------
# Check 2b: does injecting soccer change party clustering itself? Party E-I
# index of a no-soccer ablation batch (e.g. abl_500) vs. the soccer batch.
# --------------------------------------------------------------------------

def logged_party_ei(runs_by_label: dict[str, list]) -> dict[str, list[float]]:
    """Per condition: the per-seed party EI_index logged by
    src/main.py::compute_metrics (not recomputed from the pickle), so both
    batches in the comparison come from the exact same code path."""
    vals_by_label = {}
    for label, runs in runs_by_label.items():
        vals = [get_metric_from_run(run, "EI_index") for run in runs]
        vals_by_label[label] = [v for v in vals if v is not None]
    return vals_by_label


def party_ei_comparison(no_soccer_runs: dict[str, list], soccer_runs: dict[str, list]) -> pd.DataFrame:
    """Logged party EI per condition for both batches, restricted to the
    conditions present in both (in the soccer batch's order)."""
    no_soccer = logged_party_ei({l: r for l, r in no_soccer_runs.items() if l in soccer_runs})
    soccer = logged_party_ei({l: r for l, r in soccer_runs.items() if l in no_soccer_runs})
    rows = []
    for label in soccer:
        a, b = no_soccer[label], soccer[label]
        if not a and not b:
            continue
        rows.append({
            "condition": label,
            "no_soccer_mean": np.mean(a) if a else float("nan"),
            "no_soccer_std": pd.Series(a).std() if a else float("nan"),
            "no_soccer_vals": a,
            "soccer_mean": np.mean(b) if b else float("nan"),
            "soccer_std": pd.Series(b).std() if b else float("nan"),
            "soccer_vals": b,
        })
    return pd.DataFrame(rows)


def print_party_ei_comparison(df: pd.DataFrame, compare_batch_id: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Party E-I index (logged): {compare_batch_id} (no soccer) vs. soccer batch")
    print(f"{'=' * 70}")
    if df.empty:
        print("  (no shared conditions)")
        return
    for _, row in df.iterrows():
        print(f"  {row['condition']:<20} no_soccer={row['no_soccer_mean']:.3f} (+/-{row['no_soccer_std']:.3f}, n={len(row['no_soccer_vals'])})   "
              f"soccer={row['soccer_mean']:.3f} (+/-{row['soccer_std']:.3f}, n={len(row['soccer_vals'])})")


def plot_homophily_combined(party_df: pd.DataFrame, homophily_df: pd.DataFrame, batch_id: str,
                            alpha: float = 0.05) -> None:
    """Left: party E-I index per condition, no-soccer batch vs. soccer batch,
    '*' where the two differ (Mann-Whitney, p < alpha). Right: the
    soccer_homophily panel (party- vs. soccer-based E-I in the soccer batch)."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

    x = np.arange(len(party_df))
    width = 0.35
    ax_l.bar(x - width / 2, party_df["no_soccer_mean"], width, yerr=party_df["no_soccer_std"],
             label="No Soccer", color="#A6C3E0", capsize=4)
    ax_l.bar(x + width / 2, party_df["soccer_mean"], width, yerr=party_df["soccer_std"],
             label="Soccer", color="#4878A8", capsize=4)
    ax_l.axhline(0, color="black", linewidth=0.8)
    for i, row in enumerate(party_df.itertuples()):
        if significant(row.no_soccer_vals, row.soccer_vals, alpha):
            # Star over (or under) whichever bar reaches further from zero.
            candidates = [(row.no_soccer_mean, row.no_soccer_std), (row.soccer_mean, row.soccer_std)]
            mean, std = max(candidates, key=lambda ms: abs(ms[0]) + (0.0 if pd.isna(ms[1]) else ms[1]))
            draw_star(ax_l, x[i], mean, std)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(party_df["condition"], rotation=20, ha="right")
    ax_l.set_title("Party-based EI index")
    ax_l.set_ylabel("E-I index")
    ax_l.legend()

    draw_homophily(ax_r, homophily_df, alpha)
    ax_r.set_title("Party vs. soccer EI index")
    ax_r.tick_params(labelleft=True)

    save_fig(fig, "soccer_homophily_combined", batch_id)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-analyze the already-run `ablation_soccer` batch as a cheap "
                     "saliency check, without a new rich-persona experiment: (1) does "
                     "the apolitical, randomly-assigned soccer trait leak unprompted "
                     "into the model's explanations for political questions, and (2) "
                     "does the simulated follow network cluster by soccer stance the "
                     "way it clusters by party."
    )
    parser.add_argument("--batch_id", type=str, default="ablation_soccer",
                         help="Wandb group/batch id (default: the existing ablation_soccer batch).")
    parser.add_argument("--wandb_project", type=str, default="persona-simulation",
                         help="Wandb project the batch was logged to (run_persona_pipeline.py's default).")
    parser.add_argument("--exclude_run_ids", type=str, nargs="*", default=["lr8g6heb"],
                         help="Wandb run ids to skip (e.g. a run still in progress). "
                              "Defaults to the ablation_soccer batch's currently-unfinished run.")
    parser.add_argument("--compare_batch_id", type=str, default="abl_500",
                         help="No-soccer ablation batch whose party E-I index is compared against "
                              "--batch_id's in the combined homophily figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_by_label = fetch_ablation_soccer_runs(
        args.batch_id, args.wandb_project, exclude_run_ids=frozenset(args.exclude_run_ids)
    )

    leakage_df = leakage_by_condition(runs_by_label)
    print_leakage_table(leakage_df)
    if not leakage_df.empty:
        plot_leakage(leakage_df, args.batch_id)

    homophily_df = homophily_by_condition(runs_by_label)
    print_homophily_table(homophily_df)
    if not homophily_df.empty:
        plot_homophily(homophily_df, args.batch_id)

    no_soccer_runs = fetch_ablation_soccer_runs(args.compare_batch_id, args.wandb_project)
    party_df = party_ei_comparison(no_soccer_runs, runs_by_label)
    print_party_ei_comparison(party_df, args.compare_batch_id)
    if not party_df.empty and not homophily_df.empty:
        plot_homophily_combined(party_df, homophily_df, args.batch_id)


if __name__ == "__main__":
    main()
