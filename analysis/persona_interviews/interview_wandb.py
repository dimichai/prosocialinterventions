"""Shared wandb read/write helpers for persona-interview results.

Mirrors the pattern used for simulations (src/main.py / analysis/dimi_analysis.py):
config logged at wandb.init, per-run results logged as metrics (so runs are
browsable/plottable/sortable in the wandb UI without downloading anything), and
the raw results uploaded as a wandb.Artifact for exact reproduction/aggregation.
"""

import os
import tempfile

import pandas as pd
import wandb

WANDB_PROJECT = "prosocial-interventions-obf-interviews"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "wandb_cache")


def pct_yes_no_dontknow(answers: pd.Series) -> tuple[float, float, int]:
    """From a raw `{key}_answer` column (True/False/"dont_know"/None), return
    (pct_yes, pct_dont_know, n_answered) for one run."""
    is_dont_know = answers.eq("dont_know")
    n_total = int(answers.notna().sum())
    # Columns mixing bool with the "dont_know" string can't be inferred as bool
    # dtype by pandas, so a CSV round-trip (upload_results_artifact ->
    # download_results_dataframe) turns True/False into the strings "True"/"False"
    # rather than leaving them as real Python bools — map both forms to be safe.
    yes_no = answers[~is_dont_know].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    n_answered = int(yes_no.notna().sum())
    pct_yes = yes_no.mean() if n_answered else float("nan")
    pct_dont_know = (is_dont_know.sum() / n_total) if n_total else float("nan")
    return pct_yes, pct_dont_know, n_answered


def pct_recognized_and_rating(recognized: pd.Series, rating: pd.Series) -> tuple[float, float, int]:
    """From raw `{role}_therm_recognized`/`{role}_therm_rating` columns, return
    (pct_recognized, mean_rating_among_recognized, n_recognized) for one run."""
    rec_numeric = recognized.map({True: 1.0, False: 0.0})
    n_total = int(rec_numeric.notna().sum())
    pct_recognized = rec_numeric.mean() if n_total else float("nan")
    recognized_ratings = pd.to_numeric(rating[recognized == True], errors="coerce")
    mean_rating = recognized_ratings.mean() if recognized_ratings.notna().any() else float("nan")
    return pct_recognized, mean_rating, n_total


def result_metrics(
    df: pd.DataFrame,
    questions: list[tuple[str, str]],
    thermometer_targets: list[tuple[str, str]],
) -> dict:
    """Per-(key, party) and per-(role, party) result stats for this one run, as a
    flat dict of wandb metric keys (question/{key}/{party}/pct_yes,
    thermometer/{role}/{party}/rating, etc)."""
    metrics = {}

    for key, question in questions:
        if not question.strip():
            continue
        col = f"{key}_answer"
        if col not in df.columns:
            continue
        for party, group in df.groupby("party"):
            pct_yes, pct_dont_know, _ = pct_yes_no_dontknow(group[col])
            metrics[f"question/{key}/{party}/pct_yes"] = pct_yes
            metrics[f"question/{key}/{party}/pct_dont_know"] = pct_dont_know

    for role, _ in thermometer_targets:
        recognized_col = f"{role}_therm_recognized"
        rating_col = f"{role}_therm_rating"
        if recognized_col not in df.columns:
            continue
        for party, group in df.groupby("party"):
            pct_recognized, mean_rating, n = pct_recognized_and_rating(group[recognized_col], group[rating_col])
            metrics[f"thermometer/{role}/{party}/pct_recognized"] = pct_recognized
            metrics[f"thermometer/{role}/{party}/rating"] = mean_rating
            metrics[f"thermometer/{role}/{party}/n"] = n

    return metrics


def persona_population_metrics(df: pd.DataFrame) -> dict:
    """Distribution of the (ANES-derived) persona attributes for the personas
    interviewed in this run — i.e. who was asked, not how they answered. Logged
    under a "personas/" prefix so it shows as its own section in the wandb UI,
    separate from the "question/"/"thermometer/" LLM-elicited results."""
    n = len(df)
    metrics = {"personas/n": n}
    if n == 0:
        return metrics

    party_counts = df["party"].value_counts(normalize=True)
    for party in ("Democrat", "Republican", "Non-partisan"):
        metrics[f"personas/pct_{party.lower().replace('-', '_')}"] = party_counts.get(party, 0.0)

    if "feelingDemocratic" in df.columns:
        metrics["personas/avg_feeling_democratic"] = pd.to_numeric(df["feelingDemocratic"], errors="coerce").mean()
    if "feelingRepublican" in df.columns:
        metrics["personas/avg_feeling_republican"] = pd.to_numeric(df["feelingRepublican"], errors="coerce").mean()

    if "partisan" in df.columns:
        partisan = pd.to_numeric(df["partisan"], errors="coerce")
        democrats, republicans = df["party"] == "Democrat", df["party"] == "Republican"
        if democrats.any():
            metrics["personas/avg_partisan_democrats"] = partisan[democrats].mean()
        if republicans.any():
            metrics["personas/avg_partisan_republicans"] = partisan[republicans].mean()

    if "voted2020_for" in df.columns:
        metrics["personas/pct_voted_trump"] = (df["voted2020_for"] == "Donald Trump").mean()
        metrics["personas/pct_voted_biden"] = (df["voted2020_for"] == "Joe Biden").mean()

    return metrics


def upload_results_artifact(df: pd.DataFrame, name: str) -> None:
    """Upload raw per-persona interview results as a wandb.Artifact, mirroring
    the platform-pickle artifact pattern in src/main.py."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as tmp:
        df.to_csv(tmp.name, index=False)
        artifact = wandb.Artifact(name=name, type="interview_results")
        artifact.add_file(tmp.name, name="interview_results.csv")
        wandb.log_artifact(artifact)


def download_results_dataframe(run, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Download (or reuse a cached copy of) a run's interview_results artifact
    and return it as a DataFrame."""
    run_dir = os.path.join(cache_dir, run.id)
    csv_path = os.path.join(run_dir, "interview_results.csv")
    if not os.path.exists(csv_path):
        artifacts = [a for a in run.logged_artifacts() if a.type == "interview_results"]
        if not artifacts:
            raise RuntimeError(f"Run '{run.id}' has no logged interview_results artifact.")
        artifacts[0].download(root=run_dir)
    return pd.read_csv(csv_path)


def fetch_runs_by_group(project: str, group: str):
    """Fetch all runs in `project` sharing the given wandb group (batch) id."""
    return list(wandb.Api().runs(project, filters={"group": group}))
