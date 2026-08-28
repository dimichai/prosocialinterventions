import argparse
import os
import sys
import json
import random
import uuid
from datetime import datetime
from typing import Literal

import dotenv
import pandas as pd
import wandb
from openai import OpenAI, LengthFinishReasonError
from pydantic import BaseModel, ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../PersonaGeneration'))
sys.path.insert(0, os.path.dirname(__file__))

import interview_wandb  # noqa: E402
from obfuscation_labels import (  # noqa: E402
    infer_obfuscation as _infer_obfuscation,
    get_political_figure_labels,
    get_party_labels,
    build_group_context,
)
from question_battery import (  # noqa: E402
    THERMOMETER_INTRO,
    build_questions,
    build_thermometer_targets,
)


# GPT-5-family models spend part of their completion budget on hidden reasoning
# tokens before writing the visible answer, so a max_tokens value that's ample
# for gpt-4o-mini-style models can still truncate them mid-JSON.
REASONING_MODEL_PREFIXES = ("gpt-5", "openai/gpt-5", "o1", "o3", "o4")
DEFAULT_MAX_TOKENS = 16384
REASONING_MODEL_MAX_TOKENS = 32768


def _max_tokens_for_model(model: str) -> int:
    if model.lower().startswith(REASONING_MODEL_PREFIXES):
        return REASONING_MODEL_MAX_TOKENS
    return DEFAULT_MAX_TOKENS


class BooleanAnswer(BaseModel):
    choice: Literal["yes", "no"]
    explanation: str


class TraitAnswer(BaseModel):
    # Trait questions ask a persona to characterize an entire political party.
    # Under obfuscation, a non-partisan may genuinely have no basis to judge
    # the (obfuscated) group, so we let them opt out instead of forcing a "no"
    # that would be indistinguishable from genuine negativity.
    choice: Literal["yes", "no", "dont_know"]
    explanation: str


class ThermometerAnswer(BaseModel):
    recognized: bool
    rating: int | None   # 0-100, None if not recognized


def _system_message(persona: dict, group_context: str = "") -> str:
    message = (
        "Here is a description of your persona:\n"
        f"{persona['persona']}"
    )
    if group_context:
        message += f"\n\n{group_context}"
    return message


def ask_question(
    client: OpenAI, persona: dict, question: str, model: str,
    allow_dont_know: bool = False, group_context: str = "",
) -> tuple[bool | str | None, str]:
    """Send a question to the LLM and return (answer, explanation).

    `answer` is True/False for a yes/no question. When `allow_dont_know` is set
    (used for the trait battery), the persona may instead answer "dont_know",
    returned as the literal string "dont_know" rather than being coerced into
    a boolean.

    Returns (None, "<error>") if the model's response was truncated before it
    could produce valid structured output (rare, but happens on some
    OpenRouter models/personas) rather than raising and aborting the whole run.
    """

    response_format = TraitAnswer if allow_dont_know else BooleanAnswer
    instruction = (
        "Reply with 'yes', 'no', or 'dont_know' if you don't know enough about "
        "this group to have an opinion. Also provide a short explanation for your answer."
        if allow_dont_know else
        "Reply with 'yes' or 'no'. Also provide a short explanation for your answer."
    )

    for attempt in range(2):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": _system_message(persona, group_context)},
                    {"role": "user", "content": f"{question}\n\n{instruction}"},
                ],
                response_format=response_format,
                max_tokens=_max_tokens_for_model(model),
                temperature=1.0,
            )
            parsed = response.choices[0].message.parsed
            choice = parsed.choice.strip().lower()
            if choice == "dont_know":
                return "dont_know", parsed.explanation
            return choice == "yes", parsed.explanation
        except (LengthFinishReasonError, ValidationError):
            # Some OpenRouter providers truncate the completion without reporting
            # finish_reason="length", so the SDK doesn't raise LengthFinishReasonError
            # and instead fails to parse the incomplete JSON (ValidationError).
            print(f"    [warn] [id={persona.get('persona_index')}] truncated response on attempt {attempt + 1} for question: {question!r}")

    return None, "ERROR: response truncated (length limit reached) after retry"


def ask_feeling_thermometer_single(
    client: OpenAI, persona: dict, label: str, model: str, group_context: str = "",
) -> tuple[bool | None, int | None]:
    """Rate a single thermometer target in its own call, with no other target in context."""

    prompt = (
        f"{THERMOMETER_INTRO}\n\n"
        f"How would you rate: {label}\n\n"
        "Indicate whether you recognize this person, and if so, give a "
        "whole-number rating between 0 and 100."
    )

    for attempt in range(2):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": _system_message(persona, group_context)},
                    {"role": "user", "content": prompt},
                ],
                response_format=ThermometerAnswer,
                max_tokens=_max_tokens_for_model(model),
                temperature=1.0,
            )
            parsed = response.choices[0].message.parsed
            return parsed.recognized, parsed.rating
        except (LengthFinishReasonError, ValidationError):
            print(f"    [warn] [id={persona.get('persona_index')}] truncated thermometer response on attempt {attempt + 1} for {label!r}")

    return None, None


def ask_feeling_thermometer(
    client: OpenAI, persona: dict, targets: list[tuple[str, str]], model: str, group_context: str = "",
) -> dict:
    """Rate each thermometer target with a separate call (avoids order/anchoring
    effects from batching multiple targets into one comparative call)."""

    row = {}
    for role, label in targets:
        recognized, rating = ask_feeling_thermometer_single(client, persona, label, model, group_context)
        row[f"{role}_therm_recognized"] = recognized
        row[f"{role}_therm_rating"]     = rating
        print(f"    [id={persona.get('persona_index')}] [{role}_therm] recognized={recognized!r} rating={rating!r}")
    return row


def interview_personas(
    personas_file: str | None,
    questions: list[tuple[str, str]],
    thermometer_targets: list[tuple[str, str]],
    model: str,
    personas: list[dict] | None = None,
    persona_sample: int | None = None,
    seed: int = 42,
    group_context: str = "",
) -> pd.DataFrame:
    """Interview either an in-memory `personas` list, or (when not given) the
    personas loaded from `personas_file`."""

    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY_1"),
    )

    if personas is None:
        personas = json.load(open(personas_file, "r"))
    if persona_sample is not None:
        personas = random.Random(seed).sample(personas, min(persona_sample, len(personas)))

    active_questions = [(key, q) for key, q in questions if q.strip()]

    if not active_questions:
        print("No questions defined yet – fill in the QUESTIONS list and re-run.")
        return pd.DataFrame()

    print(f"Interviewing {len(personas)} persona(s) with {len(active_questions)} question(s):")
    for key, question in active_questions:
        print(f"  [{key}] {question}")
    print(f"Feeling thermometer ({len(thermometer_targets)} target(s)):")
    for role, label in thermometer_targets:
        print(f"  [{role}] {label}")

    results = []

    for i, persona in enumerate(personas):
        persona_id = persona.get("persona_index", i)
        print(f"[{i + 1}/{len(personas)}] Interviewing persona (id={persona_id})…")

        row = {
            "persona_index": persona_id,
            "persona_text":  persona.get("persona", ""),
            "party":         persona.get("party", ""),
            "age":           persona.get("age", ""),
            "gender":        persona.get("gender", ""),
            "race":          persona.get("race", ""),
            "state":         persona.get("state", ""),
            # Raw ANES-derived attributes (ground truth, not LLM-elicited) — kept
            # here so run-level population stats can be computed from this same
            # dataframe (see interview_wandb.persona_population_metrics).
            "feelingDemocratic": persona.get("feelingDemocratic"),
            "feelingRepublican": persona.get("feelingRepublican"),
            "partisan":          persona.get("partisan"),
            "voted2020_for":     persona.get("voted2020_for"),
        }

        for key, question in active_questions:
            allow_dont_know = key.startswith(("dem_", "rep_"))
            answer, explanation = ask_question(
                client, persona, question, model,
                allow_dont_know=allow_dont_know, group_context=group_context,
            )
            row[f"{key}_answer"]      = answer          # True / False / "dont_know"
            row[f"{key}_explanation"] = explanation
            print(f"    [id={persona_id}] [{key}] {answer!r} — {explanation}")

        row.update(ask_feeling_thermometer(client, persona, thermometer_targets, model, group_context))

        results.append(row)

    df = pd.DataFrame(results)
    client.close()
    return df


def aggregate_interview_runs(
    dfs: list[pd.DataFrame],
    questions: list[tuple[str, str]],
    thermometer_targets: list[tuple[str, str]],
) -> pd.DataFrame:
    """Aggregate multiple interview runs (one per seed) into per-party summary
    statistics, averaged across runs (mean ± std).

    Each run samples a different set of personas, so there's no persona-to-persona
    correspondence across runs. Instead this computes a per-party summary stat
    (yes-rate per question, mean thermometer rating) within each run, then averages
    those summary stats across runs — mirroring the aggregation convention used in
    persona_interviews_analysis_obfuscation.py (dont_know excluded from the yes/no
    rate, grouped by `party`). Free-text explanations aren't averaged and are
    dropped from this output.
    """
    active_questions = [(key, q) for key, q in questions if q.strip()]
    rows = []

    for key, label in active_questions:
        col = f"{key}_answer"
        per_party_runs: dict[str, list[dict]] = {}
        for df in dfs:
            if col not in df.columns:
                continue
            for party, group in df.groupby("party"):
                pct_yes, pct_dont_know, n = interview_wandb.pct_yes_no_dontknow(group[col])
                per_party_runs.setdefault(party, []).append(
                    {"pct_yes": pct_yes, "pct_dont_know": pct_dont_know, "n": n}
                )

        for party, run_stats in per_party_runs.items():
            pct_yes = pd.Series([r["pct_yes"] for r in run_stats])
            pct_dk = pd.Series([r["pct_dont_know"] for r in run_stats])
            ns = pd.Series([r["n"] for r in run_stats])
            rows.append({
                "metric": "question",
                "key": key,
                "label": label,
                "party": party,
                "n_runs": len(run_stats),
                "avg_n": ns.mean(),
                "pct_yes_mean": pct_yes.mean(),
                "pct_yes_std": pct_yes.std(),
                "pct_dont_know_mean": pct_dk.mean(),
                "pct_dont_know_std": pct_dk.std(),
                "pct_recognized_mean": float("nan"),
                "pct_recognized_std": float("nan"),
                "rating_mean": float("nan"),
                "rating_std": float("nan"),
            })

    for role, label in thermometer_targets:
        recognized_col = f"{role}_therm_recognized"
        rating_col = f"{role}_therm_rating"
        per_party_runs: dict[str, list[dict]] = {}
        for df in dfs:
            if recognized_col not in df.columns:
                continue
            for party, group in df.groupby("party"):
                pct_recognized, mean_rating, n = interview_wandb.pct_recognized_and_rating(
                    group[recognized_col], group[rating_col]
                )
                per_party_runs.setdefault(party, []).append(
                    {"pct_recognized": pct_recognized, "rating": mean_rating, "n": n}
                )

        for party, run_stats in per_party_runs.items():
            pct_rec = pd.Series([r["pct_recognized"] for r in run_stats])
            rating = pd.Series([r["rating"] for r in run_stats])
            ns = pd.Series([r["n"] for r in run_stats])
            rows.append({
                "metric": "thermometer",
                "key": role,
                "label": label,
                "party": party,
                "n_runs": len(run_stats),
                "avg_n": ns.mean(),
                "pct_yes_mean": float("nan"),
                "pct_yes_std": float("nan"),
                "pct_dont_know_mean": float("nan"),
                "pct_dont_know_std": float("nan"),
                "pct_recognized_mean": pct_rec.mean(),
                "pct_recognized_std": pct_rec.std(),
                "rating_mean": rating.mean(),
                "rating_std": rating.std(),
            })

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interview personas (with real or obfuscated Trump/Biden/party labels, "
                     "inferred from the personas file name) and plot yes/no + feeling-thermometer "
                     "results by party."
    )
    parser.add_argument("--personas_setting", type=str,
                         default="20260720_personas_with_bio_2000_noExtendWithAi_",
                         help="Name (without .json) of the personas file in src/ to interview. "
                              "The obfuscation condition (Neutral/Nonce/RandomReal/RandomNonce) is "
                              "inferred from this name's suffix.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                         help="OpenRouter model id used to answer as each persona.")
    parser.add_argument("--persona_sample", type=int, default=None,
                         help="Number of personas to randomly sample; omit to interview all personas.")
    parser.add_argument("--seed", type=int, nargs="+", default=[42],
                         help="One or more random seeds used when --persona_sample is set. "
                              "Each seed is run and logged to wandb as its own run "
                              "(sharing a common wandb group); aggregation across seeds "
                              "happens on the analysis side, reading back from wandb.")
    parser.add_argument("--wandb_project", type=str, default=interview_wandb.WANDB_PROJECT,
                         help="Wandb project to log interview runs to.")
    parser.add_argument("--no_log", action="store_true", default=False,
                         help="Skip wandb logging entirely (e.g. for local debugging runs).")
    parser.add_argument("--include_group_context", action="store_true", default=False,
                         help="Prepend a short paragraph to each persona's system message naming "
                              "the two rival political affiliations and their leader (see "
                              "build_group_context). Applied in every obfuscation condition, "
                              "including 'none', so it stays the only thing that's off by default "
                              "rather than a confound between conditions.")
    return parser.parse_args()


def run_interview_for_setting(
    personas_setting: str,
    model: str,
    persona_sample: int | None,
    seeds: list[int],
    wandb_group: str | None = None,
    extra_config: dict | None = None,
    log: bool = True,
    wandb_project: str = interview_wandb.WANDB_PROJECT,
    include_group_context: bool = False,
    personas: list[dict] | None = None,
    own_wandb_run: bool = True,
) -> dict:
    """Interview personas, once per seed, logging each seed's raw per-persona results
    to wandb. `personas_setting` is always required (even when `personas` is supplied
    directly) — it's used to resolve real/obfuscated labels and `group_context` via
    obfuscation_labels, independent of where the persona data itself comes from.

    By default (`personas=None`) reads personas from src/<personas_setting>.json, and
    each seed gets its own wandb run, all sharing one wandb group/batch id — aggregation
    across seeds is done later, on the analysis side, by reading the runs back from
    wandb. When `personas` is supplied directly, that in-memory list is interviewed
    instead of reading from disk.

    When `own_wandb_run=False` (used when this is one stage of a larger orchestrated
    run), this skips its own `wandb.init`/`wandb.finish()` per seed and instead logs
    into whatever wandb run is already active — for that case, `seeds` should be a
    single-element list, since the caller owns one run per seed. Returns identifying
    info about the run(s) that were logged."""

    trump_label, biden_label = get_political_figure_labels(personas_setting)
    democrats_label, republicans_label = get_party_labels(personas_setting)
    questions = build_questions(trump_label, biden_label, democrats_label, republicans_label)
    thermometer_targets = build_thermometer_targets(trump_label, biden_label, democrats_label, republicans_label)

    personas_file = os.path.join(os.path.dirname(__file__), f"../../src/{personas_setting}.json")

    obfuscation = (extra_config or {}).get("obfuscation") or _infer_obfuscation(personas_setting)
    # When enabled, applied in every condition, including "none" — the
    # label-recognition problem this solves is specific to obfuscation, but
    # injecting it unconditionally keeps every condition's prompt structure
    # identical except for the labels themselves, so obfuscation stays the only
    # manipulated variable.
    group_context = (
        build_group_context(trump_label, biden_label, democrats_label, republicans_label)
        if include_group_context else ""
    )
    base_config = {
        "obfuscation": obfuscation,
        "personas_setting": personas_setting,
        "model": model,
        "persona_sample": persona_sample,
        "num_seeds_in_batch": len(seeds),
        "trump_label": trump_label,
        "biden_label": biden_label,
        "democrats_label": democrats_label,
        "republicans_label": republicans_label,
        "include_group_context": include_group_context,
        "group_context": group_context,
    }
    if extra_config:
        base_config.update(extra_config)

    group = wandb_group or uuid.uuid4().hex[:8]
    run_ids = []

    if persona_sample is None and len(seeds) > 1:
        print("Note: --persona_sample not set, so each seed re-interviews the full "
              "population; comparing seeds captures run-to-run LLM variability rather "
              "than sampling variance.")

    for i, seed in enumerate(seeds):
        print(f"=== Seed {i + 1}/{len(seeds)} (seed={seed}) ===")

        if log and own_wandb_run:
            wandb.init(
                project=wandb_project,
                group=group,
                job_type="interview",
                tags=[obfuscation],
                name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{obfuscation}_seed{seed}",
                config={**base_config, "interview_seed": seed},
                reinit=True,
            )

        df = interview_personas(
            personas_file=personas_file,
            personas=personas,
            questions=questions,
            thermometer_targets=thermometer_targets,
            model=model,
            persona_sample=persona_sample,
            seed=seed,
            group_context=group_context,
        )

        if log:
            interview_wandb.upload_results_artifact(df, name=f"interview-results-{obfuscation}-seed{seed}")
            metrics = interview_wandb.persona_population_metrics(df)
            metrics.update(interview_wandb.result_metrics(df, questions, thermometer_targets))
            # In addition to the downloadable CSV artifact above, log the same
            # per-persona rows (individual answers + explanations) as a wandb
            # Table, so they're browsable/sortable/filterable in the run's UI
            # without downloading anything. wandb.Table requires one consistent
            # type per column, but `{key}_answer` columns mix True/False/
            # "dont_know"/None, so stringify a copy just for the table view (the
            # CSV artifact above keeps the original values for programmatic use).
            table_df = df.copy()
            answer_cols = [c for c in table_df.columns if c.endswith("_answer")]
            table_df[answer_cols] = table_df[answer_cols].astype(str)
            metrics["interview_results_table"] = wandb.Table(dataframe=table_df)
            wandb.log(metrics)
            run_ids.append(wandb.run.id)
            if own_wandb_run:
                wandb.finish()

    if log and own_wandb_run:
        # Only accurate when this function owns the run(s) it logged to — when
        # own_wandb_run=False, `wandb_project`/`group` are unused defaults, not
        # where the data actually went (the caller owns and reports that instead).
        print(f"Logged {len(seeds)} run(s) to wandb project '{wandb_project}' (group={group})")

    return {"group": group, "obfuscation": obfuscation, "run_ids": run_ids}


def main() -> None:
    args = parse_args()
    result = run_interview_for_setting(
        args.personas_setting, args.model, args.persona_sample, args.seed,
        log=not args.no_log, wandb_project=args.wandb_project,
        include_group_context=args.include_group_context,
    )
    print(result)


if __name__ == "__main__":
    main()
