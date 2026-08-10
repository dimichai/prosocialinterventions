import argparse
import os
import sys
import json
import random
from typing import Literal

import dotenv
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


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

THERMOMETER_INTRO = (
    "I’d like to get your feelings toward some of our political leaders and other "
    "people who are in the news these days. I’ll read the name of a person and I’d "
    "like you to rate that person using something we call the feeling thermometer.\n\n"
    "Ratings between 50 degrees and 100 degrees mean that you feel favorable and warm "
    "toward the person. Ratings between 0 degrees and 50 degrees mean that you don’t "
    "feel favorable toward the person and that you don’t care too much for that "
    "person. You would rate the person at the 50 degree mark if you don’t feel "
    "particularly warm or cold toward the person.\n\n"
    "If we come to a person whose name you don’t recognize, you don’t need to rate "
    "that person. Just tell me and we’ll move on to the next one."
)

# role -> display label shown in the question text
THERMOMETER_TARGETS = [
    ("biden", "Joe Biden"),
    ("trump", "Donald Trump"),
]


class BooleanAnswer(BaseModel):
    choice: Literal["yes", "no"]
    explanation: str


class TraitAnswer(BaseModel):
    # Trait questions ask a persona to characterize an entire political party.
    # A non-partisan under heavy obfuscation may genuinely have no basis to
    # judge the (obfuscated) group, so we let them opt out instead of forcing
    # a "no" that would be indistinguishable from genuine negativity.
    choice: Literal["yes", "no", "dont_know"]
    explanation: str


class ThermometerAnswer(BaseModel):
    recognized: bool
    rating: int | None   # 0-100, None if not recognized


def _system_message(persona: dict) -> str:
    return (
        "You are a user of the X social media platform. "
        "This is a platform where users share opinions and thoughts on topics of interest "
        "in the form of posts.\n\n"
        "Here is a description of your persona:\n"
        f"{persona['persona']}"
    )


def ask_question(
    client: OpenAI, persona: dict, question: str, model: str, allow_dont_know: bool = False
) -> tuple[bool | str, str]:
    """Send a question to the LLM and return (answer, explanation).

    `answer` is True/False for a yes/no question. When `allow_dont_know` is set
    (used for the trait battery), the persona may instead answer "dont_know",
    which is returned as the literal string "dont_know" rather than being
    coerced into a boolean.
    """

    response_format = TraitAnswer if allow_dont_know else BooleanAnswer
    instruction = (
        "Reply with 'yes', 'no', or 'dont_know' if you don't know enough about "
        "this group to have an opinion. Also provide a short explanation for your answer."
        if allow_dont_know else
        "Reply with 'yes' or 'no'. Also provide a short explanation for your answer."
    )

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _system_message(persona)},
            {"role": "user", "content": f"{question}\n\n{instruction}"},
        ],
        response_format=response_format,
        temperature=1.0,
    )

    parsed = response.choices[0].message.parsed
    choice = parsed.choice.strip().lower()
    if choice == "dont_know":
        return "dont_know", parsed.explanation
    return choice == "yes", parsed.explanation


def ask_feeling_thermometer_single(client: OpenAI, persona: dict, label: str, model: str) -> tuple[bool, int | None]:
    """Rate a single thermometer target in its own call, with no other target in context."""

    prompt = (
        f"{THERMOMETER_INTRO}\n\n"
        f"How would you rate: {label}\n\n"
        "Indicate whether you recognize this person, and if so, give a "
        "whole-number rating between 0 and 100."
    )

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _system_message(persona)},
            {"role": "user", "content": prompt},
        ],
        response_format=ThermometerAnswer,
        temperature=1.0,
    )

    parsed = response.choices[0].message.parsed
    return parsed.recognized, parsed.rating


def ask_feeling_thermometer(client: OpenAI, persona: dict, targets: list[tuple[str, str]], model: str) -> dict:
    """Rate each thermometer target with a separate call (avoids order/anchoring
    effects from batching multiple targets into one comparative call)."""

    row = {}
    for role, label in targets:
        recognized, rating = ask_feeling_thermometer_single(client, persona, label, model)
        row[f"{role}_therm_recognized"] = recognized
        row[f"{role}_therm_rating"]     = rating
    return row


def interview_personas(
    personas_file: str,
    output_file: str | None,
    questions: list[tuple[str, str]],
    thermometer_targets: list[tuple[str, str]],
    model: str,
    persona_sample: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:

    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY_1"),
    )

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
        print(f"[{i + 1}/{len(personas)}] Interviewing persona…")

        row = {
            "persona_index": i,
            "persona_text":  persona.get("persona", ""),
            "party":         persona.get("party", ""),
            "age":           persona.get("age", ""),
            "gender":        persona.get("gender", ""),
            "race":          persona.get("race", ""),
            "state":         persona.get("state", ""),
        }

        for key, question in active_questions:
            allow_dont_know = key.startswith(("dem_", "rep_"))
            answer, explanation = ask_question(client, persona, question, model, allow_dont_know=allow_dont_know)
            row[f"{key}_answer"]      = answer          # True / False / "dont_know"
            row[f"{key}_explanation"] = explanation

        row.update(ask_feeling_thermometer(client, persona, thermometer_targets, model))

        results.append(row)

    df = pd.DataFrame(results)
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    client.close()
    return df


def _pct_yes_no_dontknow(answers: pd.Series) -> tuple[float, float, int]:
    """From a raw `{key}_answer` column (True/False/"dont_know"/None), return
    (pct_yes, pct_dont_know, n_answered) for one run."""
    is_dont_know = answers.eq("dont_know")
    n_total = int(answers.notna().sum())
    yes_no = answers[~is_dont_know].map({True: 1.0, False: 0.0})
    n_answered = int(yes_no.notna().sum())
    pct_yes = yes_no.mean() if n_answered else float("nan")
    pct_dont_know = (is_dont_know.sum() / n_total) if n_total else float("nan")
    return pct_yes, pct_dont_know, n_answered


def _pct_recognized_and_rating(recognized: pd.Series, rating: pd.Series) -> tuple[float, float, int]:
    """From raw `{role}_therm_recognized`/`{role}_therm_rating` columns, return
    (pct_recognized, mean_rating_among_recognized, n_recognized) for one run."""
    rec_numeric = recognized.map({True: 1.0, False: 0.0})
    n_total = int(rec_numeric.notna().sum())
    pct_recognized = rec_numeric.mean() if n_total else float("nan")
    recognized_ratings = pd.to_numeric(rating[recognized == True], errors="coerce")
    mean_rating = recognized_ratings.mean() if recognized_ratings.notna().any() else float("nan")
    return pct_recognized, mean_rating, n_total


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
    persona_interviews_analysis_ablation.py (dont_know excluded from the yes/no
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
                pct_yes, pct_dont_know, n = _pct_yes_no_dontknow(group[col])
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
                pct_recognized, mean_rating, n = _pct_recognized_and_rating(
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
    parser = argparse.ArgumentParser(description="Interview personas and plot yes/no + feeling-thermometer results by party.")
    parser.add_argument("--personas_setting", type=str,
                         default="20260123_personas_with_bio_2000_noLoveHate_noPartyId_",
                         help="Name (without .json) of the personas file in src/ to interview.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                         help="OpenRouter model id used to answer as each persona.")
    parser.add_argument("--persona_sample", type=int, default=None,
                         help="Number of personas to randomly sample; omit to interview all personas.")
    parser.add_argument("--seed", type=int, nargs="+", default=[42],
                         help="One or more random seeds used when --persona_sample is set. "
                              "With multiple seeds, the interview is run once per seed and "
                              "the per-party results are averaged (mean ± std) across runs "
                              "instead of writing raw per-persona rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    personas_setting = args.personas_setting
    sample_suffix = f"_sample{args.persona_sample}" if args.persona_sample else ""
    seed_suffix = f"_avg{len(args.seed)}seeds" if len(args.seed) > 1 else ""

    personas_file = os.path.join(os.path.dirname(__file__), f"../../src/{personas_setting}.json")
    output_file   = os.path.join(os.path.dirname(__file__), "results", f"persona_interview_results_{personas_setting}{sample_suffix}{seed_suffix}.csv")

    if len(args.seed) == 1:
        df = interview_personas(
            personas_file=personas_file,
            output_file=output_file,
            questions=QUESTIONS,
            thermometer_targets=THERMOMETER_TARGETS,
            model=args.model,
            persona_sample=args.persona_sample,
            seed=args.seed[0],
        )
    else:
        if args.persona_sample is None:
            print("Note: --persona_sample not set, so each seed re-interviews the full "
                  "population; averaging captures run-to-run LLM variability rather than "
                  "sampling variance.")
        dfs = []
        for i, seed in enumerate(args.seed):
            print(f"=== Seed {i + 1}/{len(args.seed)} (seed={seed}) ===")
            dfs.append(interview_personas(
                personas_file=personas_file,
                output_file=None,
                questions=QUESTIONS,
                thermometer_targets=THERMOMETER_TARGETS,
                model=args.model,
                persona_sample=args.persona_sample,
                seed=seed,
            ))
        df = aggregate_interview_runs(dfs, QUESTIONS, THERMOMETER_TARGETS)
        df.to_csv(output_file, index=False)
        print(f"Averaged results across {len(args.seed)} seeds saved to {output_file}")


if __name__ == "__main__":
    main()
