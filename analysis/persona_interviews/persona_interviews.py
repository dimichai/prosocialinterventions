import argparse
import os
import sys
import json
import random
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
    choice: str        # "yes" or "no"
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


def ask_question(client: OpenAI, persona: dict, question: str, model: str) -> tuple[bool, str]:
    """Send a yes/no question to the LLM and return (answer_bool, explanation)."""

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _system_message(persona)},
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with 'yes' or 'no'. Also provide a short explanation for your answer."
                ),
            },
        ],
        response_format=BooleanAnswer,
    )

    parsed = response.choices[0].message.parsed
    return parsed.choice.strip().lower() == "yes", parsed.explanation


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
    output_file: str,
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
            answer, explanation = ask_question(client, persona, question, model)
            row[f"{key}_answer"]      = answer          # True / False
            row[f"{key}_explanation"] = explanation

        row.update(ask_feeling_thermometer(client, persona, thermometer_targets, model))

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    client.close()
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interview personas and plot yes/no + feeling-thermometer results by party.")
    parser.add_argument("--personas_setting", type=str,
                         default="20260123_personas_with_bio_2000_noLoveHate_noPartyId_",
                         help="Name (without .json) of the personas file in src/ to interview.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                         help="OpenRouter model id used to answer as each persona.")
    parser.add_argument("--persona_sample", type=int, default=None,
                         help="Number of personas to randomly sample; omit to interview all personas.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed used when --persona_sample is set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    personas_setting = args.personas_setting
    sample_suffix = f"_sample{args.persona_sample}" if args.persona_sample else ""

    personas_file = os.path.join(os.path.dirname(__file__), f"../../src/{personas_setting}.json")
    output_file   = os.path.join(os.path.dirname(__file__), "results", f"persona_interview_results_{personas_setting}{sample_suffix}.csv")

    df = interview_personas(
        personas_file=personas_file,
        output_file=output_file,
        questions=QUESTIONS,
        thermometer_targets=THERMOMETER_TARGETS,
        model=args.model,
        persona_sample=args.persona_sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
