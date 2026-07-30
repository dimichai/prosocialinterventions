import argparse
import os
import sys
import json
import random
import dotenv
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


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


class FeelingThermometerAnswer(BaseModel):
    biden_recognized: bool
    biden_rating: int | None   # 0-100, None if not recognized
    trump_recognized: bool
    trump_rating: int | None


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


def ask_feeling_thermometer(client: OpenAI, persona: dict, targets: list[tuple[str, str]], model: str) -> dict:
    """Send the feeling-thermometer battery (Biden, Trump) in a single call."""

    prompt = (
        f"{THERMOMETER_INTRO}\n\n"
        + "\n".join(f"How would you rate: {label}" for _, label in targets)
        + "\n\nFor each person, indicate whether you recognize them, and if so, "
          "give a whole-number rating between 0 and 100."
    )

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _system_message(persona)},
            {"role": "user", "content": prompt},
        ],
        response_format=FeelingThermometerAnswer,
    )

    parsed = response.choices[0].message.parsed
    return {
        "biden_therm_recognized": parsed.biden_recognized,
        "biden_therm_rating":     parsed.biden_rating,
        "trump_therm_recognized": parsed.trump_recognized,
        "trump_therm_rating":     parsed.trump_rating,
    }


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


def plot_by_party(
    df: pd.DataFrame,
    questions: list[tuple[str, str]],
    thermometer_targets: list[tuple[str, str]],
    personas_setting: str,
    sample_suffix: str,
) -> None:
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Convert answer columns to 0/1 — works for any dtype pandas may give us
    answer_cols    = [f"{k}_answer" for k, _ in questions if f"{k}_answer" in df.columns]
    question_texts = [t for k, t in questions if f"{k}_answer" in df.columns]

    for col in answer_cols:
        df[col] = df[col].astype(str).str.strip().str.lower().eq("true").astype(int)

    groups = sorted(df["party"].dropna().unique())

    fig, axes = plt.subplots(1, len(answer_cols), figsize=(4 * len(answer_cols), 5), sharey=True)
    if len(answer_cols) == 1:
        axes = [axes]

    for ax, col, text in zip(axes, answer_cols, question_texts):
        yes_vals = [df[df["party"] == g][col].mean() for g in groups]
        no_vals  = [1 - v for v in yes_vals]

        ax.bar(groups, yes_vals, color="#3C97DA")
        ax.bar(groups, no_vals,  color="#FE9D51", bottom=yes_vals)
        ax.set_ylim(0, 1)
        ax.set_title(text, fontsize=8, wrap=True)
        ax.tick_params(axis="x", rotation=15)

    axes[0].set_ylabel("Fraction")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#4878A8", "#D45500"]]
    fig.legend(handles, ["Yes", "No"], loc="upper right", frameon=False)
    fig.suptitle("Yes/No answers by party", fontsize=11)
    plt.tight_layout()
    answers_path = os.path.join(figs_dir, f"persona_interview_results_{personas_setting}{sample_suffix}_by_party.png")
    plt.savefig(answers_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved plot to {answers_path}")

    therm_cols = [(f"{role}_therm_rating", label) for role, label in thermometer_targets
                  if f"{role}_therm_rating" in df.columns]

    fig, axes = plt.subplots(1, len(therm_cols), figsize=(4 * len(therm_cols), 5), sharey=True)
    if len(therm_cols) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, therm_cols):
        means, errs = [], []
        for g in groups:
            subset = df.loc[df["party"] == g, col].dropna()
            if len(subset) == 0:
                means.append(float("nan")); errs.append(float("nan"))
                continue
            means.append(subset.mean())
            errs.append(1.96 * subset.std(ddof=1) / len(subset) ** 0.5)
        ax.bar(groups, means, yerr=errs, color="#3C97DA", capsize=4)
        ax.set_ylim(0, 100)
        ax.set_title(f"Feeling thermometer:\n{label}", fontsize=10)
        ax.tick_params(axis="x", rotation=15)

    axes[0].set_ylabel("Mean rating (0-100)")
    fig.suptitle("Feeling thermometer ratings by party", fontsize=11)
    plt.tight_layout()
    therm_path = os.path.join(
        figs_dir, f"persona_interview_thermometer_results_{personas_setting}{sample_suffix}_by_party.png"
    )
    plt.savefig(therm_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved plot to {therm_path}")


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

    plot_by_party(df, QUESTIONS, THERMOMETER_TARGETS, personas_setting, sample_suffix)


if __name__ == "__main__":
    main()
