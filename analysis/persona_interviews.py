#%%
import os
import sys
import json
import dotenv
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

personas_setting = 'personas'
PERSONAS_FILE = os.path.join(os.path.dirname(__file__), f"../src/{personas_setting}.json")
OUTPUT_FILE   = os.path.join(os.path.dirname(__file__), f"persona_interview_results_{personas_setting}.csv")
MODEL         = "gpt-4o-mini"


QUESTIONS = [
    ("q1", "Would you follow someone who is a member of the opposing political party?"),   # ← fill in question 1
    ("q2", "Would you follow someone who loves Trump?"),
    ("q3", "Would you follow someone who hates Trump?"),
    ("q4", "Would you follow someone who loves Biden?"),
    ("q5", "Would you follow someone who hates Biden?"),
]

#%%
class BooleanAnswer(BaseModel):
    choice: str        # "yes" or "no"
    explanation: str


def _system_message(persona: dict) -> str:
    return (
        "You are a user of the X social media platform. "
        "This is a platform where users share opinions and thoughts on topics of interest "
        "in the form of posts.\n\n"
        "Here is a description of your persona:\n"
        f"{persona['persona']}"
    )


def ask_question(client: OpenAI, persona: dict, question: str) -> tuple[bool, str]:
    """Send a yes/no question to the LLM and return (answer_bool, explanation)."""

    response = client.beta.chat.completions.parse(
        model=MODEL,
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


def interview_personas(
    personas_file: str = PERSONAS_FILE,
    output_file: str = OUTPUT_FILE,
) -> pd.DataFrame:

    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY_1"),
    )

    personas = json.load(open(personas_file, "r"))
    active_questions = [(key, q) for key, q in QUESTIONS if q.strip()]

    if not active_questions:
        print("No questions defined yet – fill in the QUESTIONS list and re-run.")
        return pd.DataFrame()

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
            answer, explanation = ask_question(client, persona, question)
            row[f"{key}_answer"]      = answer          # True / False
            row[f"{key}_explanation"] = explanation

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    client.close()
    return df


interview_personas()

#%%
df = pd.read_csv(OUTPUT_FILE)

# Convert answer columns to 0/1 — works for any dtype pandas may give us
answer_cols    = [f"{k}_answer" for k, _ in QUESTIONS if f"{k}_answer" in df.columns]
question_texts = [t for k, t in QUESTIONS if f"{k}_answer" in df.columns]

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
plt.savefig(OUTPUT_FILE.replace(".csv", "_by_party.png"), bbox_inches="tight", dpi=150)
plt.show()
