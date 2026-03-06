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

# personas_setting = 'personas'
personas_setting = '20260123_personas_with_bio_2000_noLoveHate_noPartyId_'
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

#%%

COMPARISON_FILES = {
    "Full Persona":            os.path.join(os.path.dirname(__file__), "persona_interview_results_personas.csv"),
    "No Love/Hate": os.path.join(os.path.dirname(__file__), "persona_interview_results_20260121_personas_with_bio_2000_noLoveHate_.csv"),
    "No Love/Hate & PartyId": os.path.join(os.path.dirname(__file__), "persona_interview_results_20260123_personas_with_bio_2000_noLoveHate_noPartyId_.csv"),
    "No Love/Hate, PartyId, & Vot. Beh.": os.path.join(os.path.dirname(__file__), "persona_interview_results_20260227_personas_with_bio_2000_noLoveHate_noPartyId_noVoted2020_.csv"),
}
COMPARISON_OUTPUT_SLOPE = os.path.join(os.path.dirname(__file__), "interview_results.pdf")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for k, _ in QUESTIONS:
        col = f"{k}_answer"
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().eq("true").astype(int)
    return df

dfs = {label: load_and_prepare(path) for label, path in COMPARISON_FILES.items()}

answer_cols = ['q1_answer', 'q2_answer', 'q4_answer']
question_labels = {
    'q1': "Would you follow\nan opposing-party member?",
    'q2': "Would you follow\nsomeone who loves Trump?",
    'q4': "Would you follow\nsomeone who loves Biden?",
}
question_texts = {k: t for k, t in QUESTIONS}

all_parties = sorted(set().union(*[set(df["party"].dropna().unique()) for df in dfs.values()]))
labels      = list(dfs.keys())
n_questions = len(answer_cols)
n_datasets  = len(labels)
x_ticks     = list(range(n_datasets))

party_colors = {p: c for p, c in zip(all_parties, ["#03357D", "#888888", "#D50403", "#58508D", "#FFA600"])}

# Per-panel nudges for rightmost label: {col: {party: y_offset}}
right_nudge = {"q1_answer": {"Democrat": -0.04}}

# Right margin to accommodate party labels
right_margin = 1.6

fig, axes = plt.subplots(1, n_questions, figsize=(6.4, 3.2), sharey=True)
if n_questions == 1:
    axes = [axes]

for ax_idx, (ax, col) in enumerate(zip(axes, answer_cols)):
    r_nudges = right_nudge.get(col, {})
    for party in all_parties:
        vals = [
            dfs[label][dfs[label]["party"] == party][col].mean()
            if party in dfs[label]["party"].values else float("nan")
            for label in labels
        ]
        color = party_colors.get(party, "#888888")
        ax.plot(x_ticks, vals, marker="o", color=color, linewidth=1.5,
                markersize=4, solid_capstyle="round", clip_on=False)
        if not pd.isna(vals[-1]):
            ax.text(n_datasets - 1 + 0.12, vals[-1] + r_nudges.get(party, 0), party,
                    ha="left", va="center", fontsize=6.5, color=color)

    key = col.replace("_answer", "")
    ax.set_title(question_labels.get(key, question_texts.get(key, col)),
                 fontweight='medium', pad=6)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels, rotation=30, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.4, n_datasets - 1 + right_margin)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax.set_axisbelow(True)

axes[0].set_ylabel("Fraction answering Yes")
fig.tight_layout(pad=1.2)
fig.savefig(COMPARISON_OUTPUT_SLOPE, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(COMPARISON_OUTPUT_SLOPE.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved to {COMPARISON_OUTPUT_SLOPE}")

#%%
for label, df in dfs.items():
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for col in answer_cols:
        key = col.replace("_answer", "")
        q_text = question_labels.get(key, question_texts.get(key, col)).replace("\n", " ")
        print(f"\n  {q_text}")
        for party in all_parties:
            subset = df[df["party"] == party]
            if len(subset) == 0:
                continue
            pct = subset[col].mean() * 100
            print(f"    {party:<30} {pct:5.1f}%  (n={len(subset)})")

# %%
