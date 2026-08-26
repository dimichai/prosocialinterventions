import os

import pandas as pd

# Obfuscated Trump/Biden/party labels are defined per obfuscation mode in this CSV
# (see anes_generate_personas.py, which generated the persona files).
OBFUSCATION_CSV = os.path.join(os.path.dirname(__file__), "persona_obfuscations.csv")

# Suffix anes_generate_personas.py embeds in persona filenames per obfuscation
# condition -> (obfuscation id, persona_obfuscations.csv column).
FILENAME_SUFFIX_INFO = {
    "obfNeutral_":     ("neutral",     "A_Neutral"),
    "obfNonce_":       ("nonce",       "B_Nonce"),
    "obfRandomReal_":  ("randomreal",  "C_RandomReal"),
    "obfRandomNonce_": ("randomnonce", "D_RandomNonce"),
}


def infer_obfuscation(personas_setting: str) -> str:
    """Best-effort fallback: infer the obfuscation condition ('none' if no suffix
    matches) from the persona filename convention, for callers that don't already
    know it explicitly."""
    return next((o for suffix, (o, _) in FILENAME_SUFFIX_INFO.items() if suffix in personas_setting), "none")


def lookup_obfuscated_terms(personas_setting: str, terms: list[str]) -> list[str]:
    """Translate `terms` through the obfuscation mode encoded in personas_setting."""
    column = next((c for suffix, (_, c) in FILENAME_SUFFIX_INFO.items() if suffix in personas_setting), None)
    if column is None:
        return terms
    df = pd.read_csv(OBFUSCATION_CSV).set_index("Term")
    return [df.loc[term, column] for term in terms]


def get_political_figure_labels(personas_setting: str) -> tuple[str, str]:
    """Return (trump_label, biden_label) matching the obfuscation mode encoded in personas_setting."""
    trump_label, biden_label = lookup_obfuscated_terms(personas_setting, ["Donald Trump", "Joe Biden"])
    return trump_label, biden_label


def get_party_labels(personas_setting: str) -> tuple[str, str]:
    """Return (democrats_label, republicans_label) matching the obfuscation mode encoded in personas_setting."""
    democrats_label, republicans_label = lookup_obfuscated_terms(personas_setting, ["Democrats", "Republicans"])
    return democrats_label, republicans_label


def build_group_context(
    trump_label: str, biden_label: str, democrats_label: str, republicans_label: str
) -> str:
    """A short scene-setting paragraph naming the two rival political affiliations
    and their associated leader, using whatever labels (real or obfuscated) are in
    play. Personas already have their own affiliation/leader in their bio, but under
    obfuscation the *opposing* party's label is often never mentioned anywhere in
    their persona text — so without this, the model has no basis at all for
    answering questions about that opposing label (it's just an unrecognized
    string). This gives the minimal relational grounding needed to answer, without
    revealing which (real-world) party either obfuscated label maps to."""
    return (
        f"For context: {democrats_label} and {republicans_label} are the two rival "
        "political affiliations in this society — every adult identifies with one "
        "of them, the other, or neither. "
        f"{biden_label} and {trump_label} are the two most prominent national "
        f"political leaders, with {biden_label} aligned with {democrats_label} and "
        f"{trump_label} aligned with {republicans_label}."
    )
