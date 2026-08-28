"""Shared question battery (yes/no + trait questions, feeling-thermometer
targets) used by persona_interviews.py, so real-label and obfuscated-label
conditions ask identically structured questions — the only thing that varies
between them is which labels (real or obfuscated) are plugged into the
templates below.
"""

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


def build_questions(
    trump_label: str, biden_label: str, democrats_label: str, republicans_label: str
) -> list[tuple[str, str]]:
    """Yes/no question battery, parameterized on the (real or obfuscated) labels
    in play so ablation and obfuscation runs ask identically structured questions."""
    return [
        ("q1_dem", f"Would you follow someone who is a member of {democrats_label}?"),
        ("q1_rep", f"Would you follow someone who is a member of {republicans_label}?"),
        ("q2", f"Would you follow someone who loves {trump_label}?"),
        ("q3", f"Would you follow someone who hates {trump_label}?"),
        ("q4", f"Would you follow someone who loves {biden_label}?"),
        ("q5", f"Would you follow someone who hates {biden_label}?"),
        ("dem_intelligent",  f"Do you think {democrats_label} are intelligent?"),
        ("dem_honest",       f"Do you think {democrats_label} are honest?"),
        ("dem_openminded",   f"Do you think {democrats_label} are open-minded?"),
        ("dem_generous",     f"Do you think {democrats_label} are generous?"),
        ("dem_hypocritical", f"Do you think {democrats_label} are hypocritical?"),
        ("dem_selfish",      f"Do you think {democrats_label} are selfish?"),
        ("dem_mean",         f"Do you think {democrats_label} are mean?"),
        ("rep_intelligent",  f"Do you think {republicans_label} are intelligent?"),
        ("rep_honest",       f"Do you think {republicans_label} are honest?"),
        ("rep_openminded",   f"Do you think {republicans_label} are open-minded?"),
        ("rep_generous",     f"Do you think {republicans_label} are generous?"),
        ("rep_hypocritical", f"Do you think {republicans_label} are hypocritical?"),
        ("rep_selfish",      f"Do you think {republicans_label} are selfish?"),
        ("rep_mean",         f"Do you think {republicans_label} are mean?"),
    ]


def build_thermometer_targets(
    trump_label: str, biden_label: str, democrats_label: str, republicans_label: str
) -> list[tuple[str, str]]:
    # role -> display label shown in the question text (obfuscated per condition)
    return [
        ("biden", biden_label),
        ("trump", trump_label),
        ("democrats", democrats_label),
        ("republicans", republicans_label),
    ]
