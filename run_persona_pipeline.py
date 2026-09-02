"""Unified generate -> interview -> simulate pipeline.

For one obfuscation condition and each seed in --seeds, this:
  1. Samples personas from ANES (in memory only — no local persona JSON is ever
     written; see anes_generate_personas.sample_personas/enrich_personas).
  2. Interviews them (unless --skip_interview) via persona_interviews.py.
  3. Simulates them (unless --skip_simulate) via src/main.py.

All three stages of one seed's cycle share a single wandb run (personas, interview
results, and simulation metrics/state all logged there — nothing is written to local
disk). Multiple seeds in one invocation share a wandb group (batch id). Comparing
obfuscation conditions means invoking this script once per condition.
"""

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime

import pandas as pd
import wandb

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "PersonaGeneration"))
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "persona_interviews"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import anes_generate_personas  # noqa: E402
from obfuscation_labels import (  # noqa: E402
    FILENAME_SUFFIX_INFO,
    get_political_figure_labels,
    get_party_labels,
)
import interview_wandb  # noqa: E402
import persona_interviews  # noqa: E402
import main as simulation  # noqa: E402  (src/main.py)

OBFUSCATION_CHOICES = ['none', 'neutral', 'nonce', 'randomreal', 'randomnonce']

# --ablations token -> the anes_generate_personas.py ignore_* kwarg it sets. Additive:
# any subset of these can be combined in one --ablations list (unlike --obfuscation,
# which is a single mutually-exclusive condition per invocation).
ABLATION_TO_IGNORE_KWARG = {
    "love_hate": "ignore_love_hate",
    "bio_love_hate": "ignore_bio_love_hate",
    "party_identity": "ignore_party_identity",
    "bio_party_identity": "ignore_bio_party_identity",
    "voted2020": "ignore_voted2020",
    "bio_voted2020": "ignore_bio_voted2020",
    "voted2020_year": "ignore_voted2020_year",
    "age": "ignore_age",
    "ideology": "ignore_ideology",
    "political_behaviour": "ignore_political_behaviour",
    "leisure": "ignore_leisure",
    "problems": "ignore_problems",
    "religion": "ignore_religion",
    "state": "ignore_state",
    "extend_with_ai": "ignore_extend_with_ai",
    "bio": "ignore_bio",
}

# obfuscation id -> the filename-suffix substring obfuscation_labels.py's
# infer_obfuscation/lookup_obfuscated_terms match against (reverse of
# obfuscation_labels.FILENAME_SUFFIX_INFO).
OBFUSCATION_SUFFIX_BY_ID = {obf_id: suffix for suffix, (obf_id, _) in FILENAME_SUFFIX_INFO.items()}


def upload_personas_artifact(personas: list[dict], name: str) -> None:
    """Upload the generated personas as a wandb.Artifact, tempfile-staged so nothing
    is left on local disk — mirrors interview_wandb.upload_results_artifact's pattern
    for interview results. `default=str` covers numpy scalar types (e.g. int64/float64)
    that ANES-derived fields can carry, which plain json.dump can't serialize."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=True) as tmp:
        json.dump(personas, tmp, default=str)
        tmp.flush()
        artifact = wandb.Artifact(name=name, type="personas")
        artifact.add_file(tmp.name, name="personas.json")
        wandb.log_artifact(artifact)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified pipeline: for each seed, sample personas from ANES, "
                     "optionally interview them, and optionally simulate them — all "
                     "inside one wandb run per seed."
    )

    parser.add_argument("--num_personas", type=int, default=2000,
                         help="Number of personas to sample from ANES. The full "
                              "generated set is interviewed, and the same count is "
                              "used as the simulation's target population size.")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                         help="LLM model used across every stage: AI-extended "
                              "occupation/hobbies and the biography during "
                              "generation, answering as each persona during the "
                              "interview, and driving agent actions during the "
                              "simulation.")
    parser.add_argument("--ablations", nargs="*", default=[],
                         choices=sorted(ABLATION_TO_IGNORE_KWARG),
                         help="Additive list of persona-content ablations to apply at "
                              "generation time (each corresponds to today's "
                              "--ignore_<name> flag in anes_generate_personas.py).")
    parser.add_argument("--obfuscation", choices=OBFUSCATION_CHOICES, default="none",
                         help="Obfuscation condition for this invocation — one "
                              "condition per invocation; run the script again per "
                              "condition to compare them.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                         help="One seed per full generate->interview->simulate cycle. "
                              "Each seed drives every stage's randomness and is its "
                              "own wandb run; multiple seeds in one invocation share a "
                              "wandb group (batch id).")
    parser.add_argument("--skip_interview", action="store_true", default=False,
                         help="Skip the interview stage.")
    parser.add_argument("--skip_simulate", action="store_true", default=False,
                         help="Skip the simulation stage.")
    parser.add_argument("--batch_id", type=str, default=None,
                         help="Wandb group/batch id to use instead of a fresh random "
                              "one. Pass the same value across multiple invocations "
                              "(e.g. one per --obfuscation value, or one per "
                              "--ablations combo) to group them for cross-condition, "
                              "cross-seed comparison later.")
    parser.add_argument("--experiment_label", type=str, default=None,
                         help="Free-text label (e.g. 'ablation-v1') logged as a wandb "
                              "tag and config field, for readability when browsing "
                              "runs. Not required by analysis scripts — they key on "
                              "--batch_id.")

    # Simulation-stage flags
    parser.add_argument("--simulation_steps", type=int, default=5000,
                         help="Number of steps to run the simulation for.")
    parser.add_argument("--user_link_strategy", type=str, default="on_repost_bio",
                         help="User link strategy for the simulation.")
    parser.add_argument("--timeline_select_strategy", type=str, default="other_partisan",
                         choices=['random', 'random_weighted', 'random_weighted_reversed',
                                  'bridging_attributes', 'chronological', 'other_partisan'],
                         help="Timeline selection strategy for the simulation.")
    parser.add_argument("--news_feed", type=str, default="News_Category_Dataset_v3.json",
                         help="Path to the news feed dataset.")
    parser.add_argument("--openrouter_api_key", type=int, default=None,
                         help="If None, use the OpenAI key; otherwise which "
                              "OpenRouter API key to use from env (1, 2, or 3).")
    # Simulation-prompt ablations — separate from --ablations (which is generation-
    # content-only) since these ablate what's shown in the agent's prompt at
    # simulate time, not what's in the generated persona text.
    parser.add_argument("--hide_own_persona", action="store_true", default=False,
                         help="Omit the persona description from the agent's system "
                              "prompt (neutral-agent baseline).")
    parser.add_argument("--hide_target_bio", action="store_true", default=False,
                         help="Omit the target user's bio when an agent decides "
                              "whether to follow them.")
    parser.add_argument("--hide_news_category", action="store_true", default=False,
                         help="Omit the news category when an agent decides which "
                              "news to share.")

    # Shared
    parser.add_argument("--include_group_context", action="store_true", default=False,
                         help="Prepend a short paragraph naming the two rival "
                              "political affiliations and their leader to every "
                              "persona/agent's system message (interview + simulation).")
    parser.add_argument("--wandb_project", type=str,
                         default="persona-simulation",
                         help="Wandb project to log pipeline runs to.")
    parser.add_argument("--no_log", action="store_true", default=False,
                         help="Skip wandb logging entirely (e.g. for local debugging runs).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = not args.no_log

    ablation_flags = {kwarg: (name in args.ablations) for name, kwarg in ABLATION_TO_IGNORE_KWARG.items()}
    obf_suffix = OBFUSCATION_SUFFIX_BY_ID.get(args.obfuscation, "")
    # main.py/NewsFeed.py resolve a bare filename relative to the process's cwd
    # (their standalone-CLI convention assumes cwd=src/); since this orchestrator is
    # invoked from the repo root, resolve it to an absolute path ourselves instead.
    news_feed_path = args.news_feed if os.path.isabs(args.news_feed) else os.path.join(REPO_ROOT, "src", args.news_feed)
    batch_id = args.batch_id or uuid.uuid4().hex[:8]
    print(f"Wandb batch id for this invocation: {batch_id}")

    for i, seed in enumerate(args.seeds):
        print(f"\n=== Seed {i + 1}/{len(args.seeds)} (seed={seed}) ===")

        # Personas never touch disk, but obfuscation_labels only needs a string shaped
        # like a persona filename to resolve real/obfuscated labels and infer the
        # obfuscation condition — this stands in for a real file/settings name.
        personas_setting = f"pipeline_seed{seed}_{obf_suffix}"

        # Logged into config below so per-condition analysis scripts (which fetch a
        # batch of runs and bucket by config field) can rebuild question/thermometer
        # text without re-deriving it from personas_setting themselves — mirrors what
        # persona_interviews.run_interview_for_setting used to log via its own
        # (now-skipped, since own_wandb_run=False) wandb.init call.
        trump_label, biden_label = get_political_figure_labels(personas_setting)
        democrats_label, republicans_label = get_party_labels(personas_setting)

        if log:
            tags = [args.obfuscation]
            if args.experiment_label:
                tags.append(args.experiment_label)
            wandb.init(
                project=args.wandb_project,
                group=batch_id,
                job_type="pipeline",
                tags=tags,
                name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.obfuscation}_seed{seed}",
                config={
                    # Also logged as a plain config field (in addition to being set
                    # as wandb's special `group` above) — the API-level fetching in
                    # the analysis scripts filters by `group`, but most wandb UI
                    # views only offer grouping/filtering on config fields, not the
                    # `group` value itself, so this is what makes it usable there.
                    "batch_id": batch_id,
                    "obfuscation": args.obfuscation,
                    "ablations": args.ablations,
                    "experiment_label": args.experiment_label,
                    "num_personas": args.num_personas,
                    "llm_model": args.llm_model,
                    "seed": seed,
                    "skip_interview": args.skip_interview,
                    "skip_simulate": args.skip_simulate,
                    "include_group_context": args.include_group_context,
                    "simulation_steps": args.simulation_steps,
                    "user_link_strategy": args.user_link_strategy,
                    "timeline_select_strategy": args.timeline_select_strategy,
                    "hide_own_persona": args.hide_own_persona,
                    "hide_target_bio": args.hide_target_bio,
                    "hide_news_category": args.hide_news_category,
                    "trump_label": trump_label,
                    "biden_label": biden_label,
                    "democrats_label": democrats_label,
                    "republicans_label": republicans_label,
                },
            )

        # --- Stage 1: Generate ---
        gen_args = argparse.Namespace(
            num_personas=args.num_personas,
            obfuscation=args.obfuscation,
            seed=seed,
            minimal_persona=False,
            **ablation_flags,
        )
        personas = anes_generate_personas.sample_personas(gen_args)
        personas = anes_generate_personas.enrich_personas(
            personas,
            model=args.llm_model,
            ignore_extend_with_ai=ablation_flags["ignore_extend_with_ai"],
            ignore_bio=ablation_flags["ignore_bio"],
            ignore_bio_love_hate=ablation_flags["ignore_bio_love_hate"],
            ignore_bio_party_identity=ablation_flags["ignore_bio_party_identity"],
            ignore_bio_voted2020=ablation_flags["ignore_bio_voted2020"],
        )
        print(f"Generated {len(personas)} personas.")

        if log:
            upload_personas_artifact(personas, name=f"personas-{args.obfuscation}-seed{seed}")
            wandb.log(interview_wandb.persona_population_metrics(pd.DataFrame(personas)))

        # --- Stage 2: Interview ---
        if not args.skip_interview:
            persona_interviews.run_interview_for_setting(
                personas_setting=personas_setting,
                model=args.llm_model,
                persona_sample=None,
                seeds=[seed],
                log=log,
                include_group_context=args.include_group_context,
                personas=personas,
                own_wandb_run=False,
                openrouter_api_key=args.openrouter_api_key,
            )
        else:
            print("Skipping interview stage.")

        # --- Stage 3: Simulate ---
        if not args.skip_simulate:
            simulation.run_simulation(
                simulation_size=args.num_personas,
                simulation_steps=args.simulation_steps,
                user_link_strategy=args.user_link_strategy,
                timeline_select_strategy=args.timeline_select_strategy,
                llm_model=args.llm_model,
                news_feed=news_feed_path,
                show_info=True,
                sim_path=f"{args.obfuscation}_seed{seed}",
                personas_file=personas_setting,
                personas=personas,
                seed=seed,
                openrouter_api_key=args.openrouter_api_key,
                log=log,
                own_wandb_run=False,
                include_group_context=args.include_group_context,
                wandb_project=args.wandb_project,
                hide_own_persona=args.hide_own_persona,
                hide_target_bio=args.hide_target_bio,
                hide_news_category=args.hide_news_category,
            )
        else:
            print("Skipping simulation stage.")

        if log:
            wandb.finish()

    if log:
        print(f"\nLogged {len(args.seeds)} run(s) to wandb project "
              f"'{args.wandb_project}' (group={batch_id}).")
    else:
        print(f"\nDone ({len(args.seeds)} seed(s), logging disabled).")


if __name__ == "__main__":
    main()
