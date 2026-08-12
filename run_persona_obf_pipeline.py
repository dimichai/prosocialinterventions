"""Run the full persona generation and interview pipeline across
one or more obfuscation conditions in a single command.

For each condition in --obfuscations, this:
  1. Generates personas via PersonaGeneration/anes_generate_personas.py
     (writes both the pre-bio and with-bio JSON into PersonaGeneration/, same
     as running that script directly).
  2. Copies the with-bio JSON into src/, since persona_interviews_obfuscation.py
     expects its input there (previously a manual step).
  3. Runs analysis/persona_interviews/persona_interviews_obfuscation.py against
     that file, logging one wandb run per (obfuscation, seed) to the interviews
     wandb project. All runs from one invocation of this script share a wandb
     group (batch id), so the analysis script can later pull "this whole
     comparison" back from wandb in a single query.
"""

import argparse
import os
import shutil
import sys
import uuid

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "PersonaGeneration"))
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "persona_interviews"))

import anes_generate_personas  # noqa: E402
import persona_interviews_obfuscation  # noqa: E402
import interview_wandb  # noqa: E402

OBFUSCATION_CHOICES = ['none', 'neutral', 'nonce', 'randomreal', 'randomnonce']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate personas and interview them, across one or more obfuscation conditions."
    )

    # Generation params (mirrors anes_generate_personas.py, minus --obfuscation/--seed).
    parser.add_argument("--num_personas", type=int, default=2000, help="Number of personas to generate")
    parser.add_argument("--ignore_love_hate", action='store_true', default=False, help="Whether to ignore love/hate lists")
    parser.add_argument("--ignore_bio_love_hate", action='store_true', default=False, help="Whether to ignore love/hate lists in bio")
    parser.add_argument("--ignore_party_identity", action='store_true', default=False, help="Whether to ignore party identity info")
    parser.add_argument("--ignore_bio_party_identity", action='store_true', default=False, help="Whether to ignore party identity info in bio")
    parser.add_argument("--ignore_voted2020", action='store_true', default=False, help="Whether to ignore voted2020 info")
    parser.add_argument("--ignore_bio_voted2020", action='store_true', default=False, help="Whether to ignore voted2020 info in bio")
    
    parser.add_argument("--ignore_voted2020_year", action='store_true', default=False, help="Whether to omit 'in 2020' from the voted2020 sentence (candidate/no-vote is still stated)")
    parser.add_argument("--ignore_age", action='store_true', default=False, help="Whether to omit the age sentence from the persona text")
    parser.add_argument("--ignore_extend_with_ai", action='store_true', default=False, help="Whether to skip extending personas with AI-generated occupation/hobbies")
    parser.add_argument("--ignore_bio", action='store_true', default=False, help="Whether to skip generating a biography for the persona entirely")
    parser.add_argument("--gen_seed", type=int, default=42, help="Random seed for persona sampling and AI extension choices")

    parser.add_argument("--obfuscations", nargs="+", choices=OBFUSCATION_CHOICES, default=list(OBFUSCATION_CHOICES),
                         help="Obfuscation conditions to run, in order. Defaults to all five.")

    # Interview params (mirrors persona_interviews_obfuscation.py, minus --personas_setting/--seed).
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenRouter model id used to answer as each persona.")
    parser.add_argument("--persona_sample", type=int, default=None, help="Number of personas to randomly sample; omit to interview all personas.")
    parser.add_argument("--interview_seed", type=int, nargs="+", default=[42],
                         help="One or more random seeds for the interview step. Each seed is logged "
                              "to wandb as its own run; aggregation across seeds happens on the "
                              "analysis side, reading back from wandb.")
    parser.add_argument("--wandb_project", type=str, default=interview_wandb.WANDB_PROJECT,
                         help="Wandb project to log interview runs to.")
    parser.add_argument("--no_log", action='store_true', default=False,
                         help="Skip wandb logging entirely (e.g. for local debugging runs).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gen_arg_names = [
        "num_personas", "ignore_love_hate", "ignore_party_identity", "ignore_voted2020",
        "ignore_voted2020_year", "ignore_age", "ignore_extend_with_ai", "ignore_bio",
        "ignore_bio_love_hate", "ignore_bio_party_identity", "ignore_bio_voted2020",
    ]

    src_dir = os.path.join(REPO_ROOT, "src")
    summary = []
    batch_id = uuid.uuid4().hex[:8]
    print(f"Wandb batch id for this invocation: {batch_id}")

    for i, obfuscation in enumerate(args.obfuscations):
        print(f"\n=== [{i + 1}/{len(args.obfuscations)}] obfuscation={obfuscation} ===")

        gen_args = argparse.Namespace(
            **{name: getattr(args, name) for name in gen_arg_names},
            obfuscation=obfuscation,
            seed=args.gen_seed,
        )
        gen_paths = anes_generate_personas.generate_personas_cli(gen_args)
        with_bio_path = gen_paths["personas_with_bio_file"]
        print(f"Generated: {with_bio_path}")

        personas_setting = os.path.splitext(os.path.basename(with_bio_path))[0]
        src_path = os.path.join(src_dir, os.path.basename(with_bio_path))
        shutil.copy2(with_bio_path, src_path)
        print(f"Copied to: {src_path}")

        extra_config = {name: getattr(args, name) for name in gen_arg_names}
        extra_config.update(obfuscation=obfuscation, gen_seed=args.gen_seed)

        interview_result = persona_interviews_obfuscation.run_interview_for_setting(
            personas_setting=personas_setting,
            model=args.model,
            persona_sample=args.persona_sample,
            seeds=args.interview_seed,
            wandb_group=batch_id,
            extra_config=extra_config,
            log=not args.no_log,
            wandb_project=args.wandb_project,
        )
        print(f"Interviewed -> {interview_result}")

        summary.append((obfuscation, with_bio_path, interview_result))

    print("\n=== Summary ===")
    print(f"Wandb batch id: {batch_id} (project: {args.wandb_project})")
    for obfuscation, generated_path, interview_result in summary:
        print(f"{obfuscation}:")
        print(f"  generated: {generated_path}")
        print(f"  wandb runs: {interview_result['run_ids']}")


if __name__ == "__main__":
    main()
