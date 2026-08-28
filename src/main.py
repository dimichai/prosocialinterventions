import argparse
from datetime import datetime
import dotenv
import os
import json
import pickle
import random
import time
import wandb
import numpy as np

from collections import Counter
from openai import OpenAI

from Agent import Agent
from Platform import Platform
from NewsFeed import NewsFeed
import tempfile
import networkx as nx
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.analyse_multiple import gini_coefficient, EI_index, correlations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../PersonaGeneration'))
from obfuscation_labels import infer_obfuscation, get_political_figure_labels, get_party_labels, build_group_context

dotenv.load_dotenv()

def compute_metrics(platform, step, cost_input, cost_output, cost_cached, compute_clustering=False):
    log_start_time = time.time()
    follower_distribution = [u.followers for u in platform.users]
    repost_distribution = [p.reposts for p in platform.raw_posts]
    action_counts = Counter([a['action'] for a in platform.actions])

    metrics = {"step": step}

    # EI index
    if len(platform.user_links) > 0:
        IL = sum(1 for u1, u2 in platform.user_links
                if platform.get_user(u1).persona['party'] == platform.get_user(u2).persona['party'])
        EL = len(platform.user_links) - IL
        metrics["EI_index"] = (EL - IL) / (EL + IL)

    # EI index restricted to Democrat/Republican nodes only
    dem_rep_links = [(u1, u2) for u1, u2 in platform.user_links
                      if platform.get_user(u1).persona['party'] in ('Democrat', 'Republican')
                      and platform.get_user(u2).persona['party'] in ('Democrat', 'Republican')]
    if len(dem_rep_links) > 0:
        IL_dem_rep = sum(1 for u1, u2 in dem_rep_links
                if platform.get_user(u1).persona['party'] == platform.get_user(u2).persona['party'])
        EL_dem_rep = len(dem_rep_links) - IL_dem_rep
        metrics["EI_index_dem_rep"] = (EL_dem_rep - IL_dem_rep) / (EL_dem_rep + IL_dem_rep)

    # Gini coefficients
    if sum(follower_distribution) > 0:
        metrics["gini_followers"] = gini_coefficient(follower_distribution)
    if repost_distribution and sum(repost_distribution) > 0:
        metrics["gini_reposts"] = gini_coefficient(repost_distribution)

    # Correlations (partisanship vs followers/retweets)
    if len(platform.raw_posts) > 0:
        partisans = [abs(u.persona['partisan']) for u in platform.users]
        corr_followers = np.corrcoef(partisans, follower_distribution)[0, 1]
        total_retweets = [sum(p.reposts for p in platform.raw_posts if p.author.identifier == u.identifier) for u in platform.users]
        corr_retweets = np.corrcoef(partisans, total_retweets)[0, 1]
        metrics["correlation_followers_partisan"] = corr_followers
        metrics["correlation_retweets_partisan"] = corr_retweets

    # Summary stats
    metrics["num_connections"] = len(platform.user_links)
    metrics["num_posts"] = len(platform.raw_posts)
    metrics["mean_followers"] = np.mean(follower_distribution)
    metrics["mean_reposts"] = np.mean(repost_distribution) if repost_distribution else 0

    # Estimated cost
    total_input = sum(u.used_tokens_input for u in platform.users)
    total_output = sum(u.used_tokens_output for u in platform.users)
    total_cached = sum(u.used_tokens_cached for u in platform.users)
    metrics["estimated_cost"] = ((cost_output / 1e6) * total_output) + \
        ((cost_input / 1e6) * (total_input - total_cached)) + \
        ((cost_cached / 1e6) * total_cached)
    metrics["total_tokens_input"] = total_input
    metrics["total_tokens_output"] = total_output
    metrics["total_tokens_cached"] = total_cached

    # Network density, reciprocity and modularity (cheap, computed every step)
    if len(platform.user_links) > 0:
        G = nx.DiGraph()
        G.add_nodes_from([u.identifier for u in platform.users])
        G.add_edges_from(platform.user_links)
        metrics["network_density"] = nx.density(G)
        metrics["network_reciprocity"] = nx.overall_reciprocity(G)

        democrats = {u.identifier for u in platform.users if u.persona['party'] == 'Democrat'}
        republicans = {u.identifier for u in platform.users if u.persona['party'] == 'Republican'}
        others = set(G.nodes()) - democrats - republicans
        communities = [c for c in (democrats, republicans, others) if c]
        if len(communities) > 1:
            metrics["modularity_dem_rep"] = nx.community.modularity(G, communities)

        # Clustering coefficient is expensive, so it's only computed periodically
        if compute_clustering:
            cluster_coeff = nx.clustering(G)
            metrics["avg_clustering_coefficient"] = np.mean(list(cluster_coeff.values()))

    # Action distribution
    for action_type, count in action_counts.items():
        metrics[f"action_{action_type}"] = count

    # Total reposts per news category
    for category, total_reposts in category_repost_totals(platform).items():
        metrics[f"category_reposts_{category}"] = total_reposts

    metrics["seconds_to_log"] = time.time() - log_start_time
    return metrics

def category_repost_totals(platform):
    """
    Total number of reposts received by posts sharing a news headline, grouped by news category.
    Posts not tied to a news headline (option 1 reposts of a plain post, or posts with no
    resolvable category) are excluded.
    """
    totals = Counter()
    for post in platform.raw_posts:
        if post.news_category:
            totals[post.news_category] += post.reposts
    return dict(totals)

def log_action(user, action):
    """
    Log the action taken by the user to the console.
    """

    persona_index = user.persona.get("persona_index") if user.persona else None
    log_msg = f"User {user.identifier} (persona_id={persona_index}) chose action "

    if action.option == 1:
        log_msg += "1, repost."
        log_msg += f"User reposted message with ID {action.content}\n"
    elif action.option == 2:
        log_msg += "2, post.\n"
        log_msg += f"User wrote: {action.content}\n"
    elif action.option == 3:
        log_msg += "3, do nothing.\n"
    else:
        log_msg += f"{action.option}, which is invalid.\n"

    return log_msg

def select_users(persona_path, n, personas=None):
    """
    Create a sample of `n` users for the simulation, from an in-memory `personas`
    list when given, otherwise loaded from the persona file at `persona_path`.

    Previously forced an exact 45% Democrat / 46% Republican / 9% Non-partisan split
    (per Gallup, 2025) by sampling that many from each party's sub-pool. That only
    works when the pool is large relative to `n`; now that run_persona_pipeline.py's
    --num_personas drives both how many personas are generated and the simulation's
    population size (n == len(users)), it needs the pool's *actual* party mix to
    already match those fractions almost exactly — which raw ANES sampling doesn't
    guarantee (its natural split runs closer to 47/42/11), so that approach fails
    reliably rather than rarely. Simulating the whole given pool (in whatever party
    mix it naturally has) instead of forcing a target split avoids that failure mode.
    """
    users = personas if personas is not None else json.load(open(persona_path, 'r'))
    return random.sample(users, min(n, len(users)))

def get_persona_label(personas_file, no_personas, no_bio):
    """
    Label used for the wandb run name/config and the results directory.
    When --no_personas/--no_bio are set, the persona file's identity no longer
    describes the run, so the label reflects the ablation instead of the filename.
    """
    if no_personas and no_bio:
        return "no_personas_no_bio"
    elif no_personas:
        return "no_personas"
    elif no_bio:
        return "no_bio"
    return personas_file

def run_simulation(simulation_size = 500, simulation_steps = 10000,
                user_link_strategy = "on_repost_bio",
                timeline_select_strategy = "random_weighted",
                llm_model = "gpt-4o-mini",
                news_feed = 'News_Category_Dataset_v3.json',
                show_info = True,
                sim_path="",
                personas_file = 'personas.json',
                personas = None,
                seed = None,
                openrouter_api_key = None,
                log = True,
                own_wandb_run = True,
                no_personas = False,
                no_bio = False,
                no_news_category = False,
                wandb_project = "prosocial-interventions",
                include_group_context = False):

    if seed is not None:
        random.seed(seed)

    persona_label = get_persona_label(personas_file, no_personas, no_bio)
    obfuscation = infer_obfuscation(personas_file)
    group_context = (
        build_group_context(*get_political_figure_labels(personas_file), *get_party_labels(personas_file))
        if include_group_context else ""
    )

    if log and own_wandb_run:
        wandb.init(project=wandb_project,
            name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{persona_label}",
            config={
                "simulation_size": simulation_size,
                "simulation_steps": simulation_steps,
                "user_link_strategy": user_link_strategy,
                "timeline_select_strategy": timeline_select_strategy,
                "llm_model": llm_model,
                "news_feed": news_feed,
                "personas_file": persona_label,
                "personas_file_original": personas_file,
                "persona_love_hate_lists": 'noLoveHate' not in personas_file,
                "persona_party_id": 'noPartyId' not in personas_file,
                "persona_voted2020": 'noVoted2020' not in personas_file,
                "bio_love_hate_lists": 'noLoveHate' not in personas_file and 'noBioLoveHate' not in personas_file,
                "bio_party_id": 'noPartyId' not in personas_file and 'noBioPartyId' not in personas_file,
                "bio_voted2020": 'noVoted2020' not in personas_file and 'noBioVoted2020' not in personas_file,
                "no_personas": no_personas,
                "no_bio": no_bio,
                "no_news_category": no_news_category,
                "obfuscation": obfuscation,
                "include_group_context": include_group_context,
                "group_context": group_context,
            }
        )

    costs_path = os.path.join(os.path.dirname(__file__), "model_costs.json")
    with open(costs_path, "r") as f:
        model_costs = json.load(f)
    if llm_model not in model_costs:
        print(f"Warning: No cost data for model '{llm_model}' in model_costs.json. Cost estimates will be 0.")
        cost_input = 0.0
        cost_output = 0.0
        cost_cached = 0.0
    else:
        cost_input = model_costs[llm_model]["input"]
        cost_output = model_costs[llm_model]["output"]
        cost_cached = model_costs[llm_model]["cached_input"]

    # Define the path to the persona file (unused when `personas` is supplied directly)
    persona_path = os.path.join(os.getcwd(), personas_file)
    news_feed = NewsFeed(news_feed)

    platform = Platform(user_link_strategy=user_link_strategy, timeline_select_strategy=timeline_select_strategy, show_info=show_info)

    # Ensure the right fraction of Democrats, Republicans, and non-partisans
    selected_users = select_users(persona_path, n=simulation_size, personas=personas)

    # Initialize the OpenRouter client
    model = llm_model
    if openrouter_api_key is not None:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv(f"OPENROUTER_API_KEY_{openrouter_api_key}"),
        )
    else:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    # Register users
    [platform.register_user(Agent(model, user, no_personas=no_personas, no_bio=no_bio, no_news_category=no_news_category, group_context=group_context)) for user in selected_users]
    platform.set_client(client)
    
    for i in range(simulation_steps):

        print(f"Simulation step {i + 1}")

        # Select a random user
        user = platform.sample_user()

        # Perform an action
        news_data = news_feed.get_random_news(10)
        action, prompt = user.perform_action(news_data, platform.get_timeline(user.identifier, 10))
        platform.parse_and_do_action(user.identifier, action, prompt, news_data)

        print(log_action(user, action))

        # Add snapshot of the platform for analysis
        platform.add_snapshot()
        
        if log:
            metrics = compute_metrics(platform, i + 1, cost_input, cost_output, cost_cached, compute_clustering=(i % 100 == 0))
            wandb.log(metrics)

        # Refresh client every 1000 steps
        if i % 1000 == 0 and i != 0:
            
            new_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv(f"OPENROUTER_API_KEY_{openrouter_api_key}"),
            )
            platform.set_client(new_client)
            client.close()

            client = new_client

    # Set reuse of platform
    platform.set_client(None)
    client.close()

    # Save current state of the platform to wandb
    if log:
        final_metrics = compute_metrics(platform, simulation_steps, cost_input, cost_output, cost_cached, compute_clustering=True)
        for key, value in final_metrics.items():
            wandb.summary[f"final/{key}"] = value

        # Save platform pickle as a temporary file and upload as wandb artifact
        run_label = sim_path or "run"
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=True) as tmp:
            pickle.dump(platform, tmp)
            tmp.flush()
            artifact = wandb.Artifact(
                name=f"platform-{run_label}",
                type="platform",
            )
            artifact.add_file(tmp.name, name=f"{run_label}.pkl")
            wandb.log_artifact(artifact)
        if own_wandb_run:
            wandb.finish()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--personas_file", type=str, default='personas.json', help="Path to the personas file")  
    argparser.add_argument("--user_link_strategy", type=str, default='on_repost_bio', help="User link strategy for the simulation")
    argparser.add_argument("--timeline_select_strategy", type=str, default='other_partisan', choices=['random', 'random_weighted', 'random_weighted_reversed', 'bridging_attributes', 'chronological', 'other_partisan'], help="Timeline selection strategy for the simulation")
    argparser.add_argument("--openrouter_api_key", type=int, default=None, help="If None, use OpenAI key, Which OpenRouter API key to use from env (1, 2, or 3)")
    argparser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="Which LLM model to use for the agents")
    argparser.add_argument("--news_feed", type=str, default='News_Category_Dataset_v3.json', help="Path to the news feed dataset")
    argparser.add_argument("--simulation_size", type=int, default=500, help="Number of users in the simulation")
    argparser.add_argument("--simulation_steps", type=int, default=5000, help="Number of steps to run the simulation for")
    argparser.add_argument('--no_log', action='store_true', default=False)
    argparser.add_argument('--no_personas', action='store_true', default=False, help="Omit the persona description from the agent's system prompt (neutral-agent baseline)")
    argparser.add_argument('--no_bio', action='store_true', default=False, help="Omit the target user's bio when an agent decides whether to follow them")
    argparser.add_argument('--no_news_category', action='store_true', default=False, help="Omit the news category when an agent decides which news to share")
    argparser.add_argument("--wandb_project", type=str, default="prosocial-interventions", help="Wandb project to log the run to")
    argparser.add_argument("--include_group_context", action="store_true", default=False,
        help="Prepend a short paragraph to each persona's system message naming the two "
             "rival political affiliations and their leader (mirrors the interview script's "
             "--include_group_context). Applied regardless of obfuscation condition, including "
             "'none', so it stays the only thing that's off by default rather than a confound.")

    args = argparser.parse_args()

    persona_label = get_persona_label(args.personas_file, args.no_personas, args.no_bio)
    sim_path = f"{persona_label}_{args.user_link_strategy}_{args.timeline_select_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Running simulation {sim_path}...")

    run_simulation(
        simulation_size=args.simulation_size,
        simulation_steps=args.simulation_steps,
        user_link_strategy=args.user_link_strategy,
        timeline_select_strategy=args.timeline_select_strategy,
        llm_model=args.llm_model,
        news_feed=args.news_feed,
        show_info=True, sim_path=sim_path,
        personas_file=args.personas_file,
        openrouter_api_key=args.openrouter_api_key,
        log = not args.no_log,
        no_personas=args.no_personas,
        no_bio=args.no_bio,
        no_news_category=args.no_news_category,
        wandb_project=args.wandb_project,
        include_group_context=args.include_group_context
    )