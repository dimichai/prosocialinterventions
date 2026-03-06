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

from pathlib import Path
from collections import Counter
from openai import OpenAI

from Agent import Agent
from Platform import Platform
from NewsFeed import NewsFeed

import networkx as nx
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.analyse_multiple import gini_coefficient, EI_index, correlations

dotenv.load_dotenv()

def log_action(user, action):
    """
    Log the action taken by the user to the console.
    """

    log_msg = f"User {user.identifier} chose action "

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

def select_users(persona_path, n):
    """
    Create a sample of users for the simulation from the persona file.
    """

    # According to Gallup, 45% of Americans identify as Democrats, 46% as Republicans, and 9% as other (2025)
    fraction_democrat = 0.45
    fraction_republican = 0.46
    fraction_non_partisan = 0.09

    users = json.load(open(persona_path, 'r'))

    democrat_users = [user for user in users if user['party'] == 'Democrat']
    republican_users = [user for user in users if user['party'] == 'Republican']
    non_partisan_users = [user for user in users if user['party'] == 'Non-partisan']

    # Randomly sample users from each group
    democrat_sample = random.sample(democrat_users, int(n * fraction_democrat))
    republican_sample = random.sample(republican_users, int(n * fraction_republican))
    non_partisan_sample = random.sample(non_partisan_users, int(n * fraction_non_partisan))

    return democrat_sample + republican_sample + non_partisan_sample

def run_simulation(simulation_size = 500, simulation_steps = 10000, 
                user_link_strategy = "on_repost_bio", 
                timeline_select_strategy = "random_weighted",
                llm_model = "gpt-4o-mini",
                news_feed = 'News_Category_Dataset_v3.json',
                show_info = True, 
                sim_path="", 
                personas_file = 'personas.json',
                openrouter_api_key = None, 
                log = True,
                save_full_log = False):
    
    if log:
        wandb.init(project="prosocial-interventions", 
            name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{personas_file.split('.')[0]}", 
            config={
                "simulation_size": simulation_size,
                "simulation_steps": simulation_steps,
                "user_link_strategy": user_link_strategy,
                "timeline_select_strategy": timeline_select_strategy,
                "llm_model": llm_model,
                "news_feed": news_feed,
                "personas_file": personas_file,
                "persona_love_hate_lists": 'noLoveHate' not in personas_file,
                "persona_party_id": 'noPartyId' not in personas_file,
                "persona_voted2020": 'noVoted2020' not in personas_file,
                "bio_love_hate_lists": 'noLoveHate' not in personas_file and 'noBioLoveHate' not in personas_file,
                "bio_party_id": 'noPartyId' not in personas_file and 'noBioPartyId' not in personas_file,
                "bio_voted2020": 'noVoted2020' not in personas_file and 'noBioVoted2020' not in personas_file,
            }
        )

    # Define the path to the persona file
    persona_path = os.path.join(os.getcwd(), personas_file)
    news_feed = NewsFeed(news_feed)

    filename = f"{sim_path}"

    platform = Platform(user_link_strategy=user_link_strategy, timeline_select_strategy=timeline_select_strategy, show_info=show_info)
    
    # Ensure the right fraction of Democrats, Republicans, and non-partisans
    selected_users = select_users(persona_path, n=simulation_size)

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
    [platform.register_user(Agent(model, user)) for user in selected_users]
    platform.set_client(client)
    
    for i in range(simulation_steps):

        print(f"Simulation step {i + 1}")

        # Select a random user
        user = platform.sample_user()

        # Perform an action
        action, prompt = user.perform_action(news_feed.get_random_news(10), platform.get_timeline(user.identifier, 10))
        platform.parse_and_do_action(user.identifier, action, prompt)

        print(log_action(user, action))

        # Add snapshot of the platform for analysis
        platform.add_snapshot()
        
        if log:
            log_start_time = time.time()
            follower_distribution = [u.followers for u in platform.users]
            repost_distribution = [p.reposts for p in platform.raw_posts]
            action_counts = Counter([a['action'] for a in platform.actions])

            metrics = {"step": i + 1}

            # EI index
            if len(platform.user_links) > 0:
                IL = sum(1 for u1, u2 in platform.user_links
                        if platform.get_user(u1).persona['party'] == platform.get_user(u2).persona['party'])
                EL = len(platform.user_links) - IL
                metrics["EI_index"] = (EL - IL) / (EL + IL)

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

            # Estimated cost so far (OpenAI pricing)
            total_input = sum(u.used_tokens_input for u in platform.users)
            total_output = sum(u.used_tokens_output for u in platform.users)
            total_cached = sum(u.used_tokens_cached for u in platform.users)
            metrics["estimated_cost"] = ((0.6 / 1e6) * total_output) + \
                ((0.15 / 1e6) * (total_input - total_cached)) + \
                ((0.075 / 1e6) * total_cached)
            metrics["total_tokens_input"] = total_input
            metrics["total_tokens_output"] = total_output

            # Clustering coefficient
            if len(platform.user_links) > 0:
                G = nx.DiGraph()
                G.add_nodes_from([u.identifier for u in platform.users])
                G.add_edges_from(platform.user_links)
                cluster_coeff = nx.clustering(G)
                metrics["avg_clustering_coefficient"] = np.mean(list(cluster_coeff.values()))

            # Action distribution
            for action_type, count in action_counts.items():
                metrics[f"action_{action_type}"] = count

            metrics["seconds_to_log"] = time.time() - log_start_time
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
    # except:
    #     json.dump(platform.generate_log(), open(filename + '.json', 'w'), indent=4, default=str)

    #     # Set reuse of platform
    #     platform.set_client(None)
    #     client.close()

    #     pickle.dump(platform, open(filename + '.pkl', 'wb'))
    
    if save_full_log:
        json.dump(platform.generate_log(), open(filename + '.json', 'w'), indent=4, default=str)

    # Set reuse of platform
    platform.set_client(None)
    client.close()

    # Save platform pickle and log as wandb artifact
    pkl_path = filename + '.pkl'
    pickle.dump(platform, open(pkl_path, 'wb'))

    # Save current state of the platform to wandb
    if log:
        artifact = wandb.Artifact(
            name=f"platform-{sim_path.stem}",
            type="platform",
        )
        artifact.add_file(pkl_path)
        wandb.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--personas_file", type=str, default='personas.json', help="Path to the personas file")  
    argparser.add_argument("--user_link_strategy", type=str, default='on_repost_bio', help="User link strategy for the simulation")
    argparser.add_argument("--timeline_select_strategy", type=str, default='other_partisan', help="Timeline selection strategy for the simulation") 
    argparser.add_argument("--openrouter_api_key", type=int, default=None, help="If None, use OpenAI key, Which OpenRouter API key to use from env (1, 2, or 3)")
    argparser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="Which LLM model to use for the agents")
    argparser.add_argument("--news_feed", type=str, default='News_Category_Dataset_v3.json', help="Path to the news feed dataset")
    argparser.add_argument("--simulation_size", type=int, default=500, help="Number of users in the simulation")
    argparser.add_argument("--simulation_steps", type=int, default=5000, help="Number of steps to run the simulation for")
    argparser.add_argument('--no_log', action='store_true', default=False)
    argparser.add_argument('--save_full_log', action='store_true', default=False, help="Whether to save the full log of the simulation in a json (can be large)")
    
    args = argparser.parse_args()
    
    sim_dir = f"../results/{args.personas_file.split('.')[0]}_{args.user_link_strategy}_{args.timeline_select_strategy}"
    sim_run = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sim_path = Path(sim_dir, sim_run)
    os.makedirs(sim_dir, exist_ok=True)
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
        save_full_log=args.save_full_log
    )