import argparse
from datetime import datetime
import dotenv
import os
import json
import pickle
import random

from pathlib import Path
from openai import OpenAI

from Agent import Agent
from Platform import Platform
from NewsFeed import NewsFeed

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
                   show_info = True, sim_path="",
                   personas_file = 'personas.json',
                   openrouter_api_key = None):

    # Define the path to the persona file
    persona_path = os.path.join(os.getcwd(), personas_file)
    news_feed = NewsFeed('News_Category_Dataset_v3.json')

    filename = f"{sim_path}"

    platform = Platform(user_link_strategy=user_link_strategy, timeline_select_strategy=timeline_select_strategy, show_info=show_info)
    
    # Ensure the right fraction of Democrats, Republicans, and non-partisans
    selected_users = select_users(persona_path, n=simulation_size)

    # Set client for platform to OpenAI gpt-4o-mini
    model = "gpt-4o-mini"
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

    json.dump(platform.generate_log(), open(filename + '.json', 'w'), indent=4, default=str)

    # Set reuse of platform
    platform.set_client(None)
    client.close()

    pickle.dump(platform, open(filename + '.pkl', 'wb'))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--personas_file", type=str, default='personas.json', help="Path to the personas file")  
    argparser.add_argument("--user_link_strategy", type=str, default='on_repost_bio', help="User link strategy for the simulation")
    argparser.add_argument("--timeline_select_strategy", type=str, default='other_partisan', help="Timeline selection strategy for the simulation") 
    argparser.add_argument("--openrouter_api_key", type=int, default=None, help="If None, use OpenAI key, Which OpenRouter API key to use from env (1, 2, or 3)")
    args = argparser.parse_args()
    
    sim_dir = f"../results/{args.personas_file.split('.')[0]}_{args.user_link_strategy}_{args.timeline_select_strategy}"
    sim_run = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sim_path = Path(sim_dir, sim_run)
    os.makedirs(sim_dir, exist_ok=True)
    print(f"Running simulation {sim_path}...")

    run_simulation(simulation_size=500, simulation_steps=5000,
                user_link_strategy=args.user_link_strategy, 
                timeline_select_strategy=args.timeline_select_strategy,
                show_info=True, sim_path=sim_path,
                personas_file=args.personas_file,
                openrouter_api_key=args.openrouter_api_key)