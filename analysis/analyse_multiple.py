import json
import numpy as np
import requests
from collections import Counter

to_analyze = "on_repost_bio_other_partisan_info"

output_data = {}

def gini_coefficient(data):

    n = len(data)

    if n == 0:
        return 0
    
    sum_distances = 0
    for i in range(n):
        for j in range(n):
            sum_distances += abs(data[i] - data[j])

    gini = sum_distances / (2 * n * np.sum(data))

    return gini

def EI_index(data):

    IL = 0
    EL = 0

    for user_link in data['user_links']:
        user_from = [user for user in data['users'] if user['identifier'] == user_link[0]][0]
        user_to = [user for user in data['users'] if user['identifier'] == user_link[1]][0]

        if user_from['persona']['party'] == user_to['persona']['party']:
            IL += 1
        else:
            EL += 1

    EI_index = (EL - IL) / (EL + IL)

    return EI_index

def correlations(data):

    partisans = [abs(user['persona']['partisan']) for user in data['users']]
    followers = [user['followers'] for user in data['users']]
    total_retweets_user = [sum([post['reposts'] for post in data['raw_posts'] if post['author'] == user['identifier']]) for user in data['users']]

    # Calculate the correlation between partisanship and followers
    correlation_followers = np.corrcoef(partisans, followers)[0, 1]
    # Calculate the correlation between partisanship and total retweets
    correlation_retweets = np.corrcoef(partisans, total_retweets_user)[0, 1]

    return {
        "correlation_followers": correlation_followers,
        "correlation_retweets": correlation_retweets
    }

def inequality(data):
    
    # sort users by followers
    sorted_users = sorted(data['users'], key=lambda x: x['followers'], reverse=True)

    # get the top 10% of users
    top_10_percent = int(len(sorted_users) * 0.1)
    top_10_percent_users = sorted_users[:top_10_percent]

    top_10_followers = sum([user['followers'] for user in top_10_percent_users])
    total_followers = sum([user['followers'] for user in sorted_users])

    users_with_0_followers = len([user for user in sorted_users if user['followers'] == 0])
    users_with_0or1_followers = len([user for user in sorted_users if user['followers'] <= 1])

    sorted_posts = sorted(data['raw_posts'], key=lambda x: x['reposts'], reverse=True)

    top_10_percent_posts = int(len(sorted_posts) * 0.1)
    top_10_percent_posts_data = sorted_posts[:top_10_percent_posts]
    top_10_reposts = sum([post['reposts'] for post in top_10_percent_posts_data])
    total_reposts = sum([post['reposts'] for post in sorted_posts])
    top_10_reposts_percentage = top_10_reposts / total_reposts if total_reposts > 0 else 0

    posts_with_0_reposts = len([post for post in sorted_posts if post['reposts'] == 0])
    posts_with_0or1_reposts = len([post for post in sorted_posts if post['reposts'] <= 1])

    return {
        "top_10_percent_followers": top_10_followers,
        "top_10_percent_followers_percentage": top_10_followers / total_followers,
        "percentage_users_with_0_followers": users_with_0_followers / len(sorted_users),
        "percentage_users_with_0or1_followers": users_with_0or1_followers / len(sorted_users),
        "top_10_percent_reposts": top_10_reposts,
        "top_10_percent_reposts_percentage": top_10_reposts_percentage,
        "percentage_posts_with_0_reposts": posts_with_0_reposts / len(sorted_posts),
        "percentage_posts_with_0or1_reposts": posts_with_0or1_reposts / len(sorted_posts),
        "max_followers": max([user['followers'] for user in sorted_users]),
        "min_followers": min([user['followers'] for user in sorted_users]),
        "mean_followers": np.mean([user['followers'] for user in sorted_users]),
        "median_followers": np.median([user['followers'] for user in sorted_users]),
        "std_followers": np.std([user['followers'] for user in sorted_users]),
        "max_reposts": max([post['reposts'] for post in sorted_posts]),
        "min_reposts": min([post['reposts'] for post in sorted_posts]),
        "mean_reposts": np.mean([post['reposts'] for post in sorted_posts]),
        "median_reposts": np.median([post['reposts'] for post in sorted_posts]),
        "std_reposts": np.std([post['reposts'] for post in sorted_posts]),
    }

# all_files = [
#     "on_repost_bio_random_weighted_info",
#     "on_repost_bio_bridging_attributes_info",
#     "on_repost_bio_chronological_info",
#     "on_repost_bio_random_info",
#     "on_repost_bio_random_weighted_noinfo",
#     "on_repost_bio_random_weighted_reversed_info",
#     "on_repost_posts_random_weighted_info"
# ]

# all_files = ['on_repost_bio_other_partisan_info']

# for file_to_analyze in all_files:

#     for i in range(1, 6):

#         f = open(f"../results/{file_to_analyze}_{i}.json", "r")
#         data = json.load(f)

#         follower_distribution = [user['followers'] for user in data['users']]
#         repost_distribution = [post['reposts'] for post in data['raw_posts']]
        

#         output_data[f"simulation_{i}"] = {
#             "gini_coefficient_followers": gini_coefficient(follower_distribution),
#             "gini_coefficient_reposts": gini_coefficient(repost_distribution),
#             "EI_index": EI_index(data),
#             "correlations": correlations(data),
#             "actions": Counter([action['action'] for action in data['actions']]),
#             "inequality": inequality(data),
#         }

#         f.close()

#         print(output_data[f"simulation_{i}"])

#     with open(f"../results/{file_to_analyze}_summary.json", "w") as f:
#         json.dump(output_data, f, indent=4)