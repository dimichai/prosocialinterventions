#%%
import sys
sys.path.insert(0, '..')

import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
from analysis.analyse_multiple import gini_coefficient, EI_index, correlations, inequality

log_dir = "../results/without_love_hate_lists"
log_setting = "on_repost_bio_other_partisan_info"
log_data = json.load(open(f'{log_dir}/{log_setting}_1.json'))

#%%
print(f"This simulation used {log_data['total_tokens_input']} input tokens, from which {log_data['total_tokens_cached']} cached.")
print(f"This simulation used {log_data['total_tokens_output']} output tokens.")
print(f"The predicted total costs are €{log_data['predicted_cost']}")

#%% Color graph based on labels
G = nx.DiGraph()

G.add_edges_from(log_data['user_links'])

nodes = {
    user['identifier']: {
        'label': user['identifier'],
    'neighbors': [link[1] for link in log_data['user_links'] if user['identifier'] == link[0]],
    'party': user['persona']['party'],
} for user in log_data['users']}

colors = {node: 'blue' if node_data['party'] == 'Democrat' else 'red' if node_data['party'] == 'Republican' else 'grey' for node, node_data in nodes.items()}

G.add_nodes_from([node for node in nodes])

nx.draw_kamada_kawai(G, node_color=[colors[node] for node in G.nodes()], edgecolors='black')
plt.show()

#%% Analyze multiple simulations
nr_runs = 1
output_data = {}

for i in range(1, nr_runs + 1):

    f = open(f"{log_dir}/{log_setting}_{i}.json", "r")
    data = json.load(f)

    follower_distribution = [user['followers'] for user in data['users']]
    repost_distribution = [post['reposts'] for post in data['raw_posts']]
    

    output_data[f"simulation_{i}"] = {
        "gini_coefficient_followers": gini_coefficient(follower_distribution),
        "gini_coefficient_reposts": gini_coefficient(repost_distribution),
        "EI_index": EI_index(data),
        "correlations": correlations(data),
        "actions": Counter([action['action'] for action in data['actions']]),
        "inequality": inequality(data),
    }

    f.close()

    print(output_data[f"simulation_{i}"])

with open(f"{log_dir}/{log_setting}_summary.json", "w") as f:
    json.dump(output_data, f, indent=4)
    
#%%
summary_data = json.load(open(f"{log_dir}/{log_setting}_summary.json", "r"))

mean_ei = np.mean([data['EI_index'] for simulation, data in summary_data.items()])
min_ei = np.min([data['EI_index'] for simulation, data in summary_data.items()])
max_ei = np.max([data['EI_index'] for simulation, data in summary_data.items()])
mean_gini_followers = np.mean([data['gini_coefficient_followers'] for simulation, data in summary_data.items()])
mean_gini_reposts = np.mean([data['gini_coefficient_reposts'] for simulation, data in summary_data.items()])
mean_correlation_followers = np.mean([data['correlations']['correlation_followers'] for simulation, data in summary_data.items()])
min_correlation_followers = np.min([data['correlations']['correlation_followers'] for simulation, data in summary_data.items()])
max_correlation_followers = np.max([data['correlations']['correlation_followers'] for simulation, data in summary_data.items()])
mean_correlation_retweets = np.mean([data['correlations']['correlation_retweets'] for simulation, data in summary_data.items()])
min_correlation_retweets = np.min([data['correlations']['correlation_retweets'] for simulation, data in summary_data.items()])
max_correlation_retweets = np.max([data['correlations']['correlation_retweets'] for simulation, data in summary_data.items()])
mean_number_reposts = np.mean([data['actions']['1'] for simulation, data in summary_data.items()])
mean_number_posts = np.mean([data['actions']['2'] for simulation, data in summary_data.items()])

#%% Compare different settings
settings_to_compare = ['with_love_hate_lists', 'without_love_hate_lists']

metrics_by_setting = {}
for setting in settings_to_compare:
    summary_file = f"../results/{setting}/{log_setting}_summary.json"
    with open(summary_file, "r") as f:
        summary_data = json.load(f)

    metrics_by_setting[setting] = {
        'EI': np.mean([data['EI_index'] for data in summary_data.values()]),
        'correlation_followers': np.mean([data['correlations']['correlation_followers'] for data in summary_data.values()]),
        'correlation_retweets': np.mean([data['correlations']['correlation_retweets'] for data in summary_data.values()]),
        'gini_followers': np.mean([data['gini_coefficient_followers'] for data in summary_data.values()]),
        'gini_reposts': np.mean([data['gini_coefficient_reposts'] for data in summary_data.values()]),
    }

#%%
# Create Nature-style publication figure
metric_names = ['EI', 'correlation_followers', 'correlation_retweets', 'gini_followers', 'gini_reposts']
metric_labels = ['EI Index', 'Correlation\n(Followers)', 'Correlation\n(Retweets)', 'Gini\n(Followers)', 'Gini\n(Reposts)']

# Nature-style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
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

# Nature single column width: 89mm (~3.5in), double: 183mm (~7.2in)
fig, axes = plt.subplots(1, len(metric_names), figsize=(7.2, 2.2))

x = np.arange(len(settings_to_compare))
width = 0.65

# Professional color palette (colorblind-friendly)
colors = ['#4878A8', '#E07B54']  # Muted blue and coral
setting_labels = ['With Lists', 'Without Lists']

for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
    ax = axes[idx]
    values = [metrics_by_setting[setting][metric] for setting in settings_to_compare]

    bars = ax.bar(x, values, width, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_title(label, fontweight='medium', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(setting_labels, rotation=0, ha='center')

    # Subtle zero line
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.5)

    # Clean up y-axis formatting
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Subtle grid for readability
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax.set_axisbelow(True)

plt.tight_layout(pad=1.2)
plt.savefig('metrics_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# %%
