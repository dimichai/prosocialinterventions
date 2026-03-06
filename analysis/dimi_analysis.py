#%%
import sys
sys.path.insert(0, '..')

import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
from analysis.analyse_multiple import gini_coefficient, EI_index, correlations, inequality

settings_config = {
    'with_love_hate_lists': {
        'color': '#4878A8',
        'label': 'Full Persona',
    },
    
    'without_bio_love_hate_lists': {
        'color': '#D45500',
        'label': 'No Aff. Pol.',
    },
    'without_bio_love_hate_voted2020': {
        'color': '#E88033',
        'label': 'No Aff. Pol. and Vot. Beh.',
    },
    'without_bio_love_hate_party_id': {
        'color': '#F5A666',
        'label': 'No Aff. Pol. and Pol. ID',
    
    },
    'without_love_hate_lists': {
        'color': '#2D7D2D',
        'label': 'No Aff. Pol.',
    },
    'without_love_hate_lists_and_voted2020': {
        'color': '#5AAD5A',
        'label': 'No Aff. Pol. and Vot. Beh.',
    },
    'without_love_hate_and_party_id': {
        'color': '#8DD38D',
        'label': 'No Aff. Pol. and Pol. ID',
    },
    
    'without_love_hate_lists_voted2020_party_id': {
        'color': "#C1F2C1",
        'label': 'No Aff. Pol., Pol. ID, and Vot. Beh.',
    }
}

def generate_graph_from_log(log_data):
    G = nx.DiGraph()
    G.add_edges_from(log_data['user_links'])
    
    nodes = {
    user['identifier']: {
        'label': user['identifier'],
        'neighbors': [link[1] for link in log_data['user_links'] if user['identifier'] == link[0]],
        'party': user['persona']['party'],
    } for user in log_data['users']}
    
    G.add_nodes_from([node for node in nodes])

    return G, nodes

#%% Analyze single simulations, all of them
# for setting in settings_to_analyze:
#     log_dir = f"../results/{setting}"
#     log_setting = "on_repost_bio_other_partisan_info"
#     log_data = json.load(open(f'{log_dir}/{log_setting}_1.json'))
#     print(f"Setting: {setting}")
#     print(f"This simulation used {log_data['total_tokens_input']} input tokens, from which {log_data['total_tokens_cached']} cached.")
#     print(f"This simulation used {log_data['total_tokens_output']} output tokens.")
#     print(f"The predicted total costs are €{log_data['predicted_cost']}")
#     print("")

#%% Visualize specific simulation network
run_to_analyze = 'without_love_hate_lists_voted2020_party_id'
log_dir = f"../results/{run_to_analyze}"
log_setting = "on_repost_bio_other_partisan_info"
log_data = json.load(open(f'{log_dir}/{log_setting}_1.json'))

G, nodes = generate_graph_from_log(log_data)
colors = {node: 'blue' if node_data['party'] == 'Democrat' else 'red' if node_data['party'] == 'Republican' else 'grey' for node, node_data in nodes.items()}

nx.draw_kamada_kawai(G, node_color=[colors[node] for node in G.nodes()], edgecolors='black')
plt.show()

#%% Analyze multiple simulations
# Analyzing graphs is optional and can be turned off to save time
analyze_graph = True
for setting in settings_config:
    log_dir = f"../results/{setting}"
    log_setting = "on_repost_bio_other_partisan_info"
    
    output_data = {}
    
    run_id = 0
    
    while True:
        run_id += 1
        print(f"analyzing run {run_id}... for setting {setting}")
        try:
            f = open(f"{log_dir}/{log_setting}_{run_id}.json", "r")
            
            data = json.load(f)
            
            follower_distribution = [user['followers'] for user in data['users']]
            repost_distribution = [post['reposts'] for post in data['raw_posts']]
            
            output_data[f"simulation_{run_id}"] = {
                "gini_coefficient_followers": gini_coefficient(follower_distribution),
                "gini_coefficient_reposts": gini_coefficient(repost_distribution),
                "EI_index": EI_index(data),
                "correlations": correlations(data),
                "actions": Counter([action['action'] for action in data['actions']]),
                "inequality": inequality(data),
            }
            
            if analyze_graph:
                G, nodes = generate_graph_from_log(data)
                cluster_coeff = nx.clustering(G)
                output_data[f"simulation_{run_id}"]["average_clustering_coefficient"] = np.mean(list(cluster_coeff.values()))

                # Calculate modularity between Democrats and Republicans
                democrats = {node for node in G.nodes() if node in nodes and nodes[node]['party'] == 'Democrat'}
                republicans = {node for node in G.nodes() if node in nodes and nodes[node]['party'] == 'Republican'}
                others = set(G.nodes()) - democrats - republicans
                communities = [democrats, republicans]
                if others:
                    communities.append(others)
                modularity = nx.community.modularity(G, communities)
                output_data[f"simulation_{run_id}"]["modularity_dem_rep"] = modularity
            
            f.close()
        except FileNotFoundError:
            print(f"did not find file for run {run_id}, stopping.")
            break

    with open(f"{log_dir}/{log_setting}_summary.json", "w") as f:
        json.dump(output_data, f, indent=4)
    
#%% Compare different settings

metrics_by_setting = {}
for setting in settings_config:
    summary_file = f"../results/{setting}/{log_setting}_summary.json"
    with open(summary_file, "r") as f:
        summary_data = json.load(f)

    ei_values = [data['EI_index'] for data in summary_data.values()]
    corr_followers_values = [data['correlations']['correlation_followers'] for data in summary_data.values()]
    corr_retweets_values = [data['correlations']['correlation_retweets'] for data in summary_data.values()]
    gini_followers_values = [data['gini_coefficient_followers'] for data in summary_data.values()]
    gini_reposts_values = [data['gini_coefficient_reposts'] for data in summary_data.values()]
    cluster_coeff_values = [data.get('average_clustering_coefficient', np.nan) for data in summary_data.values()]
    modularity_values = [data.get('modularity_dem_rep', np.nan) for data in summary_data.values()]

    n = len(summary_data)
    metrics_by_setting[setting] = {
        'EI': np.mean(ei_values),
        'EI_se': np.std(ei_values) / np.sqrt(n),
        'correlation_followers': np.mean(corr_followers_values),
        'correlation_followers_se': np.std(corr_followers_values) / np.sqrt(n),
        'correlation_retweets': np.mean(corr_retweets_values),
        'correlation_retweets_se': np.std(corr_retweets_values) / np.sqrt(n),
        'gini_followers': np.mean(gini_followers_values),
        'gini_followers_se': np.std(gini_followers_values) / np.sqrt(n),
        'gini_reposts': np.mean(gini_reposts_values),
        'gini_reposts_se': np.std(gini_reposts_values) / np.sqrt(n),
        'average_clustering_coefficient': np.nanmean(cluster_coeff_values),
        'average_clustering_coefficient_se': np.nanstd(cluster_coeff_values) / np.sqrt(n),
        'modularity_dem_rep': np.nanmean(modularity_values),
        'modularity_dem_rep_se': np.nanstd(modularity_values) / np.sqrt(n),
    }

#%%
metric_names = ['EI', 'average_clustering_coefficient', 'correlation_retweets']
metric_labels = ['EI Index', 'Avg. Clustering Coefficient', 'Correlation Retweets - Partisanship']

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

def plot_metrics_comparison(axes, subset_keys):
    subset = {k: settings_config[k] for k in subset_keys}
    bar_colors = [subset[s]['color'] for s in subset]
    bar_labels = [subset[s]['label'] for s in subset]
    x = np.arange(len(subset))
    width = 0.65

    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        values = [metrics_by_setting[s][metric] for s in subset]
        errors = [metrics_by_setting[s][f'{metric}_se'] for s in subset]

        ax.bar(x, values, width, color=bar_colors, edgecolor='white', linewidth=0.5,
               yerr=errors, capsize=3, error_kw={'elinewidth': 0.8, 'capthick': 0.8})

        ax.set_title(label, fontweight='medium', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#333333')
        ax.set_axisbelow(True)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6.4, 4))

# fig.suptitle("(A) Ablations on Agent's Social Bio", fontsize=10, fontweight='medium', y=1.005)

# Row 0: With Lists vs "without bio" ablations (orange)
plot_metrics_comparison(
    axes[0], ['with_love_hate_lists', 'without_bio_love_hate_lists', 'without_bio_love_hate_party_id']
)

# Row 1: With Lists vs "without lists" ablations (green)
plot_metrics_comparison(
    axes[1], ['with_love_hate_lists', 'without_love_hate_lists', 'without_love_hate_and_party_id', 'without_love_hate_lists_voted2020_party_id'],
)

fig.tight_layout(pad=1.2, h_pad=3.5)
# fig.text(0.5, 0.47, "(B) Ablations on Agent's Persona", ha='center', va='bottom', fontsize=9, fontweight='medium')
fig.savefig('metrics_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')


# %%
