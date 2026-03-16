#%%
import sys
sys.path.insert(0, '../src')

import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import wandb
from collections import defaultdict
from pathlib import Path

# Keyed by persona file suffix. 'group' determines which plot row(s) the setting appears in.
settings_config = {
    'personas.json': {
        'color': '#4878A8',
        'label': 'Full Persona',
        'group': 'both',
    },
    'noBioLoveHate_.json': {
        'color': '#D45500',
        'label': 'No AP',
        'group': 'bio',
    },
    'noBioLoveHate_noBioPartyId_.json': {
        'color': '#E88033',
        'label': 'No AP & PID',
        'group': 'bio',
    },
    'noBioLoveHate_noBioPartyId_noBioVoted2020_.json': {
        'color': '#F5A666',
        'label': 'No AP & PID & VB',
        'group': 'bio',
    },
    'noLoveHate_.json': {
        'color': '#2D7D2D',
        'label': 'No AP',
        'group': 'persona',
    },
    'noLoveHate_noPartyId_.json': {
        'color': '#5AAD5A',
        'label': 'No AP & PID',
        'group': 'persona',
    },
    'noLoveHate_noPartyId_noVoted2020_.json': {
        'color': '#8DD38D',
        'label': 'No AP & PID & VB',
        'group': 'persona',
    },
}

METRICS = {
    'EI_index': 'EI Index',
    'avg_clustering_coefficient': 'Avg. Clustering Coefficient',
    'correlation_retweets_partisan': 'Correlation Retweets - Partisanship',
}

MODEL_NAMES = [
    "gpt-4o-mini",
    "google/gemini-2.5-flash-lite",
    "mistralai/mistral-small-3.2-24b-instruct",
]

def match_persona_file(personas_file: str) -> str | None:
    """Match a personas_file config value to a settings_config key by suffix."""
    for suffix in settings_config:
        if personas_file.endswith(suffix):
            return suffix
    return None

#%% Plot setup

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

def plot_metrics_comparison(axes, subset_keys, data):
    bar_colors = [settings_config[s]['color'] for s in subset_keys]
    bar_labels = [settings_config[s]['label'] for s in subset_keys]
    x = np.arange(len(subset_keys))
    width = 0.65

    for idx, (metric, label) in enumerate(METRICS.items()):
        ax = axes[idx]
        values = [data[s][metric] for s in subset_keys]
        errors = [data[s][f'{metric}_se'] for s in subset_keys]

        ax.bar(x, values, width, color=bar_colors, edgecolor='white', linewidth=0.5,
               yerr=errors, capsize=3, error_kw={'elinewidth': 0.8, 'capthick': 0.8})

        ax.set_title(label, fontweight='medium', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#333333')
        ax.set_axisbelow(True)

#%% Load runs from wandb and plot per model

api = wandb.Api()
for model_name in MODEL_NAMES:
    runs = api.runs(
        "prosocial-interventions",
        filters={"config.llm_model": model_name},
    )
    print(f"\nFound {len(runs)} runs with llm_model='{model_name}'")

    raw_per_setting = defaultdict(lambda: defaultdict(list))

    for run in runs:
        personas_file = run.config.get("personas_file", "unknown")
        setting = match_persona_file(personas_file)
        if setting is None:
            print(f"  Warning: unknown personas_file '{personas_file}', skipping run {run.name}")
            continue

        for metric in METRICS:
            val = run.summary.get(f"final/{metric}")
            if val is None:
                hist = run.history(keys=[metric], pandas=True)
                if len(hist) > 0 and metric in hist.columns:
                    val = hist[metric].dropna().iloc[-1] if not hist[metric].dropna().empty else None
            if val is not None:
                raw_per_setting[setting][metric].append(val)

    if not raw_per_setting:
        print(f"  No metric data found, skipping plot.")
        continue

    wandb_metrics = {}
    for setting, vals in raw_per_setting.items():
        wandb_metrics[setting] = {}
        for metric in METRICS:
            v = vals.get(metric, [])
            if v:
                wandb_metrics[setting][metric] = np.nanmean(v)
                wandb_metrics[setting][f'{metric}_se'] = np.nanstd(v) / np.sqrt(len(v))
            else:
                wandb_metrics[setting][metric] = np.nan
                wandb_metrics[setting][f'{metric}_se'] = np.nan

    bio_keys = [k for k in settings_config if settings_config[k]['group'] in ('both', 'bio') and k in wandb_metrics]
    list_keys = [k for k in settings_config if settings_config[k]['group'] in ('both', 'persona') and k in wandb_metrics]
    n_rows = (1 if bio_keys else 0) + (1 if list_keys else 0)

    if n_rows == 0:
        print(f"  No matching settings found, skipping plot.")
        continue

    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(6.4, 2.0 * n_rows))
    if n_rows == 1:
        axes = [axes]

    row = 0
    if bio_keys:
        plot_metrics_comparison(axes[row], bio_keys, data=wandb_metrics)
        row += 1
    if list_keys:
        plot_metrics_comparison(axes[row], list_keys, data=wandb_metrics)

    safe_name = model_name.replace("/", "_")
    # fig.suptitle(model_name, fontsize=10, fontweight='medium')
    fig.tight_layout(pad=1.2, h_pad=3.5)
    fig.savefig(f'metrics_comparison_{safe_name}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'metrics_comparison_{safe_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved metrics_comparison_{safe_name}.pdf/png")

#%% Download platform artifacts from wandb and plot networks

PARTY_COLORS = {
    'Democrat': '#4878A8',
    'Republican': '#D45500',
}

NETWORK_MODEL = "gpt-4o-mini"
NETWORK_GROUP = "persona"

api = wandb.Api()
runs = api.runs(
    "prosocial-interventions",
    filters={"config.llm_model": NETWORK_MODEL},
)

# Pick the first run for each persona file setting
runs_by_setting = {}
for run in runs:
    personas_file = run.config.get("personas_file", "unknown")
    setting = match_persona_file(personas_file)
    if setting is not None and setting not in runs_by_setting:
        runs_by_setting[setting] = run

# Filter to selected group, maintain settings_config order
ordered_settings = [
    s for s in settings_config
    if s in runs_by_setting and settings_config[s]['group'] in ('both', NETWORK_GROUP)
]
n_cols = len(ordered_settings)

fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
if n_cols == 1:
    axes = [axes]

for idx, setting in enumerate(ordered_settings):
    run = runs_by_setting[setting]

    # Download the platform artifact (cached locally per run)
    run_artifact_dir = Path("artifacts") / run.id
    pkl_files = list(run_artifact_dir.glob("*.pkl"))
    if not pkl_files:
        artifacts = list(run.logged_artifacts())
        platform_artifact = next(a for a in artifacts if a.type == "platform")
        platform_artifact.download(root=str(run_artifact_dir))
        pkl_files = list(run_artifact_dir.glob("*.pkl"))
    with open(pkl_files[0], "rb") as f:
        platform = pickle.load(f)

    # Build network
    G = nx.DiGraph()
    G.add_nodes_from(user.identifier for user in platform.users)
    G.add_edges_from(platform.user_links)

    node_colors = [
        PARTY_COLORS.get(user.persona.get('party', ''), '#999999')
        for user in sorted(platform.users, key=lambda u: u.identifier)
    ]

    ax = axes[idx]
    nx.draw_kamada_kawai(G, ax=ax, node_color=node_colors, edgecolors='black',
                        node_size=100, width=1.0, linewidths=0.5)
    panel_letter = chr(ord('A') + idx)
    ax.set_title(f"({panel_letter}) {settings_config[setting]['label']}", fontweight='medium', pad=8, fontsize=22)

# fig.suptitle(NETWORK_MODEL, fontsize=10, fontweight='medium')
fig.tight_layout()
safe_name = NETWORK_MODEL.replace("/", "_")
fig.savefig(f'networks_{safe_name}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'networks_{safe_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved networks_{safe_name}.pdf/png")
# %%
