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
from scipy.stats import mannwhitneyu
import pandas as pd

FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

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
    'noPartyId_.json': {
        'color': '#2D7D2D',
        'label': 'No PID',
        'group': 'persona',
    },
    'noVoted2020_.json': {
        'color': '#8DD38D',
        'label': 'No VB',
        'group': 'persona',
    }
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
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'medium',
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.titleweight': 'medium',
    # Axes
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.axisbelow': True,
    # Ticks
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    # Lines & patches
    'lines.linewidth': 1.5,
    'patch.edgecolor': 'white',
    'patch.linewidth': 0.5,
    # Grid (used selectively)
    'grid.color': '#333333',
    'grid.alpha': 0.15,
    'grid.linestyle': '-',
    # Figure
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    # PDF/PS export with editable text
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def plot_metrics_comparison(axes, subset_keys, data, raw_data=None, alpha=0.05):
    """Plot bar charts with optional significance annotations vs the first (baseline) setting."""
    bar_colors = [settings_config[s]['color'] for s in subset_keys]
    bar_labels = [settings_config[s]['label'] for s in subset_keys]
    x = np.arange(len(subset_keys))
    width = 0.65
    baseline = subset_keys[0]

    # Mann-Whitney U tests: each ablation vs baseline, per metric
    sig_map = {}  # (metric_idx, bar_idx) -> significant?
    if raw_data and baseline in raw_data:
        for metric_idx, metric in enumerate(METRICS):
            baseline_vals = raw_data.get(baseline, {}).get(metric, [])
            for bar_idx, setting in enumerate(subset_keys[1:], start=1):
                ablation_vals = raw_data.get(setting, {}).get(metric, [])
                if len(baseline_vals) >= 2 and len(ablation_vals) >= 2:
                    _, p = mannwhitneyu(baseline_vals, ablation_vals, alternative='two-sided')
                    sig_map[(metric_idx, bar_idx)] = p < alpha

    for idx, (metric, label) in enumerate(METRICS.items()):
        ax = axes[idx]
        values = [data[s][metric] for s in subset_keys]
        errors = [data[s][f'{metric}_se'] for s in subset_keys]

        bars = ax.bar(x, values, width, color=bar_colors,
                      yerr=errors, capsize=4, error_kw={'elinewidth': 1.0, 'capthick': 1.0})

        # Annotate significant differences with *
        for bar_idx in range(1, len(subset_keys)):
            if sig_map.get((idx, bar_idx), False):
                bar_val = values[bar_idx]
                bar_err = errors[bar_idx]
                y_pos = bar_val + bar_err if bar_val >= 0 else bar_val - bar_err
                ax.text(x[bar_idx], y_pos, '*', ha='center', va='bottom' if bar_val >= 0 else 'top',
                        fontweight='bold', color='#333333')

        ax.set_title(label, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.grid(True)

#%% Load runs from wandb and plot per model
BIO_PERSONAS = [
    'personas.json',
    'noBioLoveHate_.json',
    'noBioLoveHate_noBioPartyId_.json',
    'noBioLoveHate_noBioPartyId_noBioVoted2020_.json',
]
PERSONAS = [
    'personas.json',
    'noLoveHate_.json',
    'noLoveHate_noPartyId_.json',
    'noLoveHate_noPartyId_noVoted2020_.json',
]

api = wandb.Api()
for model_name in MODEL_NAMES:
    runs = api.runs(
        "prosocial-interventions",
        filters={"config.llm_model": model_name},
    )
    runs = list(runs)
    print(f"\nFound {len(runs)} runs with llm_model='{model_name}'")

    # Group runs by timeline_select_strategy
    strategies = set()
    for run in runs:
        strategies.add(run.config.get("timeline_select_strategy", "unknown"))
    strategies = sorted(strategies)

    for strategy in strategies:
        strategy_runs = [r for r in runs if r.config.get("timeline_select_strategy", "unknown") == strategy]
        print(f"\n  Strategy: {strategy} ({len(strategy_runs)} runs)")

        raw_per_setting = defaultdict(lambda: defaultdict(list))

        for run in strategy_runs:
            personas_file = run.config.get("personas_file", "unknown")
            setting = match_persona_file(personas_file)
            if setting is None:
                print(f"    Warning: unknown personas_file '{personas_file}', skipping run {run.name}")
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
            print(f"    No metric data found, skipping plot.")
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

        if BIO_PERSONAS is not None:
            bio_keys = [k for k in BIO_PERSONAS if k in wandb_metrics]
        else:
            bio_keys = [k for k in settings_config if settings_config[k]['group'] in ('both', 'bio') and k in wandb_metrics]
        if PERSONAS is not None:
            list_keys = [k for k in PERSONAS if k in wandb_metrics]
        else:
            list_keys = [k for k in settings_config if settings_config[k]['group'] in ('both', 'persona') and k in wandb_metrics]
        n_rows = (1 if bio_keys else 0) + (1 if list_keys else 0)

        if n_rows == 0:
            print(f"    No matching settings found, skipping plot.")
            continue

        fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(10, 3.5 * n_rows))
        if n_rows == 1:
            axes = [axes]

        row = 0
        if bio_keys:
            plot_metrics_comparison(axes[row], bio_keys, data=wandb_metrics, raw_data=raw_per_setting)
            row += 1
        if list_keys:
            plot_metrics_comparison(axes[row], list_keys, data=wandb_metrics, raw_data=raw_per_setting)

        safe_name = model_name.replace("/", "_")
        safe_strategy = strategy.replace("/", "_")
        fig.tight_layout(pad=1.5, h_pad=4.0)
        fig.savefig(FIGS_DIR / f'metrics_comparison_{safe_name}_{safe_strategy}.pdf')
        fig.savefig(FIGS_DIR / f'metrics_comparison_{safe_name}_{safe_strategy}.png')
        plt.show()
        print(f"    Saved metrics_comparison_{safe_name}_{safe_strategy}.pdf/png")

#%% Plot networks

PERSONAS_TO_PLOT = [
    'personas.json',
    'noLoveHate_.json',
    'noLoveHate_noPartyId_.json',
    'noLoveHate_noPartyId_noVoted2020_.json',
]

PARTY_COLORS = {
    'Democrat': '#4878A8',
    'Republican': '#D45500',
}

NETWORK_MODEL = "gpt-4o-mini"
NETWORK_GROUP = "persona"

api = wandb.Api()
network_runs = list(api.runs(
    "prosocial-interventions",
    filters={"config.llm_model": NETWORK_MODEL},
))

# Group by timeline_select_strategy
network_strategies = sorted(set(r.config.get("timeline_select_strategy", "unknown") for r in network_runs))

all_platforms_by_strategy = {}
for strategy in network_strategies:
    strategy_runs = [r for r in network_runs if r.config.get("timeline_select_strategy", "unknown") == strategy]
    print(f"\nNetwork plots for strategy: {strategy} ({len(strategy_runs)} runs)")

    # Pick the first run for each persona file setting
    runs_by_setting = {}
    for run in strategy_runs:
        personas_file = run.config.get("personas_file", "unknown")
        setting = match_persona_file(personas_file)
        if setting is not None and setting not in runs_by_setting:
            runs_by_setting[setting] = run

    # Filter to PERSONAS_TO_PLOT, maintain order
    ordered_settings = [
        s for s in PERSONAS_TO_PLOT
        if s in runs_by_setting
    ]
    n_cols = len(ordered_settings)

    if n_cols == 0:
        print(f"  No matching settings found, skipping.")
        continue

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    platforms_by_setting = {}
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
        platforms_by_setting[setting] = platform

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
        ax.set_title(f"({panel_letter}) {settings_config[setting]['label']}", pad=10)

    fig.tight_layout()
    safe_name = NETWORK_MODEL.replace("/", "_")
    safe_strategy = strategy.replace("/", "_")
    fig.savefig(FIGS_DIR / f'networks_{safe_name}_{safe_strategy}.pdf')
    fig.savefig(FIGS_DIR / f'networks_{safe_name}_{safe_strategy}.png')
    plt.show()
    print(f"  Saved networks_{safe_name}_{safe_strategy}.pdf/png")

    all_platforms_by_strategy[strategy] = platforms_by_setting

#%% Cross-party follow analysis
# For each setting, compute per-party breakdown of who users follow (same vs opposing party)

PARTIES = ['Democrat', 'Republican', 'Non-partisan']

followed_colors = {'Democrat': '#03357D', 'Republican': '#D50403', 'Non-partisan': '#888888'}
OPPOSING = {'Democrat': 'Republican', 'Republican': 'Democrat'}
PANEL_PARTIES = ['Democrat', 'Republican']

def _slope_panel(ax, ax_idx, title, compute_fn, setting_labels, x_ticks):
    """Draw slope lines for each party returned by compute_fn(setting_label, party) -> (p, n)."""
    for party in PANEL_PARTIES:
        vals, errs = [], []
        for setting_label in setting_labels:
            result = compute_fn(setting_label, party)
            if result is None:
                vals.append(float('nan'))
                errs.append(float('nan'))
                continue
            p, n = result
            vals.append(p)
            errs.append(1.96 * (p * (1 - p) / n) ** 0.5)

        color = followed_colors[party]
        ax.errorbar(x_ticks, vals, yerr=errs, marker='o', color=color,
                    linewidth=1.5, markersize=5, solid_capstyle='round',
                    clip_on=False, capsize=3, capthick=1.0, elinewidth=1.0)
        if not pd.isna(vals[-1]):
            ax.text(len(x_ticks) - 1 + 0.12, vals[-1], party,
                    va='center', color=color, clip_on=False)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(setting_labels, rotation=30, ha='right', rotation_mode='anchor')
    ax.set_title(title, pad=8)
    ax.set_ylabel('Fraction of follows' if ax_idx == 0 else '')
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.grid(True)
    ax.set_xlim(-0.3, len(x_ticks) - 1 + 1.5)

for strategy in network_strategies:
    platforms_by_setting = all_platforms_by_strategy.get(strategy, {})
    if not platforms_by_setting:
        continue

    print(f"\nCross-party follows for strategy: {strategy}")

    cross_party_rows = []
    for setting, platform in platforms_by_setting.items():
        party_by_id = {user.identifier: user.persona.get('party', 'Unknown') for user in platform.users}

        for from_id, to_id in platform.user_links:
            from_party = party_by_id.get(from_id, 'Unknown')
            to_party = party_by_id.get(to_id, 'Unknown')
            cross_party_rows.append({
                'setting': settings_config[setting]['label'],
                'follower_party': from_party,
                'followed_party': to_party,
            })

    cross_party_df = pd.DataFrame(cross_party_rows)

    # Preserve setting order from PERSONAS_TO_PLOT
    setting_labels = [settings_config[s]['label'] for s in PERSONAS_TO_PLOT if s in platforms_by_setting]
    x_ticks = list(range(len(setting_labels)))

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

    # Panel 1: Total cross-party follows (fraction of follows going to the opposing party)
    def cross_party_total(setting_label, party, _df=cross_party_df):
        subset = _df[
            (_df['setting'] == setting_label) &
            (_df['follower_party'] == party)
        ]
        if subset.empty:
            return None
        opposite = OPPOSING[party]
        p = (subset['followed_party'] == opposite).mean()
        return p, len(subset)

    _slope_panel(axes[0], 0, 'Cross-party follows', cross_party_total, setting_labels, x_ticks)
    axes[0].axhline(y=0.215, color='#999999', linestyle='--', linewidth=1.0)
    axes[0].text(2.3, 0.13, '(Gopal et al., 2025)',
                 va='bottom', color='#999999', fontsize=9, fontstyle='italic', clip_on=False)

    # Panels 2–3: Democrat / Republican follows breakdown
    def party_follows_fn(setting_label, followed_party, follower_party, _df=cross_party_df):
        subset = _df[
            (_df['setting'] == setting_label) &
            (_df['follower_party'] == follower_party)
        ]
        if subset.empty:
            return None
        p = (subset['followed_party'] == followed_party).mean()
        return p, len(subset)

    for panel_idx, follower_party in enumerate(PANEL_PARTIES, start=1):
        def _make_fn(fp=follower_party, _df=cross_party_df):
            return lambda sl, fp2: party_follows_fn(sl, fp2, fp, _df)
        _slope_panel(axes[panel_idx], panel_idx, f'{follower_party} follows:', _make_fn(), setting_labels, x_ticks)

    axes[1].axhline(y=0.6711, color=followed_colors['Democrat'], linestyle='--', linewidth=1.0, alpha=0.5)
    axes[1].axhline(y=0.3289, color=followed_colors['Republican'], linestyle='--', linewidth=1.0, alpha=0.5)
    axes[1].text(-0.2, 0.6711, '(Halberstam et al., 2016)',
                 va='bottom', color=followed_colors['Democrat'], fontsize=8, fontstyle='italic', clip_on=False)
    axes[1].text(2.3, 0.35, '(Halberstam et al., 2016)',
                 va='bottom', color=followed_colors['Republican'], fontsize=8, fontstyle='italic', clip_on=False)

    axes[2].axhline(y=0.2025, color=followed_colors['Democrat'], linestyle='--', linewidth=1.0, alpha=0.5)
    axes[2].axhline(y=0.7975, color=followed_colors['Republican'], linestyle='--', linewidth=1.0, alpha=0.5)
    axes[2].text(2.0, 0.2025, '(Halberstam et al., 2016)',
                 va='bottom', color=followed_colors['Democrat'], fontsize=8, fontstyle='italic', clip_on=False)
    axes[2].text(2.0, 0.7975, '(Halberstam et al., 2016)',
                 va='bottom', color=followed_colors['Republican'], fontsize=8, fontstyle='italic', clip_on=False)

    safe_name = NETWORK_MODEL.replace("/", "_")
    safe_strategy = strategy.replace("/", "_")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f'cross_party_follows_{safe_name}_{safe_strategy}.pdf')
    fig.savefig(FIGS_DIR / f'cross_party_follows_{safe_name}_{safe_strategy}.png')
    plt.show()
    print(f"  Saved cross_party_follows_{safe_name}_{safe_strategy}.pdf/png")
#%% Ablation effect analysis: delta from baseline per removed feature
ABLATION_FILES = {
    'personas.json': None,
    '20260121_personas_with_bio_2000_noLoveHate_.json': 'Affective Polarization',
    '20260316_personas_with_bio_2000_noPartyId_.json': 'Party ID',
    '20260316_personas_with_bio_2000_noVoted2020_.json': 'Voted 2020',
}

def match_ablation_file(personas_file: str) -> str | None:
    """Match a personas_file value to an ABLATION_FILES key by suffix."""
    for key in ABLATION_FILES:
        if personas_file.endswith(key):
            return key
    return None

def plot_ablation_effects(model_name: str):
    """
    Load runs from wandb for the given model, compute the mean delta (ablation - baseline)
    per removed feature per metric, and plot horizontal bar charts — one figure per strategy.
    """
    api = wandb.Api()

    runs = list(api.runs(
        "prosocial-interventions",
        filters={"config.llm_model": model_name},
    ))

    strategies = sorted(set(r.config.get("timeline_select_strategy", "unknown") for r in runs))
    all_dfs = []

    for strategy in strategies:
        strategy_runs = [r for r in runs if r.config.get("timeline_select_strategy", "unknown") == strategy]
        print(f"\nAblation effects for {model_name}, strategy: {strategy} ({len(strategy_runs)} runs)")

        # Collect per-run metric values keyed by ablated feature
        raw = defaultdict(lambda: defaultdict(list))
        for run in strategy_runs:
            pf = run.config.get("personas_file", "")
            matched = match_ablation_file(pf)
            if matched is None:
                continue
            ablated_feature = ABLATION_FILES[matched]  # None for baseline
            for metric in METRICS:
                val = run.summary.get(f"final/{metric}")
                if val is None:
                    hist = run.history(keys=[metric], pandas=True)
                    if len(hist) > 0 and metric in hist.columns:
                        val = hist[metric].dropna().iloc[-1] if not hist[metric].dropna().empty else None
                if val is not None:
                    raw[ablated_feature][metric].append(val)

        # Compute deltas: mean(ablation) - mean(baseline) for each feature
        ablation_names = [v for v in ABLATION_FILES.values() if v is not None]
        results = []
        baseline_vals = raw.get(None, {})
        for feat in ablation_names:
            ablation_vals = raw.get(feat, {})
            for metric in METRICS:
                b = baseline_vals.get(metric, [])
                a = ablation_vals.get(metric, [])
                if b and a:
                    delta = np.mean(a) - np.mean(b)
                    se = np.sqrt(np.std(a, ddof=1)**2 / len(a) + np.std(b, ddof=1)**2 / len(b))
                    _, p = mannwhitneyu(b, a, alternative='two-sided') if len(b) >= 2 and len(a) >= 2 else (None, np.nan)
                    results.append({
                        'removed_feature': feat,
                        'metric': metric, 'delta': delta, 'se': se, 'p': p,
                    })

        df = pd.DataFrame(results)
        if df.empty:
            print(f"  No ablation data found, skipping.")
            continue

        print(df.to_string(index=False))
        all_dfs.append(df)

        colors = {'Affective Polarization': '#3845dc', 'Party ID': '#cd07b4', 'Voted 2020': '#ff6441'}
        fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.5))
        for col, (metric, label) in enumerate(METRICS.items()):
            ax = axes[col]
            metric_df = df[df['metric'] == metric].sort_values('removed_feature')
            if metric_df.empty:
                ax.set_visible(False)
                continue

            y_pos = np.arange(len(metric_df))
            bar_colors = [colors[f] for f in metric_df['removed_feature']]
            ax.barh(y_pos, metric_df['delta'], xerr=metric_df['se'],
                    color=bar_colors,
                    capsize=4, error_kw={'elinewidth': 1.0, 'capthick': 1.0})

            for i, (_, r) in enumerate(metric_df.iterrows()):
                if r['p'] < 0.05:
                    x_pos = r['delta'] + r['se'] if r['delta'] >= 0 else r['delta'] - r['se']
                    ax.text(x_pos, y_pos[i], ' *', ha='left' if r['delta'] >= 0 else 'right',
                            va='center', fontweight='bold', color='#333333')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(metric_df['removed_feature'])
            ax.axvline(x=0, color='#333333', linestyle='-', linewidth=0.5)
            ax.xaxis.grid(True)
            ax.set_title(label, pad=8)

        safe_name = model_name.replace("/", "_")
        safe_strategy = strategy.replace("/", "_")
        fig.suptitle(r"Ablation Effect ($\Delta$ from Full Persona Baseline)")
        fig.tight_layout()
        fig.savefig(FIGS_DIR / f'ablation_effects_{safe_name}_{safe_strategy}.pdf')
        fig.savefig(FIGS_DIR / f'ablation_effects_{safe_name}_{safe_strategy}.png')
        plt.show()
        print(f"  Saved ablation_effects_{safe_name}_{safe_strategy}.pdf/png")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

ablation_df = plot_ablation_effects("gpt-4o-mini")

# %%
