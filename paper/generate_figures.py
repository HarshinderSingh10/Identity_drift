import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PATHS
# ============================================================

base = r'E:\PythonProject\Identity_drift'

results = os.path.join(base, 'results', 'metrics')
out = os.path.join(base, 'paper', 'figures')

os.makedirs(out, exist_ok=True)

print("=" * 80)
print("GENERATING IEEE RESEARCH PAPER FIGURES")
print("=" * 80)
print(f"Results directory: {results}")
print(f"Output directory : {out}")
print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates, description):
    """
    Find the first matching column from a list of possible names.
    Matching is case-insensitive.
    """
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"\nCould not find {description}.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_identity(value):
    """
    Normalize identity IDs so that:
        00457
        457
        00457.0
    can be compared safely.
    """
    if pd.isna(value):
        return ""

    s = str(value).strip()

    # Remove trailing .0 from numeric-looking values
    if s.endswith(".0"):
        s = s[:-2]

    # If purely numeric, normalize leading zeros
    if s.isdigit():
        return str(int(s))

    return s


def save_figure(fig, filename):
    """
    Save both PDF and PNG versions.
    """
    pdf_path = os.path.join(out, filename + '.pdf')
    png_path = os.path.join(out, filename + '.png')

    fig.savefig(
        pdf_path,
        format='pdf',
        bbox_inches='tight'
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# ============================================================
# FIGURE 1 — PIPELINE
# ============================================================

print("\n[FIGURE 1] Research pipeline")

fig, ax = plt.subplots(figsize=(8.5, 6.8))
ax.set_axis_off()

steps = [
    'MORPH',
    'Dataset Audit',
    'Identity-Disjoint Split',
    'Face Detection + Alignment',
    'ArcFace / InsightFace',
    '512-D Embeddings',
    'Longitudinal Pair Construction',
    'Embedding Drift',
    'Verification Analysis',
    'Identity Stability Analysis',
    'Future Adaptive Re-enrollment'
]

# Top and bottom coordinates
top_y = 0.94
bottom_y = 0.06

ys = np.linspace(top_y, bottom_y, len(steps))

box_height = 0.052
box_x = 0.12
box_width = 0.76

for i, (step, y) in enumerate(zip(steps, ys)):

    box = plt.Rectangle(
        (box_x, y - box_height / 2),
        box_width,
        box_height,
        linewidth=1.2,
        edgecolor='black',
        facecolor='white'
    )

    ax.add_patch(box)

    ax.text(
        0.50,
        y,
        step,
        ha='center',
        va='center',
        fontsize=9.5
    )

    # Correct arrow placement:
    # arrow starts at bottom of current box
    # arrow ends at top of next box
    if i < len(steps) - 1:

        next_y = ys[i + 1]

        ax.annotate(
            '',
            xy=(0.50, next_y + box_height / 2),
            xytext=(0.50, y - box_height / 2),
            arrowprops=dict(
                arrowstyle='-|>',
                lw=1.2,
                color='black'
            )
        )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

save_figure(fig, 'fig1_pipeline')


# ============================================================
# FIGURE 2 — LONGITUDINAL CONCEPT
# ============================================================

print("\n[FIGURE 2] Longitudinal embedding concept")

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.set_axis_off()

x_positions = [0.22, 0.50, 0.78]

ages = ['Age t₁', 'Age t₂', 'Age t₃']
embeddings = ['Embedding e₁', 'Embedding e₂', 'Embedding e₃']

for x, age, embedding in zip(
    x_positions,
    ages,
    embeddings
):

    ax.text(
        x,
        0.82,
        'Person A',
        ha='center',
        fontsize=12,
        weight='bold'
    )

    ax.text(
        x,
        0.68,
        age,
        ha='center',
        fontsize=10
    )

    ax.text(
        x,
        0.51,
        embedding,
        ha='center',
        fontsize=10
    )

    ax.text(
        x,
        0.40,
        '(512-dimensional)',
        ha='center',
        fontsize=9
    )

# Pair comparisons

pairs_to_compare = [
    (0, 1, 'e₁ ↔ e₂'),
    (1, 2, 'e₂ ↔ e₃'),
    (0, 2, 'e₁ ↔ e₃')
]

comparison_y = [0.28, 0.18, 0.08]

for (i, j, label), y in zip(
    pairs_to_compare,
    comparison_y
):

    x1 = x_positions[i]
    x2 = x_positions[j]

    ax.annotate(
        '',
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(
            arrowstyle='<->',
            lw=1.3,
            color='black'
        )
    )

    midpoint = (x1 + x2) / 2

    ax.text(
        midpoint,
        y + 0.035,
        f'Compare {label}',
        ha='center',
        fontsize=9
    )

    ax.text(
        midpoint,
        y - 0.035,
        'Drift = 1 − cosine similarity',
        ha='center',
        fontsize=8.5
    )

ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 1)

save_figure(fig, 'fig2_longitudinal_concept')


# ============================================================
# FIGURE 3 — IDENTITY-LEVEL DRIFT DISTRIBUTION
# ============================================================

print("\n[FIGURE 3] Identity-level drift distribution")

identity_file = os.path.join(
    results,
    'stage6_identity_stability.csv'
)

if not os.path.exists(identity_file):
    raise FileNotFoundError(
        f"Could not find:\n{identity_file}"
    )

ident = pd.read_csv(identity_file)

drift_col = find_column(
    ident,
    [
        'mean_drift',
        'identity_mean_drift',
        'mean_cosine_distance',
        'cosine_distance'
    ],
    'identity-level drift column'
)

vals = pd.to_numeric(
    ident[drift_col],
    errors='coerce'
).dropna()

print(f"Using drift column: {drift_col}")
print(f"Number of identities: {len(vals):,}")

mean_value = vals.mean()
median_value = vals.median()
p95_value = np.quantile(vals, 0.95)
p99_value = np.quantile(vals, 0.99)

fig, ax = plt.subplots(figsize=(7.0, 4.8))

ax.hist(
    vals,
    bins=35,
    edgecolor='black',
    alpha=0.70
)

# Completely different colors + line styles
summary_lines = [
    ('Mean', mean_value, 'tab:red', '-'),
    ('Median', median_value, 'tab:blue', '--'),
    ('P95', p95_value, 'tab:green', '-.'),
    ('P99', p99_value, 'tab:purple', ':')
]

for label, value, color, linestyle in summary_lines:

    ax.axvline(
        value,
        color=color,
        linestyle=linestyle,
        linewidth=2.2,
        label=f'{label} = {value:.4f}'
    )

# Add small value annotations at top
ymax = ax.get_ylim()[1]

for i, (label, value, color, linestyle) in enumerate(summary_lines):

    ax.annotate(
        f'{label}\n{value:.4f}',
        xy=(value, ymax * 0.82),
        xytext=(8 + i * 3, 0),
        textcoords='offset points',
        fontsize=8,
        color=color,
        weight='bold',
        ha='left',
        va='center'
    )

ax.set_xlabel('Identity mean embedding drift')
ax.set_ylabel('Number of identities')

ax.set_title(
    'Distribution of Identity-Level Longitudinal Embedding Drift'
)

ax.grid(
    True,
    axis='y',
    alpha=0.2
)

# Put legend outside
ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    frameon=False,
    fontsize=8
)

fig.subplots_adjust(
    right=0.73
)

save_figure(
    fig,
    'fig3_identity_drift_distribution'
)


# ============================================================
# FIGURE 4 — AGE-GAP DRIFT
# ============================================================

print("\n[FIGURE 4] Age-gap drift")

age_gap_file = os.path.join(
    results,
    'stage6_age_gap_drift.csv'
)

if not os.path.exists(age_gap_file):
    raise FileNotFoundError(
        f"Could not find:\n{age_gap_file}"
    )

pairs = pd.read_csv(age_gap_file)

age_group_col = find_column(
    pairs,
    [
        'age_gap_group',
        'age_group',
        'age_gap_bin'
    ],
    'age-gap group column'
)

distance_col = find_column(
    pairs,
    [
        'cosine_distance',
        'drift',
        'embedding_drift',
        'reference_drift'
    ],
    'embedding drift column'
)

order = [
    '0-2',
    '3-4',
    '5-9',
    '10-19',
    '20+'
]

print(f"Using age group column: {age_group_col}")
print(f"Using drift column: {distance_col}")

box_data = []

sample_sizes = []

for group in order:

    subset = pd.to_numeric(
        pairs.loc[
            pairs[age_group_col].astype(str) == group,
            distance_col
        ],
        errors='coerce'
    ).dropna()

    box_data.append(subset.values)
    sample_sizes.append(len(subset))

    print(
        f"  {group}: n={len(subset):,}"
    )

fig, ax = plt.subplots(figsize=(7.4, 5.2))

bp = ax.boxplot(
    box_data,
    tick_labels=order,
    patch_artist=True,
    widths=0.55,
    showfliers=False
)

# Keep box colors distinct
box_colors = [
    'tab:blue',
    'tab:green',
    'tab:orange',
    'tab:red',
    'tab:purple'
]

for box, color in zip(
    bp['boxes'],
    box_colors
):
    box.set(
        facecolor=color,
        alpha=0.25,
        edgecolor=color,
        linewidth=1.4
    )

# Determine a safe upper position
all_values = np.concatenate(
    [x for x in box_data if len(x) > 0]
)

data_max = np.nanmax(all_values)

# Give lots of room above boxes
ax.set_ylim(
    ax.get_ylim()[0],
    data_max * 1.20
)

label_y = data_max * 1.10

for i, n in enumerate(
    sample_sizes,
    start=1
):

    ax.text(
        i,
        label_y,
        f'n = {n:,}',
        ha='center',
        va='bottom',
        fontsize=8,
        weight='bold'
    )

ax.set_xlabel(
    'Longitudinal age gap (years)'
)

ax.set_ylabel(
    'Embedding drift (1 − cosine similarity)'
)

ax.set_title(
    'Embedding Drift Across Longitudinal Age-Gap Groups'
)

ax.grid(
    True,
    axis='y',
    alpha=0.2
)

save_figure(
    fig,
    'fig4_age_gap_drift'
)


# ============================================================
# FIGURE 5 — AGE-GAP VERIFICATION FRR
# ============================================================

print("\n[FIGURE 5] Age-gap verification")

recognition_file = os.path.join(
    results,
    'stage7_age_gap_recognition.csv'
)

if not os.path.exists(recognition_file):
    raise FileNotFoundError(
        f"Could not find:\n{recognition_file}"
    )

bar = pd.read_csv(recognition_file)

split_col = find_column(
    bar,
    [
        'research_split',
        'split'
    ],
    'research split column'
)

age_group_col = find_column(
    bar,
    [
        'age_gap_group',
        'age_group',
        'age_gap_bin'
    ],
    'age-gap group column'
)

frr_col = find_column(
    bar,
    [
        'frr',
        'false_rejection_rate'
    ],
    'FRR column'
)

n_col = find_column(
    bar,
    [
        'n',
        'pairs',
        'pair_count',
        'count'
    ],
    'pair-count column'
)

bar = bar[
    bar[split_col].astype(str) == 'research_test'
].copy()

bar['_order'] = bar[age_group_col].map(
    {
        '0-2': 0,
        '3-4': 1,
        '5-9': 2,
        '10-19': 3,
        '20+': 4
    }
)

bar = bar.sort_values('_order')

fig, ax = plt.subplots(
    figsize=(7.2, 5.0)
)

x = np.arange(len(bar))

frr_percent = (
    pd.to_numeric(
        bar[frr_col],
        errors='coerce'
    ) * 100
)

sample_sizes = pd.to_numeric(
    bar[n_col],
    errors='coerce'
)

bars = ax.bar(
    x,
    frr_percent,
    width=0.62,
    alpha=0.75,
    edgecolor='black'
)

# Put sample size INTO the x-axis labels
x_labels = [
    f'{group}\n(n={int(n):,})'
    for group, n in zip(
        bar[age_group_col],
        sample_sizes
    )
]

ax.set_xticks(x)
ax.set_xticklabels(
    x_labels,
    fontsize=9
)

# Put only FRR above the bars
for i, value in enumerate(
    frr_percent
):

    ax.text(
        i,
        value + max(1.5, value * 0.04),
        f'{value:.2f}%',
        ha='center',
        va='bottom',
        fontsize=8,
        weight='bold'
    )

ax.set_ylabel(
    'False rejection rate (FRR, %)'
)

ax.set_xlabel(
    'Longitudinal age gap'
)

ax.set_title(
    'Verification False-Rejection Rate Across Age Gaps'
)

ax.set_ylim(
    0,
    max(70, frr_percent.max() * 1.18)
)

ax.grid(
    True,
    axis='y',
    alpha=0.2
)

save_figure(
    fig,
    'fig5_age_gap_verification'
)


# ============================================================
# FIGURE 6 — DRIFT VS VERIFICATION BOUNDARY
# ============================================================

print("\n[FIGURE 6] Drift vs verification boundary")

test_pairs = pairs[
    pairs[age_group_col].astype(str).notna()
].copy()

# Try to find split column in the Stage 6 pair file
pair_split_col = None

for candidate in [
    'research_split',
    'split'
]:
    if candidate in test_pairs.columns:
        pair_split_col = candidate
        break

if pair_split_col is not None:

    test_pairs = test_pairs[
        test_pairs[pair_split_col].astype(str)
        == 'research_test'
    ].copy()

vals = pd.to_numeric(
    test_pairs[distance_col],
    errors='coerce'
).dropna().values

boundary = 0.793229669333

fig, ax = plt.subplots(
    figsize=(7.2, 4.8)
)

ax.hist(
    vals,
    bins=40,
    range=(0, 1.2),
    edgecolor='black',
    alpha=0.75
)

ax.axvline(
    boundary,
    linestyle='--',
    linewidth=2.2,
    color='tab:red',
    label=(
        'Frozen verification boundary\n'
        f'drift = {boundary:.4f}'
    )
)

ax.set_xlabel(
    'Embedding drift (1 − cosine similarity)'
)

ax.set_ylabel(
    'Count of genuine test pairs'
)

ax.set_title(
    'Genuine Test-Pair Drift and Verification Boundary'
)

ax.legend(
    frameon=False
)

ax.grid(
    True,
    axis='y',
    alpha=0.2
)

save_figure(
    fig,
    'fig6_drift_verification_boundary'
)


# ============================================================
# FIGURE 7 — LONGITUDINAL TRAJECTORIES
# ============================================================

print("\n[FIGURE 7] Longitudinal trajectories")

trajectory_file = os.path.join(
    results,
    'stage6_longitudinal_trajectories.csv'
)

if not os.path.exists(trajectory_file):
    raise FileNotFoundError(
        f"Could not find:\n{trajectory_file}"
    )

traj = pd.read_csv(
    trajectory_file
)

print("\nAvailable trajectory columns:")
print(list(traj.columns))

# ------------------------------------------------------------
# Automatically identify the required columns
# ------------------------------------------------------------

trajectory_identity_col = find_column(
    traj,
    [
        'identity_id',
        'identity',
        'person_id',
        'id'
    ],
    'trajectory identity column'
)

trajectory_age_col = find_column(
    traj,
    [
        'filename_age',
        'age',
        'age_years',
        'reference_age'
    ],
    'trajectory age column'
)

trajectory_drift_col = find_column(
    traj,
    [
        'drift_from_reference',
        'reference_drift',
        'mean_reference_drift',
        'cosine_distance',
        'embedding_drift',
        'drift'
    ],
    'trajectory drift column'
)

print(f"\nIdentity column : {trajectory_identity_col}")
print(f"Age column      : {trajectory_age_col}")
print(f"Drift column    : {trajectory_drift_col}")

# Normalize identity IDs
traj['_identity_normalized'] = traj[
    trajectory_identity_col
].apply(normalize_identity)

# ------------------------------------------------------------
# FIRST inspect which identities actually exist
# ------------------------------------------------------------

available_ids = sorted(
    traj['_identity_normalized']
    .dropna()
    .unique()
)

print(
    f"\nTotal unique identities in trajectory file: "
    f"{len(available_ids):,}"
)

requested_ids = [
    '00457',
    '00744',
    '01239',
    '02128'
]

print("\nRequested identities:")

for requested in requested_ids:

    normalized = normalize_identity(requested)

    count = (
        traj['_identity_normalized']
        == normalized
    ).sum()

    print(
        f"  {requested} → {normalized}: "
        f"{count} rows"
    )

# ------------------------------------------------------------
# If requested IDs don't exist, select real identities
# automatically
# ------------------------------------------------------------

valid_requested = [
    normalize_identity(x)
    for x in requested_ids
    if normalize_identity(x) in set(available_ids)
]

if len(valid_requested) < 4:

    print(
        "\nWARNING: Some requested identities were not found."
    )

    print(
        "Selecting identities with the largest number "
        "of longitudinal observations instead."
    )

    identity_counts = (
        traj.groupby(
            '_identity_normalized'
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    fallback_ids = list(
        identity_counts.index[:10]
    )

    # Prefer identities not already selected
    for candidate in fallback_ids:

        if candidate not in valid_requested:

            valid_requested.append(candidate)

        if len(valid_requested) >= 4:
            break

selected = valid_requested[:4]

print(
    "\nFinal identities selected for Figure 7:"
)

for ident in selected:
    print(
        f"  {ident}"
    )

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(8.5, 6.5)
)

axes = axes.flatten()

for ax, ident in zip(
    axes,
    selected
):

    subset = traj[
        traj['_identity_normalized']
        == ident
    ].copy()

    # Convert plotting columns to numeric
    subset['_age'] = pd.to_numeric(
        subset[trajectory_age_col],
        errors='coerce'
    )

    subset['_drift'] = pd.to_numeric(
        subset[trajectory_drift_col],
        errors='coerce'
    )

    subset = subset.dropna(
        subset=['_age', '_drift']
    ).sort_values('_age')

    print(
        f"\nIdentity {ident}: "
        f"{len(subset)} valid trajectory points"
    )

    if len(subset) == 0:

        ax.text(
            0.5,
            0.5,
            'No valid trajectory data',
            ha='center',
            va='center',
            transform=ax.transAxes
        )

        ax.set_title(
            f'Identity {ident}'
        )

        continue

    ax.plot(
        subset['_age'],
        subset['_drift'],
        marker='o',
        linewidth=1.8,
        markersize=4
    )

    ax.set_title(
        f'Identity {ident}',
        fontsize=10
    )

    ax.set_xlabel(
        'Filename age'
    )

    ax.set_ylabel(
        'Reference drift'
    )

    ax.grid(
        True,
        alpha=0.2
    )

fig.suptitle(
    'Representative Longitudinal Embedding-Drift Trajectories',
    fontsize=12,
    y=1.01
)

fig.tight_layout()

save_figure(
    fig,
    'fig7_longitudinal_trajectories'
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FIGURE GENERATION COMPLETE")
print("=" * 80)

for filename in sorted(
    os.listdir(out)
):

    print(filename)

print("\nAll figures were saved to:")
print(out)