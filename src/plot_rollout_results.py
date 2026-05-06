"""
Plot rollout results for the 3x3 experimental matrix, with 95% Wilson
score intervals on every bar.

Reads 9 .mat files produced by rollout.py and produces three grouped bar charts
forming the success funnel:

  1) Primary   -- gripper ever within XY zone           (lowest bar to clear)
  2) Secondary -- gripper held XY zone for >= 1 sec     (sustained / hover)
  3) Tertiary  -- gripper ever within 3D zone           (full descent)

For each metric, two layouts are produced:

  * by_condition : x = test condition, bars = policy
                   -> "which policy wins on each task"
  * by_policy    : x = policy,         bars = test condition
                   -> "how each policy degrades as the world gets harder"

Error bars: 95% Wilson score intervals (asymmetric in general; safe near 0% / 100%).
Colors    : Okabe-Ito colorblind-safe palette.
Outputs   : PNG (web) + PDF (slides) into OUTPUT_DIR.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# -------------------------------------------------------------------
# Config -- edit these two paths if needed
# -------------------------------------------------------------------
RESULTS_DIR = "C:\\Users\\ahmed\\Documents\\UW-Madison\\Classes\\Spring 2026\\CS 839\\Projects\\final_project\\results\\final"  # folder containing the 9 .mat files
OUTPUT_DIR = "C:\\Users\\ahmed\\Documents\\UW-Madison\\Classes\\Spring 2026\\CS 839\\Projects\\final_project\\results\\final\\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette
# https://jfly.uni-koeln.de/color/
#
# Two palettes, one per layout, each chosen so the bars within a
# cluster read as a difficulty / robustness gradient.
# -------------------------------------------------------------------
POLICY_COLORS = {
    "no_obstacles": "#999999",  # neutral gray  -> baseline policy
    "with_obstacles": "#0072B2",  # blue          -> obstacle-aware
    "occlusions": "#009E73",  # bluish green  -> occlusion-robust
}

CONDITION_COLORS = {
    "no_obstacles_clean": "#56B4E9",  # sky blue   -> easiest test
    "with_obstacles_clean": "#E69F00",  # orange     -> harder
    "with_obstacles_occ": "#D55E00",  # vermillion -> hardest
}

# Policy labels (long form for one-bar legend, short form for x-axis ticks)
POLICY_LABELS_LEGEND = {
    "no_obstacles": "P1 (trained: no obstacles)",
    "with_obstacles": "P2 (trained: obstacles)",
    "occlusions": "P3 (trained: obstacles + occ.)",
}
POLICY_LABELS_TICK = {
    "no_obstacles": "P1\n(trained: no obstacles)",
    "with_obstacles": "P2\n(trained: obstacles)",
    "occlusions": "P3\n(trained: obstacles + occ.)",
}

# Condition labels (two-line for x-axis ticks, single-line for legend)
CONDITION_LABELS_TICK = {
    "no_obstacles_clean": "No obstacles\nno occlusions",
    "with_obstacles_clean": "Obstacles\nno occlusions",
    "with_obstacles_occ": "Obstacles\n+ occlusions",
}
CONDITION_LABELS_LEGEND = {
    "no_obstacles_clean": "No obstacles, no occlusions",
    "with_obstacles_clean": "Obstacles, no occlusions",
    "with_obstacles_occ": "Obstacles + occlusions",
}

# Canonical orderings (easy -> hard)
POLICY_ORDER = ["no_obstacles", "with_obstacles", "occlusions"]
CONDITION_ORDER = ["no_obstacles_clean", "with_obstacles_clean", "with_obstacles_occ"]


# -------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    """Wilson score 95% confidence interval for a binomial proportion.

    Args:
        k : number of successes
        n : number of trials
        z : z-value (1.96 for 95% confidence)

    Returns:
        (lower, upper) in [0, 1]
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - halfwidth), min(1.0, center + halfwidth)


def cell_stats(arr):
    """Compute (mean_pct, err_low_pct, err_high_pct) for a 0/1 success array.

    Errors are expressed as distances from the observed mean to the Wilson
    95% bounds, so they can be passed straight to matplotlib's yerr=[low, up].
    """
    arr = np.asarray(arr).astype(float).ravel()
    n = len(arr)
    if n == 0:
        return np.nan, 0.0, 0.0
    k = int(arr.sum())
    p = k / n
    lo, hi = wilson_ci(k, n)
    return 100.0 * p, 100.0 * (p - lo), 100.0 * (hi - p)


# -------------------------------------------------------------------
# Helpers for getting clean Python scalars out of loadmat() output
# -------------------------------------------------------------------
def _as_str(x):
    return str(np.atleast_1d(x).flatten()[0]).strip()


def _as_int(x):
    return int(np.atleast_1d(x).flatten()[0])


def parse_mat(path):
    """Extract (policy, condition, primary, secondary, tertiary) from a .mat file."""
    m = loadmat(path)

    policy = _as_str(
        m["trained_with_obstacles"]
    )  # 'no_obstacles' / 'with_obstacles' / 'occlusions'
    tested_occ = _as_int(m["tested_with_occlusions"])  # 0 / 1

    # The test scene isn't stored explicitly -- read it from the run_tag in the filename
    fname = os.path.basename(path)
    if "scene-no_obstacles" in fname:
        scene = "no_obstacles"
    elif "scene-with_obstacles" in fname:
        scene = "with_obstacles"
    else:
        raise ValueError(f"Cannot infer test scene from filename: {fname}")

    if scene == "no_obstacles" and tested_occ == 0:
        condition = "no_obstacles_clean"
    elif scene == "with_obstacles" and tested_occ == 0:
        condition = "with_obstacles_clean"
    elif scene == "with_obstacles" and tested_occ == 1:
        condition = "with_obstacles_occ"
    else:
        raise ValueError(
            f"Unexpected scene/occlusion combination in {fname}: "
            f"scene={scene}, tested_occ={tested_occ}"
        )

    primary = np.atleast_1d(m["primary_success"]).flatten().astype(float)
    secondary = np.atleast_1d(m["secondary_success"]).flatten().astype(float)
    tertiary = np.atleast_1d(m["tertiary_success"]).flatten().astype(float)

    return policy, condition, primary, secondary, tertiary


def collect_results(results_dir):
    """Build 3x3 grids where each cell holds (mean_pct, err_low, err_high)."""
    files = sorted(glob.glob(os.path.join(results_dir, "*.mat")))
    if len(files) != 9:
        print(f"[WARN] expected 9 .mat files in {results_dir}, found {len(files)}")

    primary_grid = {p: {} for p in POLICY_ORDER}
    secondary_grid = {p: {} for p in POLICY_ORDER}
    tertiary_grid = {p: {} for p in POLICY_ORDER}
    n_episodes = None

    for f in files:
        policy, condition, primary, secondary, tertiary = parse_mat(f)
        n_episodes = len(primary)
        primary_grid[policy][condition] = cell_stats(primary)
        secondary_grid[policy][condition] = cell_stats(secondary)
        tertiary_grid[policy][condition] = cell_stats(tertiary)

        pm, plo, phi = primary_grid[policy][condition]
        sm, slo, shi = secondary_grid[policy][condition]
        tm, tlo, thi = tertiary_grid[policy][condition]
        print(
            f"  {os.path.basename(f):<70s}  "
            f"policy={policy:<14s} cond={condition:<22s} "
            f"P={pm:5.1f}% [-{plo:.1f},+{phi:.1f}]  "
            f"S={sm:5.1f}% [-{slo:.1f},+{shi:.1f}]  "
            f"T={tm:5.1f}% [-{tlo:.1f},+{thi:.1f}]"
        )

    return primary_grid, secondary_grid, tertiary_grid, n_episodes


# -------------------------------------------------------------------
# Generic grouped-bar plotter
#
# `grid[policy][condition] = (mean_pct, err_low_pct, err_high_pct)`
#
# `outer_keys`     -> what goes on the x-axis (one cluster per key)
# `inner_keys`     -> what each cluster contains (one bar per key)
# `value_lookup`   -> function(outer, inner) -> 3-tuple from the grid
# -------------------------------------------------------------------
def _plot_grouped_bars(
    outer_keys,
    inner_keys,
    value_lookup,
    inner_color_map,
    outer_tick_labels,
    inner_legend_labels,
    xlabel,
    legend_title,
    title,
    ylabel,
    save_name,
    n_episodes,
):
    n_outer = len(outer_keys)
    n_inner = len(inner_keys)
    bar_width = 0.26
    x = np.arange(n_outer)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

    for i, inner in enumerate(inner_keys):
        trios = [value_lookup(outer, inner) for outer in outer_keys]
        heights = [t[0] for t in trios]
        yerr_low = [t[1] for t in trios]
        yerr_up = [t[2] for t in trios]

        offset = (i - (n_inner - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            heights,
            bar_width,
            color=inner_color_map[inner],
            edgecolor="black",
            linewidth=0.7,
            label=inner_legend_labels[inner],
            yerr=[yerr_low, yerr_up],
            capsize=3,
            ecolor="#333",
            error_kw=dict(linewidth=1.1, alpha=0.85, capthick=1.1),
        )
        # Value labels float above the upper error cap
        for bar, h, eu in zip(bars, heights, yerr_up):
            if np.isnan(h):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + eu + 1.5,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([outer_tick_labels[k] for k in outer_keys], fontsize=12)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=8)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_ylim(0, 115)  # extra headroom for upper caps + labels
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=11,
        title_fontsize=11,
    )

    if n_episodes:
        ax.text(
            0.99,
            0.97,
            f"n = {n_episodes} episodes / cell\nerror bars: 95% Wilson CI",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            style="italic",
            color="#555",
        )

    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(OUTPUT_DIR, f"{save_name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  -> {out}")

    plt.close(fig)


# -------------------------------------------------------------------
# Two layouts wrapping the generic plotter
# -------------------------------------------------------------------
def plot_by_condition(grid, title, ylabel, save_name, n_episodes):
    """X = test condition; bars within cluster = policy."""
    _plot_grouped_bars(
        outer_keys=CONDITION_ORDER,
        inner_keys=POLICY_ORDER,
        value_lookup=lambda cond, policy: grid[policy].get(cond, (np.nan, 0.0, 0.0)),
        inner_color_map=POLICY_COLORS,
        outer_tick_labels=CONDITION_LABELS_TICK,
        inner_legend_labels=POLICY_LABELS_LEGEND,
        xlabel="Test condition",
        legend_title="Policy",
        title=title,
        ylabel=ylabel,
        save_name=save_name,
        n_episodes=n_episodes,
    )


def plot_by_policy(grid, title, ylabel, save_name, n_episodes):
    """X = policy; bars within cluster = test condition."""
    _plot_grouped_bars(
        outer_keys=POLICY_ORDER,
        inner_keys=CONDITION_ORDER,
        value_lookup=lambda policy, cond: grid[policy].get(cond, (np.nan, 0.0, 0.0)),
        inner_color_map=CONDITION_COLORS,
        outer_tick_labels=POLICY_LABELS_TICK,
        inner_legend_labels=CONDITION_LABELS_LEGEND,
        xlabel="Policy",
        legend_title="Test condition",
        title=title,
        ylabel=ylabel,
        save_name=save_name,
        n_episodes=n_episodes,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print(f"Reading .mat files from: {os.path.abspath(RESULTS_DIR)}")
    primary_grid, secondary_grid, tertiary_grid, n_eps = collect_results(RESULTS_DIR)

    metrics = [
        (primary_grid, "Primary success: gripper reaches XY zone", "primary_success"),
        (
            secondary_grid,
            "Secondary success: gripper holds XY zone for ≥1 s",
            "secondary_success",
        ),
        (
            tertiary_grid,
            "Tertiary success: gripper reaches 3D zone (full descent)",
            "tertiary_success",
        ),
    ]

    print("\nWriting figures (grouped by test condition)...")
    for grid, title, base in metrics:
        plot_by_condition(
            grid,
            title=title,
            ylabel="Success rate (mean over episodes)",
            save_name=f"{base}_by_condition",
            n_episodes=n_eps,
        )

    print("\nWriting figures (grouped by policy)...")
    for grid, title, base in metrics:
        plot_by_policy(
            grid,
            title=title,
            ylabel="Success rate (mean over episodes)",
            save_name=f"{base}_by_policy",
            n_episodes=n_eps,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
