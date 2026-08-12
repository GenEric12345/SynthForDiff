"""Matplotlib figures (static PNGs for the paper/README).

Fixed series colors (validated categorical order — never reassigned):
  main AUC curve  blue   #2a78d6
  shuffle control orange #eb6834
  unigram         aqua   #1baf7a
  chance band     neutral gray
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_MAIN = "#2a78d6"
C_SHUFFLE = "#eb6834"
C_UNIGRAM = "#1baf7a"
C_NEUTRAL = "#7a7a75"
C_GRID = "#e6e6e2"
C_TEXT = "#1a1a19"
SURFACE = "#fcfcfb"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_TEXT, labelsize=9)


def _fig(figsize=(7.0, 4.4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    return fig, ax


def _auc_axes(ax, threshold: float):
    ax.axhline(0.5, color=C_NEUTRAL, linewidth=1.0)
    ax.axhline(threshold, color=C_NEUTRAL, linewidth=1.0, linestyle="--")
    ax.annotate("chance (0.5)", xy=(0.005, 0.5), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color=C_NEUTRAL)
    ax.annotate(f"threshold ({threshold})", xy=(0.005, threshold), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color=C_NEUTRAL)
    ax.set_xlabel("mask ratio t", color=C_TEXT, fontsize=10)
    ax.set_ylabel("AUC (real vs synthetic)", color=C_TEXT, fontsize=10)
    ax.set_ylim(0.4, 1.02)


def plot_auc_vs_t(df, threshold: float, out_path, chance_band=None) -> None:
    """df: columns t, auc, ci_lo, ci_hi. Optional chance_band: (t, auc_97.5)."""
    fig, ax = _fig()
    _auc_axes(ax, threshold)
    ax.fill_between(df["t"], df["ci_lo"], df["ci_hi"], color=C_MAIN, alpha=0.18,
                    linewidth=0, label="95% CI (bootstrap over documents)")
    ax.plot(df["t"], df["auc"], color=C_MAIN, linewidth=2.0, marker="o",
            markersize=5, label="classifier AUC")
    if chance_band is not None:
        tb, ub = chance_band
        ax.fill_between(tb, 0.5 - (np.asarray(ub) - 0.5), ub, color=C_NEUTRAL,
                        alpha=0.15, linewidth=0, label="permutation null (95%)")
    i = len(df) // 2
    ax.annotate("AUC", xy=(df["t"].iloc[i], df["auc"].iloc[i]),
                xytext=(6, 8), textcoords="offset points",
                fontsize=9, color=C_MAIN, fontweight="bold")
    ax.set_title("Real-vs-synthetic separability under the masking forward process",
                 color=C_TEXT, fontsize=11)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_controls_overlay(main_df, controls_df, threshold: float, out_path) -> None:
    """main_df: t/auc/ci_lo/ci_hi. controls_df: t, auc_shuffle, auc_unigram, perm_null_975."""
    fig, ax = _fig()
    _auc_axes(ax, threshold)
    ax.fill_between(main_df["t"], main_df["ci_lo"], main_df["ci_hi"],
                    color=C_MAIN, alpha=0.15, linewidth=0)
    series = [
        (main_df["t"], main_df["auc"], C_MAIN, "o", "-", "classifier"),
        (controls_df["t"], controls_df["auc_shuffle"], C_SHUFFLE, "s", "-",
         "shuffled tokens"),
        (controls_df["t"], controls_df["auc_unigram"], C_UNIGRAM, "^", "--",
         "unigram baseline"),
    ]
    lo, hi = ax.get_ylim()
    for x, y, c, m, ls, name in series:
        ax.plot(x, y, color=c, linewidth=2.0, marker=m, markersize=5,
                linestyle=ls, label=name)
        y_lab = float(np.clip(y.iloc[-1], lo + 0.015, hi - 0.015))
        ax.annotate(name, xy=(x.iloc[-1], y_lab), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color=c,
                    fontweight="bold", va="center")
    ub = controls_df["perm_null_975"]
    ax.fill_between(controls_df["t"], 0.5 - (ub - 0.5), ub, color=C_NEUTRAL,
                    alpha=0.15, linewidth=0, label="permutation null (95%)")
    ax.set_xlim(-0.02, max(main_df["t"]) + 0.22)  # room for direct labels
    ax.set_title("Main curve vs controls", color=C_TEXT, fontsize=11)
    ax.legend(loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_tmin_histogram(t_min, t_grid, tau: float, out_path) -> None:
    fig, ax = _fig()
    grid = list(t_grid) + [1.0]
    edges = np.array(grid + [1.05]) - 0.025
    ax.hist(t_min, bins=edges, color=C_MAIN, edgecolor=SURFACE, linewidth=2.0)
    ax.set_xlabel("t_min (smallest t with mean P(real) ≥ "
                  f"{tau:.2f}; 1.0 = never)", color=C_TEXT, fontsize=10)
    ax.set_ylabel("synthetic documents", color=C_TEXT, fontsize=10)
    ax.set_title("Distribution of per-document indistinguishability point t_min",
                 color=C_TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
