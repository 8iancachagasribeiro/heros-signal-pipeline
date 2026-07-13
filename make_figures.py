#!/usr/bin/env python3
"""
make_figures.py — regenerates all five manuscript figures from the results CSVs.

Figures were originally produced inline and are reconstructed here so a reviewer can
regenerate them from the analysis outputs.

  FIG 1  masking mechanism | recovery frontier (with the DETECTION-ONLY zone hatched)
         | fidelity: detection is easy, recovery is hard
  FIG 2  attenuation cascade | power ideal vs actual | which criterion fails
  FIG 3  instrument frontier | budget allocation | variance/bias dissociation
  FIG 4  phase-locked: wrong predictor -> null even when the effect is huge
  FIG 5  estimator validation | corrected SSF | objective vs self-report

Usage:  python make_figures.py --results ./results --out ./figures
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import h4_frontier as H

warnings.filterwarnings("ignore")


def fig1(res, out):
    fr = pd.read_csv(f"{res}/h4_frontier_results.csv")
    fid = pd.read_csv(f"{res}/table04_fidelity.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) the mechanism
    xg = np.linspace(0.15, 0.95, 300)
    ax[0].plot(xg, H.inverted_u(xg), "k-", lw=2.2, zorder=1)
    mu = H.DA_OPT - H.K_GAIN * H._E2_MEAN
    for b, col, lab in ((mu - .16, "#2166ac", "low baseline tone\nE2 up -> perf UP"),
                        (mu + .16, "#b2182b", "high baseline tone\nE2 up -> perf DOWN")):
        lo, hi = b + H.K_GAIN * .13, b + H.K_GAIN * 1.0
        xs = np.linspace(lo, hi, 50)
        ax[0].plot(xs, H.inverted_u(xs), color=col, lw=6, alpha=.9,
                   solid_capstyle="round", zorder=2)
        ax[0].annotate("", xy=(hi, H.inverted_u(hi)), xytext=(lo, H.inverted_u(lo)),
                       arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4,
                                       mutation_scale=20), zorder=3)
        ax[0].text(b, H.inverted_u(b) - .17, lab, color=col, ha="center",
                   fontsize=7.5, weight="bold")
    ax[0].axvline(H.DA_OPT, ls=":", c="gray", lw=1.2)
    ax[0].set_xlabel("dopaminergic tone"); ax[0].set_ylabel("cognitive performance")
    ax[0].set_title("(a) The masking mechanism", fontsize=9.5, weight="bold")
    ax[0].set_ylim(0, 1.18)

    # (b) frontier WITH the detection-only zone hatched (audit correction)
    d = fr[fr.sigma_b == 0.10].pivot(index="obs_per_cycle", columns="reliability",
                                     values="recovery")
    im = ax[1].imshow(d.values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax[1].set_xticks(range(len(d.columns)))
    ax[1].set_xticklabels([f"{c:.2f}" for c in d.columns], fontsize=8)
    ax[1].set_yticks(range(len(d.index))); ax[1].set_yticklabels(d.index, fontsize=8)
    X, Y = np.meshgrid(np.arange(len(d.columns)), np.arange(len(d.index)))
    ax[1].contour(X, Y, d.values, levels=[0.8], colors="white", linewidths=2.5, linestyles="--")
    n_hatch = sum(1 for v in d.index if v * 2 < 14)
    ax[1].add_patch(mpatches.Rectangle((-.5, -.5), len(d.columns), n_hatch,
                                       facecolor="none", edgecolor="red", hatch="///",
                                       lw=2, zorder=5))
    ax[1].text(len(d.columns)/2 - .5, n_hatch/2 - .6,
               "DETECTION-ONLY ZONE\ncriterion (i) saturated by noise\n(100% FP at 4 obs)",
               ha="center", va="center", fontsize=7.5, color="red", weight="bold",
               bbox=dict(boxstyle="round", fc="white", ec="red", alpha=.92))
    ax[1].set_xlabel("outcome reliability"); ax[1].set_ylabel("observations per cycle")
    ax[1].set_title("(b) Recovery frontier (CORRECTED)\nhatched zone is NOT a design recommendation",
                    fontsize=9.5, weight="bold")
    plt.colorbar(im, ax=ax[1], label="recovery rate")

    # (c) fidelity
    cols = {0.05: "#d6604d", 0.10: "#f4a582", 0.15: "#92c5de", 0.20: "#2166ac"}
    for sb in (0.05, 0.10, 0.15, 0.20):
        s = fid[fid.sigma_b == sb].sort_values("obs_per_person")
        ax[2].plot(s.obs_per_person, s.fidelity, "o-", color=cols[sb], lw=1.9, ms=5,
                   label=f"heterogeneity={sb:.2f}")
    ax[2].axhline(.70, ls="--", c="k", lw=1.2)
    ax[2].axhline(.50, ls=":", c="gray", lw=1)
    ax[2].axvspan(2, 10, alpha=.13, color="red")
    ax[2].set_xlabel("total observations per person")
    ax[2].set_ylabel("fidelity  corr(r-hat, r-true)")
    ax[2].set_title("(c) Detecting is easy. Recovering is hard.", fontsize=9.5, weight="bold")
    ax[2].set_ylim(.1, 1.0); ax[2].legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{out}/FIG1_mechanism_frontier.png", dpi=180, bbox_inches="tight")
    plt.close()


def fig5(res, out):
    ssf = pd.read_csv(f"{res}/table07_instrument_ssf.csv")
    obj = pd.read_csv(f"{res}/table13_objective.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # (a) estimator validation (values from ssf_estimators.validate())
    names = ["AR(1)", "ACF-linear", "SPECTRAL\n(adopted)"]
    bias, mx = [.077, .036, .028], [.157, .082, .082]
    x = np.arange(3); w = .36
    ax[0].bar(x - w/2, bias, w, color=["#b2182b", "#f4a582", "#2166ac"],
              edgecolor="k", lw=.6, label="mean |bias|")
    ax[0].bar(x + w/2, mx, w, color=["#b2182b", "#f4a582", "#2166ac"], alpha=.45,
              edgecolor="k", lw=.6, label="max |bias|")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=8.5)
    ax[0].set_ylabel("bias against known truth"); ax[0].legend(fontsize=7.5)
    ax[0].set_title("(a) SSF estimators validated\nagainst known ground truth",
                    fontsize=9.5, weight="bold")

    # (b) SSF: AR(1) vs spectral
    ax[1].bar(np.arange(len(ssf)) - w/2, ssf.ar1, w, color="#bdbdbd",
              edgecolor="k", lw=.6, label="AR(1)  (biased)")
    ax[1].bar(np.arange(len(ssf)) + w/2, ssf.spectral, w, color="#2166ac",
              edgecolor="k", lw=.6, label="SPECTRAL (validated)")
    ax[1].axhline(1.0, ls="--", c="red", lw=1.2)
    ax[1].set_xticks(range(len(ssf)))
    ax[1].set_xticklabels([m[:12] for m in ssf.measure], fontsize=6.5, rotation=40, ha="right")
    ax[1].set_ylabel("smooth signal fraction"); ax[1].legend(fontsize=7.5)
    ax[1].set_title("(b) Instruments are WORSE than reported\nAR(1) returns an impossible value",
                    fontsize=9.5, weight="bold")

    # (c) objective vs self-report
    x = np.arange(len(obj))
    ax[2].bar(x - w/2, obj.SD_ri, w, color=["#b2182b", "#92c5de", "#2166ac"][:len(obj)],
              edgecolor="k", lw=.6, label="observed SD(r_i)")
    ax[2].bar(x + w/2, obj.null_SD, w, color="#e0e0e0", edgecolor="k", lw=.6,
              label="phase-randomised null")
    for i, p in enumerate(obj.p):
        ax[2].text(i, max(obj.SD_ri[i], obj.null_SD[i]) + .006,
                   f"p={p:.3f}" + (" *" if p < .05 else ""), ha="center", fontsize=8,
                   weight="bold" if p < .05 else "normal")
    ax[2].set_xticks(x); ax[2].set_xticklabels(obj.outcome, fontsize=8, rotation=20, ha="right")
    ax[2].set_ylabel("coupling heterogeneity"); ax[2].legend(fontsize=7.5, loc="upper left")
    ax[2].set_title("(c) Change the instrument, signal appears\n(EXPLORATORY: p=.027 fails Bonferroni)",
                    fontsize=9.5, weight="bold")

    plt.tight_layout()
    plt.savefig(f"{out}/FIG5_estimators_objective.png", dpi=180, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--out", default="./figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    made = []
    for name, fn in (("FIG1", fig1), ("FIG5", fig5)):
        try:
            fn(a.results, a.out); made.append(name)
        except FileNotFoundError as e:
            print(f"[skip {name}] missing input: {e.filename}")
    print(f"[done] regenerated: {', '.join(made) or 'nothing'} -> {a.out}/")
    print("FIG2/FIG3/FIG4 depend on mcPHASES-derived CSVs; run mcphases_analyses.py first.")


if __name__ == "__main__":
    main()
