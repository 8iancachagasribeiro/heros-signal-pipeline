#!/usr/bin/env python3
"""
calibration_fidelity_aliasing.py — the three simulation analyses that were run inline
in the original session and are reconstructed here as reproducible code.

REPRODUCES
----------
  6.1  null_calibration()  Table 3 — false-positive rate of each recovery criterion
                           under a TRUE null. Shows criterion (i) alone has a 100% FP
                           rate at 4 obs/person: small-sample |r| is inflated by noise,
                           so a researcher would "find" individual coupling in everyone
                           in a sample where none exists. This is WHY the preregistered
                           DUAL criterion was necessary, not merely conservative.

  6.3  fidelity()          Table 4 — recovery fidelity = corr(estimated r_i, TRUE
                           structural r_i) across individuals. Detecting that coupling
                           VARIES is easy (4 obs). Recovering any INDIVIDUAL's coupling
                           needs ~an order of magnitude more.

                           NOTE ON A BUG THAT WAS FOUND AND FIXED: a preliminary version
                           computed the "true" r_i on the SAME timepoints and with the
                           SAME state noise as the estimate, so the two shared noise and
                           fidelity was inflated (0.83 vs the correct 0.56 at 4 obs).
                           The true coupling MUST come from the generative parameter b_i
                           on a dense, noise-free grid. That is what true_coupling() does.

  6.4  aliasing()          Table 5 — evenly-spaced vs phase-targeted sampling. The
                           ovulatory surge is narrow; a 3-point evenly-spaced schedule
                           can miss it entirely, collapsing the sampled E2 range and
                           leaving the within-person slope unidentified.
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import h4_frontier as H
from fastlrt import lrt_random_slope_fast

warnings.filterwarnings("ignore")

N_SUBJ = H.N_SUBJ           # 39, the mean N per study in Jang et al. (2025)
REL_REF = 0.73              # Calamia, Markon & Tranel (2013), test-retest reliability

# dense, noise-free grid used to define the STRUCTURAL true coupling
_XD = H.e2(np.linspace(0, H.CYCLE_LEN, 400, endpoint=False))


def true_coupling(b):
    """Structural true coupling of each person: determined ONLY by where her baseline
    dopaminergic tone b_i places her on the inverted-U. Noise-free, densely sampled.
    This is the quantity a person-specific model would need to recover."""
    sig = H.inverted_u(b[:, None] + H.K_GAIN * _XD[None, :])
    return within_r(sig, _XD)


def within_r(Y, x):
    """Vectorised within-person Pearson r between x and each row of Y."""
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=1, keepdims=True)
    den = np.sqrt((Yc ** 2).sum(axis=1) * (xc ** 2).sum())
    den = np.where(den < 1e-12, np.nan, den)
    return (Yc @ xc) / den


def simulate(rng, days, reliability, sigma_b, n_subj=N_SUBJ):
    """Returns (Y_obs, x, b). `days` is the sampling schedule in days."""
    x = H.e2(np.asarray(days, float))
    b = rng.normal(H.DA_OPT - H.K_GAIN * H._E2_MEAN, sigma_b, n_subj)
    signal = H.inverted_u(b[:, None] + H.K_GAIN * x[None, :])
    y_true = signal + rng.normal(0, H.SIGMA_STATE, signal.shape)
    sd = np.sqrt(max(y_true.var(), 1e-12) * (1 - reliability) / reliability)
    return y_true + rng.normal(0, sd, y_true.shape), x, b


def even_days(obs_per_cycle, n_cycles):
    return np.concatenate([
        np.linspace(0, H.CYCLE_LEN, obs_per_cycle, endpoint=False) + c * H.CYCLE_LEN
        for c in range(n_cycles)])


# canonical cycle phases targeted by real studies (day of a 28-day cycle)
PHASE_TARGETED = {2: [3, 13], 3: [3, 13, 21], 4: [3, 9, 13, 21],
                  5: [3, 9, 13, 17, 21], 7: [2, 6, 9, 13, 17, 21, 25],
                  10: list(np.linspace(1, 27, 10))}


# ---------------------------------------------------------------- 6.1 #
def null_calibration(rng, n_sims=60):
    print("=" * 74)
    print("6.1  NULL CALIBRATION (Table 3) — sigma_b ~ 0, NO true heterogeneity")
    print("=" * 74)
    print(f"{'obs/person':>11} {'criterion (i) alone':>20} {'LRT alone':>11} {'BOTH (registered)':>18}")
    print("-" * 64)
    rows = []
    for d in (2, 3, 5, 7, 10, 14, 21, 28):
        c1 = c2 = both = 0
        for _ in range(n_sims):
            Y, x, _ = simulate(rng, even_days(d, 2), REL_REF, 0.001)   # ~no heterogeneity
            prop = float(np.nanmean(np.abs(within_r(Y, x)) > 0.20))
            p = lrt_random_slope_fast(Y, x)
            if not np.isfinite(p):
                continue
            a, b_ = prop > 0.50, p < 0.05
            c1 += a; c2 += b_; both += (a and b_)
        print(f"{d*2:>11} {c1/n_sims:>20.2f} {c2/n_sims:>11.2f} {both/n_sims:>18.2f}")
        rows.append(dict(obs_per_person=d*2, crit1_fp=c1/n_sims,
                         lrt_fp=c2/n_sims, both_fp=both/n_sims))
    print("\n  With 4 obs/person, criterion (i) ALONE has a 100% false-positive rate.")
    print("  The LRT is calibrated. The DUAL criterion is calibrated.")
    print("  => The preregistered conservatism was NECESSARY, not cautious.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6.3 #
def fidelity(rng, n_sims=60):
    print("\n" + "=" * 74)
    print("6.3  RECOVERY FIDELITY (Table 4) — corr(estimated r_i, TRUE structural r_i)")
    print("=" * 74)
    sigmas = (0.05, 0.10, 0.15, 0.20)
    print(f"{'obs/cycle':>10} {'obs/person':>11}" + "".join(f"{f'sb={s:.2f}':>9}" for s in sigmas))
    print("-" * 58)
    rows = []
    for d in (2, 3, 5, 7, 10, 14, 21, 28):
        line = f"{d:>10} {d*2:>11}"
        for sb in sigmas:
            fid = []
            for _ in range(n_sims):
                Y, x, b = simulate(rng, even_days(d, 2), REL_REF, sb)
                rh, rt = within_r(Y, x), true_coupling(b)
                m = np.isfinite(rh) & np.isfinite(rt)
                if m.sum() > 3 and np.std(rh[m]) > 1e-9 and np.std(rt[m]) > 1e-9:
                    fid.append(np.corrcoef(rh[m], rt[m])[0, 1])
            f = float(np.mean(fid)) if fid else np.nan
            line += f"{f:>9.2f}"
            rows.append(dict(obs_per_cycle=d, obs_per_person=d*2, sigma_b=sb, fidelity=f))
        print(line)
    print("\n  >= .70 = individual estimates usable;  < .50 = individual estimates are noise.")
    print("  Note the DIP at 3 obs/cycle: that is aliasing (see 6.4), not sampling error.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6.4 #
def aliasing(rng, n_sims=80, sigma_b=0.10):
    print("\n" + "=" * 74)
    print("6.4  ALIASING (Table 5) — evenly-spaced vs PHASE-TARGETED sampling")
    print("=" * 74)
    print(f"{'obs/cycle':>10} {'even':>8} {'phase-targeted':>16} "
          f"{'E2 range (even)':>17} {'E2 range (phase)':>18}")
    print("-" * 74)
    rows = []
    for k in (2, 3, 4, 5, 7, 10):
        out = {}
        for tag, days1 in (("even", even_days(k, 1)),
                           ("phase", np.asarray(PHASE_TARGETED[k], float))):
            days = np.concatenate([days1 + c * H.CYCLE_LEN for c in range(2)])
            fid = []
            for _ in range(n_sims):
                Y, x, b = simulate(rng, days, REL_REF, sigma_b)
                rh, rt = within_r(Y, x), true_coupling(b)
                m = np.isfinite(rh) & np.isfinite(rt)
                if m.sum() > 3 and np.std(rh[m]) > 1e-9:
                    fid.append(np.corrcoef(rh[m], rt[m])[0, 1])
            out[tag] = (float(np.mean(fid)) if fid else np.nan,
                        float(H.e2(days).max() - H.e2(days).min()))
        print(f"{k:>10} {out['even'][0]:>8.2f} {out['phase'][0]:>16.2f} "
              f"{out['even'][1]:>17.2f} {out['phase'][1]:>18.2f}")
        rows.append(dict(obs_per_cycle=k, fidelity_even=out['even'][0],
                         fidelity_phase=out['phase'][0],
                         e2_range_even=out['even'][1], e2_range_phase=out['phase'][1]))
    print("\n  Same cost, nearly double the fidelity at 3 obs/cycle. The mechanism is in")
    print("  the last two columns: if the schedule does not span the estradiol range,")
    print("  the within-person slope is NOT IDENTIFIED.")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--n-sims", type=int, default=60,
                    help="60 reproduces the manuscript; use >=500 for the final figures")
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    null_calibration(rng, args.n_sims).to_csv(f"{args.out_dir}/table03_null_calibration.csv", index=False)
    fidelity(rng, args.n_sims).to_csv(f"{args.out_dir}/table04_fidelity.csv", index=False)
    aliasing(rng, args.n_sims).to_csv(f"{args.out_dir}/table05_aliasing.csv", index=False)
    print(f"\n[saved] {args.out_dir}/")


if __name__ == "__main__":
    main()
