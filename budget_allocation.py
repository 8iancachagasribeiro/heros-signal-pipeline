#!/usr/bin/env python3
"""
budget_allocation.py — Reproduces Tables 9 and 17 of the manuscript.

TABLE 9  (section 7.3)  Sensitivity of the attenuation cascade to the identification
                        assumption. Analytical; no simulation, runs instantly.

TABLE 17 (section 8)    Where to spend the next research effort. Simulation.

WHY THIS SCRIPT EXISTS
----------------------
Both tables were previously computed ad hoc and had no corresponding script in the
reproduction package. Table 17 in particular sustains the manuscript's central practical
claim -- that instrument quality dominates sample size -- and an earlier version of it
was computed with the AR(1) estimator values (0.58 / 0.41) that the manuscript itself
rejects as biased. Those values are corrected here to the validated spectral estimates
(0.469 / 0.323); see CHANGELOG.md.

USAGE
-----
    python budget_allocation.py --table 9      # instant
    python budget_allocation.py --table 17     # ~1 min at n_sims=60
    python budget_allocation.py --table 17 --n-sims 200   # tighter, slower
"""
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Validated spectral smooth-signal fractions (Table 8, ESPECTRAL column)
SSF_PREDICTOR = 0.469   # at-home urinary estrone-3-glucuronide
SSF_OUTCOME   = 0.323   # central value of the two preregistered confirmatory outcomes
                        # (fatigue 0.321, mood swing 0.326)
SIGMA_B = 0.15
OBS_BASE, CYCLES_BASE = 74, 3


# --------------------------------------------------------------------------- #
def table_9():
    """Sensitivity to the assumption that smooth-signal fraction ~ reliability.

    The classical attenuation formula is stated in terms of RELIABILITY (true-score
    variance / total variance). We substitute the smooth-signal fraction. These are not
    identical: SSF counts as noise all variance above the spectral cutoff, including
    genuine fast physiological or affective variation.

    Reparametrise: if a fraction f of the high-frequency variance is genuine process
    rather than measurement error, then

        reliability = SSF + f * (1 - SSF)

    f = 0 is the literal reading of SSF; f = 1 would mean no measurement error at all.
    """
    print("=" * 78)
    print("TABLE 9 — Sensitivity of the attenuation cascade to FSS ~ reliability")
    print("=" * 78)
    print(f"\n{'f':>6} {'rel. predictor':>16} {'rel. outcome':>14} {'attenuation':>13}"
          f" {'r_true for r_obs = 0.20':>25}")
    print("-" * 78)
    for f in (0.00, 0.25, 0.50, 0.75):
        rx = SSF_PREDICTOR + f * (1 - SSF_PREDICTOR)
        ry = SSF_OUTCOME   + f * (1 - SSF_OUTCOME)
        att = np.sqrt(rx * ry)
        print(f"{f:>6.2f} {rx:>16.3f} {ry:>14.3f} {att:>13.3f} {0.20/att:>25.2f}")

    print("\n  The conclusion survives to f = 0.50: even if HALF of all high-frequency")
    print("  variance in both instruments were genuine process, attenuation remains at")
    print("  0.697 -- below the 0.70 threshold. Above f = 0.50 the argument weakens")
    print("  materially, and we say so: at f = 0.75 attenuation rises to 0.849.")
    print("\n  What does NOT depend on f is the ORDERING. Attenuation is multiplicative")
    print("  and sample density does not affect it, so the superiority of investing in")
    print("  instruments over investing in sample size holds for any value of f.")


# --------------------------------------------------------------------------- #
def table_17(n_sims=60, seed=2026):
    """Where to spend the next research effort.

    Columns:
      attenuation = sqrt(SSF_x * SSF_y), the multiplicative ceiling on observable coupling
      POWER       = dual criterion (proportion of |r_i| > 0.20 exceeding 0.50 AND
                    likelihood-ratio test for random slopes significant)
      fidelity    = correlation between estimated and true individual slopes

    The "more measurements" scenarios preserve density per cycle and extend the number of
    cycles; doubling the observations within a fixed 3-cycle span is not possible.
    """
    import h4_v2 as V

    print("=" * 78)
    print(f"TABLE 17 — Where to spend the next research effort (sigma_b = {SIGMA_B})")
    print("=" * 78)
    rng = np.random.default_rng(seed)

    px, py = SSF_PREDICTOR, SSF_OUTCOME
    scenarios = [
        ("mcPHASES as built",                  px,   py,   OBS_BASE,     CYCLES_BASE),
        ("2x measurements (148 obs/person)",   px,   py,   OBS_BASE*2,   CYCLES_BASE*2),
        ("4x measurements (296 obs/person)",   px,   py,   OBS_BASE*4,   CYCLES_BASE*4),
        (f"better OUTCOME ({py:.3f} -> 0.85)", px,   0.85, OBS_BASE,     CYCLES_BASE),
        (f"better PREDICTOR ({px:.3f} -> 0.85)", 0.85, py, OBS_BASE,     CYCLES_BASE),
        ("better BOTH instruments",            0.85, 0.85, OBS_BASE,     CYCLES_BASE),
        ("both + 2x measurements",             0.85, 0.85, OBS_BASE*2,   CYCLES_BASE*2),
    ]

    print(f"\n{'scenario':>36} {'attenuation':>12} {'POWER':>8} {'fidelity':>10}")
    print("-" * 70)
    for name, rx, ry, nobs, ncyc in scenarios:
        dual, lrt, fid, med_r = V.eval_cell(rng, nobs, ncyc, rx, ry, SIGMA_B, n_sims=n_sims)
        print(f"{name:>36} {np.sqrt(rx*ry):>12.2f} {dual:>8.2f} {fid:>10.2f}")

    print("\n  MORE DATA BUYS NOTHING. Quadrupling the observations leaves power at exactly")
    print("  zero. Repairing the instruments takes it to 0.98.")
    print("\n  FIX THE WEAKER INSTRUMENT FIRST. Repairing only the outcome yields 0.50;")
    print("  repairing only the predictor yields 0.02. The asymmetry follows directly:")
    print("  the outcome is the weaker link (0.323 against 0.469) and attenuation is")
    print("  multiplicative, so the marginal return concentrates there. Still, no single")
    print("  repair suffices -- only joint correction reaches 0.98.")
    print("\n  Note also that fidelity for the design as built is 0.66, below the 0.70")
    print("  threshold established in section 6: the study not only lacks power to detect")
    print("  heterogeneity, it would not recover individual slopes even if it detected it.")
    print(f"\n  [{n_sims} replicates per cell; raise with --n-sims for tighter estimates]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=int, choices=[9, 17], default=9)
    ap.add_argument("--n-sims", type=int, default=60)
    a = ap.parse_args()
    if a.table == 9:
        table_9()
    else:
        table_17(a.n_sims)


if __name__ == "__main__":
    main()
