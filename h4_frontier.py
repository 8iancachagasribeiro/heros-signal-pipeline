#!/usr/bin/env python3
"""
H4 - Design frontier for recovering within-person hormone-cognition coupling.

PREREGISTERED GENERATIVE MODEL (H1):
    E2(t)        : normalized estradiol trajectory across the cycle
    DA_i(t)      = b_i + K * E2(t)        with b_i ~ N(mu_b, sigma_b)
                   (b_i = baseline dopaminergic tone; COMT proxy)
    performance  = invertedU(DA_i(t))     (Gaussian, peak at DA_opt)
    observed     = performance + measurement error (variance set by reliability r)

MASKING MECHANISM (why the group null arises from real individual coupling):
    b_i < DA_opt -> rising E2 moves DA toward the optimum -> POSITIVE within-person slope
    b_i > DA_opt -> rising E2 moves DA past the optimum   -> NEGATIVE within-person slope
    When b_i is centered on DA_opt, positive and negative slopes cancel in the group mean.
    The group estimate approaches zero while every individual carries substantial coupling.
    -> This is HG: the group estimate is governed by the COMPOSITIONAL BALANCE of the
       sample, not by the magnitude of individual coupling.

H2 (falsifiability boundary): masking requires directional BALANCE. Shifting the b_i
    distribution off DA_opt (balance_offset != 0) makes the group signal LEAK.

H4 (this script): recovery frontier over
    sampling density x measurement reliability x coupling heterogeneity.
    Recovery = BOTH preregistered criteria met:
      (i)  proportion of individuals with |r_i| > 0.20 exceeds 50%
      (ii) LRT for between-person variance in slope reaches p < .05
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ----------------------------- fixed model constants ----------------------------- #
CYCLE_LEN = 28.0
DA_OPT = 0.50          # optimum of the inverted-U
U_WIDTH = 0.35         # width of the inverted-U
K_GAIN = 0.15          # E2 -> DA gain: MODEST excursion, person stays on one limb
SIGMA_STATE = 0.085    # intrinsic within-person state noise, calibrated so individual coupling is MODERATE (|g| ~ .3-.5, per H1)
N_SUBJ = 39            # mean N per study (Jang et al. 2025 study characteristics)


# ----------------------------- generative components ----------------------------- #
_grid = np.arange(0.0, CYCLE_LEN, 0.25)


def _e2_raw(d):
    """Two-peak estradiol: ovulatory surge (~d13) + luteal secondary rise (~d21)."""
    ov = np.exp(-((d - 13.0) ** 2) / (2 * 2.0 ** 2))
    lu = 0.55 * np.exp(-((d - 21.0) ** 2) / (2 * 3.5 ** 2))
    return 0.15 + ov + lu


_E2_MAX = _e2_raw(_grid).max()
_E2_MEAN = (_e2_raw(_grid) / _E2_MAX).mean()


def e2(days):
    """Normalized estradiol in ~[0.1, 1.0]."""
    return _e2_raw(np.asarray(days, dtype=float) % CYCLE_LEN) / _E2_MAX


def inverted_u(da):
    """Inverted-U performance as a function of dopaminergic tone."""
    return np.exp(-((da - DA_OPT) ** 2) / (2 * U_WIDTH ** 2))


def simulate_study(rng, n_subj, obs_per_cycle, n_cycles, reliability,
                   sigma_b, balance_offset=0.0):
    """Simulate one study. Returns tidy DataFrame [subj, e2, y_true, y_obs]."""
    # sampling schedule: obs_per_cycle points, evenly spaced, across n_cycles
    days = np.concatenate([
        np.linspace(0, CYCLE_LEN, obs_per_cycle, endpoint=False) + c * CYCLE_LEN
        for c in range(n_cycles)
    ])
    x = e2(days)                                   # same schedule for all subjects
    n_obs = len(days)

    # b_i centered so that MEAN DA sits on the optimum when balance_offset = 0
    mu_b = DA_OPT - K_GAIN * _E2_MEAN + balance_offset
    b = rng.normal(mu_b, sigma_b, size=n_subj)     # baseline dopaminergic tone

    da = b[:, None] + K_GAIN * x[None, :]          # (n_subj, n_obs)
    signal = inverted_u(da)

    # intrinsic within-person state variability (sleep, stress, practice, ...)
    # without this the true within-person coupling would be ~1.0, which is unreal;
    # calibrated so median true |r_i| is MODERATE (|g| ~ .3-.5, per H1)
    y_true = signal + rng.normal(0.0, SIGMA_STATE, size=signal.shape)

    # measurement error implied by test-retest reliability on TOTAL score variance
    var_true = y_true.var()
    sd_err = np.sqrt(max(var_true, 1e-12) * (1.0 - reliability) / reliability)
    y_obs = y_true + rng.normal(0.0, sd_err, size=y_true.shape)

    return pd.DataFrame({
        "subj": np.repeat(np.arange(n_subj), n_obs),
        "e2": np.tile(x, n_subj),
        "y_true": y_true.ravel(),
        "y_obs": y_obs.ravel(),
    })


# ----------------------------- estimation / criteria ----------------------------- #
def within_person_r(df, col="y_obs"):
    """Within-person Pearson r between E2 and outcome, per subject."""
    out = []
    for _, g in df.groupby("subj", sort=False):
        if g["e2"].std() < 1e-9 or g[col].std() < 1e-9:
            out.append(0.0)
        else:
            out.append(np.corrcoef(g["e2"], g[col])[0, 1])
    return np.asarray(out)


def group_effect(df, col="y_obs"):
    """Pooled (group-level) E2->outcome association, ignoring person."""
    if df["e2"].std() < 1e-9 or df[col].std() < 1e-9:
        return 0.0
    return float(np.corrcoef(df["e2"], df[col])[0, 1])


def lrt_random_slope(df, col="y_obs"):
    """
    LRT for between-person variance in the E2 slope.
    H0: random intercept only.  H1: random intercept + random slope.
    Boundary problem -> chi-bar-square: 0.5*chi2(1) + 0.5*chi2(2).
    Returns p-value (np.nan on convergence failure).
    """
    import statsmodels.formula.api as smf
    try:
        m0 = smf.mixedlm(f"{col} ~ e2", df, groups=df["subj"]).fit(reml=False)
        m1 = smf.mixedlm(f"{col} ~ e2", df, groups=df["subj"],
                         re_formula="~e2").fit(reml=False)
        stat = 2.0 * (m1.llf - m0.llf)
        if not np.isfinite(stat) or stat < 0:
            return np.nan
        p = 0.5 * stats.chi2.sf(stat, 1) + 0.5 * stats.chi2.sf(stat, 2)
        return float(p)
    except Exception:
        return np.nan


def evaluate_cell(rng, n_sims, obs_per_cycle, n_cycles, reliability, sigma_b,
                  balance_offset=0.0, n_subj=N_SUBJ):
    """Run n_sims studies for one design cell; return recovery rate + diagnostics."""
    c1 = c2 = both = 0
    ok = 0
    grp, propr = [], []
    for _ in range(n_sims):
        df = simulate_study(rng, n_subj, obs_per_cycle, n_cycles,
                            reliability, sigma_b, balance_offset)
        r_i = within_person_r(df)
        prop = float(np.mean(np.abs(r_i) > 0.20))          # criterion (i)
        p_lrt = lrt_random_slope(df)                        # criterion (ii)
        if np.isnan(p_lrt):
            continue
        ok += 1
        a = prop > 0.50
        b_ = p_lrt < 0.05
        c1 += a
        c2 += b_
        both += (a and b_)
        grp.append(group_effect(df))
        propr.append(prop)
    if ok == 0:
        return dict(recovery=np.nan, crit1=np.nan, crit2=np.nan,
                    mean_group_r=np.nan, mean_prop=np.nan, n_ok=0)
    return dict(recovery=both / ok, crit1=c1 / ok, crit2=c2 / ok,
                mean_group_r=float(np.mean(grp)), mean_prop=float(np.mean(propr)),
                n_ok=ok)


# ----------------------------- validation of the mechanism ----------------------------- #
def validate_masking(seed=0):
    """Sanity check: does the calibrated model reproduce a group NULL with real individual coupling?"""
    rng = np.random.default_rng(seed)
    print("=" * 74)
    print("STEP A - does the generative model actually MASK? (HG / H1 sanity check)")
    print("=" * 74)
    print(f"{'sigma_b':>8} {'group r':>9} {'|group g|':>10} {'med |r_i|':>10} "
          f"{'% |r_i|>.2':>11} {'% pos slope':>12}")
    for sigma_b in [0.05, 0.10, 0.15, 0.20]:
        gr, mr, pr, ps = [], [], [], []
        for _ in range(200):
            df = simulate_study(rng, N_SUBJ, 28, 2, 0.73, sigma_b)  # dense, reliable
            r_i = within_person_r(df, "y_true")                     # true coupling
            gr.append(group_effect(df, "y_true"))
            mr.append(np.median(np.abs(r_i)))
            pr.append(np.mean(np.abs(r_i) > 0.20))
            ps.append(np.mean(r_i > 0))
        g_r = float(np.mean(gr))
        # Hedges' g equivalent of a correlation: g = 2r/sqrt(1-r^2)
        g_eff = abs(2 * g_r / np.sqrt(max(1 - g_r ** 2, 1e-9)))
        print(f"{sigma_b:>8.2f} {g_r:>9.3f} {g_eff:>10.3f} {np.mean(mr):>10.3f} "
              f"{100*np.mean(pr):>10.1f}% {100*np.mean(ps):>11.1f}%")
    print("\n  Expected: group r ~ 0 (|g| < .10) while median |r_i| is substantial")
    print("  and positive/negative slopes are ~50/50 (bidirectional cancellation).")

    print("\n" + "=" * 74)
    print("STEP B - H2 boundary: does directional IMBALANCE make the group signal LEAK?")
    print("=" * 74)
    print(f"{'balance':>9} {'group r':>9} {'|group g|':>10} {'% pos slope':>12}  verdict")
    for off in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]:
        gr, ps = [], []
        for _ in range(200):
            df = simulate_study(rng, N_SUBJ, 28, 2, 0.73, 0.10, balance_offset=off)
            r_i = within_person_r(df, "y_true")
            gr.append(group_effect(df, "y_true"))
            ps.append(np.mean(r_i > 0))
        g_r = float(np.mean(gr))
        g_eff = abs(2 * g_r / np.sqrt(max(1 - g_r ** 2, 1e-9)))
        verdict = "MASKED (null)" if g_eff < 0.10 else "LEAKS (detectable)"
        print(f"{off:>9.2f} {g_r:>9.3f} {g_eff:>10.3f} {100*np.mean(ps):>11.1f}%  {verdict}")
    print("\n  H2 predicts: masking holds only under directional balance;")
    print("  asymmetry leaks a detectable group signal. The boundary is where |g| crosses .10.")


# ----------------------------- H4 grid ----------------------------- #
def run_frontier(seed, n_sims, densities, reliabilities, sigmas, n_cycles):
    rng = np.random.default_rng(seed)
    rows = []
    total = len(sigmas) * len(densities) * len(reliabilities)
    i = 0
    for sb in sigmas:
        for d in densities:
            for r in reliabilities:
                i += 1
                res = evaluate_cell(rng, n_sims, d, n_cycles, r, sb)
                rows.append(dict(sigma_b=sb, obs_per_cycle=d, n_cycles=n_cycles,
                                 total_obs=d * n_cycles, reliability=r, **res))
                print(f"  [{i:>3}/{total}] sigma_b={sb:.2f} obs/cycle={d:>2} "
                      f"rel={r:.2f} -> recovery={res['recovery']:.2f} "
                      f"(crit1={res['crit1']:.2f} crit2={res['crit2']:.2f})", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-cycles", type=int, default=2)
    ap.add_argument("--out", default="/mnt/user-data/outputs/h4_frontier_results.csv")
    args = ap.parse_args()

    validate_masking(args.seed)
    if args.validate_only:
        return

    if args.quick:
        densities = [2, 5, 14]
        reliabilities = [0.55, 0.73, 0.85]
        sigmas = [0.10]
        n_sims = 20
    else:
        densities = [2, 3, 5, 7, 10, 14, 21, 28]
        reliabilities = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        sigmas = [0.05, 0.10, 0.15, 0.20]
        n_sims = args.n_sims

    print("\n" + "=" * 74)
    print(f"STEP C - H4 DESIGN FRONTIER  (n_sims={n_sims}, n_cycles={args.n_cycles}, "
          f"N={N_SUBJ}/study)")
    print("=" * 74)
    df = run_frontier(args.seed, n_sims, densities, reliabilities, sigmas, args.n_cycles)
    df.to_csv(args.out, index=False)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
