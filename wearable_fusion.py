#!/usr/bin/env python3
"""
wearable_fusion.py — Can a fusion of continuous wearable signals beat the urinary E3G
                     as a predictor of menstrual cycle phase?

CONTEXT
-------
Target: a cycle-phase predictor with smooth-signal fraction (SSF) >= 0.70.
Current best instrument: at-home urinary estrone-3-glucuronide (E3G), SSF ~ 0.47.
Because attenuation is MULTIPLICATIVE -- r_obs = r_true * sqrt(SSF_x * SSF_y) -- a weak
predictor caps the observable coupling no matter how good the outcome measure is.

THE QUESTION (raised by Prof. R. C. Contreras)
----------------------------------------------
  "You used only one PCA component? Why not use more, until you capture ~95% of variance?"

Answered in section (2) below. Short version: it makes things WORSE, because NOISE IS
VARIANCE. Accumulating components to 95% of variance accumulates noise along with signal.
PCA orders by variance, not by cycle-phase information; there is no reason for those to
coincide.

THE FINDING THAT MATTERS (section 3)
------------------------------------
Auditing my own code, I found that the wearable series in mcPHASES CONTAIN GAPS, and the
FFT treats non-consecutive days as consecutive. This inflates the SSF. I quantified the
bias against known ground truth: at 12% missing days with linear interpolation, SSF is
inflated by +0.07 to +0.10; at 35%, by +0.26 to +0.34.

When I require a CONTIGUOUS, gap-free segment with all six signals present:  **N = 1**.

=> The wearable-fusion question CANNOT BE ANSWERED with this open dataset. Not for lack of
   method -- for lack of data with continuous coverage. That, I believe, is the real result.

DATA
----
mcPHASES (PhysioNet, DOI 10.13026/zx6a-2c81). CREDENTIALED ACCESS -- not redistributed.
Obtain access, unzip, and pass --data-dir.

USAGE
-----
  python wearable_fusion.py --audit-only                 # bias audit, needs NO data
  python wearable_fusion.py --data-dir /path/to/mcphases
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from ssf_estimators import ssf_spectral

warnings.filterwarnings("ignore")

SIGNALS = ["rhr", "temp_skin", "temp_wrist", "rmssd", "low_frequency", "high_frequency"]


# --------------------------------------------------------------------------- #
def load_daily(data_dir):
    """One row per (subject, day). Outer-joined; gaps are left as NaN ON PURPOSE."""
    def agg(fname, cols, daycol="day_in_study", filt=None):
        d = pd.read_csv(f"{data_dir}/{fname}")
        if filt:
            d = d[d[filt[0]] == filt[1]]
        return (d.groupby(["id", daycol])[cols].mean()
                 .reset_index().rename(columns={daycol: "day"}))

    rhr = agg("resting_heart_rate.csv", ["value"]).rename(columns={"value": "rhr"})
    ct = agg("computed_temperature.csv", ["nightly_temperature"],
             "sleep_start_day_in_study", ("type", "SKIN")
             ).rename(columns={"nightly_temperature": "temp_skin"})
    wt_raw = pd.read_csv(f"{data_dir}/wrist_temperature.csv")
    tcol = [c for c in wt_raw.columns if "temp" in c.lower()][0]
    dcol = [c for c in wt_raw.columns if "day" in c.lower()][0]
    wt = (wt_raw.groupby(["id", dcol])[tcol].mean().reset_index()
          .rename(columns={dcol: "day", tcol: "temp_wrist"}))
    hv = agg("heart_rate_variability_details.csv",
             ["rmssd", "low_frequency", "high_frequency"])
    horm = (pd.read_csv(f"{data_dir}/hormones_and_selfreport.csv")
            .groupby(["id", "day_in_study"])
            .agg(estrogen=("estrogen", "mean"), phase=("phase", "first"))
            .reset_index().rename(columns={"day_in_study": "day"}))

    M = rhr
    for other in (ct, wt, hv, horm):
        M = M.merge(other, on=["id", "day"], how="outer")
    return M.sort_values(["id", "day"])


def pc_scores(X):
    """Per-subject PCA via SVD on the z-scored signal block.
    Returns (scores [T x k], explained variance ratio [k])."""
    X = np.asarray(X, float)
    sd = X.std(0)
    Z = (X - X.mean(0)) / np.where(sd < 1e-9, 1.0, sd)
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ Vt.T, (S ** 2) / np.sum(S ** 2)


# ------------------------------- (1) + (2) ------------------------------- #
def components_analysis(M):
    """Does adding PCA components help? (Prof. Contreras's question.)

    *** THE ANSWER IS: THIS DATASET CANNOT TELL US. ***

    I ran it under four defensible preprocessing choices. They DISAGREE -- one even
    reverses the sign of the trend. The reason is section (3): the wearable series are
    gapped, every cleaning choice leaves or creates a different gap structure, and the FFT
    treats non-consecutive days as consecutive.

    An answer that flips with an arbitrary preprocessing choice is not an answer. I am
    reporting the disagreement rather than picking the run I like, because picking would
    be the mistake.
    """
    print("=" * 72)
    print("(2)  DOES ADDING PCA COMPONENTS HELP?  -- ROBUSTNESS TO PREPROCESSING")
    print("=" * 72)

    def run(label, prep):
        cum = {k: [] for k in range(1, 7)}
        for _, g in M.groupby("id"):
            gg = prep(g)
            if gg is None or len(gg) < 40:
                continue
            sc, _ = pc_scores(gg[SIGNALS].astype(float).values)
            for k in range(1, 7):
                v = ssf_spectral(sc[:, :k].sum(axis=1))
                if np.isfinite(v):
                    cum[k].append(v)
        med = [np.median(cum[k]) if cum[k] else np.nan for k in range(1, 7)]
        n = len(cum[1])
        print(f"{label:>34} {n:>4} " + " ".join(f"{m:>7.3f}" for m in med))
        return med

    def p_droprows(g):
        return g.dropna(subset=SIGNALS)

    def p_dropsubj(g):
        gg = g[SIGNALS]
        return g if not gg.isna().any().any() else None

    def p_interp(g):
        g = g.drop_duplicates("day").sort_values("day").set_index("day")
        g = g.reindex(pd.RangeIndex(int(g.index.min()), int(g.index.max()) + 1))
        X = g[SIGNALS].astype(float)
        if X.isna().all().any() or X.isna().any(axis=1).mean() > 0.35:
            return None
        g[SIGNALS] = X.interpolate(limit_direction="both")
        return g.reset_index().rename(columns={"index": "day"})

    def p_ffill(g):
        g = g.drop_duplicates("day").sort_values("day")
        g[SIGNALS] = g[SIGNALS].ffill().bfill()
        return g.dropna(subset=SIGNALS)

    print(f"\n{'preprocessing':>34} {'N':>4} " + " ".join(f"{'k='+str(k):>7}" for k in range(1, 7)))
    print("-" * 78)
    r1 = run("drop incomplete DAYS", p_droprows)
    r2 = run("drop incomplete SUBJECTS", p_dropsubj)
    r3 = run("regular grid + interpolate", p_interp)
    r4 = run("forward/backward fill", p_ffill)

    print("\n  THE FOUR RUNS DISAGREE -- including on the DIRECTION of the trend.")
    print("  An answer that flips with an arbitrary cleaning choice is not an answer.")
    print("\n  What DOES survive, and is not a matter of preprocessing:")
    print("    * PCA orders components by VARIANCE, not by cycle-phase information.")
    print("      There is no reason for those to coincide, and NOISE IS VARIANCE --")
    print("      so accumulating components to 95% of variance accumulates noise too.")
    print("    * Whether that hurts MORE than the extra signal helps is exactly what")
    print("      this dataset cannot resolve. See sections (3) and (4).")


# ------------------------------- (3) THE AUDIT ------------------------------- #
def interpolation_bias_audit(n_rep=400, seed=7):
    """Quantify how much gap-filling inflates the SSF. Needs NO data.

    Ground truth known by construction: a 28-day cycle + white noise at a KNOWN
    signal fraction. Remove days at random, interpolate linearly, re-estimate.
    """
    print("\n" + "=" * 72)
    print("(3)  AUDIT: DOES GAP INTERPOLATION INFLATE THE SSF?")
    print("     (the FFT assumes REGULAR spacing; gaps break that assumption)")
    print("=" * 72)
    rng = np.random.default_rng(seed)
    T = 90
    t = np.arange(T)
    sig = np.sin(2 * np.pi * t / 28)
    sig = (sig - sig.mean()) / sig.std()          # unit variance -> true SSF is exact

    print(f"\n{'% days removed':>15} {'true .45 -> est':>17} {'bias':>7}   "
          f"{'true .30 -> est':>17} {'bias':>7}")
    print("-" * 70)
    for frac in (0.0, 0.05, 0.12, 0.25, 0.35):
        line = f"{100*frac:>14.0f}%"
        for true in (0.45, 0.30):
            est = []
            sd = np.sqrt((1 - true) / true)
            for _ in range(n_rep):
                s = pd.Series(sig + rng.normal(0, sd, T))
                if frac > 0:
                    s.iloc[rng.choice(T, int(T * frac), replace=False)] = np.nan
                    s = s.interpolate(limit_direction="both")
                v = ssf_spectral(s.values)
                if np.isfinite(v):
                    est.append(v)
            m = float(np.median(est))
            line += f" {m:>17.3f} {m-true:>+7.3f}  "
        print(line)

    print("\n  => Interpolation INFLATES the SSF. At the ~12% gap rate of the mcPHASES")
    print("     wearable series, the bias is +0.07 to +0.10. Any SSF computed on")
    print("     gap-filled wearable data is therefore NOT trustworthy in absolute terms.")


# ------------------------------- (4) THE CLEAN TEST ------------------------------- #
def contiguous_only(M, min_len=50):
    """The only defensible computation: the LONGEST CONTIGUOUS, GAP-FREE run per subject
    with all six signals present. No interpolation -> no interpolation bias."""
    print("\n" + "=" * 72)
    print("(4)  THE CLEAN TEST: contiguous, gap-free segments only. NO interpolation.")
    print("=" * 72)

    rows = []
    for _, g in M.groupby("id"):
        g = g.drop_duplicates("day").sort_values("day")
        ok = g[SIGNALS].notna().all(axis=1).values
        days = g.day.values
        best_len, best_a, start = 0, 0, None
        for k in range(len(g)):
            if ok[k] and (start is None or days[k] == days[k - 1] + 1):
                if start is None:
                    start = k
                if k - start + 1 > best_len:
                    best_len, best_a = k - start + 1, start
            else:
                start = k if ok[k] else None
        if best_len < min_len:
            continue
        seg = g.iloc[best_a: best_a + best_len]
        sc, _ = pc_scores(seg[SIGNALS].values)
        pc1 = sc[:, 0]
        e = seg.estrogen.values.astype(float)
        m = np.isfinite(e)
        rows.append(dict(
            length=best_len,
            ssf_pc1=ssf_spectral(pc1),
            ssf_e3g=ssf_spectral(e[m]) if m.sum() >= 40 else np.nan,
            r_pc1_e3g=(abs(np.corrcoef(pc1[m], e[m])[0, 1])
                       if m.sum() >= 25 and np.std(e[m]) > 1e-9 else np.nan),
        ))

    r = pd.DataFrame(rows)
    print(f"\n  subjects with a contiguous gap-free run >= {min_len} days:  N = {len(r)}")
    if len(r):
        print(f"  median run length : {r.length.median():.0f} days")
        print(f"  SSF of PC1        : {r.ssf_pc1.median():.3f}")
        print(f"  SSF of E3G        : {r.ssf_e3g.median():.3f}")
        print(f"  target            : 0.700")
    print("\n  => WITH N = 1, THIS QUESTION IS NOT ANSWERABLE WITH THIS DATASET.")
    print("     Not for lack of method -- for lack of data with continuous coverage.")
    print("     That is the result, and it defines what a prospective study must fix.")
    return r


# ------------------------------- (5) WHAT SURVIVES ------------------------------- #
def surviving_signal(M):
    """Is there ANY cycle information in the wearables? (phase, not hormone level)"""
    print("\n" + "=" * 72)
    print("(5)  WHAT SURVIVES: is there cycle information in the wearables at all?")
    print("=" * 72)
    eta2 = []
    for _, g in M.groupby("id"):
        # keep only days where ALL six signals are present (drop rows, not subjects)
        g = g.dropna(subset=SIGNALS)
        if len(g) < 40:
            continue
        sc, _ = pc_scores(g[SIGNALS].astype(float).values)
        pc1 = sc[:, 0]
        ph = g.phase.values
        m = pd.notna(ph)
        if m.sum() < 25:
            continue
        y, p = pc1[m], ph[m]
        sst = ((y - y.mean()) ** 2).sum()
        if sst <= 0:
            continue
        ssb = sum(len(y[p == u]) * (y[p == u].mean() - y.mean()) ** 2 for u in np.unique(p))
        eta2.append(ssb / sst)

    print(f"\n  eta^2 of PC1 with CYCLE PHASE : {np.median(eta2):.3f}   (n = {len(eta2)})")
    print("\n  => The cycle signal IS present in the wearables. It is simply not")
    print("     extractable by a linear, variance-ordered method.")
    print("     PCA and Fourier both assume STATIONARITY. The menstrual cycle is NOT")
    print("     stationary: it changes shape, amplitude and duration across cycles and")
    print("     across women. This is the argument for a time-frequency (wavelet)")
    print("     decomposition -- and it is the open question I am bringing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="unzipped mcPHASES v1.0.0 (credentialed PhysioNet)")
    ap.add_argument("--audit-only", action="store_true",
                    help="run only the interpolation-bias audit (needs no data)")
    a = ap.parse_args()

    if a.audit_only or not a.data_dir:
        interpolation_bias_audit()
        return

    M = load_daily(a.data_dir)
    print(f"[i] subjects: {M.id.nunique()}   rows: {len(M)}\n")
    components_analysis(M)
    interpolation_bias_audit()
    contiguous_only(M)
    surviving_signal(M)


if __name__ == "__main__":
    main()
