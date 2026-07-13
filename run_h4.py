#!/usr/bin/env python3
"""H4 design frontier - full grid, vectorized, using the validated fast LRT."""
import warnings, sys, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import h4_frontier as H
from fastlrt import lrt_random_slope_fast

N_SUBJ = H.N_SUBJ
DENSITIES = [2, 3, 5, 7, 10, 14, 21, 28]          # observations per cycle
RELIABILITIES = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
SIGMAS = [0.05, 0.10, 0.15, 0.20]                  # coupling heterogeneity (sd of baseline tone)
N_CYCLES = 2
N_SIMS = 50
SEED = 7


def simulate_matrix(rng, n_subj, obs_per_cycle, n_cycles, reliability, sigma_b,
                    balance_offset=0.0):
    """Returns (Y_obs (n_subj,n), Y_true, x)."""
    days = np.concatenate([
        np.linspace(0, H.CYCLE_LEN, obs_per_cycle, endpoint=False) + c * H.CYCLE_LEN
        for c in range(n_cycles)
    ])
    x = H.e2(days)
    mu_b = H.DA_OPT - H.K_GAIN * H._E2_MEAN + balance_offset
    b = rng.normal(mu_b, sigma_b, size=n_subj)
    da = b[:, None] + H.K_GAIN * x[None, :]
    signal = H.inverted_u(da)
    y_true = signal + rng.normal(0.0, H.SIGMA_STATE, size=signal.shape)
    sd_err = np.sqrt(max(y_true.var(), 1e-12) * (1.0 - reliability) / reliability)
    y_obs = y_true + rng.normal(0.0, sd_err, size=y_true.shape)
    return y_obs, y_true, x


def within_r_matrix(Y, x):
    """Vectorized within-person Pearson r between x and each row of Y."""
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=1, keepdims=True)
    num = Yc @ xc
    den = np.sqrt((Yc ** 2).sum(axis=1) * (xc ** 2).sum())
    den = np.where(den < 1e-12, np.nan, den)
    return num / den


def eval_cell(args):
    sb, d, rel, seed = args
    rng = np.random.default_rng(seed)
    c1 = c2 = both = 0
    grp, props = [], []
    for _ in range(N_SIMS):
        Yo, Yt, x = simulate_matrix(rng, N_SUBJ, d, N_CYCLES, rel, sb)
        r_i = within_r_matrix(Yo, x)
        prop = float(np.nanmean(np.abs(r_i) > 0.20))
        p = lrt_random_slope_fast(Yo, x)
        if not np.isfinite(p):
            continue
        a = prop > 0.50; b_ = p < 0.05
        c1 += a; c2 += b_; both += (a and b_)
        props.append(prop)
        grp.append(float(np.corrcoef(np.tile(x, N_SUBJ), Yo.ravel())[0, 1]))
    return dict(sigma_b=sb, obs_per_cycle=d, n_cycles=N_CYCLES,
                total_obs_per_subj=d*N_CYCLES, reliability=rel,
                recovery=both/N_SIMS, crit1_effectsize=c1/N_SIMS, crit2_lrt=c2/N_SIMS,
                mean_group_r=float(np.mean(grp)) if grp else np.nan,
                mean_prop_detected=float(np.mean(props)) if props else np.nan)


def main():
    from multiprocessing import Pool
    sb_only = float(sys.argv[1]) if len(sys.argv) > 1 else None
    sigmas = [sb_only] if sb_only else SIGMAS
    jobs = []
    k = 0
    for sb in sigmas:
        for d in DENSITIES:
            for rel in RELIABILITIES:
                k += 1
                jobs.append((sb, d, rel, SEED * 1000 + k))
    t0 = time.time()
    with Pool(processes=4) as pool:
        rows = pool.map(eval_cell, jobs)
    df = pd.DataFrame(rows)
    tag = f"_sb{sb_only}" if sb_only else ""
    df.to_csv(f"/home/claude/h4_part{tag}.csv", index=False)
    print(f"[done] {len(df)} cells in {time.time()-t0:.0f}s -> h4_part{tag}.csv", flush=True)
    return


def _old_main():
    rng = np.random.default_rng(SEED)
    rows = []
    total = len(SIGMAS) * len(DENSITIES) * len(RELIABILITIES)
    i = 0
    t0 = time.time()
    for sb in SIGMAS:
        for d in DENSITIES:
            for rel in RELIABILITIES:
                i += 1
                c1 = c2 = both = 0
                grp, props = [], []
                for _ in range(N_SIMS):
                    Yo, Yt, x = simulate_matrix(rng, N_SUBJ, d, N_CYCLES, rel, sb)
                    r_i = within_r_matrix(Yo, x)
                    prop = float(np.nanmean(np.abs(r_i) > 0.20))       # criterion (i)
                    p = lrt_random_slope_fast(Yo, x)                    # criterion (ii)
                    if not np.isfinite(p):
                        continue
                    a = prop > 0.50
                    b_ = p < 0.05
                    c1 += a; c2 += b_; both += (a and b_)
                    props.append(prop)
                    # group-level pooled correlation
                    xf = np.tile(x, N_SUBJ); yf = Yo.ravel()
                    grp.append(float(np.corrcoef(xf, yf)[0, 1]))
                rows.append(dict(
                    sigma_b=sb, obs_per_cycle=d, n_cycles=N_CYCLES,
                    total_obs_per_subj=d * N_CYCLES, reliability=rel,
                    recovery=both / N_SIMS, crit1_effectsize=c1 / N_SIMS,
                    crit2_lrt=c2 / N_SIMS,
                    mean_group_r=float(np.mean(grp)),
                    mean_prop_detected=float(np.mean(props)),
                ))
                el = time.time() - t0
                print(f"[{i:>3}/{total}] sb={sb:.2f} obs/cyc={d:>2} rel={rel:.2f} "
                      f"-> recovery={both/N_SIMS:.2f} (size={c1/N_SIMS:.2f} lrt={c2/N_SIMS:.2f}) "
                      f"| {el:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("/mnt/user-data/outputs/h4_frontier_results.csv", index=False)
    print(f"\n[saved] h4_frontier_results.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
