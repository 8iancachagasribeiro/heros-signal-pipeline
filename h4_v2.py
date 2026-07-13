#!/usr/bin/env python3
"""
H4 v2 - Design frontier WITH PREDICTOR NOISE.

The mcPHASES analysis (TODO 1) exposed a structural limitation of the original H4 grid:
it assumed the predictor (E2) was measured without error. In reality, at-home urine E3G
carries only ~.58 smooth signal. Because attenuation is MULTIPLICATIVE,

        r_observed = r_true * sqrt( R_x * R_y )

predictor quality enters the frontier on equal footing with outcome quality. The original
grid therefore OVERSTATES achievable recovery and would prescribe an under-powered design.

This script:
  (1) re-computes the frontier over  density x R_outcome x R_predictor
  (2) answers the question a real researcher planning a study actually asks:
      GIVEN A FIXED BUDGET, where does the next unit of effort buy the most recovery -
      more measurements, a better outcome instrument, or a better hormone assay?
"""
import warnings, sys, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import h4_frontier as H
from run_h4 import within_r_matrix
from fastlrt import lrt_random_slope_fast

N_SUBJ = 42
N_SIMS = 60
SEED = 17

_xd = H.e2(np.linspace(0, H.CYCLE_LEN, 400, endpoint=False))


def true_coupling(b):
    return within_r_matrix(H.inverted_u(b[:, None] + H.K_GAIN * _xd[None, :]), _xd)


def simulate(rng, n_obs, n_cycles, R_x, R_y, sigma_b, n_subj=N_SUBJ):
    """n_obs = TOTAL paired observations per person, spread over n_cycles."""
    span = H.CYCLE_LEN * n_cycles
    days = np.sort(rng.choice(np.arange(int(span)), size=min(n_obs, int(span)),
                              replace=False)).astype(float)
    x_true = H.e2(days)
    b = rng.normal(H.DA_OPT - H.K_GAIN * H._E2_MEAN, sigma_b, n_subj)
    sig = H.inverted_u(b[:, None] + H.K_GAIN * x_true[None, :])
    y_true = sig + rng.normal(0, H.SIGMA_STATE, sig.shape)
    # outcome instrument noise
    sdy = np.sqrt(max(y_true.var(), 1e-12) * (1 - R_y) / R_y)
    y_obs = y_true + rng.normal(0, sdy, y_true.shape)
    # PREDICTOR instrument noise  <-- the correction the real data forced
    sdx = np.sqrt(max(x_true.var(), 1e-12) * (1 - R_x) / R_x)
    x_obs = x_true + rng.normal(0, sdx, x_true.shape)
    return y_obs, x_obs, b


def eval_cell(rng, n_obs, n_cycles, R_x, R_y, sigma_b, n_sims=N_SIMS):
    """Returns (power_dual, power_lrt, fidelity, median_obs_r)."""
    dual = lrt = 0
    fid, medr = [], []
    for _ in range(n_sims):
        yo, xo, b = simulate(rng, n_obs, n_cycles, R_x, R_y, sigma_b)
        rh = within_r_matrix(yo, xo)
        rt = true_coupling(b)
        prop = float(np.nanmean(np.abs(rh) > 0.20))
        p = lrt_random_slope_fast(yo, xo)
        if np.isfinite(p):
            if p < 0.05:
                lrt += 1
                if prop > 0.50:
                    dual += 1
        m = np.isfinite(rh) & np.isfinite(rt)
        if m.sum() > 3 and np.std(rh[m]) > 1e-9 and np.std(rt[m]) > 1e-9:
            fid.append(np.corrcoef(rh[m], rt[m])[0, 1])
        medr.append(np.nanmedian(np.abs(rh)))
    return (dual / n_sims, lrt / n_sims,
            float(np.mean(fid)) if fid else np.nan,
            float(np.mean(medr)))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    if mode == "grid":
        # frontier over R_predictor x R_outcome, at realistic density (mcPHASES-like)
        Rx = [0.40, 0.55, 0.58, 0.70, 0.85, 0.99]
        Ry = [0.40, 0.55, 0.70, 0.85, 0.99]
        rows = []
        for sb in [0.10, 0.20]:
            for rx in Rx:
                for ry in Ry:
                    d, l, f, mr = eval_cell(rng, 74, 3, rx, ry, sb)
                    rows.append(dict(sigma_b=sb, R_predictor=rx, R_outcome=ry,
                                     n_obs=74, power_dual=d, power_lrt=l,
                                     fidelity=f, median_obs_r=mr,
                                     attenuation=np.sqrt(rx * ry)))
                    print(f"sb={sb} Rx={rx:.2f} Ry={ry:.2f} -> dual={d:.2f} "
                          f"lrt={l:.2f} fid={f:.2f} atten={np.sqrt(rx*ry):.2f}", flush=True)
        pd.DataFrame(rows).to_csv("/mnt/user-data/outputs/h4v2_predictor_grid.csv", index=False)

    elif mode == "budget":
        # THE BUDGET QUESTION: starting from the mcPHASES design, which upgrade buys most?
        base = dict(n_obs=74, n_cycles=3, R_x=0.58, R_y=0.41)
        scenarios = {
            "mcPHASES as-built":                dict(base),
            "2x measurements (148 obs)":        dict(base, n_obs=148, n_cycles=6),
            "4x measurements (296 obs)":        dict(base, n_obs=296, n_cycles=11),
            "better OUTCOME (R_y .41->.85)":    dict(base, R_y=0.85),
            "better PREDICTOR (R_x .58->.85)":  dict(base, R_x=0.85),
            "better BOTH instruments":          dict(base, R_x=0.85, R_y=0.85),
            "better both + 2x measurements":    dict(base, R_x=0.85, R_y=0.85, n_obs=148, n_cycles=6),
        }
        print(f"{'scenario':>34} {'atten':>7} {'power':>7} {'fidelity':>9} {'med |r_obs|':>12}")
        print("-" * 76)
        rows = []
        for name, cfg in scenarios.items():
            d, l, f, mr = eval_cell(rng, cfg["n_obs"], cfg["n_cycles"],
                                    cfg["R_x"], cfg["R_y"], 0.15)
            at = np.sqrt(cfg["R_x"] * cfg["R_y"])
            print(f"{name:>34} {at:>7.2f} {d:>7.2f} {f:>9.2f} {mr:>12.3f}")
            rows.append(dict(scenario=name, **cfg, attenuation=at,
                             power_dual=d, power_lrt=l, fidelity=f, median_obs_r=mr))
        pd.DataFrame(rows).to_csv("/mnt/user-data/outputs/h4v2_budget.csv", index=False)
        print()
        print("sigma_b fixed at 0.15 (moderate heterogeneity). Power = preregistered dual criterion.")

    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
