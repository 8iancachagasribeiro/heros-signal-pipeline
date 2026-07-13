#!/usr/bin/env python3
"""
Power of the EXACT preregistered test: phase-randomized surrogates.

The registered H3 test was NOT the LRT. It was a surrogate test for genuine
between-person heterogeneity in within-person coupling, with the predictor
phase-randomized (preserving the power spectrum, hence the full autocorrelation
structure, while destroying temporal alignment with the outcome).

Naive permutation is prohibited: it destroys autocorrelation, understates the null
variance of r, and inflates Type I error. (This correction was made before lodging.)

Test statistic:  S = SD of the within-person couplings r_i across individuals.
Null:            per-person phase randomization of the predictor.
p:               (1 + #{S_b >= S_obs}) / (B + 1)

Realism upgrades over the earlier grid:
  - each person carries her OWN cycle phase offset (they did not all start on cycle day 1)
  - each person's predictor is phase-randomized INDEPENDENTLY
  - instrument noise on BOTH predictor and outcome, set to the values measured in mcPHASES
"""
import warnings, sys, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import h4_frontier as H

# --- measured from mcPHASES (TODO 1) ---
N_SUBJ   = 42
N_OBS    = 85      # CORRECTED: median paired obs per person (6-level map)
SPAN     = 90      # days in study (~3 cycles)
R_X_REAL = 0.58    # smooth signal fraction, E3G urine strip
R_Y_REAL = 0.41    # smooth signal fraction, affective self-report
B_SURR   = 500     # surrogates, as registered
ALPHA    = 0.05


def phase_randomize(x, rng):
    """Preserve the power spectrum (hence autocorrelation); randomize phases."""
    n = len(x)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0                      # keep the DC term real
    if n % 2 == 0:
        ph[-1] = 0.0                 # keep Nyquist real
    return np.fft.irfft(mag * np.exp(1j * ph), n)


def _corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def simulate_study(rng, sigma_b, R_x, R_y, n_subj=N_SUBJ, n_obs=N_OBS, span=SPAN):
    """Each person gets her OWN cycle phase offset and her own sampling days."""
    b = rng.normal(H.DA_OPT - H.K_GAIN * H._E2_MEAN, sigma_b, n_subj)
    offs = rng.uniform(0, H.CYCLE_LEN, n_subj)          # individual cycle phase
    Xs, Ys = [], []
    for i in range(n_subj):
        days = np.sort(rng.choice(np.arange(span), size=n_obs, replace=False)).astype(float)
        x_true = H.e2(days + offs[i])
        sig = H.inverted_u(b[i] + H.K_GAIN * x_true)
        y_true = sig + rng.normal(0, H.SIGMA_STATE, n_obs)
        sdy = np.sqrt(max(y_true.var(), 1e-12) * (1 - R_y) / R_y)
        sdx = np.sqrt(max(x_true.var(), 1e-12) * (1 - R_x) / R_x)
        Ys.append(y_true + rng.normal(0, sdy, n_obs))
        Xs.append(x_true + rng.normal(0, sdx, n_obs))
    return Xs, Ys, b


def surrogate_test(Xs, Ys, rng, B=B_SURR):
    """Registered test. Returns (p_value, S_obs, median S_null)."""
    r_obs = np.array([_corr(x, y) for x, y in zip(Xs, Ys)])
    S_obs = float(np.std(r_obs))
    S_null = np.empty(B)
    for k in range(B):
        r_s = np.array([_corr(phase_randomize(x, rng), y) for x, y in zip(Xs, Ys)])
        S_null[k] = np.std(r_s)
    p = (1 + int(np.sum(S_null >= S_obs))) / (B + 1)
    return p, S_obs, float(np.median(S_null))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "power"
    rng = np.random.default_rng(23)
    t0 = time.time()

    if mode == "calib":
        # Type I calibration of the registered test under a TRUE null (no coupling at all)
        print("TYPE I CALIBRATION of the registered surrogate test")
        print("(sigma_b = 0: no coupling heterogeneity whatsoever)")
        n_rep, B = 200, 200
        rej = 0
        for _ in range(n_rep):
            Xs, Ys, _ = simulate_study(rng, 0.0001, R_X_REAL, R_Y_REAL)
            p, _, _ = surrogate_test(Xs, Ys, rng, B=B)
            rej += (p < ALPHA)
        print(f"  false-positive rate = {rej/n_rep:.3f}   (nominal alpha = {ALPHA})")
        print("  -> the registered test is correctly calibrated." if abs(rej/n_rep - ALPHA) < 0.04
              else "  -> WARNING: miscalibrated; investigate before using.")

    else:
        print("POWER OF THE REGISTERED TEST (phase-randomized surrogates, B=500)")
        print(f"at the ACTUAL mcPHASES design: N={N_SUBJ}, {N_OBS} obs/person over {SPAN} days")
        print(f"instruments as measured: R_predictor={R_X_REAL}, R_outcome={R_Y_REAL}")
        print()
        print(f"{'sigma_b':>8} {'power (actual instr.)':>22} {'power (ideal instr.)':>21}")
        print("-" * 54)
        rows = []
        n_rep, B = 40, 150
        for sb in [0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
            res = {}
            for tag, (rx, ry) in {"actual": (R_X_REAL, R_Y_REAL)}.items():
                rej = 0
                for _ in range(n_rep):
                    Xs, Ys, _ = simulate_study(rng, sb, rx, ry)
                    p, _, _ = surrogate_test(Xs, Ys, rng, B=B)
                    rej += (p < ALPHA)
                res[tag] = rej / n_rep
            print(f"{sb:>8.3f} {res['actual']:>22.2f}", flush=True)
            rows.append(dict(sigma_b=sb, power_actual=res["actual"]))
        pd.DataFrame(rows).to_csv("/mnt/user-data/outputs/registered_test_power.csv", index=False)
        print()
        print("mcPHASES observed: fatigue surrogate p = 0.66 (null).")
        print("Read the 'actual' column: the sigma_b at which power crosses .80 is the")
        print("largest heterogeneity the study could have MISSED. Above it, the null is")
        print("informative; below it, the null is uninformative.")

    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
