#!/usr/bin/env python3
"""
ssf_estimators.py — Smooth Signal Fraction (SSF) estimators, with validation.

WHAT THIS IS
------------
The SSF is the fraction of a series' observed variance carried by a temporally
STRUCTURED (smooth) component. For detecting a CYCLE-LOCKED coupling this is the
quantity that matters: no white component -- of any origin (measurement error,
genuine white state fluctuation, or ordinal quantisation) -- can carry a cycle-locked
signal. It is NOT classical psychometric reliability and must not be reported as such.

THREE ESTIMATORS
----------------
1. AR(1) closed form:      R = rho(1)^2 / rho(2)
   Assumes the true signal is AR(1). It is NOT: a cycle-locked symptom (e.g. menstrual
   cramps) is a periodic, sharply-peaked process. This estimator is BIASED and can
   return values > 1 (impossible for a variance fraction). It produced 1.063 for cramps
   in the v2 manuscript and was replaced.

2. ACF linear extrapolation to lag 0:   R = 2*rho(1) - rho(2)
   Better, but still biased: the autocorrelation of ANY differentiable process has zero
   derivative at lag 0 (it is even, with a maximum there), so rho(k) ~ 1 - a*k^2 --
   QUADRATIC, not linear.

3. SPECTRAL (adopted):
   White noise has a FLAT power spectrum; a cycle-locked signal (period ~28 d) has no
   power at high frequency. The high-frequency plateau of the periodogram IS the noise
   floor. Makes NO assumption about the signal's shape.
   BIAS CORRECTION: periodogram ordinates of white noise are EXPONENTIALLY distributed,
   and the median of an exponential is ln(2) ~ 0.693 times its mean. Using the raw median
   underestimates the noise floor by ~30%. We divide by ln(2).

VALIDATION RESULT (see validate() below; three signal shapes x four noise levels):
    estimator          mean |bias|   max |bias|
    AR(1)                0.077         0.157
    ACF-linear           0.036         0.082
    SPECTRAL (adopted)   0.028         0.082
"""
import numpy as np

LN2 = np.log(2.0)


# ----------------------------------------------------------------------------- #
def ssf_ar1(y):
    """AR(1) closed form. BIASED for cyclic processes; can exceed 1. Reported for
    comparison only -- do not use for inference."""
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if len(y) < 25 or np.std(y) < 1e-12:
        return np.nan
    y = y - y.mean(); n = len(y); v = np.dot(y, y) / n
    r1 = np.dot(y[:-1], y[1:]) / n / v
    r2 = np.dot(y[:-2], y[2:]) / n / v
    return r1 ** 2 / r2 if r2 > 0.02 else np.nan


def ssf_acf_linear(y):
    """Linear extrapolation of the ACF to lag 0. Biased (true ACF is quadratic near 0)."""
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if len(y) < 25 or np.std(y) < 1e-12:
        return np.nan
    y = y - y.mean(); n = len(y); v = np.dot(y, y) / n
    r1 = np.dot(y[:-1], y[1:]) / n / v
    r2 = np.dot(y[:-2], y[2:]) / n / v
    return 2 * r1 - r2


def ssf_spectral(y, f_cut=0.25):
    """ADOPTED ESTIMATOR. Spectral separation with exponential-median bias correction.

    f_cut is in cycles/day. 0.25 => periods shorter than 4 days are treated as the
    noise band. A 28-day cycle (f = 0.036) is far below this, so no signal leaks in.
    """
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    n = len(y)
    if n < 25 or np.std(y) < 1e-12:
        return np.nan
    y = y - y.mean()
    P = (np.abs(np.fft.rfft(y)) ** 2) / n          # periodogram
    f = np.fft.rfftfreq(n, d=1.0)
    P = P[1:]; f = f[1:]                            # drop the DC term
    hi = f > f_cut
    if hi.sum() < 4:
        return np.nan
    noise_psd = np.median(P[hi]) / LN2              # <-- exponential-median correction
    return float(np.clip(1.0 - noise_psd * len(P) / P.sum(), 0.0, 1.0))


# ----------------------------------------------------------------------------- #
def validate(n_rep=200, seed=7, verbose=True):
    """Recover a KNOWN smooth signal fraction from synthetic data.

    Three signal shapes, four true SSF levels. Reports absolute bias per estimator.
    Reproduces the validation table in Methods 2.3 / Figure 5(a).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(90)
    shapes = {
        "smooth sinusoid (28d)": np.sin(2 * np.pi * t / 28),
        "two-peak (like E2)":    np.exp(-((t % 28 - 13) ** 2) / 8)
                                 + 0.55 * np.exp(-((t % 28 - 21) ** 2) / 24),
        "sharp pulse (cramps)":  (np.minimum(t % 28, 28 - (t % 28)) < 2.5).astype(float),
    }
    err = {"ar1": [], "lin": [], "spec": []}
    if verbose:
        print(f"{'signal':>22} {'TRUE':>6} {'AR(1)':>8} {'ACF-lin':>9} {'SPECTRAL':>10}")
        print("-" * 60)
    for name, s in shapes.items():
        s = (s - s.mean()) / s.std()
        for true in (0.30, 0.40, 0.55, 0.70):
            est = {"ar1": [], "lin": [], "spec": []}
            sd = np.sqrt((1 - true) / true)
            for _ in range(n_rep):
                y = s + rng.normal(0, sd, len(s))
                est["ar1"].append(ssf_ar1(y))
                est["lin"].append(ssf_acf_linear(y))
                est["spec"].append(ssf_spectral(y))
            a, l, sp = (np.nanmedian(est[k]) for k in ("ar1", "lin", "spec"))
            for k, v in zip(("ar1", "lin", "spec"), (a, l, sp)):
                err[k].append(abs(v - true))
            if verbose and true in (0.40, 0.70):
                print(f"{name:>22} {true:>6.2f} {a:>8.2f} {l:>9.2f} {sp:>10.2f}")
    if verbose:
        print("-" * 60)
        print(f"{'MEAN |BIAS|':>22} {'':>6} {np.mean(err['ar1']):>8.3f} "
              f"{np.mean(err['lin']):>9.3f} {np.mean(err['spec']):>10.3f}")
        print(f"{'MAX |BIAS|':>22} {'':>6} {np.max(err['ar1']):>8.3f} "
              f"{np.max(err['lin']):>9.3f} {np.max(err['spec']):>10.3f}")
    return err


if __name__ == "__main__":
    validate()
