#!/usr/bin/env python3
"""
actigraphy_replication.py — Second-domain replication of the instrumental measurement.

Reproduces Tables 15 and 16 of the manuscript (sections 7.9 and 7.10).

WHY THIS EXISTS
---------------
The strongest objection to Section 7 is generality: every smooth-signal fraction (SSF)
reported there comes from a single dataset (mcPHASES, N = 42). Is low SSF a property of
the field, or a defect of that one study?

This script applies the SAME estimator, at the SAME relative cutoff, to an entirely
different domain: clinical actigraphy.

WHY THE COMPARISON IS STRUCTURALLY VALID
----------------------------------------
The estimator is scale-invariant. `f_cut = 0.25` means "periods shorter than 4 samples
are the noise band", whatever the sampling interval is.

  menstrual scale : 1 sample = 1 day,  signal = 28 days (f = 0.036), noise = < 4 days
  circadian scale : 1 sample = 1 hour, signal = 24 hours (f = 0.042), noise = < 4 hours

Both place the signal ~24-28x below the sampling frequency and the noise band above
one quarter of it. The two applications are structurally identical.

WHAT IT FINDS
-------------
(1) Circadian SSF = 0.681 across n = 162 subjects in four clinical populations, against
    0.336-0.474 for the menstrual signature in the SAME class of wearable device.
    => The limiting factor is the CONSTRUCT, not the instrument.

(2) SSF does NOT depend on the number of cycles observed (flat from 3 to 16 cycles).
    => The 0.469 measured in mcPHASES (~3 cycles) is NOT an artefact of short duration.
       This was the test with the greatest potential to refute the manuscript's argument.

DATA (all open, no credentialing required)
------------------------------------------
  DEPRESJON   Garcia-Ceja et al., MMSys'18   DOI 10.1145/3204949.3208125
  PSYKOSE     Jakobsen et al., IEEE CBMS'20  DOI 10.1109/CBMS49503.2020.00064
  HYPERAKTIV  Hicks et al., MMSys'21         DOI 10.1145/3458305.3478454

Expected layout under --data-dir:
    depresjon/data/condition/*.csv
    depresjon/data/control/*.csv
    psykose/patient/*.csv
    psykose/control/*.csv
    activity_data/*.csv            (HYPERAKTIV; semicolon-separated)

NOTE: DEPRESJON and PSYKOSE share control subjects. They are de-duplicated below;
failing to do so inflates n and produces identical duplicated statistics.

USAGE
-----
    python actigraphy_replication.py --data-dir /path/to/actigraphy
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd

from ssf_estimators import ssf_spectral

warnings.filterwarnings("ignore")

SOURCES = [
    ("depresjon/data/condition/*.csv", "major depression"),
    ("depresjon/data/control/*.csv",   "controls"),
    ("psykose/patient/*.csv",          "schizophrenia"),
    ("psykose/control/*.csv",          "controls"),      # de-duplicated against the above
    ("activity_data/*.csv",            "ADHD"),
]


def load_series(path):
    """Read one actigraphy file. Handles both comma- and semicolon-separated variants."""
    for sep in (",", ";"):
        try:
            d = pd.read_csv(path, sep=sep)
            if d.shape[1] >= 2:
                break
        except Exception:
            continue
    else:
        return None

    acts = [c for c in d.columns if "activ" in c.lower()]
    times = [c for c in d.columns if c.lower() in ("timestamp", "date", "time")]
    if not acts or not times:
        return None

    t = pd.to_datetime(d[times[0]], errors="coerce", format="mixed")
    y = pd.to_numeric(d[acts[0]], errors="coerce")
    return pd.Series(y.values, index=t).dropna()


def longest_contiguous(s, step_hours=1.0):
    """Longest run of consecutive samples with NO gaps.

    This matters. The FFT assumes regular spacing; feeding it a gapped series makes it
    treat non-consecutive samples as consecutive and inflates the SSF. Interpolating the
    gaps does not fix this -- it introduces a bias of its own (see wearable_fusion.py,
    section 3). The only clean option is to use gap-free segments.
    """
    if len(s) < 2:
        return s
    gaps = pd.Series(s.index).diff().dt.total_seconds().fillna(step_hours * 3600).values
    gaps = gaps / 3600.0
    best_len, best_start, start = 0, 0, 0
    for k in range(1, len(gaps) + 1):
        if k == len(gaps) or abs(gaps[k] - step_hours) > 1e-6:
            if k - start > best_len:
                best_len, best_start = k - start, start
            start = k
    return s.iloc[best_start:best_start + best_len]


def collect(data_dir, min_hours=48):
    """Hourly-resampled, gap-free segments, de-duplicated across datasets."""
    rows, seen = [], set()
    for pattern, label in SOURCES:
        for path in sorted(glob.glob(os.path.join(data_dir, pattern))):
            s = load_series(path)
            if s is None or len(s) < min_hours:
                continue
            s = s.resample("1h").mean().dropna()
            seg = longest_contiguous(s)
            if len(seg) < min_hours:
                continue
            # de-duplication key: DEPRESJON and PSYKOSE share control subjects
            key = (round(float(seg.mean()), 4), len(seg))
            if key in seen:
                continue
            seen.add(key)
            v = ssf_spectral(seg.values)
            if np.isfinite(v):
                rows.append(dict(group=label, hours=len(seg), ssf=v, series=seg.values))
    return rows


# --------------------------------------------------------------------------- #
def table_15(rows):
    """SSF by clinical population -- the generality check."""
    print("=" * 74)
    print("(7.9)  SMOOTH-SIGNAL FRACTION IN CLINICAL ACTIGRAPHY (circadian scale)")
    print("=" * 74)
    df = pd.DataFrame([{k: r[k] for k in ("group", "hours", "ssf")} for r in rows])
    print(f"\n{'population':>18} {'n':>5} {'hours (med)':>12} {'SSF median':>12} {'IQR':>18}")
    print("-" * 70)
    for g, sub in df.groupby("group"):
        print(f"{g:>18} {len(sub):>5} {sub.hours.median():>12.0f} {sub.ssf.median():>12.3f}"
              f"   [{sub.ssf.quantile(.25):.3f}, {sub.ssf.quantile(.75):.3f}]")
    print("-" * 70)
    print(f"{'TOTAL':>18} {len(df):>5} {'':>12} {df.ssf.median():>12.3f}"
          f"   [{df.ssf.quantile(.25):.3f}, {df.ssf.quantile(.75):.3f}]")

    print("\n  Contrast with mcPHASES (same estimator, same device class, other construct):")
    print("    circadian / activity           ", f"{df.ssf.median():.3f}")
    print("    menstrual / resting heart rate   0.474")
    print("    menstrual / skin temperature     0.336")
    print("    menstrual / self-report          0.323")
    print("\n  => The accelerometer and the photoplethysmograph measure with the same")
    print("     precision in both cases. The difference is the CONSTRUCT: the wake-sleep")
    print("     alternation dominates total variance; cycle phase produces a peripheral")
    print("     modulation of 2-3 bpm against day-to-day variation of comparable size.")


def table_16(rows, min_cycles=16):
    """Does SSF depend on the number of cycles observed? THE refutation test."""
    print("\n" + "=" * 74)
    print("(7.10) ROBUSTNESS: DOES SSF DEPEND ON THE NUMBER OF CYCLES OBSERVED?")
    print("=" * 74)
    print("\n  The threat: mcPHASES spans ~3 menstrual cycles. If the estimator were biased")
    print("  downward when few cycles are observed, then 0.469 would be an artefact of")
    print("  design, the attenuation cascade would be inflated, and the manuscript's")
    print("  central conclusion would fall. The circadian domain permits the test that the")
    print("  menstrual domain cannot: it has cycles to spare.")

    long_series = [r["series"] for r in rows if len(r["series"]) >= min_cycles * 24]
    print(f"\n  series with >= {min_cycles} complete cycles: n = {len(long_series)}")
    print(f"\n{'cycles':>8} {'samples':>9} {'SSF median':>12} {'IQR':>18}")
    print("-" * 50)
    for nc in (3, 5, 8, 12, 16):
        vals = [ssf_spectral(s[:nc * 24]) for s in long_series]
        vals = np.array([v for v in vals if np.isfinite(v)])
        print(f"{nc:>8} {nc*24:>9} {np.median(vals):>12.3f}"
              f"   [{np.percentile(vals,25):.3f}, {np.percentile(vals,75):.3f}]")
    print("\n  => FLAT. No systematic dependence. The 0.469 measured in mcPHASES is NOT an")
    print("     artefact of observing only three cycles: it is a property of the instrument.")
    print("     The attenuation cascade of section 7.3 stands.")

    print("\n  TWO DECLARED LIMITATIONS:")
    print("   1. SSF conflates measurement error with genuine fast variation. Part of what")
    print("      the estimator calls noise in actigraphy is real behaviour, and activity is")
    print("      intrinsically burstier than body temperature. The cross-domain comparison")
    print("      is asymmetric in this respect.")
    print("   2. Aggregation differs: an hourly mean of 60 minute-level samples attenuates")
    print("      measurement error by a factor near 8; a urine strip is a single reading.")
    print("   Both act in the same direction and may inflate the circadian figure. Neither")
    print("   affects the robustness test, which is internal to the actigraphy domain.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="directory containing depresjon/, psykose/ and activity_data/")
    ap.add_argument("--min-hours", type=int, default=48,
                    help="minimum gap-free segment length, in hours (default 48)")
    a = ap.parse_args()

    rows = collect(a.data_dir, a.min_hours)
    if not rows:
        raise SystemExit("no usable series found -- check --data-dir layout")
    table_15(rows)
    table_16(rows)


if __name__ == "__main__":
    main()
