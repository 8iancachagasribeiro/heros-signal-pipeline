# Reproducibility package
## *The averaged individual does not exist* — manuscript v4

**Bianca Chagas Ribeiro** · Preregistration: OSF, 8 July 2026 (embargoed)
Package built 11 July 2026.

---

## READ THIS FIRST — two honest notices

**1. Several analyses in the original session were run inline and never saved as code.**
They have been reconstructed here as proper scripts (`ssf_estimators.py`,
`mcphases_analyses.py`, `calibration_fidelity_aliasing.py`, `make_figures.py`) and
**re-executed to confirm they reproduce the manuscript numbers exactly**. The
reconstruction is documented rather than hidden, because a reviewer is entitled to know
which code was archived from the outset and which was recovered afterwards.

**2. mcPHASES is NOT included in this package.**
It is credentialed-access PhysioNet data (DOI `10.13026/zx6a-2c81`) under a data use
agreement that does not permit redistribution. The simulation analyses — which carry the
paper's theoretical claims (HG, H1, H2, H4) — require **no data at all** and run
standalone. See `data/README_DATA.txt`.

---

## Environment

Python **3.12.3**. Install with:

```bash
pip install -r requirements.txt
```

| library | version | role |
|---|---|---|
| numpy | 2.4.4 | all numerics |
| scipy | 1.17.1 | optimisation, chi-bar-square, Cholesky |
| pandas | 3.0.2 | tabular I/O |
| matplotlib | 3.10.8 | figures |
| statsmodels | 0.14.6 | **validation only** — used to check `fastlrt.py`; not needed to reproduce results |

Manuscript build (optional): Node **v22.22.2** + the `docx` npm package.

---

## How to reproduce, in order

```bash
# 1. Validate the SSF estimator against known ground truth  (Methods 2.3, Fig 5a)
python ssf_estimators.py

# 2. Generative model: does it mask? does H2 hold?          (Tables 1, 2)
python h4_frontier.py --validate-only

# 3. Null calibration, fidelity, aliasing                   (Tables 3, 4, 5)
python calibration_fidelity_aliasing.py --out-dir ./results

# 4. The H4 recovery frontier                              (Fig 1b)
python run_h4.py                     # NOTE: run one sigma_b at a time, see below

# 5. Instrument frontier + budget allocation               (Fig 3, Table 12)
python h4_v2.py grid
python h4_v2.py budget

# 6. The registered surrogate test: calibration + power    (Table 11)
python registered_test_power.py calib
python registered_test_power.py power

# 7. SIGMA_STATE sensitivity (why the model-based bound was withdrawn)
python sigma_sweep.py 0.04,0.085,0.15,0.25

# 8. EMPIRICAL — requires credentialed mcPHASES            (Tables 7, 9, 10, 13; Fig 2, 4)
python mcphases_analyses.py --data-dir /path/to/mcphases-...-1.0.0 --out-dir ./results

# 9. Regenerate figures from the results
python make_figures.py --results ./results --out ./figures
```

---

## FILE-BY-FILE

### Code — simulation core (no data required)

| file | what it is | manuscript |
|---|---|---|
| **`h4_frontier.py`** | The generative model and its constants: two-peak E2 trajectory, `DA_i(t) = b_i + K·E2(t)`, inverted-U, state noise, measurement error. `--validate-only` runs the two sanity checks: **does it mask?** (Table 1) and **does H2 hold?** (Table 2). Everything else imports from here. | §3, Tables 1–2 |
| **`fastlrt.py`** | Fast likelihood-ratio test for between-person slope variance. Exploits the balanced design (identical marginal covariance across subjects → invert once, not N times). Chi-bar-square null (½χ²₁ + ½χ²₂). **Validated against `statsmodels`: 8/9 decision agreement; the single divergence was a case where `statsmodels` failed to converge and this implementation found a better optimum (Δlog-lik +2.17).** ~10× faster. | Methods §2.2 |
| **`run_h4.py`** | The H4 recovery grid: density × reliability × heterogeneity, 224 cells. Dual criterion as preregistered. | §6.2, Fig 1b |
| **`calibration_fidelity_aliasing.py`** | *(reconstructed)* Three analyses: **null calibration** (criterion (i) alone = 100% FP at 4 obs/person — this is why the dual criterion was necessary); **recovery fidelity** (detection ≠ recovery); **aliasing** (phase-targeted vs evenly-spaced). Contains a documented note on a fidelity bug that was found and fixed. | §§6.1, 6.3, 6.4; Tables 3–5 |
| **`h4_v2.py`** | Frontier **with predictor noise as a third axis** — the correction the real data forced. `grid` = instrument frontier; `budget` = where the next unit of research effort buys the most. | §8, Fig 3, Table 12 |
| **`registered_test_power.py`** | The **exact preregistered test**: phase-randomised surrogates (preserve the power spectrum, hence the full autocorrelation; destroy temporal alignment). `calib` → Type I = **0.050** against nominal 0.05. `power` → power at the real design. | §7.7, Table 11 |
| **`sigma_sweep.py`** | Sensitivity of the results to `SIGMA_STATE`, the one free parameter the analyst chose. **The power curves do NOT collapse when re-expressed in true effect size — which is why the model-based bound was withdrawn and replaced by a model-free one.** | §7.7, §10 |

### Code — instruments and empirical (mcPHASES required)

| file | what it is | manuscript |
|---|---|---|
| **`ssf_estimators.py`** | *(reconstructed)* The three smooth-signal-fraction estimators and, critically, `validate()` — which recovers a **known** SSF from synthetic data across three signal shapes. Result: AR(1) mean bias .077 (and returns an **impossible** 1.063 for cramps); ACF-linear .036; **spectral .028 (adopted)**. Documents the exponential-median ln(2) bias correction. | Methods §2.3, Fig 5a |
| **`mcphases_analyses.py`** | *(reconstructed)* **Every** empirical analysis, in one script: sample description, instrument SSF, the preregistered differential prediction, the **disattenuation + person-level bootstrap that shows the prediction CANNOT be tested**, the phase-locked finding, the model-free bound, and objective-vs-self-report. Carries the correct **six-level** ordinal map (the earlier five-level map silently dropped "Not at all" and produced N=39/median 74 instead of N=41/median 85). | §7 entire |
| **`make_figures.py`** | *(reconstructed)* Regenerates the figures from the results CSVs, including the **audit-corrected Fig 1b** with the detection-only zone hatched. | all figures |

### Data

| path | contents |
|---|---|
| `data/README_DATA.txt` | Why mcPHASES is not here, and how to obtain it. |
| `results/` | All output CSVs (written by the scripts above). |
| `figures/` | Regenerated figures. |

---

## Claim → code → output

| manuscript claim | produced by | output |
|---|---|---|
| Group \|g\|=.081 while individual \|g\|=.32 (H1) | `h4_frontier.py --validate-only` | stdout, Table 1 |
| Masking breaks at balance offset .05 (H2) | `h4_frontier.py --validate-only` | stdout, Table 2 |
| Criterion (i) alone: **100% FP at 4 obs** | `calibration_fidelity_aliasing.py` | `table03_null_calibration.csv` |
| Fidelity crosses .70 only at ~28 obs/person | `calibration_fidelity_aliasing.py` | `table04_fidelity.csv` |
| 3-obs evenly-spaced aliases (.31 vs .58) | `calibration_fidelity_aliasing.py` | `table05_aliasing.csv` |
| Spectral estimator is least biased | `ssf_estimators.py` | stdout, Fig 5a |
| SSF: predictor .469, outcome .323 → attenuation **.389** | `mcphases_analyses.py` | `table07_instrument_ssf.csv` |
| Differential prediction **cannot be tested** (all CIs span 0 and .10) | `mcphases_analyses.py` | `table10_disattenuated.csv` |
| Cramps: \|r\| with E2 level .075, η² with phase **.162** | `mcphases_analyses.py` | `fig04_phase_locked.csv` |
| Model-free: SD(r_i)=.1401 vs null median .1410, p=.515 | `mcphases_analyses.py` | `modelfree_bound.csv` |
| Skin temp p=.027 (exploratory; fails Bonferroni) | `mcphases_analyses.py` | `table13_objective.csv` |
| Power = 0.00 with real instruments, .90–1.00 with ideal | `h4_v2.py` | `h4v2_predictor_grid.csv` |
| **4× more measurements buys ZERO power** | `h4_v2.py budget` | `h4v2_budget.csv` |
| Registered test Type I = **0.050** | `registered_test_power.py calib` | stdout |
| Model-based bound withdrawn (curves don't collapse) | `sigma_sweep.py` | stdout |

---

## Known limitations of this package

1. **Simulation resolution.** Grids ran at 50–60 sims/cell and power at 40 replicates × 150 surrogates (compute-bound). The central effects are large (power 0.00 vs 0.90+) and do not depend on this, but **the final frontier figures must be re-run at ≥500** before submission. Every script takes `--n-sims`.
2. **`run_h4.py` memory/time.** The full 224-cell grid exceeds a short timeout on a single core. Run one `sigma_b` at a time (`python run_h4.py 0.10`).
3. **The ad-hoc scale factor** in the model-free minimum-detectable-effect injection is arbitrary; this is why the script reports the SD(r_i) **actually achieved**, not the injected parameter. Declared in Methods.
4. **`statsmodels` is a validation dependency only.** If you only want to reproduce the results, `fastlrt.py` needs nothing beyond numpy/scipy.

---

## Files you already have locally (NOT duplicated here)

These exist on the author's machine and in `/mnt/user-data/outputs`; they are **not** part of the code package and are listed only so they can be located:

| file | what it is |
|---|---|
| `ARTIGO_individuo_medio_ABNT_v4.docx` / `.pdf` | **the current manuscript** (24 pp, ABNT) |
| `ARTIGO_individuo_medio_ABNT_v2/v3.*` | superseded drafts — keep for provenance, do not circulate |
| `MANUSCRIPT_averaged_individual_v1.md` | the original English draft, pre-audit — superseded |
| `manuscript_skeleton_averaged_individual.md` | the working skeleton — superseded |
| `AUDIT_manuscript_v1.md` | first audit (found the missing differential prediction) |
| `AUDIT2_manuscript_v3.md` | **second audit** (found the disattenuation trap) |
| `RESPOSTA_revisao_gemini.md` | response to external review (the estimator correction) |
| `TODO1_mcphases_frontier_conclusion.md` | the instrument analysis write-up |
| `registered_test_conclusion.md` | the registered-test power write-up |
| `FIG1_mecanismo_fronteira.png`, `FIG4_phase_locked.png`, `FIG5_estimador_objetivo.png` | **current figures** (Portuguese, in the manuscript) |
| `figure_h4_design_frontier.png` | **superseded** by FIG1 — this is the version *without* the detection-only hatching. Do not use. |
| `figure_mcphases_on_frontier.png`, `figure_instrument_frontier.png` | Figures 2 and 3 as they appear in the manuscript |
| `build_paper.js` | the ABNT manuscript builder (Node + docx) |
| **`RAP_idiographic_forecasting_ABNT.*`, `step0_inventory.py`, `step1_ergodicity_test.py`, `build_abnt.js`, `*_inventory.csv`** | **These belong to the OTHER paper** (the actigraphy pre-registration, OSF `ewyp7`). Keep them in a separate project folder. |
| `step1_ergodicity.py`, `run_step0.py`, `reader.py` | earlier/duplicate versions of the actigraphy scripts — superseded |

---

## Before submission (blocking)

- [ ] Re-run all grids at **≥500 replicates**.
- [ ] Complete the mcPHASES author list from the PhysioNet dataset page.
- [ ] Deposit this package (Zenodo/OSF) and put the DOI in the manuscript.
