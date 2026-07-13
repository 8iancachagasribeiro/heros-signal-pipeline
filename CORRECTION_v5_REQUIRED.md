# CORRECTION REQUIRED IN MANUSCRIPT v5

**Found by the reproducibility package, 11 July 2026.**

## The error

Table 7's SSF values for the OBJECTIVE (wearable) measures were computed on a
**different and invalid subset**: only the days paired with an estrogen reading. Those
days contain gaps (~4% non-consecutive). **The FFT assumes regularly-spaced sampling**;
concatenating non-consecutive days distorts the spectrum and therefore the noise floor.

The AR(1) and ACF-linear columns, meanwhile, were computed on the FULL daily series.
**Table 7 therefore compares estimators across different data.**

## Correct values (all estimators, same full daily series)

| measure | AR(1) | ACF-lin | **SPECTRAL** | manuscript said |
|---|---|---|---|---|
| E3G (predictor) | 0.587 | 0.538 | **0.469** | 0.469 ✓ |
| fatigue | 0.500 | 0.381 | **0.321** | 0.321 ✓ |
| mood swing | 0.637 | 0.478 | **0.326** | 0.326 ✓ |
| cramps | 0.964 | 0.688 | **0.494** | 0.494 ✓ |
| **resting heart rate** | 0.595 | 0.480 | **0.474** | **0.772 ✗ WRONG** |
| **skin temperature** | 0.442 | 0.392 | **0.336** | **0.574 ✗ WRONG** |

**Attenuation is UNCHANGED at 0.389.** Every core claim of the paper survives: the
predictor and outcome SSFs, the attenuation cascade, the power collapse, the budget
result, the disattenuation trap. **Sections 1-7.7 and 8-11 require no change.**

## The consequence: SECTION 7.8 MUST BE REWRITTEN

§7.8 currently argues that the paper's prescription is *demonstrated inside the dataset*:
"change the instrument and the signal appears."

**With the correct values, that argument does not hold.**

- Skin temperature **shows** coupling (p = .027) with SSF = **0.336**.
- Fatigue **shows nothing** with SSF = **0.321**.
- These are nearly identical. Instrument quality cannot explain the contrast.
- Resting heart rate has the **best** SSF of all outcomes (0.474) and shows **no**
  coupling at all.

**The honest reading reverses:** what distinguishes skin temperature from fatigue is
**not the instrument — it is the construct**. Skin temperature is genuinely cycle-locked
(almost certainly via progesterone's luteal rise); self-reported daily fatigue is not.

### Recommended rewrite of §7.8

Retitle from *"the prescription demonstrated within the dataset"* to something like
*"objective versus self-report: the construct, not the instrument"*, and state:

1. Instrument quality is **necessary but not sufficient** — resting HR has the best SSF
   and shows nothing. (This was already in the manuscript and remains correct.)
2. Instrument quality is **not the operative variable here** — temperature's SSF is
   barely above fatigue's, yet only temperature couples. (**NEW; replaces the old claim.**)
3. Skin temperature's coupling is plausibly **progesterone-mediated**, not estradiol-
   mediated. (Already flagged as a caveat; now it moves to the centre of the
   interpretation.)
4. The result therefore does **not** demonstrate the design prescription. It demonstrates
   that a cycle-locked construct can be found where a self-reported one cannot — which is
   suggestive for future instrument choice, but is **not** evidence that better
   instruments would have rescued the fatigue analysis.

This is a smaller claim. It is the one the data support.

### Also update

- **Abstract**: remove "Demonstramos a prescrição dentro do próprio conjunto de dados".
- **Discussion §9**: the sentence crediting §7.8 as demonstrating the prescription.
