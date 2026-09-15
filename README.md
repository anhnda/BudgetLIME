# BudgetLIME — Finite-Budget Certification for LIME-Style Local Surrogates

Reference implementation for the paper *Finite-Budget Certification for LIME-Style
Local Surrogates*. The code computes a **detection floor** for the dense
ordinary-least-squares core of degree-`K` LIME surrogates: coefficients whose
fitted magnitude clears the floor have **certified signs** at the chosen query
budget; coefficients below it are reported as **unresolved** rather than zero.

The whole codebase is one inequality (Theorem 1) read in two directions. The
certified floor carries the two Bernstein leakage terms **explicitly** (nothing
absorbed into a sub-1 constant):

```
||beta_hat - beta||_inf  <=  floor(N, rho)
floor(N, rho) = C_est * [ sigma_obs*sqrt(2L/N)          # query noise
                          + sqrt(2*m_>K*L/N)            # leakage, sub-Gaussian
                          + (2/3)*B*L/N ]               # leakage, sub-exponential
L = log(SPLIT * pK / delta),   C_est measured per run,   B = ||r_>K||_inf
```

* **Forward** — the guarantee: `|beta_hat_S| > floor`  ⇒  the sign of `beta_S` is correct.
* **Backward** — the budget rule (Eq. 8), a planning heuristic: `N ≳ 2 C_budget² σ_eff² log(SPLIT·pK/δ) / β_min²` with `σ_eff = σ_obs + C_m√m_>K`.

Note the asymmetry: the **certified floor** uses `C_est` (per-run) and the two
explicit Bernstein terms; the **budget prediction** uses the empirical `C_m` /
`C_budget` planning constants. `C_m` never enters a certified radius.

Everything that is neither a forward sign check nor a backward budget check
(set nesting, count monotonicity) is treated as a *workflow diagnostic* and
reported separately, never as theorem evidence.

---

## Repository structure

The code separates a single pure-NumPy numerical core from the only Torch-dependent
file (the black-box model wrappers). Nothing heavy runs on import; a model is
constructed only inside a driver's `main()` after arguments are supplied.

| File | Torch? | Role |
|------|--------|------|
| `bl_core.py` | no | The shared numerical core. Degree-`K` Walsh feature machinery (`p_K`, `feature_subsets`, `design_matrix`, `standardize_columns`, `sample_masks`), dense OLS (`ols_fit`), the floor (`sigma_eff`, `floor_value`, `certified_set`), the budget rule (`predict_budget`, `feasibility_floor`, `plan_budget`), pilot estimation of `sigma_eff` (`estimate_mismatch_from_residual`, `pilot_N0`), and the prefix-ladder diagnostic (`sweep_prefix_ladder`). The frozen planning constants (`C_M`, `C_BUDGET`) plus the per-run `C_est` live here in `CONSTANTS` and are frozen after Tier-1 calibration. |
| `bl_models.py` | **yes** | The only Torch code. Two query-only black boxes: `TextClassifier` (sentence + token mask → class probability, `σ_obs > 0`) and `ImageClassifier` (image + cell mask → class logit, `σ_obs ≈ 0`). These map `(input, binary mask) → model output` and nothing else; all certification math is in `bl_core.py`. |
| `tier1_synthetic.py` | no | **Tier 1 — synthetic, ground truth known.** Calibrates the two constants and proves the two-direction collapse: leakage linchpin (fixes `C_M`), forward SDR-collapse curve, backward budget-constant recovery (fixes `C_BUDGET`), and the regime grid spanning the two axes of `σ_eff`. This is the *only* place constants are fit. |
| `tier1b_cert_transfer.py` | no | **Tier 1b — Cest transfer study (referee R1.4).** Design-only (no model): measures the forward floor constant `Cest = max{γ^{-1/2}, ‖Σ̂⁻¹‖∞}` as a function of the coordinate-to-budget ratio `pK/N`. Three questions: (Q1) confirms Cest grows with `pK` at fixed `N` (the referee's premise); (Q2) tests whether Cest **collapses onto one curve in `pK/N`** across very different `(d, K)` — the condition under which a frozen value transfers; (Q3) evaluates Cest at the actually deployed points (`d=30`/`K=1`, `d=49`/`K=1`, `K=2` enumeration) and reports the gap against the frozen `C_FLOOR`. |
| `tier2_blackbox.py` | yes | **Tier 2 — black-box classifiers.** With constants frozen, tests the guarantee on real query-only models: forward sign-flip stability over a prefix-nested budget ladder, backward budget planning, and the **exact-β sign-correctness check** (enumerate the full `2^d` mask cube for short inputs, `d ≤ 13`, to recover the exact projection and directly verify signs). |
| `tier2b_reseed.py` | yes | **Tier 2b — independent-reseeding audit.** Breaks the shared-mask coupling of the nested ladder: at a single fixed `N`, runs `R` independent seeds to measure (A) the cross-seed sign-violation rate vs the `1/pK` target and (B) the stratified Jaccard stability of the certified set above vs inside the unresolved band. Includes a `selftest` mode (pure NumPy, no models). |
| `tier3_feasibility.py` | no | **Tier 3 — feasibility (Reading 2).** Shows why `K=2` costs *more* despite *lower* noise: moving pairwise structure into the fit lowers `σ_eff`, but `pK` jumps `~ d²/2`, lifting the feasibility floor `~ pK`. Produces the resolution-budget vs feasibility-floor crossing curves. |
| `baselines.py` | yes | Compares the floor against a per-coordinate bootstrap CI and the single-coordinate Wald interval on the **same** fit and mask bank, isolating the certification criterion. Demonstrates the Wald degeneracy as `σ_obs → 0` that the mismatch term `C_m√m` repairs. |

### The two frozen constants (`bl_core.CONSTANTS`)

| Constant | Default | Role |
|----------|---------|------|
| `C_M` | `0.795` (mean over d∈{15,24,30,49}, frozen from Tier-1 large-N rows, N ≥ 2000) | **Empirical planning/calibration quantity only — NOT a certificate constant.** The certified floor now carries the two Bernstein leakage terms explicitly (`√(2mL/N)+(2/3)BL/N`), so `C_M` no longer sets any certified radius; it enters only the backward budget rule (`predict_budget`) and pilot reporting. Was absorbed into `σ_eff` pre-fix, which under-covered the leading term since 0.795<1. |
| `C_FLOOR` | `1.0` | Floor-bound constant (forward); theory = 1 for the orthonormal ±1 design, empirical ≥ 1 expected. |
| `C_BUDGET` | `1.508` (mean over d∈{15,24,30,49}) | Budget-rule constant (backward), back-solved at Tier 1 under the split log factor (R1.2). Re-calibrate after any floor/estimator change. |

`C_FLOOR` (the bound) and `C_BUDGET` (the budget invert) are deliberately kept as
separate objects: inverting the budget with `C_FLOOR` lands below the feasibility
floor and is physically meaningless.

---

## Correctness fixes (revision R1.2 / R1.5 / R1.6)

Three referee correctness points are implemented at the single source of truth
(`bl_core.py`), so every tier inherits them without re-deriving anything. All
three are auditable against the pre-revision numbers.

* **R1.2 — δ-budget accounting.** Theorem 1 was stated at `1 − δ` while its proof
  spends several `1 − δ` events (design conditioning, the sub-Gaussian
  query-noise maximum, and the Lemma 1 mismatch bound). The floor now carries
  the union-bounded factor `log(SPLIT·pK/δ)` computed once in
  `bl.log_pk_over_delta(...)` and routed through `floor_value`, `predict_budget`,
  and every downstream comparator. `DELTA_SPLIT` (default **3**) counts the
  probabilistic events; the pilot scale is handled as Theorem 1's explicit
  *conditioning hypothesis* (set `DELTA_SPLIT = 4` to also budget the pilot
  event). With `δ = 1/pK` this turns the old `2 log pK` into
  `2 log pK + log SPLIT`, a bounded additive correction that vanishes as `pK`
  grows (`< 8%` on the floor at `pK = 50`, `< 2%` at `pK ≈ 1225`). Passing
  `split=1, delta=1/pK` reproduces the old factor exactly.

* **R1.5 — sub-exponential Bernstein term in Lemma 1.** The leakage bound is
  Bernstein, not purely sub-Gaussian:
  `η_N ≤ sqrt(2 m log(.)/N) + (2/3) B log(.)/N` with `B = ‖r_>K‖_∞`. Both terms
  are now computed (`bl.leakage_bound_terms`), and the domination regime
  `N ≳ (2B²/9m) log(.)` (`bl.leakage_domination_N`) is *verified* rather than
  asserted. Tier 1's `leakage` calibration prints the sub-exponential term,
  `N_dom`, and a per-cell `dom?` flag; Tier 2 reports the fraction of items run
  inside the dominated regime (`dom%` column).

* **R1.6 — negative mismatch estimate + clipping bias.** The held-out mismatch
  estimate is a difference of variances and can be negative.
  `bl.estimate_mismatch_detail` now returns the **raw** (possibly negative)
  value, the clipped value fed to the floor, a `negative` flag, and a plug-in
  `B_hat` for the R1.5 check. Clipping at zero only *raises* `σ_eff`
  (conservative); the raw value is retained solely for reporting. Tier 2 logs
  the per-cell fraction of pilot draws with `m_hat < 0` (`neg%` column),
  expected nonzero mainly in near-degenerate / deterministic cells — the
  conditional-clause mechanism of Theorem 1 (the ViSoBERT/zero cell).

Reproducing the affected numbers:

```bash
# R1.2 + R1.5: recalibrated C_M / C_BUDGET under the split log factor,
#              with the Bernstein-domination table.
python tier1_synthetic.py leakage      # C_m + sub-exp term + N_dom + dom? flags
python tier1_synthetic.py backward     # C_budget back-solved vs the new floor

# R1.5 + R1.6: neg% and dom% columns in the Tier-2 guarantee table.
python tier2_blackbox.py image --beta_min 0.05 --N_ladder 512,1000,2000,4000 \
    --images_dir image_samples --glob "*.JPEG"
python tier2_blackbox.py nlp --K 1 --references mask,pad,zero \
    --backbones distilbert,roberta,visobert --beta_min 0.02 \
    --N_ladder 512,1000,2000,4000 --sentences text_samples/sst2_short.txt
```

To reproduce the **pre-revision** floor for a direct before/after comparison,
call the floor with `split=1`:
`bl.floor_value(s_eff, d, N, K, split=1)`.

### R1.4 — the floor constant is measured per run, not frozen

The referee's most serious point was that `Cest = max{γ^{-1/2}, ‖Σ̂⁻¹‖∞}` — the
constant in the forward floor (Eq. 6) — is a max absolute row sum of the inverse
Gram that grows with `pK`, yet was calibrated at `(d=30, K=1)` and frozen for
`d=49` and `K=2`. The Tier-1b study (`tier1b_cert_transfer.py`) settles it
empirically:

* **Q1** confirms `Cest` grows with `pK` at fixed `N` (the concern is real).
* **Q2** shows `Cest` does **not** collapse onto one curve in `pK/N`: at a fixed
  ratio it still splits by `pK` (40–55% within-bin spread across `(d, K)`), so
  no single frozen scalar is correct.
* **Q3** measures the empirical `Cest` at the deployed points: **1.5–3.8×**, not
  the frozen `C_FLOOR = 1.0`. A forward floor computed at `C_FLOOR = 1.0` is
  therefore anti-conservative by those factors.

**Resolution (Path A).** `Cest` is now **measured per run** from the realized
design (`bl.realized_cest(Z, K)` / `bl.floor_from_design(Z, s_eff, K)`) — a pure,
model-free, reference-free quantity read off the same Gram the OLS fit uses, in
milliseconds. Every certified decision (Tier 2 forward ladder, the exact-β check,
Tier 2b reseeding, the baseline comparison) now uses `floor_from_design`, so
there is **no transfer claim to defend**: nothing is transferred, each run reports
its own design constant. `C_FLOOR` is retained only as the orthonormal ideal and
as a fallback when the Gram is too ill-conditioned to invert. The Tier-2 report
prints, per cell, the realized `Cest` and the floor-inflation factor vs the old
frozen-`C_FLOOR` floor, so the before/after is visible.

> **Consequence to expect on re-run.** The realized floor is larger than the old
> frozen floor by ~1.5–3.8×, most at low budgets / high `pK/N`. Certified sets
> will shrink accordingly — this is the method reporting its honest resolution,
> not a regression. The forward sign-flip and exact-β guarantees must be
> re-generated under this floor before being reported.

---

## Correctness fixes (this revision batch — floor / estimator / pilot / enumeration)

These change the floor and the estimator, so **all tables must be regenerated**
and the two planning constants **re-calibrated** (Tier 1) before any number is
reported.

* **Two-term certified floor.** The floor no longer folds the leakage into
  `C_m√m · sqrt(2L/N)` (which only upper-bounds the leading Lemma-1 term when
  `C_m ≥ 1`, whereas the calibrated `C_m = 0.795 < 1`). `certified_floor` /
  `floor_from_design(..., sigma_obs, m_hat, B)` carry
  `C_est·[σ_obs√(2L/N) + √(2mL/N) + (2/3)BL/N]`. The `B/N` sub-exponential term
  is carried at all `N`, so no "dominated throughout `N ≳ pK`" assumption is
  needed. `C_m` survives only as the backward planning quantity.
* **Intercept-augmented OLS / Gram.** `p_K` counts the intercept (`S = ∅`), so
  `ols_fit`, the Gram, `C_est` (`realized_cest` / `measure_conditioning`) and the
  baselines now use the full augmented design `[1, χ_i, χ_iχ_j, …]`
  (`design_matrix(..., intercept=True)`). Regressing `y − ȳ` on un-centered
  Walsh columns is only equal to full OLS at the population, not at finite `N`.
* **Upper-confidence pilot.** `sigma_eff_ucb` now (a) spends `δ/4 = 1/(4pK)` on
  the pilot event (`DELTA_SPLIT_UCB = 4`), not the whole `1/pK`, and (b) uses a
  **known** range bound `B_known` for the empirical-Bernstein term (probability
  outputs ⇒ `B_known = 1`) instead of the random sample maximum. Cross-fitting is
  used for **every** `K` (not only `K = 2`).
* **Enumeration = i.i.d. with replacement.** The exact-β tier caches the full
  cube once (`exact_cube`) and draws i.i.d. cube indices **with replacement**
  (`iid_from_cube`) for the pilot and the run, instead of taking a prefix of a
  shuffled cube (which is sampling *without* replacement). Scoring
  (`false_sign_rate`) now checks **every** run-certified coordinate against
  `sgn(β_exact)` (with a zero-tolerance), not only the run∩exact intersection.
* **No anti-conservative fallback.** `realized_cest` **raises** on an
  ill-conditioned Gram (run unresolved) instead of returning `C_FLOOR = 1`;
  `ols_fit` and `realized_cest` share one cutoff `COND_MAX = 1e8`.
* **Deterministic NLP `σ_obs`.** A `.eval()` forward pass is deterministic, so
  `σ_obs ≈ 0` and the NLP floor is mismatch-driven, like the image tier — the
  probability output does not by itself make `σ_obs > 0`. Genuine query-noise
  regimes are the synthetic tier (or deliberately stochastic inference).

> **Re-calibration required.** Because the floor and the estimator changed, the
> two planning constants were re-calibrated by running `python tier1_synthetic.py all`
> under this code and reading the new `C_M`, `C_BUDGET` and the `C_est` d-sweep.
> The values now in `bl_core.Constants` (`C_M = 0.795`, `C_BUDGET = 1.508`, mean
> over d∈{15,24,30,49}) are those re-calibrated numbers; re-run Tier 1 and update
> `Constants` again after any further floor/estimator change before regenerating
> Tier 2 / 2b / 3 / baselines.

---

The numerical core and the synthetic / feasibility / self-test paths need only
**NumPy**. The black-box drivers additionally need PyTorch, torchvision,
transformers, Pillow, and (optionally) SciPy and tqdm.

```bash
# Core only — runs Tier 1, Tier 3, and the Tier-2b self-test
pip install numpy

# Full — adds the real black-box models (Tier 2 / 2b / baselines)
pip install numpy torch torchvision transformers pillow scipy tqdm
```

> **Note on PyTorch:** the model-backed drivers (`tier2_blackbox.py`,
> `tier2b_reseed.py nlp|image`, `baselines.py`) import Torch and will download
> Hugging Face / torchvision weights on first use. The NumPy-only paths below
> never touch Torch.

GPU is used automatically if available (`cuda`), otherwise CPU.

### Data layout expected by the black-box drivers

The drivers read inputs from plain files; defaults can be overridden on the CLI:

```
text_samples/sst2_short.txt     # one sentence per line (Tier 2 NLP default)
sst2_samples.txt                # one sentence per line (Tier 2b / baselines default)
image_samples/*.JPEG            # input images (Tier 2 image default)
benchmark_50/*.JPEG             # input images (Tier 2b / baselines image default)
```

Short sentences (`d ≤ 13` free tokens) are required for the exact-β check, since
it enumerates the full `2^d` mask cube.

---

## Quick start

### 1. NumPy-only (no Torch, no downloads)

```bash
# Tier 1: calibrate the two constants and show the two-direction collapse
python tier1_synthetic.py all          # or: leakage | forward | backward | grid

# Tier 3: feasibility — why K=2 costs more with less noise
python tier3_feasibility.py

# Tier 2b pipeline end-to-end on synthetic mock probes (no models)
python tier2b_reseed.py selftest
```

`python tier1_synthetic.py leakage` reproduces `C_m ≈ 0.80` (revised Table 2,
frozen from the large-N rows under the R1.2 split normalizer; the all-N mean
0.78 is printed alongside for transparency, and the small-N drift is the R1.5
sub-exp term). The full `all` run also recovers `C_budget ≈ 1.51`.

### 2. Black-box, using the default model

The default model in `tier2_blackbox.py` is **DistilBERT on SST-2**
(`distilbert-base-uncased-finetuned-sst-2-english`), default `--dataset sst2`.
The minimal run uses every default (`--K 1`, `β_min = 0.02`,
`--N_ladder 512,1000,2000,4000`, references `mask,pad,zero`):

```bash
# Tier 2 forward + backward on the default NLP model (DistilBERT / SST-2)
python tier2_blackbox.py nlp \
    --backbones distilbert \
    --sentences text_samples/sst2_short.txt
```

Exact-β sign-correctness check (the strongest, direct correctness test) on the
same default model, at `K=2` so pairwise interactions are certified:

```bash
python tier2_blackbox.py exact-nlp --K 2 --subset 10 --max_d 13 \
    --sentences text_samples/sst2_short.txt
```

Independent-reseeding audit on the default model at a single fixed budget:

```bash
python tier2b_reseed.py nlp --backbones distilbert --references mask \
    --N 2000 --R 40 --K 1 --subset 10 --sentences sst2_samples.txt
```

Baseline comparison (floor vs bootstrap CI vs Wald) on the default model:

```bash
python baselines.py nlp --backbones distilbert --references mask \
    --N 2000 --B 200 --subset 10 --sentences sst2_samples.txt
```

---

## Full reproduction

### Tier 1 — calibration (synthetic)

```bash
python tier1_synthetic.py all
```

Outputs: `C_M` from the leakage linchpin, the forward SDR-collapse crossing
`x_0.5` (a point on the shared curve; `x_0.5 < 1` is expected), `C_BUDGET` from
the backward rule, and a calibration summary. These constants are then **frozen**
and reused verbatim downstream.

### Tier 2 — black-box guarantee (frozen constants)

NLP (probabilistic backbones, exercise the query-noise half of `σ_eff`):

```bash
python tier2_blackbox.py nlp --K 1 --references mask,pad,zero \
    --backbones distilbert,roberta,visobert --beta_min 0.02 \
    --N_ladder 512,1000,2000,4000 --sentences text_samples/sst2_short.txt
```

Images (deterministic backbones, exercise the mismatch half of `σ_eff`; `7×7`
grid → `d = 49`):

```bash
python tier2_blackbox.py image --beta_min 0.05 --N_ladder 512,1000,2000,4000 \
    --backbones resnet50,resnet18,vit_b_16 --references white,black,mean \
    --images_dir image_samples --glob "*.JPEG"
```

Exact-β check, NLP `K=2` (enumerable short sentences only):

```bash
python tier2_blackbox.py exact-nlp --K 2 --subset 10 --max_d 13 \
    --sentences text_samples/sst2_short.txt
```

Visualize
```bash
cd visualize
./visualize.sh
```

The Tier-2 report separates the **guarantee** (certified sign flips / checks,
target 0) and the **backward** realized/target ratio (target ≤ 1) from the
appendix **diagnostics** (count-monotone, set-nested), which are near-guaranteed
by the prefix construction.

### Tier 2b — independent reseeding

```bash
# NLP, weakest-signal cell
python tier2b_reseed.py nlp --backbones visobert --references zero \
    --N 2000 --R 40 --K 1 --subset 10 --sentences sst2_samples.txt

# Image, strongest-signal cell (deterministic backbone)
python tier2b_reseed.py image --backbones resnet50 --references mean \
    --N 2000 --R 40 --subset 10 --images_dir benchmark_50 --glob "*.JPEG"
```

Reports the cross-seed sign-violation rate vs `1/pK` and the above-band vs
in-band Jaccard, plus a `booktabs` LaTeX table for the appendix.

### Tier 3 — feasibility

```bash
python tier3_feasibility.py
```

### Baselines

```bash
python baselines.py image --backbones resnet50 --references mean \
    --N 2000 --B 200 --subset 10 --images_dir benchmark_50 --glob "*.JPEG"
```

---

## Key CLI flags

| Flag | Drivers | Meaning |
|------|---------|---------|
| `mode` (positional) | all black-box | `nlp` / `image` / `exact-nlp` / `selftest` as applicable. |
| `--K` | tier2, tier2b, baselines | Surrogate degree (1 = main effects, 2 = + pairwise). |
| `--backbones` | all black-box | Comma list or `all`. NLP: `distilbert,roberta,visobert`. Image: `resnet50,resnet18,vit_b_16`. |
| `--references` | all black-box | Comma list or `all`. NLP: `mask,pad,zero`. Image: `white,black,mean`. |
| `--dataset` | NLP drivers | Selects the fine-tuned checkpoint (default `sst2`). |
| `--beta_min` | tier2 | Target resolution level (default `0.02` NLP, `0.05` image). |
| `--N_ladder` | tier2 | Prefix-nested budgets (default `512,1000,2000,4000`). |
| `--N` | tier2b, baselines | Single fixed budget. |
| `--R` | tier2b | Number of independent seeds. |
| `--B` | baselines | Bootstrap resamples. |
| `--subset` | all black-box | Cap items per cell. |
| `--max_d` | exact-nlp | Enumerate the cube only when `d ≤ max_d` (`2^max_d` calls/probe); larger `d` is skipped, never approximated. |
| `--grid` | image drivers | Superpixel grid (default `7` → `d = 49`). |

---

## Design notes

* **One numerical core.** Every tier calls `bl_core.py` for the floor, OLS,
  certified set, and `σ_eff`; the drivers only supply the black-box query closure.
* **Constants fit once.** `(C_M, C_BUDGET)` are calibrated in Tier 1 and frozen;
  nothing downstream re-fits them.
* **`σ_eff` is the only data-dependent input** to the floor and is estimated from
  a cross-fitted pilot (`pilot_N0 = max(500, 6 pK)`); the guarantee is conditional
  on this pilot upper-bounding the true scale.
* **No exact-β fallback.** When `d > 13` the cube is not enumerable, so the
  probe is skipped for the exact check rather than substituting a random-bank
  estimate as "ground truth".

## Citation 

See the paper for the full
author list and references.