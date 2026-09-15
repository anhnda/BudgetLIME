# BudgetLIME — Finite-Budget Certification for LIME-Style Local Surrogates

Reference implementation for the paper *Finite-Budget Certification for LIME-Style
Local Surrogates*. The code computes a **detection floor** for the dense
ordinary-least-squares core of degree-`K` LIME surrogates: coefficients whose
fitted magnitude clears the floor have **certified signs** at the chosen query
budget; coefficients below it are reported as **unresolved** rather than zero.

## The idea

LIME-style local surrogates estimate feature effects from a finite number of
masked-model queries, so fitted coefficients are estimates, not exact effects.
With only a few hundred or a few thousand masks, a small real contribution can be
hidden by sampling fluctuation, while noise or surrogate mismatch can make a weak
coordinate look important. A finite-budget explanation should therefore report not
only its coefficients but the **resolution** at which they can be trusted.

This codebase studies the dense degree-`K` OLS core of first- and second-order
surrogates under uniform product masks with no locality kernel. In that setting
the estimand is a Walsh–Fourier / Banzhaf-type object: the low-degree centered
Walsh projection of the reference-conditioned masked response `g_rho`, not the
proximity-weighted target of canonical kernel LIME. Everything reduces to one
inequality (Theorem 1), read in two directions. The certified floor keeps the two
Bernstein leakage terms **explicit** (nothing folded into a sub-1 constant):

```
||beta_hat - beta||_inf  <=  floor(N, rho)
floor(N, rho) = C_est * [ sigma_obs*sqrt(2L/N)          # query noise
                          + sqrt(2*m_>K*L/N)            # leakage, sub-Gaussian
                          + (2/3)*B*L/N ]               # leakage, sub-exponential
L = log(SPLIT * pK / delta),   C_est measured per run,   B = ||r_>K||_inf
```

* **Forward** — the guarantee: `|beta_hat_S| > floor`  ⇒  the sign of `beta_S` is correct. Every true effect with `|beta_S| > 2·floor` is detected.
* **Backward** — the budget rule (Eq. 8), a planning heuristic: `N ≳ 2 C_budget² σ_eff² log(SPLIT·pK/δ) / β_min²` with `σ_eff = σ_obs + C_m√m_>K`.

Note the asymmetry: the **certified floor** uses `C_est` (per-run) and the two
explicit Bernstein terms; the **budget prediction** uses the empirical `C_m` /
`C_budget` planning constants. `C_m` never enters a certified radius. The floor
is a **resolution limit, not a sparsity threshold**: sub-floor coefficients are
unresolved, not zero. It is also **reference-relative** — a smaller floor under a
different reference certifies a different reference-conditioned target, not a more
faithful version of the same one.

Everything that is neither a forward sign check nor a backward budget check
(set nesting, count monotonicity) is treated as a *workflow diagnostic* and
reported separately, never as theorem evidence.

## Formulation

Fix an input `x`, a target output coordinate, and a partition of `x` into `d`
interpretable units. A reference operator `rho` fills masked-out units, giving the
masked response `g_rho(z) = f_c(Phi_rho(x, z))` for a binary mask `z`. Using the
centered Walsh coding `chi_S(z) = prod_{i in S} 2(z_i - 1/2)` with masks drawn
i.i.d. from the uniform product measure, the characters are orthonormal and
`g_rho` expands in the Walsh basis. A degree-`K` surrogate keeps the `pK = sum_{k<=K} C(d,k)`
terms with `|S| <= K`: main effects at `K = 1`, plus pairwise interactions at
`K = 2`.

* **Estimand.** `beta_{<=K, rho}` is the coefficient vector of the population
  L²(µ) projection of `g_rho` onto the degree-`K` span. The mismatch residual
  `r_{>K, rho}` collects the higher-order part the surrogate cannot represent;
  its energy is `m_{>K, rho} = ||r_{>K, rho}||²`. This residual is deterministic
  once `(f, x, rho)` is fixed and population-orthogonal to every fitted character.

* **Observation model.** We observe `y_t = g_rho(z_t) + eps_t`, where `eps_t` is
  query noise (scale `sigma_obs`). The dense OLS estimator decomposes into the
  target plus a **mismatch-leakage** term and a **query-noise** term:
  `beta_hat = beta + Sigma_hat^{-1}(X'r/N) + Sigma_hat^{-1}(X'eps/N)`. The
  query-noise term vanishes in expectation and concentrates at rate
  `sigma_obs·sqrt(log pK / N)`; the leakage is population-zero but nonzero at
  finite `N` and scales with `sqrt(m_{>K, rho})`. Both defeat a budget, and the
  floor accounts for both.

Two regularity conditions carry the certificate: design conditioning (the realized
Gram is invertible and well-conditioned, from which the per-run `C_est` is read)
and concentration (mean-zero sub-Gaussian query noise, bounded mismatch features).
These are exactly what is needed to estimate every coordinate of the dense
surrogate — no sparsity or incoherence assumptions. The floor's failure budget
`delta` is split across the events (`nu = 3` base, `nu = 4` with the mismatch
pilot). Appendix C supplies a one-sided empirical-Bernstein upper-confidence bound
for `m_{>K, rho}`; a Ridge-penalized variant is derived in Appendix E.

## Repository structure

The code separates a single pure-NumPy numerical core from the only Torch-dependent
file (the black-box model wrappers). Nothing heavy runs on import; a model is
constructed only inside a driver's `main()` after arguments are supplied.

| File | Torch? | Role |
|------|--------|------|
| `bl_core.py` | no | The shared numerical core. Degree-`K` Walsh feature machinery (`p_K`, `feature_subsets`, `design_matrix`, `standardize_columns`, `sample_masks`), dense OLS (`ols_fit`), the floor (`sigma_eff`, `floor_value`, `certified_set`), the budget rule (`predict_budget`, `feasibility_floor`, `plan_budget`), pilot estimation of `sigma_eff` (`estimate_mismatch_from_residual`, `pilot_N0`), and the prefix-ladder diagnostic (`sweep_prefix_ladder`). The frozen planning constants (`C_M`, `C_BUDGET`) plus the per-run `C_est` live here in `CONSTANTS`. |
| `bl_models.py` | **yes** | The only Torch code. Two query-only black boxes: `TextClassifier` (sentence + token mask → class probability, `σ_obs > 0`) and `ImageClassifier` (image + cell mask → class logit, `σ_obs ≈ 0`). These map `(input, binary mask) → model output` and nothing else; all certification math is in `bl_core.py`. |
| `tier1_synthetic.py` | no | **Tier 1 — synthetic, ground truth known.** Calibrates the two constants and shows the two-direction collapse: leakage linchpin (fixes `C_M`), forward SDR-collapse curve, backward budget-constant recovery (fixes `C_BUDGET`), and the regime grid spanning the two axes of `σ_eff`. This is the *only* place constants are fit. |
| `tier1b_cert_transfer.py` | no | **Tier 1b — `C_est` transfer study.** Design-only (no model): measures the forward floor constant `C_est = max{γ^{-1/2}, ‖Σ̂⁻¹‖∞}` as a function of the coordinate-to-budget ratio `pK/N`. Confirms `C_est` grows with `pK` at fixed `N`, that it does **not** collapse onto one curve in `pK/N` across different `(d, K)`, and reports its value at the deployed points against the ideal `C_FLOOR`. |
| `tier2_blackbox.py` | yes | **Tier 2 — black-box classifiers.** With constants frozen, tests the guarantee on real query-only models: forward sign-flip stability over a prefix-nested budget ladder, backward budget planning, and the **exact-β sign-correctness check** (enumerate the full `2^d` mask cube for short inputs, `d ≤ 13`, to recover the exact projection and directly verify signs). |
| `tier2b_reseed.py` | yes | **Tier 2b — independent-reseeding audit.** Breaks the shared-mask coupling of the nested ladder: at a single fixed `N`, runs `R` independent seeds to measure (A) the cross-seed sign-violation rate vs the `1/pK` target and (B) the stratified Jaccard stability of the certified set above vs inside the unresolved band. Includes a `selftest` mode (pure NumPy, no models). |
| `tier3_feasibility.py` | no | **Tier 3 — feasibility.** Shows why `K=2` costs *more* despite *lower* noise: moving pairwise structure into the fit lowers `σ_eff`, but `pK` jumps `~ d²/2`, lifting the feasibility floor `~ pK`. Produces the resolution-budget vs feasibility-floor crossing curves. |
| `baselines.py` | yes | Compares the floor against a per-coordinate bootstrap CI and the single-coordinate Wald interval on the **same** fit and mask bank, isolating the certification criterion. Demonstrates the Wald degeneracy as `σ_obs → 0` that the mismatch term `C_m√m` repairs. |

### The frozen constants (`bl_core.CONSTANTS`)

| Constant | Default | Role |
|----------|---------|------|
| `C_M` | `0.795` (mean over d∈{15,24,30,49}, frozen from Tier-1 large-N rows, N ≥ 2000) | **Empirical planning quantity only — NOT a certificate constant.** The certified floor carries the two Bernstein leakage terms explicitly (`√(2mL/N)+(2/3)BL/N`), so `C_M` sets no certified radius; it enters only the backward budget rule (`predict_budget`) and pilot reporting. |
| `C_FLOOR` | `1.0` | Floor-bound constant (forward); theory = 1 for the orthonormal ±1 design, empirical ≥ 1 expected. Retained as the orthonormal ideal and a fallback. |
| `C_BUDGET` | `1.508` (mean over d∈{15,24,30,49}) | Budget-rule constant (backward), back-solved at Tier 1 under the split log factor. |

`C_FLOOR` (the bound) and `C_BUDGET` (the budget invert) are deliberately kept as
separate objects: inverting the budget with `C_FLOOR` lands below the feasibility
floor and is physically meaningless.

The forward floor constant `C_est = max{γ^{-1/2}, ‖Σ̂⁻¹‖∞}` is **measured per run**
from the realized design (`bl.realized_cest(Z, K)` / `bl.floor_from_design(Z, s_eff, K)`)
— a pure, model-free, reference-free quantity read off the same Gram the OLS fit
uses. It grows with `pK` (empirically 1.5–3.8× the ideal 1.0 at deployed points)
and does not collapse onto a single `pK/N` curve, so it is never frozen or
transferred: each run reports its own design constant. `realized_cest` **raises**
on an ill-conditioned Gram (run unresolved) rather than falling back to `C_FLOOR`;
`ols_fit` and `realized_cest` share one cutoff `COND_MAX = 1e8`.

## Results

The floor is validated across three tiers.

* **Synthetic (Tier 1).** Population coefficients are known, so the two planning
  constants are calibrated and the floor's scale is checked. `C_M` and `C_BUDGET`
  are empirically stable across `d ∈ {15, 24, 30, 49}` (cross-`d` CoV 0.042 and
  0.035) and across the noise / mismatch / budget regimes, while the per-run
  `C_est` rises monotonically with `d` (CoV 0.192, 1.44 → 2.26) and cannot be a
  single transferred scalar.

* **Black-box (Tier 2).** On ResNet-50/18 and ViT-B/16 (ImageNet, `7×7` grid,
  `d = 49`) and DistilBERT / RoBERTa (SST-2) and ViSoBERT (VSFC), with the
  leakage/budget constants frozen and `C_est` measured per run: **all nine image
  cells and all nine NLP cells show zero forward sign reversals** as the budget
  grows over the nested ladder `512 ⊂ 1000 ⊂ 2000 ⊂ 4000`. Pooled NLP is
  `0/7,713 = 0.00%`, including the near-degenerate ViSoBERT/zero cell (`0/840`).
  Coefficients that do not clear the realized floor are reported as unresolved.

* **Enumeration (Tier 2b / exact-β).** For short inputs (`d ≤ 13`) the full `2^d`
  cube is enumerated to recover the exact projection, then finite-budget masks are
  drawn i.i.d. with replacement. Across 27 enumerable runs at `K = 2`, the
  surrogate certifies **170 coordinates with zero false signs (0/170)**, including
  pairwise interactions. Every coordinate the theorem promises to detect
  (`|beta_S| > 2·floor`) is recovered (**power 1.000**, 87/87); relative to the
  tighter full-cube floor the deployment budget resolves 103/240 coordinates
  (power 0.429), the rest reported as unresolved — the honest finite-budget
  resolution cost, not a guarantee failure.

* **Independent reseeding (Tier 2b audit).** At fixed `N = 2000` over `R = 40`
  independent seeds, the strong image cell (ResNet-50/mean) is perfectly stable
  (per-coordinate 0/109, per-run 0/40). In the weak ViSoBERT/zero cell, the plain
  pilot shows some near-threshold churn, but under the conservative
  mismatch-energy **UCB pilot** there are **no cross-seed sign disagreements**
  (0/297 per-coordinate, 0/160 per-run). The floor-passing-set Jaccard is 1.000
  above the factor-two margin under both pilots: set churn is concentrated near
  the certification threshold, not above it. Cross-seed agreement measures
  stability, not correctness — correctness is claimed only in the enumeration tier.

* **Feasibility (Tier 3).** Moving from `K = 1` to `K = 2` lowers `σ_eff` (pairwise
  structure leaves the mismatch residual) but raises the design from `p1 = d+1` to
  `p2 ≈ d²/2`, so the feasibility requirement `N ≳ pK` grows quadratically. Below
  that floor the dense pairwise surrogate is not well-posed and its interaction
  coefficients should not be read as a reliable interaction map.

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

`python tier1_synthetic.py leakage` reproduces `C_m ≈ 0.80` (Table 2, frozen from
the large-N rows; the all-N mean 0.78 is printed alongside for transparency). The
full `all` run also recovers `C_budget ≈ 1.51`.

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

See the paper for the full author list and references.