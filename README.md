# BudgetLIME — Finite-Budget Certification for LIME-Style Local Surrogates

Reference implementation for the paper *Finite-Budget Certification for LIME-Style
Local Surrogates*. The code computes a **detection floor** for the dense
ordinary-least-squares core of degree-`K` LIME surrogates: coefficients whose
fitted magnitude clears the floor have **certified signs** at the chosen query
budget; coefficients below it are reported as **unresolved** rather than zero.

The whole codebase is one inequality (Theorem 1) read in two directions:

```
||beta_hat - beta||_inf  <=  floor(N, rho)
floor(N, rho) = C_floor * sigma_eff * sqrt(2 log pK / N)
sigma_eff     = sigma_obs + C_m * sqrt(m_>K)
```

* **Forward** — the guarantee: `|beta_hat_S| > floor`  ⇒  the sign of `beta_S` is correct.
* **Backward** — the budget rule (Eq. 8): `N ≳ 2 C_budget² σ_eff² log pK / β_min²`.

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
| `bl_core.py` | no | The shared numerical core. Degree-`K` Walsh feature machinery (`p_K`, `feature_subsets`, `design_matrix`, `standardize_columns`, `sample_masks`), dense OLS (`ols_fit`), the floor (`sigma_eff`, `floor_value`, `certified_set`), the budget rule (`predict_budget`, `feasibility_floor`, `plan_budget`), pilot estimation of `sigma_eff` (`estimate_mismatch_from_residual`, `pilot_N0`), and the prefix-ladder diagnostic (`sweep_prefix_ladder`). **The two and only constants** live here in `CONSTANTS` and are frozen after Tier-1 calibration. |
| `bl_models.py` | **yes** | The only Torch code. Two query-only black boxes: `TextClassifier` (sentence + token mask → class probability, `σ_obs > 0`) and `ImageClassifier` (image + cell mask → class logit, `σ_obs ≈ 0`). These map `(input, binary mask) → model output` and nothing else; all certification math is in `bl_core.py`. |
| `tier1_synthetic.py` | no | **Tier 1 — synthetic, ground truth known.** Calibrates the two constants and proves the two-direction collapse: leakage linchpin (fixes `C_M`), forward SDR-collapse curve, backward budget-constant recovery (fixes `C_BUDGET`), and the regime grid spanning the two axes of `σ_eff`. This is the *only* place constants are fit. |
| `tier2_blackbox.py` | yes | **Tier 2 — black-box classifiers.** With constants frozen, tests the guarantee on real query-only models: forward sign-flip stability over a prefix-nested budget ladder, backward budget planning, and the **exact-β sign-correctness check** (enumerate the full `2^d` mask cube for short inputs, `d ≤ 13`, to recover the exact projection and directly verify signs). |
| `tier2b_reseed.py` | yes | **Tier 2b — independent-reseeding audit.** Breaks the shared-mask coupling of the nested ladder: at a single fixed `N`, runs `R` independent seeds to measure (A) the cross-seed sign-violation rate vs the `1/pK` target and (B) the stratified Jaccard stability of the certified set above vs inside the unresolved band. Includes a `selftest` mode (pure NumPy, no models). |
| `tier3_feasibility.py` | no | **Tier 3 — feasibility (Reading 2).** Shows why `K=2` costs *more* despite *lower* noise: moving pairwise structure into the fit lowers `σ_eff`, but `pK` jumps `~ d²/2`, lifting the feasibility floor `~ pK`. Produces the resolution-budget vs feasibility-floor crossing curves. |
| `baselines.py` | yes | Compares the floor against a per-coordinate bootstrap CI and the single-coordinate Wald interval on the **same** fit and mask bank, isolating the certification criterion. Demonstrates the Wald degeneracy as `σ_obs → 0` that the mismatch term `C_m√m` repairs. |
| `git.push` | — | Convenience git push snippet. |

### The two frozen constants (`bl_core.CONSTANTS`)

| Constant | Default | Role |
|----------|---------|------|
| `C_M` | `1.0` (calibrated to **1.24** in Tier 1) | Leakage constant (Lemma 1); enters `σ_eff`. |
| `C_FLOOR` | `1.0` | Floor-bound constant (forward); theory = 1 for the orthonormal ±1 design, empirical ≥ 1 expected. |
| `C_BUDGET` | `1.81` | Budget-rule constant (backward), back-solved at Tier 1. |

`C_FLOOR` (the bound) and `C_BUDGET` (the budget invert) are deliberately kept as
separate objects: inverting the budget with `C_FLOOR` lands below the feasibility
floor and is physically meaningless.

---

## Installation

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

`python tier1_synthetic.py leakage` reproduces `C_m = 1.24` (paper Table 2);
the full `all` run also recovers `C_budget ≈ 1.81`.

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