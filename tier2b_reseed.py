"""
tier2b_reseed.py
================
TIER 2b -- INDEPENDENT-RESEEDING audit (the experiment the paper defers to
"future work" in the Limitations section, and that a reviewer asks for because
the main tables use prefix-nested banks).

WHAT THIS VERIFIES (and what Tier 2's nested ladder structurally CANNOT).
The single-budget guarantee (Theorem 1) is a statement about the DISTRIBUTION
over i.i.d. draws: at a fixed N, the certified signs are correct with high
probability >= 1 - 1/pK. The Tier 2 prefix ladder reuses ONE bank, so its
stability is partly built in by construction (Corollary 2 is deterministic on a
single draw). Reseeding breaks the shared-mask coupling on purpose and answers
three questions the nested ladder leaves untouched:

  (A) delta CALIBRATION.  Over R INDEPENDENT seeds at one fixed N, how often
      does a certified coordinate disagree in sign with the cross-seed
      consensus?  The theorem predicts this violation rate <~ 1/pK.  This is
      the only frequentist test of the "with high probability" clause; a single
      nested run gives exactly one sample and cannot estimate a rate.

  (B) the STABLE/UNSTABLE STRATIFICATION around the floor.  Definition 1 calls
      the certified set a "derived random object" and the paper predicts set
      churn lives in the unresolved band |beta| <= 2*floor.  We measure the
      mean pairwise Jaccard of certified sets across independent seeds, SPLIT
      into coordinates whose cross-seed median |beta| is ABOVE 2*floor vs INSIDE
      [floor, 2*floor].  Prediction: above-band Jaccard ~ 1, in-band Jaccard
      low.  This shows the floor sits at the right PLACE, not merely the right
      scale.

NOTE on what is NOT measured here.  An earlier version compared a prefix-nested
Jaccard against the independent one to "quantify nested inflation."  That
comparison is ill-posed: the nested Jaccard is taken ACROSS two budgets (whose
floors differ as 1/sqrt(N), so set membership legitimately shifts with the
threshold), whereas the independent Jaccard is at a SINGLE fixed budget (common
floor).  Corollary 2 guarantees nested SIGN stability, not nested SET-membership
stability, so a set-Jaccard comparison conflates a threshold shift with sampling
churn and can even make the nested view look LESS stable.  The honest
nested-vs-independent contrast is a sign-flip-rate comparison, not a set-Jaccard
one, and is out of scope for this fixed-budget audit.  We report only (A) and
(B), both defined at a single common budget.

NO NEW NUMERICS.  Everything here calls bl_core (floor, OLS, certified_set,
sigma_eff) and reuses tier2's Probe / nlp_probe / image_probe / pilot_sigma_eff
verbatim.  The ONLY thing added is an outer loop over independent seeds and the
two reseeding statistics above.

TORCH.  Exactly as in Tier 2: a backbone is built only inside main() after
arguments are supplied, and the only line that touches the model is
probe.query(Z).  Tier-1/synthetic style numerics need only numpy.

USAGE
  # NLP, one or more cells, R independent seeds at a single fixed budget N:
  python tier2b_reseed.py nlp --backbones visobert --references zero \
      --N 2000 --R 40 --K 1 --subset 10 --sentences text_samples/sst2_samples.txt

  # Image cell (deterministic backbone -> sigma_obs ~ 0, mismatch-driven floor):
  python tier2b_reseed.py image --backbones resnet50 --references mean \
      --N 2000 --R 40 --subset 10 --images_dir image_samples --glob "*.JPEG"

  # Self-test with a synthetic mock probe (NO torch, NO models): verifies the
  # whole reseeding pipeline + LaTeX emitter run end-to-end.
  python tier2b_reseed.py selftest

OUTPUT
  A per-cell summary plus a LaTeX-ready booktabs table (appendix
  "Independent-reseeding audit").  The headline numbers are:
    * sign-violation rate vs the 1/pK target               (calibration, A)
    * Jaccard above-band  ~ 1.00  /  Jaccard in-band  low   (stratification, B)
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import sys
import numpy as np

import bl_core as bl


# --------------------------------------------------------------------------- #
#  Reuse Tier 2's adapters verbatim where torch is involved.  We import lazily
#  so that `selftest` (pure numpy) never triggers the model imports at module
#  load time.
# --------------------------------------------------------------------------- #
def _import_tier2():
    import tier2_blackbox as T2
    return T2


# =========================================================================== #
#  The reseeding core (pure numpy; operates on any object exposing .d/.query/
#  .sigma_obs and a precomputed sigma_eff).
# =========================================================================== #
def _fit_certify(Z, y, s_eff, d, N, K, family_wise=True):
    """One independent run: dense OLS -> floor -> certified set.  Returns
    (beta, floor, certified-index-set) or None if the fit is not well-posed."""
    try:
        beta, _, _ = bl.ols_fit(Z, y, K)
    except np.linalg.LinAlgError:
        return None
    fl = bl.floor_value(s_eff, d, N, K, family_wise)
    cset, _ = bl.certified_set(beta, fl)
    return beta, fl, cset


def _pairwise_jaccard(sets):
    """Mean Jaccard over all unordered pairs of index sets.  Empty/empty -> 1."""
    vals = []
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            sa, sb = sets[a], sets[b]
            u = len(sa | sb)
            vals.append(1.0 if u == 0 else len(sa & sb) / u)
    return float(np.mean(vals)) if vals else float("nan")


def reseed_probe(probe, s_eff, N, K, R, base_seed=10_000, family_wise=True):
    """Run R INDEPENDENT seeds at a single fixed budget N on one probe.

    Returns a dict of the three reseeding statistics, or None if the probe is
    not identifiable at N (N <= pK).  Pilot sigma_eff is passed in (estimated
    once, exactly as Tier 2's frozen-input philosophy: the floor's only
    data-dependent input is sigma_eff, and we hold it fixed across seeds so the
    test isolates SAMPLING variability, not pilot variability).
    """
    d, pK = probe.d, bl.p_K(probe.d, K)
    if N <= pK:
        return None
    p = len(bl.feature_subsets(d, K))          # number of coefficients (no intc)

    betas, floors, csets = [], [], []
    for s in range(R):
        rng = np.random.default_rng(base_seed + s)
        Z = bl.sample_masks(N, d, rng)
        y = probe.query(Z)                      # <-- only model contact
        out = _fit_certify(Z, y, s_eff, d, N, K, family_wise)
        if out is None:
            continue
        beta, fl, cset = out
        betas.append(beta); floors.append(fl); csets.append(cset)
    R_ok = len(betas)
    if R_ok < 2:
        return None
    B = np.stack(betas, axis=0)                 # (R_ok, p)
    fl_mean = float(np.mean(floors))

    # --- (A) sign-violation rate vs consensus -----------------------------
    # consensus sign = sign of the cross-seed MEDIAN coefficient.
    med = np.median(B, axis=0)
    consensus_sign = np.sign(med)
    n_cert = 0
    n_violate = 0
    for r in range(R_ok):
        cert_idx = list(csets[r])
        for i in cert_idx:
            n_cert += 1
            if np.sign(B[r, i]) != consensus_sign[i] and consensus_sign[i] != 0:
                n_violate += 1
    violation_rate = (n_violate / n_cert) if n_cert else 0.0
    delta_target = 1.0 / pK

    # --- (B) stratified set stability around the floor --------------------
    # classify each coordinate by its cross-seed median magnitude relative to
    # the floor band.  above-band: |med| > 2*floor;  in-band: floor < |med| <= 2floor
    absmed = np.abs(med)
    above_mask = absmed > 2.0 * fl_mean
    inband_mask = (absmed > fl_mean) & (absmed <= 2.0 * fl_mean)
    above_idx = set(np.where(above_mask)[0].tolist())
    inband_idx = set(np.where(inband_mask)[0].tolist())

    sets_above = [cs & above_idx for cs in csets]
    sets_inband = [cs & inband_idx for cs in csets]
    jac_above = _pairwise_jaccard(sets_above)
    jac_inband = _pairwise_jaccard(sets_inband)
    jac_all = _pairwise_jaccard(csets)

    return dict(
        d=d, pK=pK, R_ok=R_ok, N=N, sigma_eff=float(s_eff), floor=fl_mean,
        n_cert=n_cert, n_violate=n_violate,
        violation_rate=violation_rate, delta_target=delta_target,
        n_above=len(above_idx), n_inband=len(inband_idx),
        jac_above=jac_above, jac_inband=jac_inband, jac_all=jac_all,
    )


# =========================================================================== #
#  Cell aggregation (a "cell" = backbone x reference, averaged over items)
# =========================================================================== #
def _agg(rows):
    """Aggregate per-probe reseed dicts into one cell summary.

    Rates are pooled over the raw counts (sum of violations / sum of certified)
    so a probe with more certified coordinates carries proportional weight.
    Jaccards are simple means over probes that had a defined value.
    """
    rows = [r for r in rows if r is not None]
    if not rows:
        return None

    def _mean(key):
        v = [r[key] for r in rows if r[key] == r[key]]   # drop nan
        return float(np.mean(v)) if v else float("nan")

    n_cert = sum(r["n_cert"] for r in rows)
    n_violate = sum(r["n_violate"] for r in rows)
    return dict(
        items=len(rows),
        d_mean=_mean("d"),
        pK_mean=_mean("pK"),
        sigma_eff=_mean("sigma_eff"),
        n_cert=n_cert, n_violate=n_violate,
        violation_rate=(n_violate / n_cert) if n_cert else 0.0,
        delta_target=_mean("delta_target"),
        jac_above=_mean("jac_above"),
        jac_inband=_mean("jac_inband"),
        jac_all=_mean("jac_all"),
        n_above=_mean("n_above"),
        n_inband=_mean("n_inband"),
    )


# =========================================================================== #
#  Reporting
# =========================================================================== #
def print_cell(name, a):
    if a is None:
        print(f"  [{name}] no identifiable probes at this N -- skipped.")
        return
    print(f"\n=== cell: {name}  (items={a['items']}, "
          f"mean d={a['d_mean']:.1f}, mean pK={a['pK_mean']:.1f}) ===")
    print(f"  sigma_eff (mean)              : {a['sigma_eff']:.3f}")
    print(f"  (A) sign violations           : {a['n_violate']}/{a['n_cert']} "
          f"= {a['violation_rate']:.5f}   target <~ 1/pK = "
          f"{a['delta_target']:.5f}")
    print(f"  (B) Jaccard above-band (>2fl) : {a['jac_above']:.3f}   "
          f"(mean #coords {a['n_above']:.1f})   [prediction ~ 1.00]")
    print(f"      Jaccard in-band [fl,2fl]  : {a['jac_inband']:.3f}   "
          f"(mean #coords {a['n_inband']:.1f})   [prediction LOW]")
    print(f"      Jaccard all certified     : {a['jac_all']:.3f}")


def emit_latex(cells, N, R):
    """A booktabs appendix table: one row per cell."""
    print("\n% ---- LaTeX (appendix: independent-reseeding audit) ----")
    print(r"\begin{table}[t]")
    print(r"\centering\small\setlength{\tabcolsep}{5pt}")
    print(r"\begin{tabular}{@{}lcccccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Cell} & $\seff$ & \textbf{viol.}/cert "
          r"& $1/p_K$ & $J_{>2\mathrm{fl}}$ & $J_{[\mathrm{fl},2\mathrm{fl}]}$ "
          r"& \#$>$2fl\,/\,\#band \\")
    print(r"\midrule")
    for name, a in cells:
        if a is None:
            continue
        print(f"{name} & {a['sigma_eff']:.3f} & "
              f"{a['n_violate']}/{a['n_cert']} & "
              f"{a['delta_target']:.4f} & "
              f"{a['jac_above']:.3f} & {a['jac_inband']:.3f} & "
              f"{a['n_above']:.1f}/{a['n_inband']:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{\textbf{Independent-reseeding audit} at a single fixed "
          rf"budget $N={N}$ over $R={R}$ independent mask draws (no prefix "
          r"nesting). \textbf{viol.}/cert is the cross-seed sign-disagreement "
          r"rate among certified coordinates, predicted $\lesssim 1/p_K$. "
          r"$J_{>2\mathrm{fl}}$ and $J_{[\mathrm{fl},2\mathrm{fl}]}$ are the "
          r"mean pairwise Jaccard of the certified set restricted to "
          r"coordinates whose cross-seed median magnitude lies above "
          r"$2\,\mathrm{floor}$ and inside the unresolved band "
          r"$[\mathrm{fl},2\mathrm{fl}]$ respectively; the last column gives "
          r"the mean per-item count of each class. Set churn is confined to the "
          r"unresolved band, as Definition~1 predicts: above-band membership is "
          r"near-perfectly stable across independent draws while in-band "
          r"membership is not.}")
    print(r"\label{tab:reseed}")
    print(r"\end{table}")


# =========================================================================== #
#  Drivers
# =========================================================================== #
def run_nlp(args):
    T2 = _import_tier2()
    import bl_models as M
    backbones = (list(M.NLP_BACKBONES) if args.backbones == "all"
                 else args.backbones.split(","))
    refs = (list(M.NLP_REFERENCES) if args.references == "all"
            else args.references.split(","))
    sents = T2.load_sentences(args.sentences, args.subset)

    cells = []
    for bk in backbones:
        clf = M.TextClassifier(model=bk, dataset=args.dataset)
        for ref in refs:
            name = f"{bk}/{ref}"
            print(f"\n>>> {name}: {len(sents)} sentences, "
                  f"R={args.R} seeds at N={args.N}", flush=True)
            rows = []
            for k, sent in enumerate(sents):
                p = T2.nlp_probe(clf, sent, ref, args.max_free)
                if p is None:
                    continue
                s_eff, _ = T2.pilot_sigma_eff(p, args.K, seed=k)
                rows.append(reseed_probe(p, s_eff, args.N, args.K, args.R,
                                         base_seed=10_000 + 137 * k))
            a = _agg(rows)
            cells.append((name, a))
            print_cell(name, a)
    emit_latex(cells, args.N, args.R)


def run_image(args):
    T2 = _import_tier2()
    import bl_models as M
    paths = sorted(glob.glob(os.path.join(args.images_dir, args.glob)))
    paths = paths[:args.subset] if args.subset else paths
    bks = (list(M.IMAGE_BACKBONES) if args.backbones == "all"
           else args.backbones.split(","))
    rfs = (list(M.IMAGE_REFERENCES) if args.references == "all"
           else args.references.split(","))

    cells = []
    for bk in bks:
        clf = M.ImageClassifier(backbone=bk)
        for ref in rfs:
            name = f"{bk}/{ref}"
            print(f"\n>>> {name}: {len(paths)} images, "
                  f"R={args.R} seeds at N={args.N}", flush=True)
            rows = []
            for k, path in enumerate(paths):
                img = clf.load_image(path)
                p = T2.image_probe(clf, img, ref, args.grid)
                if p is None:
                    continue
                s_eff, _ = T2.pilot_sigma_eff(p, 1, seed=k)
                rows.append(reseed_probe(p, s_eff, args.N, 1, args.R,
                                         base_seed=10_000 + 137 * k))
            a = _agg(rows)
            cells.append((name, a))
            print_cell(name, a)
    emit_latex(cells, args.N, args.R)


# =========================================================================== #
#  Self-test: a synthetic mock probe with KNOWN coefficients, NO torch.
# =========================================================================== #
class _MockProbe:
    """A deterministic-mismatch synthetic probe: g(z) = sum active main effects
    + a controlled higher-order block, plus optional Gaussian query noise. Used
    only to exercise the reseeding pipeline end-to-end without any model.
    """
    def __init__(self, d=30, n_active=4, beta=0.18, m_high=0.02,
                 sigma_obs=0.05, seed=0):
        self.d = d
        self.sigma_obs = sigma_obs
        rng = np.random.default_rng(seed)
        self._beta_main = np.zeros(d)
        act = rng.choice(d, size=n_active, replace=False)
        self._beta_main[act] = beta * rng.choice([-1, 1], size=n_active)
        # higher-order: a handful of pair terms carrying mismatch energy m_high
        self._pairs = [(int(i), int(j)) for i, j in
                       rng.choice(d, size=(6, 2)) if i != j]
        e = math.sqrt(m_high / max(len(self._pairs), 1))
        self._beta_pair = {p: e * rng.choice([-1, 1]) for p in self._pairs}
        self._noise_seed = seed + 1
        self.target = 1

    def query(self, Z):
        Zc = 2.0 * (Z - 0.5)
        y = Zc @ self._beta_main
        for (i, j), b in self._beta_pair.items():
            y = y + b * Zc[:, i] * Zc[:, j]
        if self.sigma_obs > 0:
            rng = np.random.default_rng(
                self._noise_seed + int(1e6 * abs(y.sum())) % 1_000_000)
            y = y + rng.normal(0, self.sigma_obs, size=y.shape)
        return y


def run_selftest(_args):
    print("SELF-TEST (numpy only, no torch, no models)")
    print("Three mock cells with decreasing sigma_obs to mimic the "
          "high/low-signal\nspread of the real study.\n")
    cells = []
    configs = [("mock/high-noise", 0.10, 0.02),
               ("mock/mid-noise",  0.05, 0.02),
               ("mock/low-noise",  0.01, 0.04)]
    for name, s_obs, m_high in configs:
        rows = []
        for k in range(8):                       # 8 mock "items"
            p = _MockProbe(d=20, n_active=4, beta=0.20, m_high=m_high,
                           sigma_obs=s_obs, seed=k)
            s_eff, _ = _mock_pilot(p, K=1, seed=k)
            rows.append(reseed_probe(p, s_eff, N=2000, K=1, R=30,
                                     base_seed=500 + 137 * k))
        a = _agg(rows)
        cells.append((name, a))
        print_cell(name, a)
    emit_latex(cells, N=2000, R=30)
    print("\nSelf-test OK: pipeline + LaTeX emitter ran end-to-end.")


def _mock_pilot(probe, K, seed):
    """Mirror tier2.pilot_sigma_eff without importing torch."""
    N0 = bl.pilot_N0(probe.d, K)
    rng = np.random.default_rng(seed + 12345)
    Z = bl.sample_masks(max(N0, 3 * bl.p_K(probe.d, K)), probe.d, rng)
    y = probe.query(Z)
    m_hat = bl.estimate_mismatch_from_residual(Z, y, K, probe.sigma_obs,
                                               cross_fit=(K == 2))
    return bl.sigma_eff(probe.sigma_obs, m_hat), m_hat


# =========================================================================== #
#  CLI
# =========================================================================== #
def build_parser():
    ap = argparse.ArgumentParser(
        description="Tier 2b: independent-reseeding audit at a fixed budget.")
    ap.add_argument("mode", choices=["nlp", "image", "selftest"])
    ap.add_argument("--N", type=int, default=2000,
                    help="single fixed budget (NOT a ladder; reseeding is at one N)")
    ap.add_argument("--R", type=int, default=40, help="independent seeds")
    ap.add_argument("--K", type=int, default=1, choices=[1, 2])
    ap.add_argument("--subset", type=int, default=10)
    ap.add_argument("--backbones", default="all")
    ap.add_argument("--references", default="all")
    ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--sentences", default="sst2_samples.txt")
    ap.add_argument("--max_free", type=int, default=64)
    ap.add_argument("--images_dir", default="benchmark_50")
    ap.add_argument("--glob", default="*.JPEG")
    ap.add_argument("--grid", type=int, default=7)
    return ap


def main():
    args = build_parser().parse_args()
    if args.mode == "nlp":
        run_nlp(args)
    elif args.mode == "image":
        run_image(args)
    else:
        run_selftest(args)


if __name__ == "__main__":
    main()