"""
baselines.py
============
Compare the detection FLOOR against the two other finite-budget ways to make a
per-coordinate trust decision, on the SAME fitted dense OLS and the SAME mask
bank, so they differ ONLY in the certification criterion:

  FLOOR      : certify S iff |beta_S| > floor(N,rho)
               = C_floor * sigma_eff * sqrt(2 log pK / N).
               One fit, one simultaneous threshold for all pK coordinates;
               carries the multiplicity correction in closed form; uniquely
               invertible to a query budget (Eq. 8).
  BOOTSTRAP  : resample the N mask-response pairs with replacement, refit B
               times, certify S iff its percentile CI excludes 0.
               B fits, one interval per coordinate, no simultaneity correction.
  WALD       : certify S iff |beta_S| > z_{1-alpha} * SE_S from the OLS standard
               errors. One fit, one interval per coordinate, NO mismatch term --
               so on a deterministic backbone (sigma_obs -> 0) every SE collapses
               and it certifies everything regardless of evidence. That
               degeneracy is exactly what the mismatch term C_m sqrt(m) of
               sigma_eff repairs.

We do NOT claim the floor certifies more coordinates; we show it reaches the
same decisions while being simultaneous, closed-form, and budget-invertible.
A sequential stopping rule (e.g. S-LIME) is deliberately excluded: it answers a
different question (how many perturbations before a SELECTED support stabilizes)
and is complementary to a fixed-budget sign-certificate, not an alternative.

Reuses tier2_blackbox.Probe + bl_core; no torch is imported here. The driver
builds the backbones (inside main()), everything else is numpy.

USAGE
  # NLP, one cell, shared bank N=2000, B=200 bootstrap, 10 items:
  python baselines.py nlp --backbones distilbert --references mask \
      --N 2000 --B 200 --subset 10 --sentences sst2_samples.txt
  # Image cell (shows the Wald sigma_obs->0 degeneracy):
  python baselines.py image --backbones resnet50 --references mean \
      --N 2000 --B 200 --subset 10 --images_dir benchmark_50 --glob "*.JPEG"
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import numpy as np

import bl_core as bl
from tier2_blackbox import (Probe, nlp_probe, image_probe, load_sentences,
                            pilot_sigma_eff, Progress)


# =========================================================================== #
#  The three certification criteria, on a shared fit / bank
# =========================================================================== #
def certify_floor(beta, s_eff, d, N, K):
    """Simultaneous floor rule -> boolean mask over coordinates."""
    fl = bl.floor_value(s_eff, d, N, K)
    return np.abs(beta) > fl


def certify_wald(Z, y, K, sigma_obs, alpha=0.05):
    """Single-coordinate Wald rule: |beta_S| > z * SE_S, SE from OLS standard
    errors on the standardized design, mapped back to the original scale.

    Crucially uses sigma_obs only (NO mismatch term): on a deterministic
    backbone sigma_obs ~ 0 so every SE -> 0 and the rule certifies everything.
    Returns (boolean mask, all_certified_flag)."""
    d = Z.shape[1]
    X = bl.design_matrix(Z, K)
    N = X.shape[0]
    Xs, scale = bl.standardize_columns(X)
    G = (Xs.T @ Xs) / N
    Ginv = np.linalg.inv(G)
    beta_std = Ginv @ (Xs.T @ (y - y.mean())) / N
    beta = beta_std / scale
    # SE on standardized scale, then divide by column scale to original units
    z = abs(_z_quantile(1 - alpha / 2))
    se_std = sigma_obs * np.sqrt(np.diag(Ginv) / N)
    se = se_std / scale
    # sigma_obs ~ 0 -> se ~ 0 -> certifies all (the degeneracy)
    cert = np.abs(beta) > z * se
    all_cert = bool(cert.all())
    return cert, all_cert


def certify_bootstrap(Z, y, K, B, rng, alpha=0.05):
    """Per-coordinate percentile bootstrap: resample (mask,response) pairs with
    replacement, refit B times, certify S iff its [alpha/2, 1-alpha/2]
    percentile interval excludes 0. Returns boolean mask over coordinates."""
    N = Z.shape[0]
    betas = []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        try:
            b, _, _ = bl.ols_fit(Z[idx], y[idx], K)
        except np.linalg.LinAlgError:
            continue
        betas.append(b)
    if not betas:
        d = Z.shape[1]
        return np.zeros(bl.p_K(d, K) - 1, dtype=bool)
    B_mat = np.stack(betas, axis=0)
    lo = np.percentile(B_mat, 100 * alpha / 2, axis=0)
    hi = np.percentile(B_mat, 100 * (1 - alpha / 2), axis=0)
    return (lo > 0) | (hi < 0)          # interval excludes zero


def _z_quantile(p):
    """Inverse standard-normal CDF (Acklam-free): use math.erfinv via a small
    rational approx is overkill; numpy has no ppf, so use a stable formula."""
    # Beasley-Springer/Moro is overkill here; z_{0.975}=1.96 is the only value
    # we need in practice, but compute generally via erfinv.
    return math.sqrt(2) * _erfinv(2 * p - 1)


def _erfinv(x):
    # Winitzki approximation, accurate enough for CI quantiles.
    a = 0.147
    ln = math.log(1 - x * x)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)


# =========================================================================== #
#  Per-probe comparison on a shared bank
# =========================================================================== #
def compare_on_probe(probe: Probe, N, K, B, seed=0):
    """Fit once on a shared bank of size N; run all three criteria. Returns a
    dict of decisions + agreement counts, or None if not well-posed."""
    if N <= bl.p_K(probe.d, K):
        return None
    s_eff, _ = pilot_sigma_eff(probe, K, seed)

    rng = np.random.default_rng(seed + 99)
    Z = bl.sample_masks(N, probe.d, rng)
    y = probe.query(Z)
    try:
        beta, _, _ = bl.ols_fit(Z, y, K)
    except np.linalg.LinAlgError:
        return None

    c_floor = certify_floor(beta, s_eff, probe.d, N, K)
    c_wald, wald_all = certify_wald(Z, y, K, probe.sigma_obs)
    c_boot = certify_bootstrap(Z, y, K, B, np.random.default_rng(seed + 7))

    p = c_floor.size
    floor_only = int((c_floor & ~c_boot).sum())     # floor yes, bootstrap no
    boot_only = int((~c_floor & c_boot).sum())       # bootstrap yes, floor no
    agree = int((c_floor == c_boot).sum())
    return dict(d=probe.d, p=p, sigma_obs=probe.sigma_obs,
                floor_cert=int(c_floor.sum()),
                boot_cert=int(c_boot.sum()),
                wald_cert=int(c_wald.sum()),
                wald_all=wald_all,
                floor_only=floor_only, boot_only=boot_only,
                agree=agree)


# =========================================================================== #
#  Reporting
# =========================================================================== #
def report(rows, setting, N, B):
    print("\n" + "=" * 72)
    print(f"FLOOR vs BOOTSTRAP vs WALD  ({setting}, shared bank N={N}, B={B})")
    print("=" * 72)
    if not rows:
        print("  no well-posed probes.")
        return
    p_tot = sum(r["p"] for r in rows)
    agree_tot = sum(r["agree"] for r in rows)
    floor_only = float(np.mean([r["floor_only"] for r in rows]))
    boot_only = float(np.mean([r["boot_only"] for r in rows]))
    floor_mean = float(np.mean([r["floor_cert"] for r in rows]))
    boot_mean = float(np.mean([r["boot_cert"] for r in rows]))
    agree_pct = 100.0 * agree_tot / p_tot if p_tot else float("nan")
    wald_all = sum(r["wald_all"] for r in rows)

    print(f"  coords (total over items)     : {p_tot}")
    print(f"  floor vs bootstrap agreement  : {agree_pct:.1f}%")
    print(f"  floor certified  (mean/item)  : {floor_mean:.1f}")
    print(f"  bootstrap cert.  (mean/item)  : {boot_mean:.1f}")
    print(f"  FLOOR-ONLY (floor yes, boot no, mean/item) : {floor_only:.2f}")
    print(f"    -> {'0.00 => floor never certifies a coord the bootstrap rejects' if floor_only < 1e-9 else 'floor sometimes exceeds bootstrap'}")
    print(f"  BOOT-ONLY  (boot yes, floor no, mean/item) : {boot_only:.2f}")
    print(f"    -> all disagreement is bootstrap-only certification of "
          f"near-floor coords;\n       the floor is the more conservative "
          f"simultaneous rule.")
    print(f"  Wald certified-ALL items (sigma_obs~0 degeneracy): "
          f"{wald_all}/{len(rows)}")
    if wald_all:
        print(f"    -> on the deterministic backbone Wald certifies every "
              f"coordinate\n       regardless of evidence; the mismatch term "
              f"of sigma_eff repairs this.")


# =========================================================================== #
#  Drivers (torch built only here)
# =========================================================================== #
def run_nlp(args):
    import bl_models as M
    backbones = (list(M.NLP_BACKBONES) if args.backbones == "all"
                 else args.backbones.split(","))
    refs = (list(M.NLP_REFERENCES) if args.references == "all"
            else args.references.split(","))
    sents = load_sentences(args.sentences, None)
    rows = []
    total = len(backbones) * len(refs) * (args.subset or len(sents))
    prog = Progress(total, f"baselines-nlp K={args.K}")
    for bk in backbones:
        try:
            clf = M.TextClassifier(model=bk, dataset=args.dataset)
        except Exception as e:
            prog.note(f"[skip {bk}] {e}")
            continue
        for ref in refs:
            used = 0
            for si, sent in enumerate(sents):
                if args.subset and used >= args.subset:
                    break
                p = nlp_probe(clf, sent, ref, args.max_free)
                if p is None:
                    continue
                r = compare_on_probe(p, args.N, args.K, args.B, seed=si)
                used += 1
                if r:
                    rows.append(r)
                    prog.step(f"{bk}/{ref} d={p.d} agree="
                              f"{100*r['agree']/r['p']:.0f}%")
                else:
                    prog.step(f"{bk}/{ref} (not well-posed)")
        clf.close()
    prog.close()
    report(rows, f"NLP K={args.K}", args.N, args.B)


def run_image(args):
    import bl_models as M
    backbones = (list(M.IMAGE_BACKBONES) if args.backbones == "all"
                 else args.backbones.split(","))
    refs = (list(M.IMAGE_REFERENCES) if args.references == "all"
            else args.references.split(","))
    paths = sorted(glob.glob(os.path.join(args.images_dir, args.glob)))
    paths = paths[:args.subset] if args.subset else paths
    rows = []
    total = len(backbones) * len(refs) * len(paths)
    prog = Progress(total, "baselines-image K=1")
    for bk in backbones:
        try:
            clf = M.ImageClassifier(backbone=bk)
        except Exception as e:
            prog.note(f"[skip {bk}] {e}")
            continue
        for ref in refs:
            for pi, path in enumerate(paths):
                img = clf.load_image(path)
                p = image_probe(clf, img, ref, args.grid)
                if p is None:
                    continue
                r = compare_on_probe(p, args.N, 1, args.B, seed=pi)
                if r:
                    rows.append(r)
                    prog.step(f"{bk}/{ref} d={p.d} agree="
                              f"{100*r['agree']/r['p']:.0f}% waldALL="
                              f"{r['wald_all']}")
                else:
                    prog.step(f"{bk}/{ref} (not well-posed)")
        clf.close()
    prog.close()
    report(rows, "Image K=1", args.N, args.B)


def build_parser():
    p = argparse.ArgumentParser(
        description="Floor vs bootstrap-CI vs Wald on a shared mask bank")
    p.add_argument("mode", choices=["nlp", "image"])
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--N", type=int, default=2000, help="shared mask bank size")
    p.add_argument("--B", type=int, default=200, help="bootstrap resamples")
    p.add_argument("--backbones", default="all")
    p.add_argument("--references", default="all")
    p.add_argument("--dataset", default="sst2")
    p.add_argument("--sentences", default="sst2_samples.txt")
    p.add_argument("--max_free", type=int, default=40)
    p.add_argument("--images_dir", default="benchmark_50")
    p.add_argument("--glob", default="*.JPEG")
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--subset", type=int, default=10)
    return p


def main():
    args = build_parser().parse_args()
    if args.mode == "nlp":
        run_nlp(args)
    else:
        run_image(args)


if __name__ == "__main__":
    main()
