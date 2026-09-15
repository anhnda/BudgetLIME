"""
tier2b_reseed.py
================
Independent-reseeding STABILITY audit.

The reference sign is the cross-seed majority, so these rates measure stability,
not correctness.  The exact-beta enumeration tier is the direct correctness
check.

Two pilot modes are supported:
  * plain: cross-fitted planning point estimate; empirical stability diagnostic.
  * ucb: one-sided held-out mismatch-energy UCB.  The UCB is plugged directly
    into the two-term certified floor with split=4.  A valid B_pop bound is still
    required separately; for probability-output NLP probes it is derived from
    |g|<=1 via the generic Parseval bound.

Every well-posed seed is included in the item-level disagreement denominator,
including seeds whose certified set is empty.
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import sys
import numpy as np

import bl_core as bl
# Reuse the Tier-2 probe adapters / pilot / progress so both tiers share ONE
# definition of a probe and of the pilot sigma_eff (no divergence between tiers).
from tier2_blackbox import (
    Progress, Probe, nlp_probe, image_probe, pilot_sigma_eff,
    pilot_mismatch_ucb, probe_B_pop_bound, load_sentences,
)


# =========================================================================== #
#  Per-probe audit
# =========================================================================== #
def audit_probe(probe: Probe, N, R, K, seed0=0, ucb=False,
                B_pop_ub=None):
    """Run one fixed-budget independent-reseeding audit.

    ``ucb=True`` uses the direct mismatch-energy UCB and split=4.  It requires a
    genuine population residual sup bound.  If ``B_pop_ub`` is not supplied, a
    bound is derived automatically only when the probe has a known bounded output
    (e.g. NLP class probabilities in [0,1]).

    ``ucb=False`` preserves the historical plug-in row for stability diagnostics:
    the cross-fitted m_hat and sample residual max are used as empirical floor
    ingredients, and the output must not be described as theorem-level coverage.
    """
    if N <= bl.p_K(probe.d, K):
        return None

    if ucb:
        s_eff_plan, u = pilot_mismatch_ucb(probe, K, seed=seed0)
        m_for_floor = u.m_ucb
        B_for_floor = B_pop_ub
        if B_for_floor is None:
            B_for_floor = probe_B_pop_bound(probe, K)
        if B_for_floor is None:
            raise ValueError(
                "UCB certification needs a valid B_pop upper bound. "
                "Probability-output NLP probes derive one automatically; "
                "for logit probes pass B_pop_ub explicitly."
            )
        split = bl.CONSTANTS.DELTA_SPLIT_UCB
    else:
        s_eff_plan, est = pilot_sigma_eff(probe, K, seed=seed0, detail=True)
        m_for_floor = est.m_hat
        # Historical empirical plug-in only: sample max is not a theorem B_pop UB.
        B_for_floor = est.B_hat if B_pop_ub is None else B_pop_ub
        split = bl.CONSTANTS.DELTA_SPLIT

    res = bl.reseed_audit(
        query_fn=probe.query,
        d=probe.d,
        N=N,
        R=R,
        K=K,
        sigma_obs_ub=probe.sigma_obs,
        m_bound=m_for_floor,
        B_pop_bound=B_for_floor,
        seed0=seed0 + 1000,
        split=split,
    )
    return res if (res is not None and res.well_posed) else None


# =========================================================================== #
#  Aggregation across probes within a cell
# =========================================================================== #
def _pool_cell(cell, results):
    """Pool counts over probes; Jaccards/band sizes are averaged over probes."""
    n_cc = sum(r.n_cert_checks for r in results)
    n_vi = sum(r.n_viol for r in results)
    n_ru = sum(r.n_runs for r in results)
    n_rd = sum(r.n_run_disagree for r in results)
    pk = int(np.median([r.pK for r in results]))

    def _mn(xs):
        xs = [x for x in xs if x is not None and not math.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "cell": cell,
        "pK": pk,
        "reference": 1.0 / pk,
        "n_probes": len(results),
        "viol": n_vi,
        "cert": n_cc,
        "viol_rate": (n_vi / n_cc) if n_cc else float("nan"),
        "run_disagree": n_rd,
        "runs": n_ru,
        "run_disagree_rate": (n_rd / n_ru) if n_ru else float("nan"),
        "jac_above": _mn([r.jaccard_above for r in results]),
        "jac_band": _mn([r.jaccard_band for r in results]),
        "n_above": _mn([r.n_above for r in results]),
        "n_band": _mn([r.n_band for r in results]),
    }


def _fmt(x, nd=3):
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# =========================================================================== #
#  Reporting
# =========================================================================== #
def report_reseed(rows, N, R, K, pilot="plain"):
    print("\n" + "=" * 78)
    if pilot == "ucb":
        plabel = (
            "held-out mismatch-energy UCB (split=4); m_UCB is plugged directly "
            "into the two-term floor"
        )
    else:
        plabel = (
            "cross-fitted plug-in planning pilot (split=3); stability diagnostic, "
            "not a theorem-level coverage test"
        )
    print(f"TIER 2b -- INDEPENDENT-RESEEDING STABILITY AUDIT "
          f"(N={N}, R={R}, K={K})")
    print(f"pilot: {plabel}")
    print("=" * 78)

    if not rows:
        print("No cell produced a result.")
        return

    print(f"  {'cell':>20} {'pK':>4} | {'dis/cert':>12} {'rate':>7} "
          f"{'1/pK ref':>8} | {'run_dis/runs':>14} {'rate':>7} | "
          f"{'J>2fl':>6} {'J[fl,2fl]':>9} {'#>2fl/#band':>12}")
    for r in rows:
        vc = f"{r['viol']}/{r['cert']}"
        rd = f"{r['run_disagree']}/{r['runs']}"
        print(f"  {r['cell']:>20} {r['pK']:>4} | "
              f"{vc:>12} {_fmt(r['viol_rate'],4):>7} "
              f"{r['reference']:>8.4f} | "
              f"{rd:>14} {_fmt(r['run_disagree_rate'],4):>7} | "
              f"{_fmt(r['jac_above']):>6} {_fmt(r['jac_band']):>9} "
              f"{_fmt(r['n_above'],1):>5}/{_fmt(r['n_band'],1):<5}")

    tot_vi = sum(r["viol"] for r in rows)
    tot_cc = sum(r["cert"] for r in rows)
    tot_rd = sum(r["run_disagree"] for r in rows)
    tot_ru = sum(r["runs"] for r in rows)
    ref = float(np.median([r["reference"] for r in rows]))
    pc_rate = (tot_vi / tot_cc) if tot_cc else float("nan")
    rd_rate = (tot_rd / tot_ru) if tot_ru else float("nan")

    print("\n  Cross-seed sign disagreement (STABILITY, not correctness):")
    print(f"    per-coordinate: {tot_vi}/{tot_cc} = {_fmt(pc_rate,4)}")
    print(f"    per-run/item  : {tot_rd}/{tot_ru} = {_fmt(rd_rate,4)}")
    print(f"    nominal 1/pK reference scale: ~{ref:.4f} "
          "(shown for context, not a theorem calibration target)")

    vals_a = [r["jac_above"] for r in rows if not math.isnan(r["jac_above"])]
    vals_b = [r["jac_band"] for r in rows if not math.isnan(r["jac_band"])]
    ja = float(np.mean(vals_a)) if vals_a else float("nan")
    jb = float(np.mean(vals_b)) if vals_b else float("nan")
    print("  Set stability:")
    print(f"    Jaccard above 2*floor = {_fmt(ja)}; "
          f"in [floor,2floor] = {_fmt(jb)}")
    print("  NOTE: two seeds can agree and both be wrong. Exact-beta enumeration "
          "is the correctness check.")
    _latex_table(rows, N, R, K)


def _latex_table(rows, N, R, K):
    print("\n  % ---- booktabs row(s) for Table 4 ----")
    print("  \\begin{tabular}{lrrrrrr}")
    print("  \\toprule")
    print("  Cell & $p_K$ & per-coord & per-run & $1/p_K$ & "
          "$J_{>2\\mathrm{fl}}$ & $J_{[\\mathrm{fl},2\\mathrm{fl}]}$ \\\\")
    print("  \\midrule")
    for r in rows:
        print(f"  {r['cell']} & {r['pK']} & {_fmt(r['viol_rate'],4)} & "
              f"{_fmt(r['run_disagree_rate'],4)} & {r['reference']:.4f} & "
              f"{_fmt(r['jac_above'])} & {_fmt(r['jac_band'])} \\\\")
    print("  \\bottomrule")
    print("  \\end{tabular}")


# =========================================================================== #
#  NLP / image drivers
# =========================================================================== #
def run_nlp(args):
    import bl_models as M
    backbones = (list(M.NLP_BACKBONES) if args.backbones == "all"
                 else args.backbones.split(","))
    refs = (list(M.NLP_REFERENCES) if args.references == "all"
            else args.references.split(","))
    sents = load_sentences(args.sentences, args.subset)
    rows = []
    prog = Progress(len(backbones) * len(refs) * len(sents),
                    f"tier2b-nlp N={args.N} R={args.R}")
    for bk in backbones:
        try:
            clf = M.TextClassifier(model=bk, dataset=args.dataset)
        except Exception as e:
            prog.note(f"[skip {bk}] {e}")
            continue
        for ref in refs:
            cell_results = []
            for si, sent in enumerate(sents):
                p = nlp_probe(clf, sent, ref, args.max_free)
                if p is None:
                    prog.step(f"{bk}/{ref} (skipped: d out of range)")
                    continue
                res = audit_probe(
                    p, args.N, args.R, args.K, seed0=si,
                    ucb=(args.pilot == "ucb"), B_pop_ub=args.B_pop_ub,
                )
                if res is not None:
                    cell_results.append(res)
                    prog.step(f"{bk}/{ref} d={p.d} "
                              f"dis={res.n_viol}/{res.n_cert_checks} "
                              f"rundis={res.n_run_disagree}/{res.n_runs}")
                else:
                    prog.step(f"{bk}/{ref} d={p.d} (not identifiable)")
            if cell_results:
                rows.append(_pool_cell(f"{bk}/{ref}", cell_results))
        clf.close()
    prog.close()
    report_reseed(rows, args.N, args.R, args.K, args.pilot)


def run_image(args):
    import bl_models as M
    paths = sorted(glob.glob(os.path.join(args.images_dir, args.glob)))
    paths = paths[:args.subset] if args.subset else paths
    bks = (list(M.IMAGE_BACKBONES) if args.backbones == "all"
           else args.backbones.split(","))
    rfs = (list(M.IMAGE_REFERENCES) if args.references == "all"
           else args.references.split(","))
    rows = []
    n_items = len(bks) * len(rfs) * len(paths)
    if n_items == 0:
        print(f"  [tier2b-image] glob '{os.path.join(args.images_dir, args.glob)}'"
              f" matched {len(paths)} files -> nothing to run.")
    else:
        # Each item = R independent audits at N, so a heavy R is SLOW (e.g. R=40,
        # N=2000, ResNet-50 ~ 2 min/image). The bar advances once PER IMAGE, so a
        # long pause before the first tick is expected, not a hang.
        print(f"  [tier2b-image] {len(paths)} images x {len(bks)}x{len(rfs)} cells,"
              f" R={args.R} seeds each (~expect minutes per image at large R).",
              flush=True)
    prog = Progress(max(n_items, 1),
                    f"tier2b-image N={args.N} R={args.R}")
    for bk in bks:
        try:
            clf = M.ImageClassifier(backbone=bk)
        except Exception as e:
            prog.note(f"[skip {bk}] {e}")
            continue
        for ref in rfs:
            cell_results = []
            for pi, path in enumerate(paths):
                img = clf.load_image(path)
                p = image_probe(clf, img, ref, args.grid)
                if p is None:
                    prog.step(f"{bk}/{ref} (skipped)")
                    continue
                try:
                    res = audit_probe(
                        p, args.N, args.R, 1, seed0=pi,
                        ucb=(args.pilot == "ucb"), B_pop_ub=args.B_pop_ub,
                    )
                except ValueError as e:
                    prog.step(f"{bk}/{ref} d={p.d} (skipped: {e})")
                    continue
                if res is not None:
                    cell_results.append(res)
                    prog.step(f"{bk}/{ref} d={p.d} "
                              f"dis={res.n_viol}/{res.n_cert_checks} "
                              f"rundis={res.n_run_disagree}/{res.n_runs}")
                else:
                    prog.step(f"{bk}/{ref} (not identifiable)")
            if cell_results:
                rows.append(_pool_cell(f"{bk}/{ref}", cell_results))
        clf.close()
    prog.close()
    report_reseed(rows, args.N, args.R, 1, args.pilot)


# =========================================================================== #
#  Self-test: pure-numpy synthetic probe, no models, no downloads
# =========================================================================== #
def _make_synthetic_probe(d, n_active, m_resid, sigma_obs, seed,
                          amp_lo=0.15, amp_hi=0.5, degenerate=False):
    """Bounded pure-numpy probe for end-to-end plain/UCB smoke tests.

    Query noise is Rademacher +/-sigma_obs, hence conditionally mean-zero,
    sigma_obs-sub-Gaussian, and bounded.  The response therefore has a known
    absolute bound, allowing the UCB path to exercise a genuine R_val/B_pop bound.
    This self-test is not the Gaussian-noise Tier-1 experiment.
    """
    rng = np.random.default_rng(seed)
    beta = np.zeros(d)
    active = rng.choice(d, size=min(n_active, d), replace=False)
    beta[active] = (
        rng.choice([-1.0, 1.0], size=len(active))
        * rng.uniform(amp_lo, amp_hi, size=len(active))
    )
    if degenerate:
        beta *= 0.15
    trip = rng.choice(d, size=min(3, d), replace=False)
    amp = math.sqrt(max(m_resid, 0.0))

    def query(Z):
        Zc = 2.0 * (Z - 0.5)
        main = Zc @ beta
        hi = amp * np.prod(Zc[:, trip], axis=1) if len(trip) == 3 else 0.0
        if sigma_obs > 0:
            rr = np.random.default_rng()
            noise = sigma_obs * rr.choice([-1.0, 1.0], size=Z.shape[0])
        else:
            noise = 0.0
        return main + hi + noise

    output_abs_bound = float(np.abs(beta).sum() + abs(amp) + abs(sigma_obs))
    return Probe(
        d=d, target=0, query_fn=query, sigma_obs=sigma_obs,
        output_abs_bound=output_abs_bound,
    ), beta


def run_selftest(args):
    print("=" * 72)
    print("TIER 2b SELF-TEST -- pure numpy, no models. Exercises the full "
          "reseed\n           pipeline on synthetic probes with known structure.")
    print("           Expected: strong -> stable; near-floor -> band churn;")
    print("                     degenerate -> per-run rate rises toward 1/pK.")
    print("=" * 72)
    cells = [
        # strong signal, low noise: everything above 2*floor, near-perfect
        ("synth/strong",    dict(d=12, n_active=4, m_resid=0.02, sigma_obs=0.02,
                                 amp_lo=0.30, amp_hi=0.60)),
        # near-floor amplitudes: coords land in the unresolved band -> J_band < 1
        ("synth/nearfloor", dict(d=12, n_active=6, m_resid=0.05, sigma_obs=0.05,
                                 amp_lo=0.05, amp_hi=0.14)),
        # degenerate (ViSoBERT/zero analogue): near-constant response, pilot
        # strained -> per-coordinate / per-run rate approaches 1/pK
        ("synth/degenerate", dict(d=12, n_active=4, m_resid=0.08, sigma_obs=0.02,
                                  amp_lo=0.10, amp_hi=0.20, degenerate=True)),
    ]
    rows = []
    for name, cfg in cells:
        results = []
        for si in range(args.subset or 8):
            probe, _ = _make_synthetic_probe(seed=si, **cfg)
            res = audit_probe(
                probe, args.N, args.R, 1, seed0=si,
                ucb=(args.pilot == "ucb"), B_pop_ub=args.B_pop_ub,
            )
            if res is not None:
                results.append(res)
        if results:
            rows.append(_pool_cell(name, results))
    report_reseed(rows, args.N, args.R, 1, args.pilot)


# =========================================================================== #
#  CLI
# =========================================================================== #
def build_parser():
    p = argparse.ArgumentParser(description="Tier 2b independent-reseeding audit")
    p.add_argument("mode", choices=["nlp", "image", "selftest"])
    p.add_argument("--N", type=int, default=2000, help="single fixed budget")
    p.add_argument("--R", type=int, default=40, help="number of independent seeds")
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--backbones", default="all")
    p.add_argument("--references", default="all")
    p.add_argument("--dataset", default="sst2")
    p.add_argument("--sentences", default="sst2_samples.txt")
    p.add_argument("--max_free", type=int, default=40)
    p.add_argument("--images_dir", default="benchmark_50")
    p.add_argument("--glob", default="*.JPEG")
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--subset", type=int, default=10)
    p.add_argument("--pilot", choices=["plain", "ucb"], default="plain",
                   help="plain = plug-in stability diagnostic (split=3); "
                        "ucb = held-out mismatch-energy UCB plugged directly "
                        "into the two-term floor (split=4)")
    p.add_argument(
        "--B_pop_ub", type=float, default=None,
        help="optional valid upper bound on ||r_{>K}||_inf. Probability-output "
             "NLP probes derive one automatically; logit-image UCB mode requires "
             "this argument."
    )
    return p


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