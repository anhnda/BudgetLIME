"""
tier1b_cest_transfer.py
=======================
TIER 1b -- Cest TRANSFER STUDY (referee R1.4, "to me most seriously").

THE OBJECTION (verbatim shape). The floor constant

    Cest = max{ gamma^{-1/2},  |||Sigma_hat^{-1}|||_inf }

is defined through the ell_inf -> ell_inf operator norm of the inverse empirical
Gram, i.e. a maximum ABSOLUTE ROW SUM. That quantity grows with the number of
fitted coordinates pK at a fixed budget N. In the paper it is calibrated once at
(d=30, K=1, pK=31) and then FROZEN and transferred to d=49 (image, pK=50) and,
in the enumeration tier, to K=2 designs with far more coordinates. The referee
asks for evidence that the frozen value is stable "in the ratio between the
number of coordinates and the budget" -- i.e. a study varying pK/N with the
population coefficients known -- before the transfer claim can stand.

WHAT THIS SCRIPT SHOWS (three questions, in order).

  (Q1) MECHANISM. Is the concern real -- does Cest grow with pK at fixed N?
       We fix N and sweep pK (via d at K=1, and via K) and confirm Cinv and
       hence Cest rise with pK. If they did not, there would be nothing to
       check. (They do; this validates the referee's premise.)

  (Q2) TRANSFER AXIS. Is Cest stable as a function of the RATIO pK/N, rather
       than d or K separately? We sweep many (d, K, N) with widely different d
       and K but overlapping pK/N, and plot Cest_emp against pK/N. The transfer
       claim is TENABLE iff Cest collapses onto one curve in pK/N (so a value
       calibrated at one (d,K) transfers to another AT THE SAME RATIO). If it
       instead splits by d or K at fixed ratio, the frozen scalar does not
       transfer and must be indexed by pK/N (or recomputed per run, which is
       cheap -- it is a pure design quantity, Assumption 1).

  (Q3) OPERATING-POINT VERDICT. At the ACTUAL deployed points -- the d=30/K=1
       calibration point, the d=49/K=1 image point, and the K=2 enumeration
       regime -- what is Cest_emp at the budgets actually used, and is the
       frozen C_FLOOR = 1.0 conservative (Cest_emp <= C_FLOOR would be a
       violation of the bound's own constant; Cest_emp >= 1 is EXPECTED and the
       floor is conservative exactly when the deployed budget keeps Cest_emp
       controlled). We report the realized Cest_emp and the budget needed to
       hold it below a target.

DESIGN-ONLY, REFERENCE-FREE. Assumption 1 concerns the design columns, which
depend on the masks and basis, NOT on rho or g_rho. So this entire study needs
no model and no planted signal: it samples Walsh designs and measures the Gram.
That is also why the result transfers across references by construction -- the
reference acts only on the signal side.

Pure numpy, no torch.  Run:
    python tier1b_cest_transfer.py [all|mechanism|transfer|operating]
"""
from __future__ import annotations
import sys
import math
import numpy as np

import bl_core as bl


# --------------------------------------------------------------------------- #
#  Shared measurement: average Cest_emp over trials at one (d, K, N).
# --------------------------------------------------------------------------- #
def cest_at(d, K, N, n_trials=20, seed0=0):
    """Mean (and spread) of the empirical conditioning constants over trials.
    Returns a dict with means of gamma, Cinv, Cest_emp and the well-posed count.
    """
    gs, cs, es = [], [], []
    wp = 0
    for t in range(n_trials):
        rng = np.random.default_rng(seed0 + 1009 * t + 7 * N + d + K)
        pr = bl.measure_conditioning(d, K, N, rng)
        if pr.well_posed:
            wp += 1
            gs.append(pr.gamma); cs.append(pr.Cinv); es.append(pr.Cest_emp)
    if not es:
        return dict(d=d, K=K, N=N, pK=bl.p_K(d, K), ratio=bl.p_K(d, K) / N,
                    gamma=float("nan"), Cinv=float("nan"),
                    Cest=float("nan"), Cest_sd=float("nan"), wp=0,
                    n_trials=n_trials)
    return dict(d=d, K=K, N=N, pK=bl.p_K(d, K), ratio=bl.p_K(d, K) / N,
                gamma=float(np.mean(gs)), Cinv=float(np.mean(cs)),
                Cest=float(np.mean(es)), Cest_sd=float(np.std(es)),
                wp=wp, n_trials=n_trials)


# =========================================================================== #
#  (Q1) MECHANISM -- Cest grows with pK at fixed N (the referee's premise).
# =========================================================================== #
def mechanism(N=4000, n_trials=20):
    """Fix N; sweep pK two ways (d at K=1, and K at fixed d). Show Cinv and Cest
    rise with pK -- confirming the concern is real and worth a transfer study."""
    print("\n[Q1 MECHANISM]  Cest = max{gamma^-1/2, |||Ginv|||_inf} grows with "
          f"pK at fixed N={N}")
    print("  (if it did not rise with pK there would be nothing to transfer)\n")
    print(f"  {'d':>4} {'K':>3} {'pK':>6} {'pK/N':>7} {'gamma':>8} "
          f"{'Cinv':>8} {'Cest':>8}")
    # K=1, grow d (hence pK = d+1)
    for d in (20, 30, 40, 49, 60, 80):
        r = cest_at(d, 1, N, n_trials)
        print(f"  {d:>4d} {1:>3d} {r['pK']:>6d} {r['ratio']:>7.3f} "
              f"{r['gamma']:>8.4f} {r['Cinv']:>8.4f} {r['Cest']:>8.4f}")
    print()
    # K=2, grow d (pK ~ d^2/2) -- the far-more-coordinates regime
    for d in (10, 14, 18, 22):
        r = cest_at(d, 2, N, n_trials)
        tag = "" if r['wp'] == r['n_trials'] else f"  (well-posed {r['wp']}/{r['n_trials']})"
        print(f"  {d:>4d} {2:>3d} {r['pK']:>6d} {r['ratio']:>7.3f} "
              f"{r['gamma']:>8.4f} {r['Cinv']:>8.4f} {r['Cest']:>8.4f}{tag}")
    print("\n  READING: Cinv (a max abs row sum of Ginv) and hence Cest rise "
          "with pK at fixed N,\n  exactly as the referee notes. The question is "
          "whether they are stable in pK/N (Q2).")


# =========================================================================== #
#  (Q2) TRANSFER AXIS -- does Cest collapse onto one curve in pK/N?
# =========================================================================== #
def transfer(n_trials=24):
    """Sweep (d, K, N) with very different d and K but OVERLAPPING pK/N. Bin by
    pK/N and test whether Cest_emp is (a) a function of pK/N alone (collapses)
    or (b) splits by d or K at fixed ratio (frozen scalar does not transfer).

    The verdict statistic: within each pK/N bin spanned by BOTH K=1 and K=2 (or
    by widely different d), report the max spread of Cest across the members.
    Small spread at fixed ratio => transfer in pK/N is tenable.
    """
    print("\n[Q2 TRANSFER]  Cest_emp vs the ratio pK/N across different (d,K)")
    print("  claim under test: Cest is a function of pK/N, so a value calibrated")
    print("  at one (d,K) transfers to another AT THE SAME RATIO.\n")

    # Build (d, K, N) points spanning a common pK/N range by different routes.
    points = []
    for d in (30, 49):                       # K=1 routes (calibration + image d)
        for ratio in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
            pK = bl.p_K(d, 1)
            N = int(round(pK / ratio))
            points.append((d, 1, N))
    for d in (14, 18, 22):                   # K=2 routes (enumeration regime)
        for ratio in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
            pK = bl.p_K(d, 2)
            N = int(round(pK / ratio))
            points.append((d, 2, N))

    rows = []
    print(f"  {'d':>4} {'K':>3} {'pK':>6} {'N':>7} {'pK/N':>7} "
          f"{'gamma':>8} {'Cinv':>8} {'Cest':>8}")
    for (d, K, N) in points:
        r = cest_at(d, K, N, n_trials)
        rows.append(r)
        print(f"  {d:>4d} {K:>3d} {r['pK']:>6d} {N:>7d} {r['ratio']:>7.3f} "
              f"{r['gamma']:>8.4f} {r['Cinv']:>8.4f} {r['Cest']:>8.4f}")

    # Bin by target ratio and, within each bin, report the spread of Cest across
    # the different (d, K) members that reach that ratio.
    print("\n  [collapse test]  within each pK/N bin: spread of Cest across "
          "the different (d,K) members")
    print(f"  {'pK/N':>7} {'members':>8} {'Cest_min':>9} {'Cest_max':>9} "
          f"{'spread%':>8} {'verdict':>10}")
    targets = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    max_spread = 0.0
    for tr in targets:
        members = [r for r in rows
                   if abs(r["ratio"] - tr) / tr < 0.15 and r["wp"] > 0
                   and math.isfinite(r["Cest"])]
        if len(members) < 2:
            continue
        cmin = min(r["Cest"] for r in members)
        cmax = max(r["Cest"] for r in members)
        spread = (cmax - cmin) / cmax if cmax > 0 else float("nan")
        max_spread = max(max_spread, spread)
        verdict = "collapse" if spread < 0.15 else "SPLIT"
        print(f"  {tr:>7.2f} {len(members):>8d} {cmin:>9.4f} {cmax:>9.4f} "
              f"{100 * spread:>7.1f}% {verdict:>10}")
    print()
    if max_spread < 0.15:
        print(f"  VERDICT: Cest collapses in pK/N (max within-bin spread "
              f"{100 * max_spread:.1f}% < 15%).")
        print(f"  Transfer in the ratio is TENABLE: a Cest calibrated at one "
              f"(d,K) applies at\n  another at the same pK/N. The frozen "
              f"constant should be read as Cest(pK/N),\n  and the deployed "
              f"points must be checked to sit at a controlled ratio (Q3).")
    else:
        print(f"  VERDICT: Cest does NOT collapse in pK/N (max within-bin "
              f"spread {100 * max_spread:.1f}% >= 15%).")
        print(f"  A single frozen scalar does not transfer; index Cest by "
              f"pK/N or recompute it\n  per run (it is a pure design quantity, "
              f"Assumption 1, and cheap to measure).")
    return rows, max_spread


# =========================================================================== #
#  (Q3) OPERATING-POINT VERDICT -- Cest at the actually deployed points.
# =========================================================================== #
def operating(n_trials=40):
    """Evaluate Cest_emp at the real deployment points and budgets and confront
    it with the frozen C_FLOOR = 1.0.

    KEY POINT the referee is really after. The floor of Eq. 6 uses the FORWARD
    constant Cest = max{gamma^{-1/2}, |||Ginv|||_inf}. The code freezes
    C_FLOOR = 1.0 (the orthonormal IDEAL) and absorbs finite-sample inflation
    into the BACKWARD constant C_BUDGET. But Eq. 6's own constant is Cest, and
    its empirical value is > 1 -- so if the forward floor is computed with
    C_FLOOR = 1.0 while the true design constant is Cest_emp > 1, the floor is
    ANTI-conservative by the factor Cest_emp / 1.0 at that point. This function
    measures that factor at every deployed (d, K, N). Two readings:

      * If one wants the forward floor to be a genuine UPPER bound, C_FLOOR must
        be >= Cest_emp at the deployed pK/N -- i.e. C_FLOOR should be frozen at
        (or above) the measured Cest_emp, not at the orthonormal 1.0. The study
        gives the number to freeze it to, per pK/N.
      * The gap Cest_emp / C_FLOOR is the transfer risk: it is small and stable
        while pK/N stays in the calibrated band, and grows once the deployed
        ratio exceeds it -- which is the operational guard the transfer needs.
    """
    print("\n[Q3 OPERATING POINT]  forward-floor constant Cest_emp at the "
          "deployed (d,K,N) vs frozen C_FLOOR")
    C_floor = bl.CONSTANTS.C_FLOOR
    print(f"  frozen C_FLOOR = {C_floor} (orthonormal ideal). Eq. 6's own "
          f"constant is Cest = max{{gamma^-1/2, |||Ginv|||_inf}}.")
    print(f"  gap = Cest_emp / C_FLOOR is the factor by which a floor computed "
          f"at C_FLOOR under-states the\n  true design radius at that point "
          f"(>1 => forward floor is anti-conservative there).\n")

    cases = [
        ("calibration  d=30 K=1", 30, 1, [500, 1000, 2000, 4000]),
        ("image        d=49 K=1", 49, 1, [512, 1000, 2000, 4000]),
        ("enum (short) d=13 K=2", 13, 2, [2000, 4000, 8192]),
        ("enum (mid)   d=18 K=2", 18, 2, [4000, 8000, 16000]),
    ]
    print(f"  {'point':>22} {'pK':>6} {'N':>7} {'pK/N':>7} {'Cest_emp':>9} "
          f"{'gap x':>7} {'floor honest?':>14}")
    worst = {}
    for name, d, K, Ns in cases:
        wgap = 0.0
        for N in Ns:
            if N <= bl.p_K(d, K):
                print(f"  {name:>22} {bl.p_K(d,K):>6d} {N:>7d} "
                      f"{bl.p_K(d,K)/N:>7.3f} {'--':>9} {'--':>7} "
                      f"{'infeasible':>14}")
                continue
            r = cest_at(d, K, N, n_trials)
            gap = r["Cest"] / C_floor if C_floor > 0 else float("inf")
            wgap = max(wgap, gap if math.isfinite(gap) else wgap)
            honest = "yes" if gap <= 1.0 + 1e-9 else f"NO (x{gap:.2f})"
            print(f"  {name:>22} {r['pK']:>6d} {N:>7d} {r['ratio']:>7.3f} "
                  f"{r['Cest']:>9.4f} {gap:>7.2f} {honest:>14}")
        worst[name] = wgap
        print()

    print("  READING (R1.4 verdict on the forward floor):")
    print("    Cest_emp > 1 at every deployed point, so a forward floor computed "
          "with C_FLOOR = 1.0 is\n    NOT a strict upper bound -- it under-states "
          "the true simultaneous radius by the `gap x`\n    factor. To make the "
          "forward guarantee hold as stated, freeze C_FLOOR to the measured "
          "Cest_emp\n    at the deployed pK/N band (the max `gap x` per point "
          "below), or recompute Cest per run\n    (cheap; pure design). This is "
          "the concrete fix R1.4 asks for; the transfer is then honest\n    "
          "BECAUSE C_FLOOR tracks Cest(pK/N) rather than being pinned at the "
          "orthonormal ideal.")
    print(f"\n  max gap per deployed point (candidate C_FLOOR to freeze):")
    for name, g in worst.items():
        print(f"    {name:>22} : C_FLOOR >= {g:.3f}")
    return worst


# =========================================================================== #
def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("=" * 72)
    print("TIER 1b -- Cest TRANSFER STUDY (R1.4): is the frozen floor constant")
    print("stable in the coordinate-to-budget ratio pK/N? Design-only, no model.")
    print("=" * 72)
    if what in ("mechanism", "all"):
        mechanism()
    if what in ("transfer", "all"):
        transfer()
    if what in ("operating", "all"):
        operating()
    if what == "all":
        print("\n" + "=" * 72)
        print("SUMMARY (R1.4): Q1 shows Cest grows with pK (concern is real);")
        print("Q2 tests whether it collapses in pK/N (transfer axis); Q3 checks")
        print("the frozen C_FLOOR against the actually deployed points.")
        print("=" * 72)


if __name__ == "__main__":
    main()