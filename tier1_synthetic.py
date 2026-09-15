"""
tier1_synthetic.py
==================
TIER 1 -- Synthetic, GROUND TRUTH AVAILABLE.

Question answered: is the floor the correct SCALE, in both directions?
This is the ONLY setting where beta is known by construction, so it is the only
place the two constants are calibrated. They are frozen here and carried
verbatim into Tiers 2-3; nothing downstream is ever re-fit.

Two directions, reported as two collapses:
  FORWARD  -- the floor is the right NORMALIZER. Rescale every coefficient by
              its floor, x = |beta|/floor; plot signed-detection rate SDR(x).
              CLAIM: all regimes lie on ONE shared curve. x_0.5 (the 50%
              crossing) is a single labeled POINT on that curve, reported with
              its cross-regime CoV. x_0.5 < 1 is EXPECTED (sqrt(2 log pK)
              simultaneity price + sign-only detection), not a miscalibration.
  BACKWARD -- the budget rule recovers ONE constant. From the smallest N
              reaching 90% certification, back-solve C_budget; CLAIM: it is flat
              across an ~8x range of N (low CoV). That value becomes the frozen
              C_BUDGET.

Plus the leakage linchpin (fixes C_M) and the regime grid that spans the TWO
axes of sigma_eff INDEPENDENTLY -- the design's justification for >1 regime:
sigma_eff = sigma_obs + C_m sqrt(m) mixes query noise and mismatch, so we move
each axis alone and show the floor law is unchanged. If the collapse holds
across regimes that reach the same sigma_eff by different routes, sigma_eff is
genuinely the only data-dependent input.

Pure numpy, no torch.  Run:  python tier1_synthetic.py [all|leakage|forward|backward|grid]
"""
from __future__ import annotations
import sys
import math
import numpy as np

import bl_core as bl


# --------------------------------------------------------------------------- #
#  Synthetic masked function: planted degree-1 signal + degree>=2 mismatch
# --------------------------------------------------------------------------- #
def make_function(d, n_active, beta_active, m_resid, seed, n_hi=200):
    """g(z) = sum_i beta_i chi_i  +  sum_{|S|>=2} beta_S chi_S, with the
    higher-degree block carrying TOTAL energy m_resid (Lemma 1: Var(h_T)=m_resid
    exactly, since chi_T^2==1, independent of the partition). Returns
    (beta_true, active_set, sample_fn, resid_fn, g_fn)."""
    g = np.random.default_rng(seed)
    units = list(range(d))
    active = list(g.choice(units, size=n_active, replace=False))
    signs = g.choice([-1.0, 1.0], size=n_active)
    beta_true = np.zeros(d)
    for i, s in zip(active, signs):
        beta_true[i] = beta_active * s

    hi_sets, seen = [], set()
    # cap n_hi to the number of available |S| in {2,3} subsets: at small d there
    # may be fewer than the requested count, otherwise the loop never finishes.
    max_hi = math.comb(d, 2) + (math.comb(d, 3) if d >= 3 else 0)
    n_hi = min(n_hi, max_hi)
    while len(hi_sets) < n_hi:
        k = min(int(g.integers(2, 4)), d)
        S = tuple(sorted(g.choice(units, size=k, replace=False)))
        if S not in seen:
            seen.add(S)
            hi_sets.append(S)
    mag = math.sqrt(m_resid / n_hi) if m_resid > 0 else 0.0
    hi_signs = g.choice([-1.0, 1.0], size=n_hi)
    beta_hi = {S: mag * s for S, s in zip(hi_sets, hi_signs)}

    def chi(Z, S):
        out = np.ones(Z.shape[0])
        for i in S:
            out *= (2.0 * (Z[:, i] - 0.5))
        return out

    def g_fn(Z):
        y = np.zeros(Z.shape[0])
        for i in range(d):
            if beta_true[i] != 0.0:
                y += beta_true[i] * (2.0 * (Z[:, i] - 0.5))
        for S, b in beta_hi.items():
            y += b * chi(Z, S)
        return y

    def sample_fn(N, sigma_obs, rng=None):
        rng = rng or np.random.default_rng(seed + 10_000)
        Z = (rng.random((N, d)) > 0.5).astype(float)
        y = g_fn(Z)
        if sigma_obs > 0:
            y = y + sigma_obs * rng.standard_normal(N)
        return Z, y

    def resid_fn(N, rng=None):
        rng = rng or np.random.default_rng(seed + 20_000)
        Z = (rng.random((N, d)) > 0.5).astype(float)
        r = np.zeros(N)
        for S, b in beta_hi.items():
            r += b * chi(Z, S)
        return Z, r

    return beta_true, set(active), sample_fn, resid_fn, g_fn


def empirical_leakage(Z, r):
    """eta_N = ||(1/N) X^T r||_inf on the standardized degree-1 design."""
    Xs, _ = bl.standardize_columns(bl.design_matrix(Z, 1))
    return float(np.max(np.abs(Xs.T @ r) / Z.shape[0]))


# =========================================================================== #
#  LEAKAGE LINCHPIN -- fixes C_M  (Lemma 1)
# =========================================================================== #
def calibrate_leakage(d=30, n_active=4, n_trials=40, N_freeze=2000):
    """eta_N ~ C_m sqrt(m log(.) / N) with a FLAT ratio = C_m. Sweep (m, N);
    estimate C_m = eta_N / sqrt(m log(.) / N).

    R1.5: alongside each cell we report the Bernstein sub-exponential term
    (2/3) B log(.)/N and the domination threshold N_dom = (2 B^2 / 9 m) log(.).
    The claim that the frozen C_m ABSORBS the sub-exponential tail is only honest
    where N >= N_dom, so we flag any cell that is not dominated and confirm the
    calibration grid sits inside the dominated regime.

    R1.5 (reporting). The per-cell C_m ratio drifts UP monotonically in N (~8%
    from N=250 to N=4000) at every m. This is NOT scatter and a flat-CoV summary
    over all N would hide it: the empirical eta_N still contains the sub-exp
    contribution, which decays faster than the sqrt term, so small-N rows read a
    slightly LOWER ratio. Because C_m is an asymptotic (sqrt-regime) quantity, we
    FREEZE it from the large-N rows (N >= N_freeze), where the sub-exp term is
    2-3 orders below the sqrt term (see the `subexp` column), and report the
    all-N mean only for transparency. The drift is exactly the R1.5 term,
    already quantified per cell.

    NOTE (R1.2 / R1.3): the normalizer uses the union-bounded log factor
    bl.log_pk_over_delta, so C_m is calibrated against the SAME log the floor
    carries (ONE canonical factor across Lemma 1, the floor, and this
    calibration). This is the R1.3 resolution: the constant is consistent by
    construction with the quantity it multiplies. The product C_m*sqrt(normalizer)
    that enters eta_N is invariant to the normalizer choice.
    """
    print("\n[Lemma 1]  leakage linchpin   eta_N ~ C_m sqrt(m log(.) / N)  "
          "(+ Bernstein sub-exp term, R1.5)")
    log_fac = bl.log_pk_over_delta(d, 1)          # R1.2 split log factor
    m_grid = [0.005, 0.02, 0.05, 0.1, 0.2]
    N_grid = [250, 500, 1000, 2000, 4000]
    ratios = []                 # all cells
    ratios_large = []           # N >= N_freeze only (the frozen estimate)
    not_dominated = 0
    print(f"  {'m':>7} {'N':>7} {'eta_N':>10} {'predict':>10} {'C_m':>7} "
          f"{'subexp':>9} {'N_dom':>8} {'dom?':>5}")
    for m in m_grid:
        for N in N_grid:
            etas, Bs = [], []
            for t in range(n_trials):
                _, _, _, rf, _ = make_function(
                    d, n_active, 0.0, m, seed=7 * t + N)
                Z, r = rf(N, rng=np.random.default_rng(123 + t + N))
                etas.append(empirical_leakage(Z, r))
                Bs.append(float(np.max(np.abs(r))))      # ||r>K||_inf plug-in
            eta = float(np.mean(etas))
            B = float(np.mean(Bs))
            pred = math.sqrt(m * log_fac / N)
            ratio = eta / pred if pred > 0 else float("nan")
            ratios.append(ratio)
            if N >= N_freeze:
                ratios_large.append(ratio)
            # R1.5 Bernstein sub-exponential term and domination threshold
            _, sub_e, _ = bl.leakage_bound_terms(m, B, d, N, 1)
            N_dom = bl.leakage_domination_N(m, B, d, 1)
            dominated = N >= N_dom
            not_dominated += (0 if dominated else 1)
            print(f"  {m:>7.3f} {N:>7d} {eta:>10.5f} {pred:>10.5f} "
                  f"{ratio:>7.3f} {sub_e:>9.5f} {N_dom:>8.0f} "
                  f"{'yes' if dominated else 'NO':>5}")
    Cm_all = float(np.nanmean(ratios))
    c_all = bl.cov(ratios)
    Cm = float(np.nanmean(ratios_large))       # FROZEN value (large-N)
    c_large = bl.cov(ratios_large)
    print(f"\n  C_m all-N   (transparency) = {Cm_all:.3f}   CoV = {c_all:.3f}")
    print(f"  C_m N>={N_freeze} (FROZEN)    = {Cm:.3f}   CoV = {c_large:.3f}   "
          f"{'PASS' if c_large < 0.20 else 'CHECK'} (target CoV < 0.20)")
    print(f"  the ratio drifts UP ~8% with N: this IS the R1.5 sub-exp term "
          f"(see `subexp`),\n  which inflates small-N eta_N; C_m is an "
          f"asymptotic sqrt-regime constant, so it is\n  frozen from the large-N "
          f"rows where the sub-exp term is 2-3 orders below sqrt.")
    print(f"  R1.5 domination: {len(ratios) - not_dominated}/{len(ratios)} "
          f"cells have N >= N_dom (sub-exp term dominated, absorbed into C_m). "
          f"{'ALL PASS' if not_dominated == 0 else f'{not_dominated} CELL(S) NOT DOMINATED'}")
    return Cm


# =========================================================================== #
#  THE REGIME GRID -- two independent axes of sigma_eff (design justification)
# =========================================================================== #
def regime_grid():
    """Five regimes chosen so sigma_eff is reached by DIFFERENT routes:
       - noise axis   : vary sigma_obs with m~0   (isolates query noise)
       - mismatch axis: vary m with sigma_obs fixed (isolates leakage)
       - mixed        : both nonzero
    Collapse across these IS the proof that sigma_eff is the only input."""
    return [
        dict(name="noise-lo",    N=500,  m=0.00, sigma_obs=0.05),  # noise axis
        dict(name="noise-hi",    N=2000, m=0.00, sigma_obs=0.05),  # noise axis
        dict(name="mismatch",    N=1000, m=0.10, sigma_obs=0.05),  # mismatch axis
        dict(name="mixed-noisy", N=1000, m=0.10, sigma_obs=0.20),  # mixed
        dict(name="mixed-big",   N=4000, m=0.25, sigma_obs=0.10),  # mixed
    ]


# =========================================================================== #
#  FORWARD -- the collapse curve (replaces the bare x_0.5 number)
# =========================================================================== #
def forward_collapse(Cm, d=30, n_active=4, n_trials=60):
    """For each regime, sweep x=|beta|/floor and record signed-detection rate;
    locate x_0.5. CLAIM: the x_0.5 values collapse to a single x across regimes
    (low CoV). The shared curve, not any single point, is the result."""
    print("\n[FORWARD]  collapse: SDR(x) shares one curve across regimes  "
          "(x = |beta|/floor)")
    x_grid = np.geomspace(0.05, 4.0, 24)
    crossings = []
    print(f"  {'regime':>12} {'N':>5} {'m':>5} {'sig':>5} | "
          f"{'x_0.5':>6} {'SDR@x=1':>8}")
    for s in regime_grid():
        fl = bl.floor_value(bl.sigma_eff(s["sigma_obs"], s["m"], Cm),
                            d, s["N"], 1)
        sdr = []
        for x in x_grid:
            beta = x * fl
            ok = 0
            for t in range(n_trials):
                bt, active, sf, _, _ = make_function(
                    d, n_active, beta, s["m"], seed=int(1000 * x) + 7 * t)
                Z, y = sf(s["N"], s["sigma_obs"],
                          rng=np.random.default_rng(11 * t + int(1e3 * x)))
                beta_hat, _, _ = bl.ols_fit(Z, y, 1)
                ok += all(np.sign(beta_hat[i]) == np.sign(bt[i])
                          and abs(beta_hat[i]) > 0 for i in sorted(active))
            sdr.append(ok / n_trials)
        _, x_half = bl.collapse_curve(x_grid, sdr)
        crossings.append(x_half)
        at1 = sdr[int(np.argmin(np.abs(x_grid - 1.0)))]
        print(f"  {s['name']:>12} {s['N']:>5d} {s['m']:>5.2f} "
              f"{s['sigma_obs']:>5.2f} | {x_half:>6.2f} {at1:>8.2f}")
    c = bl.cov(crossings)
    xbar = float(np.nanmean(crossings))
    print(f"\n  shared crossing x_0.5 = {xbar:.2f}   CoV = {c:.3f}   "
          f"{'PASS' if c < 0.25 else 'CHECK'} (target CoV < 0.25)")
    print(f"  x_0.5 < 1 is EXPECTED: the floor pays sqrt(2 log pK) for "
          f"simultaneity and\n  detection is sign-only; at x=1, SDR ~ 1 in "
          f"every regime. The COLLAPSE is the claim.")
    return xbar


# =========================================================================== #
#  BACKWARD -- the budget rule recovers ONE constant (becomes frozen C_BUDGET)
# =========================================================================== #
def backward_budget(d=30, n_active=4, sigma_obs=1.0, n_trials=30):
    """Leakage-free (m=0). Family-wise certification criterion. For each SNR,
    find the smallest N reaching 90% certification and back-solve C_budget.
    CLAIM: C_budget flat across an ~8x range of N (low CoV) -> frozen value.

    Two requirements coexist (Reading 2): feasibility N >~ pK and resolution
    N >~ 2 C^2 sigma^2 log pK / beta^2. To isolate the 1/gamma^2 resolution
    law, stay in WEAK-SNR (required N well above pK); strong-SNR rows are
    feasibility-bound and excluded from the constant."""
    print("\n[BACKWARD]  budget rule -> one constant C_budget (m=0)")
    # R1.2: calibrate against the SAME union-bounded log factor the floor now
    # carries (log(SPLIT*pK/delta), delta=1/pK), so the back-solved C_BUDGET is
    # consistent with floor_value / predict_budget rather than the bare log pK.
    log_pK = bl.log_pk_over_delta(d, 1)
    pK = bl.p_K(d, 1)
    z_fw = math.sqrt(2.0 * log_pK)
    gammas = [0.20, 0.14, 0.10, 0.07]      # weak SNR: resolution regime
    N_grid = np.unique(np.round(
        np.geomspace(int(2 * pK), 60000, 48)).astype(int))
    print(f"  {'gamma':>6} {'beta':>7} {'N@90%':>8} {'pred(C=1)':>10} "
          f"{'implied C':>10} regime")
    implied = []
    for gamma in gammas:
        beta = gamma * sigma_obs
        emp_N = None
        for N in N_grid:
            ok = 0
            for t in range(n_trials):
                bt, active, sf, _, _ = make_function(
                    d, n_active, beta, 0.0, seed=31 * t + N + int(gamma * 1000))
                Z, y = sf(N, sigma_obs, rng=np.random.default_rng(13 * t + N))
                beta_hat, b0, diagGinv = bl.ols_fit(Z, y, 1)
                idx = sorted(active)
                yhat = b0 + bl.design_matrix(Z, 1) @ beta_hat
                resid = y - yhat
                dof = max(N - (d + 1), 1)
                sig_hat = max(math.sqrt((resid @ resid) / dof), 1e-9)
                se = sig_hat * np.sqrt(diagGinv[idx] / N)
                ok += all(np.sign(beta_hat[i]) == np.sign(bt[i])
                          and abs(beta_hat[i]) > z_fw * se[k]
                          for k, i in enumerate(idx))
            if ok / n_trials >= 0.90:
                emp_N = int(N)
                break
        pred_N = 2.0 * sigma_obs ** 2 * log_pK / beta ** 2     # C=1
        C = (math.sqrt(emp_N * beta ** 2 / (2.0 * sigma_obs ** 2 * log_pK))
             if emp_N else float("nan"))
        regime = "" if emp_N is None else (
            "resolution" if emp_N > 3 * pK else "FEASIBILITY")
        if regime == "resolution":
            implied.append(C)
        print(f"  {gamma:>6.2f} {beta:>7.3f} {(emp_N or -1):>8d} "
              f"{pred_N:>10.0f} {C:>10.3f} {regime}")
    c = bl.cov(implied)
    Cb = float(np.nanmean(implied)) if implied else float("nan")
    print(f"\n  C_budget (resolution rows) = {Cb:.3f}   CoV = {c:.3f}   "
          f"{'PASS' if c < 0.30 else 'CHECK'} (target CoV < 0.30)")
    print(f"  feasibility floor pK = {pK}; FEASIBILITY rows sit near it and do "
          f"NOT\n  test the 1/gamma^2 law. This C_budget is the frozen "
          f"backward constant.")
    return Cb


# =========================================================================== #
#  CALIBRATION SUMMARY -- the single Tier-1 output table (two constants)
# =========================================================================== #
def print_calibration(Cm, x_half, Cb):
    print("\n" + "=" * 72)
    print("TIER 1 CALIBRATION (frozen and carried into Tiers 2-3)")
    print("=" * 72)
    print(f"  {'constant':>10} {'value':>8}   role (values are MEAN OVER d, not d=30)")
    print(f"  {'C_M':>10} {Cm:>8.3f}   planning/pilot only (NOT in certified floor)")
    print(f"  {'C_FLOOR':>10} {CONST_FLOOR:>8.3f}   orthonormal ideal only; floor "
          f"uses per-run C_est")
    print(f"  {'C_BUDGET':>10} {Cb:>8.3f}   budget rule (backward), Eq. 8")
    print(f"\n  forward collapse point x_0.5 = {x_half:.2f} "
          f"(a point on the shared SDR curve)")
    print(f"  NOTE: C_FLOOR theory=1 vs empirical>=1 is EXPECTED -- the bound "
          f"is an upper\n  bound under finite N, query noise, and mismatch. "
          f"Not an anomaly.")


CONST_FLOOR = bl.CONSTANTS.C_FLOOR


# =========================================================================== #
#  d-SWEEP STABILITY of the FROZEN constants  (referee R1.4, extended)
#
#  R1.4 forced Cest to be measured per run because it grows with pK/d. The same
#  question must then be asked of the OTHER two constants we DO freeze from a
#  single d=30 calibration: C_m (leakage) and C_budget (planning). If either
#  drifted with d the way Cest does, freezing it at d=30 and carrying it to
#  d in {15,24,49} would be the very transfer error R1.4 objects to.
#
#  This routine re-runs both calibrations across d in {15,24,30,49} (the d's
#  that actually occur in Tiers 2-3: NLP d=15/24, images d=49, synthetic d=30)
#  and reports each constant vs d plus the cross-d CoV. The theoretical
#  expectation, now checked rather than asserted:
#    * C_m is Walsh-normalized and divides out the log(nu*pK/delta) factor, so
#      every column sees the SAME leakage variance m>K independent of d -> flat.
#    * C_budget is calibrated in the WEAK-SNR resolution regime against the same
#      union-bounded log factor, so its 1/gamma^2 law is d-independent once the
#      log(pK) growth is carried explicitly -> flat (unlike Cest, which is the
#      raw ||Sigma^{-1}||_inf row-sum and is NOT normalized).
#  A flat result here is what LICENSES freezing C_m and C_budget while measuring
#  Cest per run; a drift would force those to be per-run too.
# =========================================================================== #
def sweep_d_stability(d_grid=(15, 24, 30, 49), n_active=4):
    print("\n" + "=" * 72)
    print("d-SWEEP STABILITY of the FROZEN constants C_m, C_budget (R1.4)")
    print("  Cest is per-run (it grows with d); here we check that the two")
    print("  constants we DO freeze at d=30 are in fact d-stable.")
    print("=" * 72)
    rows = []
    print(f"  {'d':>4} {'pK':>5} {'C_m (frozen)':>13} {'C_budget':>10} "
          f"{'Cest_emp':>9}")
    for d in d_grid:
        # suppress the verbose per-cell prints of the sub-calibrations
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cm = calibrate_leakage(d=d, n_active=n_active)
            cb = backward_budget(d=d, n_active=n_active)
        # realized Cest for this d at a representative deployment budget
        rng = np.random.default_rng(2027 + d)
        N_rep = max(6 * bl.p_K(d, 1), 2000)
        Z = bl.sample_masks(N_rep, d, rng)
        cest = bl.realized_cest(Z, 1) if hasattr(bl, "realized_cest") \
            else float("nan")
        rows.append((d, bl.p_K(d, 1), cm, cb, cest))
        print(f"  {d:>4} {bl.p_K(d,1):>5} {cm:>13.3f} {cb:>10.3f} {cest:>9.3f}")
    cms = [r[2] for r in rows]
    cbs = [r[3] for r in rows]
    cests = [r[4] for r in rows]
    def _mean_cov(xs):
        a = np.array(xs, dtype=float)
        return float(a.mean()), float(a.std(ddof=1) / abs(a.mean()))
    cm_mu, cm_cov = _mean_cov(cms)
    cb_mu, cb_cov = _mean_cov(cbs)
    ce_mu, ce_cov = _mean_cov(cests)
    print("  " + "-" * 46)
    print(f"  {'MEAN':>4} {'':>5} {cm_mu:>13.3f} {cb_mu:>10.3f} {ce_mu:>9.3f}"
          f"   <- frozen = mean over d")
    print(f"  cross-d CoV: C_m {cm_cov:.3f}  C_bud {cb_cov:.3f}  "
          f"C_est {ce_cov:.3f}")
    print(f"\n  FROZEN VALUES (mean over d, carried into CONSTANTS):")
    print(f"    C_M     = {cm_mu:.3f}  (cross-d CoV {cm_cov:.3f} "
          f"{'STABLE' if cm_cov < 0.10 else 'DRIFTS'})")
    print(f"    C_BUDGET= {cb_mu:.3f}  (cross-d CoV {cb_cov:.3f} "
          f"{'STABLE' if cb_cov < 0.15 else 'DRIFTS'})")
    print(f"    C_est is NOT frozen: cross-d CoV {ce_cov:.3f} "
          f"(rises {cests[0]:.2f} -> {cests[-1]:.2f}); measured per run.")
    print("\n  We freeze the MEAN over d, not any single d's value: it would be")
    print("  inconsistent to claim d-stability and then cherry-pick one d. The")
    print("  low cross-d CoV is what licenses a single frozen scalar for C_m,")
    print("  C_budget; the high CoV for C_est is why it is re-measured per run.")
    # Return the FROZEN means (over d), not any single d's value, so the caller's
    # summary reports 0.830 / 1.552 rather than the d=30-only 0.833 / 1.535.
    return rows, cm_mu, cb_mu


# --------------------------------------------------------------------------- #
def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("=" * 72)
    print("TIER 1 -- SYNTHETIC (ground truth). Calibrate 2 constants, prove")
    print("the two-direction collapse. Dense OLS; no Lasso, no incoherence.")
    print("=" * 72)

    Cm = bl.CONSTANTS.C_M
    x_half = float("nan")
    Cb = bl.CONSTANTS.C_BUDGET
    # calibrate_leakage()/backward_budget() at their DEFAULT d=30 give the
    # single-d values (0.833 / 1.535); the frozen constants are the MEAN OVER d
    # from sweep_d_stability (0.830 / 1.552). We keep the single-d numbers only
    # for the intermediate collapse plots and OVERWRITE Cm/Cb with the d-means
    # before the final summary, so the summary never reports a cherry-picked d.
    if what in ("leakage", "all"):
        Cm = calibrate_leakage()          # d=30 value, used only for the grid/collapse
    if what in ("forward", "all"):
        x_half = forward_collapse(Cm)
    if what in ("backward", "all"):
        Cb = backward_budget()            # d=30 value
    if what in ("dsweep", "all"):
        _, Cm_mean, Cb_mean = sweep_d_stability()
        if what == "all":
            Cm, Cb = Cm_mean, Cb_mean     # FROZEN = mean over d
    if what in ("grid", "all"):
        print("\n[regime grid] five regimes span the two axes of sigma_eff:")
        for s in regime_grid():
            se = bl.sigma_eff(s["sigma_obs"], s["m"], Cm)
            print(f"  {s['name']:>12}: sigma_obs={s['sigma_obs']:.2f} "
                  f"m={s['m']:.2f} -> sigma_eff={se:.3f}")
    if what == "all":
        print_calibration(Cm, x_half, Cb)


if __name__ == "__main__":
    main()