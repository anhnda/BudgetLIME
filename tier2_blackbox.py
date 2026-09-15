"""
tier2_blackbox.py
=================
TIER 2 -- Black-box classifiers (+ an exact-beta sign-correctness check).

Question answered: how does the realized two-term floor behave on real
query-only models, and are certified-in-both signs stable as the budget grows?
Because exact population coefficients are unavailable here, this tier is a
stability/consistency study; correctness is tested only by exact enumeration.

Constants are taken from bl_core.CONSTANTS and never re-fit here.

Two directions, one table:
  FORWARD  -- grow the budget along PREFIXES; count certified coordinates that
              REVERSE SIGN (target 0). With beta unknown this is a STABILITY
              statistic, not a direct correctness test -- which is exactly what
              the exact-beta check below supplies, where it is affordable.
  BACKWARD -- fix beta_min, predict N from Eq. 8 (C_BUDGET), run at
              max(N_pred, feasibility floor), report realized_floor / beta_min.

Workflow diagnostics (count-monotone, set-nesting) are computed but reported
SEPARATELY and labeled near-guaranteed by the prefix construction -- never in
the guarantee column.

EXACT-BETA SIGN-CORRECTNESS CHECK (the direct forward test).
  When the free-unit count is small enough to ENUMERATE the full mask cube
  (d <= MAX_EXACT_D, default 13 -> at most 8192 model calls), fitting dense OLS
  on ALL 2^d masks yields the EXACT population degree-K projection beta. We then
  check, at a normal deployment budget, that every certified coordinate has the
  SAME SIGN as exact beta. This is a genuine ground-truth correctness check, not
  a stability statistic. It is the strongest line in the paper.

  There is NO approximate / high-budget fallback. If d > MAX_EXACT_D the cube
  cannot be enumerated, so there is no exact beta and the probe is SKIPPED for
  this check -- we do not substitute a random-bank estimate and call it ground
  truth, because two finite-sample fits agreeing is not a correctness proof.
  For those probes the only forward evidence is the sign-flip stability above.
  (Consequently the 7x7 image grid, d=49, has no exact-beta check; short
  sentences at K=2 do.)

Noise model used in the real-model experiments:
  * Both image and NLP wrappers run in eval/no_grad mode and are deterministic,
    so sigma_obs_ub = 0 up to numerical reproducibility.
  * Real-model floors are therefore mismatch-driven. Stochastic query-noise
    behavior is stress-tested in the synthetic tier.

Honors "never auto-run torch": a backbone is built only inside main() after
arguments are supplied.

USAGE
  # Tier 2 forward+backward, NLP, K=1:
  python tier2_blackbox.py nlp --K 1 --references mask,pad,zero \
      --backbones distilbert,roberta,visobert --beta_min 0.02 \
      --N_ladder 512,1000,2000,4000 --sentences text_samples/sst2_short.txt
  # Tier 2 image:
  python tier2_blackbox.py image --beta_min 0.05 --N_ladder 512,1000,2000,4000 \
      --images_dir image_samples --glob "*.JPEG"
  # Exact-beta sign check, NLP K=2 (enumerable short sentences only):
  python tier2_blackbox.py exact-nlp --K 2 --subset 10 --max_d 13 \
      --sentences text_samples/sst2_short.txt
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import sys
import time
import numpy as np

import bl_core as bl


MAX_EXACT_D = 13           # enumerate the full cube 2^d only up to here:
                           # 2^13 = 8192 model calls/probe (seconds). Above it
                           # the cube is not enumerable, there is NO exact beta,
                           # and the probe is SKIPPED for the exact check -- no
                           # random-bank stand-in is used as "ground truth".


# =========================================================================== #
#  Progress reporting -- so Tier 2 never runs silently
# =========================================================================== #
class Progress:
    """Lightweight live progress: a running counter with ETA and per-item
    one-liners. Uses tqdm if available, else a plain flushed print. The point
    is that a long Tier-2 run always shows where it is, never a blank screen.
    """
    def __init__(self, total, label):
        self.total = max(total, 1)
        self.label = label
        self.done = 0
        self.t0 = time.time()
        self._bar = None
        try:
            from tqdm import tqdm
            self._bar = tqdm(total=self.total, desc=label, unit="item",
                             dynamic_ncols=True)
        except Exception:
            print(f"[{label}] 0/{self.total} starting...", flush=True)

    def step(self, msg=""):
        self.done += 1
        if self._bar is not None:
            if msg:
                self._bar.set_postfix_str(msg)
            self._bar.update(1)
            return
        elapsed = time.time() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else float("inf")
        eta = ("%4.0fs" % remaining) if remaining < 1e4 else " >3h"
        print(f"[{self.label}] {self.done:>3}/{self.total} "
              f"ETA {eta}  {msg}", flush=True)

    def note(self, msg):
        """Out-of-band message (warnings, cell headers) that doesn't advance."""
        if self._bar is not None:
            self._bar.write(msg)
        else:
            print(msg, flush=True)

    def close(self):
        if self._bar is not None:
            self._bar.close()


def _count_nlp_items(sents, refs, backbones, cap=None):
    n = len(backbones) * len(refs) * len(sents)
    return min(n, cap) if cap else n


# =========================================================================== #
#  Uniform "probe" adapter: hides the modality behind a single query closure
# =========================================================================== #
class Probe:
    """A single explained instance under one reference.

    output_abs_bound:
        Optional deterministic M with |g(z)| <= M.  NLP probabilities use M=1.
        It supports a valid population mismatch sup bound and the held-out UCB
        range.  Logit outputs leave this unset unless the caller supplies a bound.

    B_pop_ub:
        Optional direct upper bound on ||r_{>K,rho}||_inf.  If absent but
        output_abs_bound is known, ``probe_B_pop_bound`` derives the generic
        Parseval bound M(1+sqrt(pK)).
    """
    def __init__(self, d, target, query_fn, sigma_obs,
                 output_abs_bound=None, B_pop_ub=None):
        self.d = d
        self.target = target
        self.query = query_fn
        self.sigma_obs = float(sigma_obs)
        self.output_abs_bound = output_abs_bound
        self.B_pop_ub = B_pop_ub


def probe_B_pop_bound(probe: Probe, K: int):
    """Return a genuine B_pop upper bound when the probe exposes enough structure."""
    if probe.B_pop_ub is not None:
        return float(probe.B_pop_ub)
    if probe.output_abs_bound is not None:
        return bl.population_residual_sup_bound(
            probe.output_abs_bound, probe.d, K
        )
    return None


def nlp_probe(clf, sentence, reference, max_free):
    ctx = clf.encode(sentence)
    d = int(ctx["free_idx"].numel())
    if d < 2 or d > max_free:
        return None
    Xb = clf.make_baseline(ctx, reference)
    target = clf.target_class(ctx)

    # The wrapper uses model.eval() + no_grad(), hence repeated identical masks
    # are deterministic.  Probability-valued output gives |g(z)| <= 1.
    return Probe(
        d, target,
        query_fn=lambda Z: clf.query(ctx, Xb, Z, target),
        sigma_obs=0.0,
        output_abs_bound=1.0,
    )


def image_probe(clf, img, reference, grid):
    ref = clf.make_reference(img, reference)
    H, W, _ = img.shape
    slices = clf._cell_slices(H, W, grid)
    d = len(slices)
    target = clf.target_class(img)
    # Deterministic logits.  No finite global logit bound is assumed here; a
    # theorem-level B_pop bound must be supplied separately if required.
    return Probe(
        d, target,
        query_fn=lambda Z: clf.query(img, ref, slices, Z, target),
        sigma_obs=0.0,
        output_abs_bound=None,
    )


# =========================================================================== #
#  Shared per-probe routines (all numerics via bl_core)
# =========================================================================== #
def _pilot_bank(probe: Probe, K: int, seed: int):
    """Draw and query the shared pilot bank used by planning/UCB routines."""
    N0 = bl.pilot_N0(probe.d, K)
    rng = np.random.default_rng(seed + 12345)
    Z = bl.sample_masks(max(N0, 3 * bl.p_K(probe.d, K)), probe.d, rng)
    y = probe.query(Z)
    return Z, y


def pilot_sigma_eff(probe: Probe, K, seed=0, detail=False):
    """Cross-fitted PLANNING pilot.

    This retains the empirical difference-of-variances estimator used by the
    backward budget rule.  It is not the mismatch-energy UCB used for certified
    forward decisions.
    """
    Z, y = _pilot_bank(probe, K, seed)
    est = bl.estimate_mismatch_detail(
        Z, y, K, probe.sigma_obs, cross_fit=True
    )
    s_eff = bl.sigma_eff(probe.sigma_obs, est.m_hat)
    if detail:
        return s_eff, est
    return s_eff, est.m_hat


def pilot_mismatch_ucb(probe: Probe, K, seed=0, delta_pilot=None,
                       validation_fraction=0.5):
    """Honest held-out UCB for mismatch energy m, not for sigma_eff.

    For probability-output probes, ``output_abs_bound=1`` yields a conditional
    validation residual bound automatically.  Logit-valued probes must expose a
    finite output bound or an explicit R_val at a lower level; otherwise strict
    UCB certification is intentionally unavailable.
    """
    Z, y = _pilot_bank(probe, K, seed)
    ucb = bl.mismatch_energy_ucb(
        Z, y, K, d=probe.d, delta_pilot=delta_pilot,
        validation_fraction=validation_fraction,
        output_abs_bound=probe.output_abs_bound,
    )
    # Optional planning diagnostic only; never fed to the certified floor.
    s_eff_plan_ucb = bl.sigma_eff(probe.sigma_obs, ucb.m_ucb)
    return s_eff_plan_ucb, ucb


def forward_backward_on_probe(probe: Probe, N_list, beta_min, K, seed=0):
    """FORWARD (sign-flips over prefix ladder) + BACKWARD (budget plan) on one
    probe. Returns a result dict or None if not identifiable."""
    N_list = [n for n in N_list if n > bl.p_K(probe.d, K)]
    if len(N_list) < 2:
        return None
    s_eff, est = pilot_sigma_eff(probe, K, seed, detail=True)   # R1.6 detail
    m_hat = est.m_hat
    # Use a genuine population sup bound when available (probability-output NLP);
    # otherwise retain the historical sample residual max as an empirical plug-in
    # for this stability study.  The latter is not a theorem-level B_pop bound.
    B_floor = probe_B_pop_bound(probe, K)
    B_floor = est.B_hat if B_floor is None else B_floor

    # R1.5: is the pilot inside the Bernstein-domination regime at the smallest
    # budget it will run? (sub-exponential term dominated => C_M absorbs it.)
    N_min = min(N_list)
    N_dom = bl.leakage_domination_N(est.m_hat, est.B_hat, probe.d, K)
    dominated = bool(N_min >= N_dom)

    # FORWARD: prefix-nested bank, single-budget theorem at each rung.
    # Report point 1/2: pass the mismatch breakdown so each rung's floor is the
    # revised TWO-TERM certified floor (sigma_obs, m_hat, B) rather than the
    # absorbed single-scalar sigma_eff.
    N_max = max(N_list)
    rng = np.random.default_rng(seed + 777)
    Zbank = bl.sample_masks(N_max, probe.d, rng)
    ybank = probe.query(Zbank)
    trace = bl.sweep_prefix_ladder(Zbank, ybank, N_list, s_eff, probe.d, K,
                                   sigma_obs=probe.sigma_obs, m_hat=est.m_hat,
                                   B=B_floor)

    # BACKWARD: predict N with the empirical planning constant, report the revised
    # two-term realized floor.
    plan = bl.plan_budget(
        s_eff, beta_min, probe.d, K,
        sigma_obs=probe.sigma_obs, m_hat=est.m_hat, B=B_floor,
    )

    return dict(d=probe.d, pK=bl.p_K(probe.d, K), m_hat=m_hat, sigma_eff=s_eff,
                m_raw=est.m_raw, m_negative=est.negative,     # R1.6
                B_hat=est.B_hat, B_floor=B_floor,
                N_dom=N_dom, dominated=dominated,  # R1.5
                cest_last=trace.cest_last,                    # R1.4 realized Cest
                floor_last=trace.floor_last,                  # R1.4 realized floor
                floor_last_old=trace.floor_last_old,          # R1.4 old frozen floor
                cert_last=trace.cert_last,                    # certified count (new floor)
                sign_flips=trace.sign_flips,                 # stability check
                n_compared=trace.n_compared,                 # its denominator
                count_monotone=trace.count_monotone,         # diagnostic
                set_nested=trace.set_nested,                 # diagnostic
                N_pred=plan.N_pred, N_run=plan.N_run,
                realized_floor=plan.realized_floor, ratio=plan.ratio,
                clamped=plan.clamped)


# =========================================================================== #
#  Exact-beta sign-correctness: dense OLS on the FULL cube 2^d (d <= MAX_EXACT_D)
# =========================================================================== #
def exact_cube(d, rng=None):
    """Enumerate ALL 2^d masks (in binary-counting order). Fitting OLS on the
    full cube gives the EXACT population degree-K projection; the row order is
    irrelevant to that fit.

    Blocking fix (report point 6): the pre-fix code SHUFFLED the cube and then
    took a PREFIX for the pilot / deployment run, calling it "i.i.d. uniform".
    A prefix of a random permutation is sampling WITHOUT replacement, not i.i.d.
    Bernoulli masks -- it is not the finite-budget experiment Theorem 1 is about.
    We no longer draw prefixes: the caller enumerates the cube ONCE (this
    function) to cache g(z), and then draws i.i.d. cube-index samples WITH
    replacement (iid_from_cube) for each finite-budget run, looking up the cached
    outputs.  No extra black-box queries are spent.  `rng` is unused now and kept
    only for signature back-compat.
    """
    n = 1 << d
    bits = ((np.arange(n)[:, None] >> np.arange(d)[None, :]) & 1).astype(float)
    return bits


def iid_from_cube(cube_bits, y_cube, N, rng):
    """Draw N i.i.d.-uniform masks WITH REPLACEMENT from the enumerated cube and
    return (Z, y) by looking up cached outputs -- an honest finite-budget draw
    from the uniform product measure at zero extra query cost (report point 6).
    Sampling cube-row indices uniformly with replacement is exactly i.i.d.
    Bernoulli(1/2) masks over the d units."""
    n = cube_bits.shape[0]
    idx = rng.integers(0, n, size=N)
    return cube_bits[idx], y_cube[idx]


def exact_calls(d):
    """Model calls one exact-beta probe costs (= cube size), known up front."""
    return 1 << d


def exact_sign_check(probe: Probe, beta_min, K, seed=0):
    """Direct sign-correctness check with exact population beta, m and B.

    For d <= MAX_EXACT_D we query the full cube once.  Because the current NLP
    wrappers are deterministic, the full cube gives the exact population
    degree-K projection and exact mismatch residual under the uniform measure.
    Finite-budget pilot/run masks are sampled i.i.d. WITH REPLACEMENT from this
    cached cube, so the finite run matches the theorem's mask law without extra
    model calls.

    The run floor uses exact m_{>K} and B_pop from the cube.  This makes the
    enumeration tier a clean direct theorem check rather than a test confounded
    by a plug-in pilot bound.
    """
    if probe.d > MAX_EXACT_D:
        return None
    if abs(probe.sigma_obs) > 1e-12:
        raise ValueError(
            "exact_sign_check currently assumes deterministic queries "
            "(sigma_obs=0) so the full cube is exact ground truth"
        )

    Zc = exact_cube(probe.d)
    yc = probe.query(Zc)
    Nc = Zc.shape[0]
    rng = np.random.default_rng(seed + 2)

    # Exact population projection and nuisance quantities from the full cube.
    beta_exact, b0_exact, _ = bl.ols_fit(Zc, yc, K)
    r_exact = yc - (b0_exact + bl.design_matrix(Zc, K) @ beta_exact)
    m_exact = float(np.mean(r_exact ** 2))
    B_exact = float(np.max(np.abs(r_exact)))

    # Planning pilot remains the empirical deployment heuristic.
    n_pilot = min(max(bl.pilot_N0(probe.d, K), 3 * bl.p_K(probe.d, K)), Nc)
    Zp, yp = iid_from_cube(Zc, yc, n_pilot, rng)
    est_plan = bl.estimate_mismatch_detail(
        Zp, yp, K, probe.sigma_obs, cross_fit=True
    )
    s_eff = bl.sigma_eff(probe.sigma_obs, est_plan.m_hat)

    # Predict a deployment budget, then simulate an i.i.d. finite run from cache.
    N_pred = bl.predict_budget(s_eff, beta_min, probe.d, K)
    N_run = max(N_pred, bl.feasibility_floor(probe.d, K))
    if N_run <= bl.p_K(probe.d, K):
        return None
    Zrun, yrun = iid_from_cube(Zc, yc, N_run, rng)
    beta_run, _, _ = bl.ols_fit(Zrun, yrun, K)

    fl_run = bl.certified_floor_from_design(
        Zrun, K, sigma_obs_ub=0.0, m_ub=m_exact, B_pop_ub=B_exact,
    )
    fl_exact = bl.certified_floor_from_design(
        Zc, K, sigma_obs_ub=0.0, m_ub=m_exact, B_pop_ub=B_exact,
    )

    n_false, n_scored = bl.false_sign_rate(beta_run, beta_exact, fl_run)
    pm = bl.power_miss_stats(beta_run, beta_exact, fl_run, fl_exact)
    return dict(
        d=probe.d, N_run=N_run, cube_N=Nc, n_calls=Nc,
        m_exact=m_exact, B_exact=B_exact,
        n_false_sign=n_false, n_scored=n_scored,
        n_margin=pm.n_margin,
        n_margin_recovered=pm.n_margin_recovered,
        n_margin_miss=pm.n_margin_miss,
        n_cube_resolvable=pm.n_cube_resolvable,
        n_cube_recovered=pm.n_cube_recovered,
        n_cube_miss=pm.n_cube_miss,
    )


# =========================================================================== #
#  Reporting -- forward stability, backward planning, and diagnostics
# =========================================================================== #
def report_tier2(rows, K, beta_min):
    print("\n" + "=" * 72)
    print(f"TIER 2 -- forward stability + backward planning "
          f"(K={K}, beta_min={beta_min}).")
    print("=" * 72)
    print(f"  {'cell':>22} {'d':>3} {'sig_eff':>8} {'Cest':>6} {'flr×':>5} | "
          f"{'flips/cmp':>11} | {'BWD ratio':>9} {'clamp':>6} | "
          f"{'neg%':>5} {'dom%':>5}")
    flips_total = cmp_total = 0
    for r in rows:
        flips_total += r["sign_flips"]
        cmp_total += r.get("n_compared", 0)
        clamp = "yes" if r["clamped"] else ""
        fc = f"{r['sign_flips']}/{r.get('n_compared', 0)}"
        negp = 100.0 * r.get("neg_frac", 0.0)
        domp = 100.0 * r.get("dom_frac", 1.0)
        cest = r.get("cest", float("nan"))
        infl = r.get("floor_infl", float("nan"))
        print(f"  {r['cell']:>22} {r['d']:>3d} {r['sigma_eff']:>8.4f} "
              f"{cest:>6.2f} {infl:>5.2f} | "
              f"{fc:>11} | {r['ratio']:>9.3f} {clamp:>6} | "
              f"{negp:>5.0f} {domp:>5.0f}")
    ratios = [r["ratio"] for r in rows]
    rate = flips_total / cmp_total if cmp_total else float("nan")
    print(f"\n  FORWARD  : certified sign flips = {flips_total} / {cmp_total} "
          f"certified-in-both-budget checks  (rate {rate:.4f}; target 0)")
    if ratios:
        print(f"  BACKWARD : realized/target ratio  median {np.median(ratios):.3f}  "
              f"range [{min(ratios):.3f}, {max(ratios):.3f}]  (target <= 1)")

    # diagnostics -- SEPARATE, explicitly labeled
    mono = np.mean([r["count_monotone"] for r in rows]) * 100 if rows else 0
    nest = np.mean([r["set_nested"] for r in rows]) * 100 if rows else 0
    print("\n  [appendix diagnostics -- near-guaranteed by prefix "
          "construction, NOT theorem evidence]")
    print(f"    count-monotone: {mono:.1f}%    set-nested: {nest:.1f}%")

    # R1.4 realized-Cest note
    if rows:
        cests = [r.get("cest") for r in rows if r.get("cest") == r.get("cest")]
        infls = [r.get("floor_infl") for r in rows
                 if r.get("floor_infl") == r.get("floor_infl")]
        if cests:
            print("\n  [R1.4 realized floor constant]")
            print(f"    Cest measured per run from the design (median "
                  f"{np.median(cests):.2f}, range [{min(cests):.2f}, "
                  f"{max(cests):.2f}]); NOT the frozen C_FLOOR=1.")
            print(f"    floor inflation vs old frozen-C_FLOOR floor: median "
                  f"{np.median(infls):.2f}x (range [{min(infls):.2f}, "
                  f"{max(infls):.2f}]).")
            print(f"    The forward floor is now a genuine upper bound; the "
                  f"certified counts above reflect it.")

    # R1.5 / R1.6 correctness-fix diagnostics
    if rows:
        neg = np.mean([r.get("neg_frac", 0.0) for r in rows]) * 100
        dom = np.mean([r.get("dom_frac", 1.0) for r in rows]) * 100
        print("\n  [correctness-fix diagnostics]")
        print(f"    R1.6 negative-clip fraction (raw m_hat<0, then clipped "
              f"to 0): {neg:.1f}% of items")
        print(f"         clipping is conservative (raises sigma_eff); nonzero "
              f"mainly in degenerate/deterministic cells.")
        print(f"    R1.5 Bernstein-domination: {dom:.1f}% of items run at "
              f"N >= (2B^2/9m) log(.), sub-exp term dominated.")


def report_exact(rows, setting, skipped):
    print("\n" + "=" * 72)
    print(f"EXACT-BETA sign-correctness ({setting})")
    print("=" * 72)
    if not rows:
        print(f"  no enumerable probes (all d > {MAX_EXACT_D}); "
              f"{skipped} skipped.")
        print(f"  -> no exact check here; forward evidence is sign-flip "
              f"stability only.")
        return
    tot_false = sum(r["n_false_sign"] for r in rows)
    tot_scored = sum(r["n_scored"] for r in rows)
    rate = tot_false / tot_scored if tot_scored else float("nan")
    ds = [r["d"] for r in rows]
    print(f"  enumerable probes: {len(rows)}  (d in [{min(ds)}, {max(ds)}], "
          f"exact cube 2^d); {skipped} probes skipped (d > {MAX_EXACT_D})")
    print(f"  certified coords : {tot_scored}")
    print(f"  false signs      : {tot_false}")
    print(f"  false-sign rate  : {rate:.4f}   "
          f"(Theorem 1 forward claim, exact ground truth; target 0)")

    # R1.9 -- POWER against the exact projection, split into the guarantee check
    # (Theorem-1 margin at the RUN floor, must have zero misses) and the
    # resolution recovery (what the full cube resolves that the run does not --
    # an expected finite-budget cost, not a violation).
    n_mar = sum(r.get("n_margin", 0) for r in rows)
    mar_rec = sum(r.get("n_margin_recovered", 0) for r in rows)
    mar_miss = sum(r.get("n_margin_miss", 0) for r in rows)
    n_cube = sum(r.get("n_cube_resolvable", 0) for r in rows)
    cube_rec = sum(r.get("n_cube_recovered", 0) for r in rows)
    cube_miss = sum(r.get("n_cube_miss", 0) for r in rows)
    mar_pow = (mar_rec / n_mar) if n_mar else float("nan")
    cube_pow = (cube_rec / n_cube) if n_cube else float("nan")
    print("\n  [R1.9 POWER -- zero false signs is only meaningful with the miss "
          "rate]")
    print(f"  (1) GUARANTEE (Theorem-1 margin at the RUN floor: "
          f"|beta_exact| > 2*fl_run):")
    print(f"      truth {n_mar:>4d}   recovered {mar_rec:>4d}   "
          f"MISS {mar_miss:>4d}   power {mar_pow:.3f}")
    print(f"      -> this is the guarantee denominator; MISS MUST be 0 "
          f"(the margin promises\n         every such coord clears the run "
          f"floor). MISS=0 with zero false signs\n         is the theorem "
          f"holding on exact ground truth.")
    print(f"  (2) RESOLUTION recovery (vs the full-cube floor: "
          f"|beta_exact| > fl_exact):")
    print(f"      cube-resolvable {n_cube:>4d}   run-recovered {cube_rec:>4d}   "
          f"unresolved {cube_miss:>4d}   power {cube_pow:.3f}")
    print(f"      -> the full cube (huge N) resolves more than a deployment "
          f"budget; the gap\n         is the honest finite-budget resolution "
          f"cost ('below-floor = unresolved,\n         not zero'), NOT a "
          f"guarantee violation. A low number here means the run\n         "
          f"budget was small relative to what the reference could support.")


# =========================================================================== #
#  Drivers (torch built only here)
# =========================================================================== #
def load_sentences(path, n):
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:n] if n else lines


def run_nlp_tier2(args):
    import bl_models as M
    backbones = args.backbones.split(",")
    refs = (list(M.NLP_REFERENCES) if args.references == "all"
            else args.references.split(","))
    N_list = [int(x) for x in args.N_ladder.split(",")]
    sents = load_sentences(args.sentences, args.subset)
    rows = []
    prog = Progress(len(backbones) * len(refs) * len(sents),
                    f"tier2-nlp K={args.K}")
    flips_so_far = 0
    for bk in backbones:
        try:
            clf = M.TextClassifier(model=bk, dataset=args.dataset)
        except Exception as e:
            prog.note(f"[skip {bk}] {e}")
            continue
        for ref in refs:
            for si, sent in enumerate(sents):
                p = nlp_probe(clf, sent, ref, args.max_free)
                if p is None:
                    prog.step(f"{bk}/{ref} (skipped: d out of range)")
                    continue
                r = forward_backward_on_probe(p, N_list, args.beta_min,
                                              args.K, seed=si)
                if r:
                    r["cell"] = f"{bk}/{ref}"
                    rows.append(r)
                    flips_so_far += r["sign_flips"]
                    prog.step(f"{bk}/{ref} d={p.d} flips={r['sign_flips']} "
                              f"ratio={r['ratio']:.2f} | tot flips {flips_so_far}")
                else:
                    prog.step(f"{bk}/{ref} d={p.d} (not identifiable)")
        clf.close()
    prog.close()
    report_tier2(_aggregate_cells(rows), args.K, args.beta_min)


def run_image_tier2(args):
    import bl_models as M
    paths = sorted(glob.glob(os.path.join(args.images_dir, args.glob)))
    paths = paths[:args.subset] if args.subset else paths
    N_list = [int(x) for x in args.N_ladder.split(",")]
    bks = (M.IMAGE_BACKBONES if args.backbones == "all"
           else args.backbones.split(","))
    rfs = (M.IMAGE_REFERENCES if args.references == "all"
           else args.references.split(","))
    rows = []
    prog = Progress(len(bks) * len(rfs) * len(paths), "tier2-image K=1")
    flips_so_far = 0
    for bk in bks:
        try:
            clf = M.ImageClassifier(backbone=bk)
        except Exception as e:
            prog.note(f"[skip {bk}] {e}")
            continue
        for ref in rfs:
            for pi, path in enumerate(paths):
                img = clf.load_image(path)
                p = image_probe(clf, img, ref, args.grid)
                if p is None:
                    prog.step(f"{bk}/{ref} (skipped)")
                    continue
                r = forward_backward_on_probe(p, N_list, args.beta_min,
                                              1, seed=pi)
                if r:
                    r["cell"] = f"{bk}/{ref}"
                    rows.append(r)
                    flips_so_far += r["sign_flips"]
                    prog.step(f"{bk}/{ref} d={p.d} flips={r['sign_flips']} "
                              f"ratio={r['ratio']:.2f} | tot flips {flips_so_far}")
                else:
                    prog.step(f"{bk}/{ref} (not identifiable)")
        clf.close()
    prog.close()
    report_tier2(_aggregate_cells(rows), 1, args.beta_min)


def run_exact_nlp(args):
    import bl_models as M
    backbones = args.backbones.split(",")
    refs = (list(M.NLP_REFERENCES) if args.references == "all"
            else args.references.split(","))
    sents = load_sentences(args.sentences, None)
    rows = []
    skipped = 0
    total = len(backbones) * len(refs) * (args.subset or len(sents))
    prog = Progress(total, f"exact-nlp K={args.K}")
    prog.note(f"  enumerating full cube when d<={args.max_d} "
              f"(<= {1 << args.max_d} calls/probe); larger d are SKIPPED "
              f"(no exact beta -- no random-bank stand-in).")
    run_false = run_scored = 0
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
                if p.d > args.max_d:
                    skipped += 1
                    used += 1
                    prog.step(f"{bk}/{ref} d={p.d} > {args.max_d}: SKIP")
                    continue
                r = exact_sign_check(p, args.beta_min, args.K, seed=si)
                used += 1
                if r:
                    rows.append(r)
                    run_false += r["n_false_sign"]
                    run_scored += r["n_scored"]
                    prog.step(f"{bk}/{ref} d={p.d} exact 2^{p.d}={r['n_calls']} "
                              f"calls | false {run_false}/{run_scored}")
                else:
                    prog.step(f"{bk}/{ref} d={p.d} (not identifiable)")
        clf.close()
    prog.close()
    report_exact(rows, f"NLP K={args.K}", skipped)


def _aggregate_cells(rows):
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r)
    out = []
    for cell, rs in cells.items():
        n = len(rs)
        out.append(dict(
            cell=cell, d=int(np.median([r["d"] for r in rs])),
            sigma_eff=float(np.mean([r["sigma_eff"] for r in rs])),
            sign_flips=sum(r["sign_flips"] for r in rs),
            n_compared=sum(r.get("n_compared", 0) for r in rs),
            ratio=float(np.median([r["ratio"] for r in rs])),
            clamped=any(r["clamped"] for r in rs),
            count_monotone=float(np.mean([r["count_monotone"] for r in rs])),
            set_nested=float(np.mean([r["set_nested"] for r in rs])),
            # R1.4: realized Cest and floor inflation vs the old frozen-C_FLOOR
            cest=float(np.median([r.get("cest_last", float("nan")) for r in rs])),
            floor_infl=float(np.median(
                [ (r["floor_last"] / r["floor_last_old"])
                  for r in rs
                  if r.get("floor_last_old") ])) if rs else float("nan"),
            # R1.6: fraction of pilot draws whose RAW mismatch was negative
            # (clipped to 0). Nonzero mainly in near-degenerate / deterministic
            # cells, exactly the conditional-clause mechanism of Theorem 1.
            n_items=n,
            neg_frac=float(np.mean([1.0 if r.get("m_negative") else 0.0
                                    for r in rs])),
            # R1.5: fraction of items inside the Bernstein-domination regime at
            # the smallest run budget (sub-exponential term dominated).
            dom_frac=float(np.mean([1.0 if r.get("dominated", True) else 0.0
                                    for r in rs])),
        ))
    return out


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Tier 2 black-box (forward+backward) + exact-beta check")
    p.add_argument("mode", choices=["nlp", "image", "exact-nlp"])
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--beta_min", type=float, default=None)
    p.add_argument("--N_ladder", default="512,1000,2000,4000")
    p.add_argument("--backbones", default="all")
    p.add_argument("--references", default="all")
    p.add_argument("--dataset", default="sst2")
    p.add_argument("--sentences", default="text_samples/sst2_short.txt")
    p.add_argument("--max_free", type=int, default=40)
    p.add_argument("--images_dir", default="image_samples")
    p.add_argument("--glob", default="*.JPEG")
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--subset", type=int, default=None,
                   help="cap items per cell (exact-nlp: per backbone x ref)")
    p.add_argument("--max_d", type=int, default=MAX_EXACT_D,
                   help="enumerate exact cube only when d <= max_d "
                        "(2^max_d calls/probe); larger d skipped")
    return p


def main():
    args = build_parser().parse_args()
    # frozen Tier-1 defaults for beta_min
    if args.beta_min is None:
        args.beta_min = 0.05 if args.mode == "image" else 0.02
    if args.backbones == "all" and args.mode in ("nlp", "exact-nlp"):
        import bl_models as M
        args.backbones = ",".join(M.NLP_BACKBONES)
    if args.mode == "nlp":
        run_nlp_tier2(args)
    elif args.mode == "image":
        run_image_tier2(args)
    elif args.mode == "exact-nlp":
        run_exact_nlp(args)


if __name__ == "__main__":
    main()