"""
tier2_blackbox.py
=================
TIER 2 -- Black-box classifiers (+ an exact-beta sign-correctness check).

Question answered: does the GUARANTEE hold on real query-only models, with the
constants frozen from Tier 1?

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

Complementary noise coverage (stated, not incidental):
  * Image backbones are DETERMINISTIC (sigma_obs ~ 0) -> exercise the MISMATCH
    half of sigma_eff.
  * NLP backbones are PROBABILISTIC (sigma_obs > 0) -> exercise the QUERY-NOISE
    half. Together they cover both terms of sigma_eff on real models.

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
    """A single explained instance (sentence or image) under one reference.

    Exposes:
      d        : number of free interpretable units
      target   : explained class index
      query(Z) : (N, d) binary masks -> (N,) model outputs  (the black box)
      sigma_obs: query-noise scale (0 for deterministic backbones)
    """
    def __init__(self, d, target, query_fn, sigma_obs):
        self.d = d
        self.target = target
        self.query = query_fn
        self.sigma_obs = sigma_obs


def nlp_probe(clf, sentence, reference, max_free):
    ctx = clf.encode(sentence)
    d = int(ctx["free_idx"].numel())
    if d < 2 or d > max_free:
        return None
    Xb = clf.make_baseline(ctx, reference)
    target = clf.target_class(ctx)
    # probabilistic backbone: estimate sigma_obs by repeated queries
    rng = np.random.default_rng(0)
    Zp = bl.sample_masks(16, d, rng)
    cols = [clf.query(ctx, Xb, Zp, target) for _ in range(8)]
    sigma_obs = float(np.stack(cols, 0).std(axis=0).mean())
    return Probe(d, target,
                 query_fn=lambda Z: clf.query(ctx, Xb, Z, target),
                 sigma_obs=sigma_obs)


def image_probe(clf, img, reference, grid):
    ref = clf.make_reference(img, reference)
    H, W, _ = img.shape
    slices = clf._cell_slices(H, W, grid)
    d = len(slices)
    target = clf.target_class(img)
    # deterministic backbone -> sigma_obs ~ 0; the floor is mismatch-driven
    return Probe(d, target,
                 query_fn=lambda Z: clf.query(img, ref, slices, Z, target),
                 sigma_obs=0.0)


# =========================================================================== #
#  Shared per-probe routines (all numerics via bl_core)
# =========================================================================== #
def pilot_sigma_eff(probe: Probe, K, seed=0):
    """Cross-fitted pilot -> sigma_eff (the only data-dependent input)."""
    N0 = bl.pilot_N0(probe.d, K)
    rng = np.random.default_rng(seed + 12345)
    Z = bl.sample_masks(max(N0, 3 * bl.p_K(probe.d, K)), probe.d, rng)
    y = probe.query(Z)
    m_hat = bl.estimate_mismatch_from_residual(Z, y, K, probe.sigma_obs,
                                               cross_fit=(K == 2))
    return bl.sigma_eff(probe.sigma_obs, m_hat), m_hat


def forward_backward_on_probe(probe: Probe, N_list, beta_min, K, seed=0):
    """FORWARD (sign-flips over prefix ladder) + BACKWARD (budget plan) on one
    probe. Returns a result dict or None if not identifiable."""
    N_list = [n for n in N_list if n > bl.p_K(probe.d, K)]
    if len(N_list) < 2:
        return None
    s_eff, m_hat = pilot_sigma_eff(probe, K, seed)

    # FORWARD: prefix-nested bank, single-budget theorem at each rung
    N_max = max(N_list)
    rng = np.random.default_rng(seed + 777)
    Zbank = bl.sample_masks(N_max, probe.d, rng)
    ybank = probe.query(Zbank)
    trace = bl.sweep_prefix_ladder(Zbank, ybank, N_list, s_eff, probe.d, K)

    # BACKWARD: predict N, clamp, realized floor
    plan = bl.plan_budget(s_eff, beta_min, probe.d, K)

    return dict(d=probe.d, pK=bl.p_K(probe.d, K), m_hat=m_hat, sigma_eff=s_eff,
                sign_flips=trace.sign_flips,                 # THE guarantee
                n_compared=trace.n_compared,                 # its denominator
                count_monotone=trace.count_monotone,         # diagnostic
                set_nested=trace.set_nested,                 # diagnostic
                N_pred=plan.N_pred, N_run=plan.N_run,
                realized_floor=plan.realized_floor, ratio=plan.ratio,
                clamped=plan.clamped)


# =========================================================================== #
#  Exact-beta sign-correctness: dense OLS on the FULL cube 2^d (d <= MAX_EXACT_D)
# =========================================================================== #
def exact_cube(d, rng):
    """Enumerate ALL 2^d masks (shuffled). Fitting OLS on this gives the EXACT
    population degree-K projection -- the order is irrelevant to the fit, but we
    shuffle so a PREFIX (reused for pilot/run) is i.i.d. uniform rather than the
    structured binary-counting order, which would give a singular pilot Gram."""
    n = 1 << d
    bits = ((np.arange(n)[:, None] >> np.arange(d)[None, :]) & 1).astype(float)
    rng.shuffle(bits)
    return bits


def exact_calls(d):
    """Model calls one exact-beta probe costs (= cube size), known up front."""
    return 1 << d


def exact_sign_check(probe: Probe, beta_min, K, seed=0):
    """Direct sign-correctness on ONE small-d probe.

    Enumerate the cube ONCE (the only model query for this probe); reuse a
    PREFIX for the pilot (sigma_eff) and the deployment-budget run estimate, and
    the FULL cube for exact beta. A certified coordinate is scored only if exact
    beta also resolves it. Returns a dict, or None if d is too large / the run
    budget is not well-posed.
    """
    if probe.d > MAX_EXACT_D:
        return None
    Zc = exact_cube(probe.d, np.random.default_rng(seed + 2))
    yc = probe.query(Zc)                       # the one expensive call
    Nc = Zc.shape[0]

    # pilot sigma_eff from a prefix
    n_pilot = min(max(bl.pilot_N0(probe.d, K), 3 * bl.p_K(probe.d, K)), Nc)
    m_hat = bl.estimate_mismatch_from_residual(
        Zc[:n_pilot], yc[:n_pilot], K, probe.sigma_obs, cross_fit=(K == 2))
    s_eff = bl.sigma_eff(probe.sigma_obs, m_hat)

    # deployment-budget run estimate, a prefix of the same cube
    plan = bl.plan_budget(s_eff, beta_min, probe.d, K)
    N_run = min(plan.N_run, Nc)
    if N_run <= bl.p_K(probe.d, K):
        return None
    beta_run, _, _ = bl.ols_fit(Zc[:N_run], yc[:N_run], K)
    fl_run = bl.floor_value(s_eff, probe.d, N_run, K)

    # EXACT beta on the full cube
    beta_exact, _, _ = bl.ols_fit(Zc, yc, K)
    fl_exact = bl.floor_value(s_eff, probe.d, Nc, K)

    n_false, n_scored = bl.false_sign_rate(beta_run, beta_exact,
                                           fl_run, fl_exact)
    return dict(d=probe.d, N_run=N_run, cube_N=Nc, n_calls=Nc,
                n_false_sign=n_false, n_scored=n_scored)


# =========================================================================== #
#  Reporting -- guarantee and diagnostics kept in SEPARATE tables
# =========================================================================== #
def report_tier2(rows, K, beta_min):
    print("\n" + "=" * 72)
    print(f"TIER 2 -- guarantee table (K={K}, beta_min={beta_min}). "
          f"Constants frozen from Tier 1.")
    print("=" * 72)
    print(f"  {'cell':>22} {'d':>3} {'sig_eff':>8} | "
          f"{'flips/cmp':>11} | {'BWD ratio':>9} {'clamp':>6}")
    flips_total = cmp_total = 0
    for r in rows:
        flips_total += r["sign_flips"]
        cmp_total += r.get("n_compared", 0)
        clamp = "yes" if r["clamped"] else ""
        fc = f"{r['sign_flips']}/{r.get('n_compared', 0)}"
        print(f"  {r['cell']:>22} {r['d']:>3d} {r['sigma_eff']:>8.4f} | "
              f"{fc:>11} | {r['ratio']:>9.3f} {clamp:>6}")
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
        out.append(dict(
            cell=cell, d=int(np.median([r["d"] for r in rs])),
            sigma_eff=float(np.mean([r["sigma_eff"] for r in rs])),
            sign_flips=sum(r["sign_flips"] for r in rs),
            n_compared=sum(r.get("n_compared", 0) for r in rs),
            ratio=float(np.median([r["ratio"] for r in rs])),
            clamped=any(r["clamped"] for r in rs),
            count_monotone=float(np.mean([r["count_monotone"] for r in rs])),
            set_nested=float(np.mean([r["set_nested"] for r in rs])),
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
