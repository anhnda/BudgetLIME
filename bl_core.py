"""
bl_core.py
==========
Single shared numerical core for the finite-budget LIME certification
experiments, rewritten to the consolidated experimental design.

DESIGN PRINCIPLE (one inequality, two directions).
The paper proves exactly one bound (Theorem 1):

    ||beta_hat - beta||_inf  <=  floor(N, rho)
    floor(N, rho) = Cest * sigma_eff * sqrt(2 log pK / N)
    sigma_eff     = sigma_obs + Cm * sqrt(m>K,rho)

Every experiment is a reading of this in ONE of two directions:
  * FORWARD  (the guarantee): |beta_hat_S| > floor  =>  sign is correct.
  * BACKWARD (the budget rule, Eq. 8): N >~ 2 Cest^2 sigma_eff^2 log pK / beta_min^2.
Anything that is not a forward sign check or a backward budget check
(set-nesting, count-monotonicity) is a WORKFLOW DIAGNOSTIC, never theorem
evidence, and is reported separately.

SINGLE SOURCE OF CONSTANTS.
The old codebase floated three constants (Cest=1 theoretical, 1.81 synthetic,
C_budget=3.5). Here there are exactly two, defined once in CONSTANTS, calibrated
on synthetic data (Tier 1) and FROZEN everywhere downstream. They are never
re-fit on real data. The two roles of the estimator constant are kept explicit
and separate:
  * C_FLOOR  -- the constant in the floor BOUND, used forward.
  * C_BUDGET -- the constant that makes the budget rule LAND at the target,
                used backward (back-solved from the signed-detection transition).
Using C_FLOOR to invert the budget produces N_pred below the feasibility floor
~pK and is physically meaningless; the two constants are therefore distinct
objects, related by the conservative direction realized_floor <= beta_min.

Pure numpy. No torch. The black-box model wrappers live in the driver files and
call into this module; this module never imports torch.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np


# =========================================================================== #
#  CONSTANTS -- the single source of truth (calibrated at Tier 1, frozen here)
# =========================================================================== #
@dataclass(frozen=True)
class Constants:
    """The two and only floor constants, plus the design's fixed parameters.

    C_FLOOR and C_M enter the floor BOUND (forward direction). C_BUDGET enters
    the budget RULE (backward direction). C_FLOOR's theoretical value is 1 for
    the orthonormal +-1 design; the empirical value is expected to be >= 1
    because the bound is an upper bound under finite N, query noise, and
    mismatch -- this gap is not an anomaly and is stated once, here.
    """
    C_FLOOR: float = 1.0       # floor-bound constant; theory = 1 (orthonormal)
    C_M: float = 1.0           # leakage constant (Lemma 1 exact-variance identity)
    C_BUDGET: float = 1.81     # budget-rule constant, back-solved at Tier 1
    P_KEEP: float = 0.5        # centered +-1 Walsh design the floor assumes
    Z_ALPHA: float = 1.96      # single pre-registered coord (two-sided 95%)


CONSTANTS = Constants()


# =========================================================================== #
#  Degree-K feature machinery  (identical across every setting and tier)
# =========================================================================== #
def p_K(d: int, K: int = 1) -> int:
    """Candidate coefficient count INCLUDING the intercept (paper's pK)."""
    if K == 1:
        return d + 1
    if K == 2:
        return 1 + d + d * (d - 1) // 2
    raise ValueError("only K in {1, 2} supported")


def feature_subsets(d: int, K: int):
    """Ordered non-empty subsets |S| <= K: singletons then pairs (i<j)."""
    subs = [(i,) for i in range(d)]
    if K >= 2:
        subs += list(combinations(range(d), 2))
    return subs


def design_matrix(Z: np.ndarray, K: int) -> np.ndarray:
    """Centered Walsh design over |S| <= K, no intercept column (centering of y
    absorbs it). Main effects chi_i = 2(z_i - 1/2) in {-1,+1}; pair columns are
    products of +-1 columns (hence also +-1). Returns (N, pK-1)."""
    Zc = 2.0 * (Z - 0.5)
    N, d = Zc.shape
    if K == 1:
        return Zc
    pair_cols = [Zc[:, i] * Zc[:, j] for (i, j) in combinations(range(d), 2)]
    if pair_cols:
        return np.concatenate([Zc, np.stack(pair_cols, axis=1)], axis=1)
    return Zc


def standardize_columns(X: np.ndarray):
    scale = np.sqrt((X ** 2).mean(axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    return X / scale, scale


def sample_masks(N: int, d: int, rng: np.random.Generator,
                 p_keep: float = None) -> np.ndarray:
    p_keep = CONSTANTS.P_KEEP if p_keep is None else p_keep
    return (rng.random((N, d)) > (1.0 - p_keep)).astype(float)


# =========================================================================== #
#  Dense OLS  (the paper's estimator; closed-form normal equations)
# =========================================================================== #
def ols_fit(Z: np.ndarray, y: np.ndarray, K: int):
    """Column-standardized dense OLS with intercept over the degree-<=K design.

    Returns (beta on ORIGINAL scale, intercept, diag(Ginv)). Raises
    LinAlgError if N <= pK (dense fit not well-posed -- Assumption 1).
    """
    d = Z.shape[1]
    X = design_matrix(Z, K)
    N = X.shape[0]
    if N <= p_K(d, K):
        raise np.linalg.LinAlgError(
            f"N={N} <= pK={p_K(d, K)}: dense K={K} fit not well-posed")
    Xs, scale = standardize_columns(X)
    y_mean = y.mean()
    G = (Xs.T @ Xs) / N
    if np.linalg.cond(G) > 1e8:
        raise np.linalg.LinAlgError("Gram ill-conditioned (N too small)")
    Ginv = np.linalg.inv(G)
    beta_std = Ginv @ (Xs.T @ (y - y_mean)) / N
    return beta_std / scale, y_mean, np.diag(Ginv)


# =========================================================================== #
#  THE FLOOR  (forward direction) -- one function, used everywhere
# =========================================================================== #
def sigma_eff(sigma_obs: float, m_hat: float, C_m: float = None) -> float:
    """sigma_eff = sigma_obs + C_m sqrt(m): the ONLY data-dependent input to the
    floor. Query noise and mismatch enter through this single scalar."""
    C_m = CONSTANTS.C_M if C_m is None else C_m
    return sigma_obs + C_m * math.sqrt(max(m_hat, 0.0))


def floor_value(s_eff: float, d: int, N: int, K: int = 1,
                family_wise: bool = True, C_floor: float = None) -> float:
    """floor(N, rho) = C_floor * sigma_eff * sqrt(2 log pK / N) (family-wise),
    or the single pre-registered-coordinate variant with z_{1-alpha}."""
    C_floor = CONSTANTS.C_FLOOR if C_floor is None else C_floor
    if family_wise:
        return C_floor * s_eff * math.sqrt(2.0 * math.log(p_K(d, K)) / N)
    return C_floor * s_eff * CONSTANTS.Z_ALPHA / math.sqrt(N)


def certified_set(beta: np.ndarray, fl: float):
    """Forward rule: certify coordinate S iff |beta_hat_S| > floor. Returns
    (index set, sign vector)."""
    idx = np.where(np.abs(beta) > fl)[0]
    return set(idx.tolist()), np.sign(beta)


# =========================================================================== #
#  THE BUDGET RULE  (backward direction) -- Eq. 8, with the budget constant
# =========================================================================== #
def predict_budget(s_eff: float, beta_min: float, d: int, K: int = 1,
                   family_wise: bool = True, C_budget: float = None) -> int:
    """Backward: smallest N that pushes the floor below beta_min, family-wise.

        N_pred = ceil( C_budget^2 * sigma_eff^2 * (2 log pK) / beta_min^2 )

    C_budget (NOT C_floor) is the back-solved signed-detection constant. Using
    C_floor here under-predicts below the feasibility floor ~pK.
    """
    C_budget = CONSTANTS.C_BUDGET if C_budget is None else C_budget
    norm = 2.0 * math.log(p_K(d, K)) if family_wise else CONSTANTS.Z_ALPHA ** 2
    return int(math.ceil(C_budget ** 2 * s_eff ** 2 * norm / beta_min ** 2))


def feasibility_floor(d: int, K: int = 1, c: float = 3.0) -> int:
    """Design-conditioning floor ~ c * pK below which the dense fit is singular
    (Reading 2). A target below this is run feasibility-CLAMPED, not infeasible."""
    return int(math.ceil(c * p_K(d, K)))


@dataclass
class BudgetPlan:
    """The backward-direction result for one item/cell."""
    N_pred: int
    N_run: int
    realized_floor: float
    ratio: float            # realized_floor / beta_min  (target <= 1)
    clamped: bool           # ran at feasibility floor, not the resolution budget


def plan_budget(s_eff: float, beta_min: float, d: int, K: int = 1,
                family_wise: bool = True) -> BudgetPlan:
    """Full backward plan: predict N, clamp to feasibility, report realized
    floor and the conservative ratio."""
    N_pred = predict_budget(s_eff, beta_min, d, K, family_wise)
    feas = feasibility_floor(d, K)
    N_run = max(N_pred, feas)
    fl = floor_value(s_eff, d, N_run, K, family_wise)
    return BudgetPlan(N_pred=N_pred, N_run=N_run, realized_floor=fl,
                      ratio=fl / beta_min, clamped=(N_pred < feas))


# =========================================================================== #
#  PILOT estimation of the only data-dependent input (sigma_eff)
# =========================================================================== #
def estimate_mismatch_from_residual(Z: np.ndarray, y: np.ndarray, K: int,
                                    sigma_obs: float,
                                    cross_fit: bool = True) -> float:
    """m_hat>K = held-out residual variance minus sigma_obs^2 (Appendix C).

    Upper-biased (conservative): the held-out residual contains both genuine
    mismatch and pilot estimation error. cross_fit removes the in-sample
    pK/N inflation -- the recommended default at K=2.
    """
    if cross_fit:
        n = Z.shape[0]
        half = n // 2
        resid = np.empty(n)
        for tr, te in [(slice(0, half), slice(half, n)),
                       (slice(half, n), slice(0, half))]:
            beta, b0, _ = ols_fit(Z[tr], y[tr], K)
            yhat = b0 + design_matrix(Z[te], K) @ beta
            resid[te] = y[te] - yhat
        mse = float((resid ** 2).mean())
    else:
        beta, b0, _ = ols_fit(Z, y, K)
        yhat = b0 + design_matrix(Z, K) @ beta
        mse = float(((y - yhat) ** 2).mean())
    return max(mse - sigma_obs ** 2, 0.0)


def pilot_N0(d: int, K: int = 1) -> int:
    """Cross-fitted pilot size N0 = max(500, 6 pK)."""
    return max(500, 6 * p_K(d, K))


# =========================================================================== #
#  FORWARD evidence containers + collapse-curve machinery (Tier 1 / exact-beta)
# =========================================================================== #
@dataclass
class CollapsePoint:
    """One (x, signed-detection-rate) sample on the shared collapse curve."""
    x: float            # |beta| / floor
    sdr: float          # signed-detection rate at this x


def collapse_curve(x_grid, sdr_values):
    """Bundle an SDR(x) curve and return its 50%-crossing by linear interp.
    The crossing x_0.5 is a single labeled POINT on the shared curve, reported
    only alongside its cross-regime spread -- it is not a standalone constant.
    Returns (list[CollapsePoint], x_0.5 or nan)."""
    pts = [CollapsePoint(float(x), float(s)) for x, s in zip(x_grid, sdr_values)]
    x_half = float("nan")
    for k in range(1, len(sdr_values)):
        if sdr_values[k - 1] < 0.5 <= sdr_values[k]:
            lo, hi = x_grid[k - 1], x_grid[k]
            plo, phi = sdr_values[k - 1], sdr_values[k]
            x_half = lo + (0.5 - plo) * (hi - lo) / (phi - plo)
            break
    return pts, x_half


def cov(values) -> float:
    """Coefficient of variation std/mean -- the collapse-tightness statistic.
    Low CoV across regimes IS the claim; the mean value is secondary."""
    a = np.asarray([v for v in values if v is not None and not math.isnan(v)],
                   dtype=float)
    if a.size < 2 or a.mean() == 0:
        return float("nan")
    return float(a.std() / a.mean())


def false_sign_rate(beta_run: np.ndarray, beta_exact: np.ndarray, fl: float,
                    fl_exact: float):
    """Direct forward test against a known/exact beta. A coordinate is scored
    only if BOTH the run and the exact fit resolve it (an exact-unresolved coord
    carries no trustworthy ground-truth sign). Returns (n_false, n_scored)."""
    run_cert = np.abs(beta_run) > fl
    exact_cert = np.abs(beta_exact) > fl_exact
    scored = run_cert & exact_cert
    false = scored & (np.sign(beta_run) != np.sign(beta_exact))
    return int(false.sum()), int(scored.sum())


# =========================================================================== #
#  WORKFLOW DIAGNOSTICS (kept SEPARATE from the guarantee, by design)
# =========================================================================== #
@dataclass
class DiagnosticTrace:
    """Secondary, prefix-construction-coupled diagnostics. Reported in the
    appendix, explicitly labeled as near-guaranteed by nested prefixes -- NOT
    theorem evidence. The guarantee proper is sign_flips (forward)."""
    count_monotone: bool = True
    set_nested: bool = True
    sign_flips: int = 0          # THE guarantee: certified coords reversing sign
    n_compared: int = 0          # denominator: certified-in-both-budgets checks
    floor_first: float = None
    floor_last: float = None
    cert_first: int = None
    cert_last: int = None


def sweep_prefix_ladder(Zbank: np.ndarray, ybank: np.ndarray, N_list,
                        s_eff: float, d: int, K: int,
                        family_wise: bool = True) -> DiagnosticTrace:
    """Apply the single-budget theorem at each rung of a prefix-nested ladder.

    Primary output: sign_flips (the guarantee -- must be 0). Secondary outputs:
    count_monotone / set_nested (workflow diagnostics, near-guaranteed by the
    prefix construction and therefore reported as such, not as evidence).
    """
    tr = DiagnosticTrace()
    prev_set = prev_beta = None
    prev_count = -1
    for N in N_list:
        beta, _, _ = ols_fit(Zbank[:N], ybank[:N], K)
        fl = floor_value(s_eff, d, N, K, family_wise)
        cur_set, _ = certified_set(beta, fl)
        if tr.floor_first is None:
            tr.floor_first, tr.cert_first = fl, len(cur_set)
        tr.floor_last, tr.cert_last = fl, len(cur_set)
        if len(cur_set) < prev_count:
            tr.count_monotone = False
        if prev_set is not None and len(prev_set - cur_set) > 1:
            tr.set_nested = False
        if prev_beta is not None:
            inter = list(prev_set & cur_set)
            tr.n_compared += len(inter)
            tr.sign_flips += sum(np.sign(beta[i]) != np.sign(prev_beta[i])
                                 for i in inter) if inter else 0
        prev_set, prev_beta, prev_count = cur_set, beta, len(cur_set)
    return tr
