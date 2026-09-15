"""
bl_core.py
==========
Shared numerical core for the finite-budget LIME-style surrogate experiments.

FINAL THEORY/CODE SPLIT
-----------------------
Forward certification and backward planning are deliberately separate.

CERTIFICATE (Theorem 1):
    floor = C_est(Z) * [
        sigma_obs_ub * sqrt(2 L / N)
        + sqrt(2 m_ub L / N)
        + (2/3) B_pop_ub L / N
    ],
    L = log(2 * nu * pK / delta).

C_est is recomputed from the realized augmented Walsh Gram for every run.
C_m and C_budget NEVER enter a certified forward decision.

PLANNING (Eq. 8):
    sigma_eff_plan = sigma_obs + C_m * sqrt(m_plan)
is an empirical leading-order scale used with C_budget to predict a starting
query budget.  After the design exists, the realized two-term certificate is
always recomputed.

The UCB pilot implemented below upper-bounds the mismatch ENERGY m directly
from independent held-out squared residuals. It does not subtract a noise
variance and it does not convert the bound into a certified sigma_eff. The
validation residual range R_val used by the empirical-Bernstein pilot is
distinct from B_pop_ub = ||r_{>K,rho}||_inf in the theorem.

Pure numpy. Black-box wrappers live in the driver files.
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
    """Fixed planning constants and design parameters.

    C_M and C_BUDGET are empirical *planning* factors calibrated across
    d in {15, 24, 30, 49}.  They never enter a certified forward radius.

    C_FLOOR=1 is only the orthonormal population ideal, useful for historical
    diagnostics.  The actual forward constant C_est is always measured from the
    realized augmented Gram and a bad Gram causes the run to be unresolved.

    DELTA_SPLIT=3 covers the theorem events {design conditioning, query-noise
    maximum, mismatch leakage}.  When an independently held-out mismatch-energy
    UCB is used, DELTA_SPLIT_UCB=4 also budgets the pilot event.
    """
    C_FLOOR: float = 1.0
    C_M: float = 0.795
    C_BUDGET: float = 1.508
    P_KEEP: float = 0.5
    Z_ALPHA: float = 1.96
    DELTA_SPLIT: int = 3
    DELTA_SPLIT_UCB: int = 4


CONSTANTS = Constants()


# Shared ill-conditioning cutoff (report point 9): ols_fit and realized_cest MUST
# use the SAME threshold, otherwise a design can pass the fit (cond < COND_MAX)
# yet be treated differently by the floor.  A design above this cutoff is declared
# NOT well-posed -> the run is UNRESOLVED / no certificate, never an
# anti-conservative fallback to C_FLOOR = 1.
COND_MAX: float = 1e8


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


def log_pk_over_delta(d: int, K: int = 1, delta: float = None,
                      split: int = None) -> float:
    """R1.2 -- the union-bounded log factor that the floor carries.

    Theorem 1's proof allocates delta across `split` failure events, so the
    per-event level is delta/split.  The coordinate maxima are two-sided, hence
    the concentration log is

        log( 2 * split * pK / delta ).

    The experiments use delta = 1/pK (the paper's choice), giving

        log( 2 * split * pK^2 ) = 2 log pK + log(2 * split).

    The factor 2 inside the logarithm accounts for the two tails.  It is distinct
    from the factor 2 in sqrt(2 L / N), which comes from the Chernoff/Bernstein
    exponent and must remain in the radius.
    """
    pk = p_K(d, K)
    delta = (1.0 / pk) if delta is None else delta
    split = CONSTANTS.DELTA_SPLIT if split is None else split
    return math.log(2.0 * split * pk / delta)


def feature_subsets(d: int, K: int):
    """Ordered non-empty subsets |S| <= K: singletons then pairs (i<j)."""
    subs = [(i,) for i in range(d)]
    if K >= 2:
        subs += list(combinations(range(d), 2))
    return subs


def design_matrix(Z: np.ndarray, K: int, intercept: bool = False) -> np.ndarray:
    """Centered Walsh design over |S| <= K. Main effects chi_i = 2(z_i - 1/2) in
    {-1,+1}; pair columns are products of +-1 columns (hence also +-1).

    Blocking fix (report point 5): the paper's pK COUNTS the intercept
    (S = emptyset), and the theorem's Gram / C_est are over the FULL augmented
    Walsh design [1, chi_i, chi_i chi_j, ...].  At the population the main/pair
    columns are mean zero, but at FINITE N they are not exactly mean zero, so
    fitting y - y_mean on the un-centered Walsh columns is NOT algebraically
    equivalent to full OLS with an intercept.  Passing intercept=True prepends the
    all-ones column so the design (and the Gram, C_est, and the fit) is the exact
    augmented Walsh design the theorem defines.

    intercept=False returns the (N, pK-1) non-intercept block (kept for the few
    call sites that genuinely want only the effect columns, e.g. plotting the
    leakage on effect columns).  intercept=True returns the (N, pK) augmented
    design with the intercept as column 0.
    """
    Zc = 2.0 * (Z - 0.5)
    N, d = Zc.shape
    if K == 1:
        eff = Zc
    else:
        pair_cols = [Zc[:, i] * Zc[:, j] for (i, j) in combinations(range(d), 2)]
        eff = (np.concatenate([Zc, np.stack(pair_cols, axis=1)], axis=1)
               if pair_cols else Zc)
    if intercept:
        return np.concatenate([np.ones((N, 1)), eff], axis=1)
    return eff


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
    """Dense OLS over the FULL augmented degree-<=K Walsh design (report point 5).

    The design is [1, chi_i, chi_i chi_j, ...] with the intercept as a genuine
    fitted column (S = emptyset), so this is EXACT full OLS with intercept -- not
    "regress y - y_mean on un-centered Walsh columns", which only coincides with
    it at the population where the Walsh columns are mean zero.  Columns are
    standardized before inversion (the intercept column standardizes to itself,
    scale 1) and the coefficients mapped back to the original scale.

    Returns (beta on ORIGINAL scale for the NON-intercept coords, intercept,
    diag(Ginv) for the non-intercept coords).  The non-intercept ordering matches
    feature_subsets(d, K), so certified_set indexing is unchanged.  Raises
    LinAlgError if N <= pK or the augmented Gram is ill-conditioned (Assumption 1).
    """
    d = Z.shape[1]
    Xa = design_matrix(Z, K, intercept=True)      # (N, pK) augmented
    N = Xa.shape[0]
    if N <= p_K(d, K):
        raise np.linalg.LinAlgError(
            f"N={N} <= pK={p_K(d, K)}: dense K={K} fit not well-posed")
    Xs, scale = standardize_columns(Xa)           # scale[0] == 1 (intercept)
    G = (Xs.T @ Xs) / N
    if np.linalg.cond(G) > COND_MAX:
        raise np.linalg.LinAlgError("Gram ill-conditioned (N too small)")
    Ginv = np.linalg.inv(G)
    beta_std_full = Ginv @ (Xs.T @ y) / N
    beta_full = beta_std_full / scale             # original scale, incl intercept
    intercept = float(beta_full[0])
    beta = beta_full[1:]                          # non-intercept effect coords
    diag_ginv = np.diag(Ginv)[1:]                 # drop the intercept coordinate
    return beta, intercept, diag_ginv


# =========================================================================== #
#  R1.4 -- design-conditioning probe: the empirical Cest and its ingredients
#          as functions of the coordinate-to-budget ratio pK/N.
#
#  The referee's most serious point: Cest = max{gamma^{-1/2}, |||Ginv|||_inf}
#  is a max ABSOLUTE ROW SUM of the inverse empirical Gram, which grows with the
#  number of fitted coordinates pK at fixed budget N. It is calibrated at
#  (d=30, K=1, pK=31) and then FROZEN and transferred to d=49 and to K=2 designs
#  with far more coordinates. This probe measures gamma = lambda_min(Sigma_hat),
#  Cinv = |||Sigma_hat^{-1}|||_inf, and Cest = max{gamma^{-1/2}, Cinv} DIRECTLY
#  from a sampled Walsh design, so Tier 1 can chart Cest vs pK/N and test whether
#  the frozen value is stable in that ratio (the transfer claim) rather than in
#  d or K separately.
# =========================================================================== #
def op_inf_norm(M: np.ndarray) -> float:
    """||| M |||_inf = max_i sum_j |M_ij|  (the ell_inf -> ell_inf operator
    norm, i.e. the maximum absolute row sum)."""
    return float(np.max(np.abs(M).sum(axis=1)))


@dataclass
class ConditioningProbe:
    """R1.4 design-conditioning measurement at one (d, K, N) point.

    gamma     : lambda_min(Sigma_hat)         (Assumption 1 lower bound)
    Cinv      : |||Sigma_hat^{-1}|||_inf       (max abs row sum -- grows with pK)
    Cest_emp  : max{gamma^{-1/2}, Cinv}        (the empirical floor constant)
    ratio     : pK / N                         (the transfer axis)
    well_posed: whether the Gram was invertible at this (N, pK)
    """
    d: int
    K: int
    N: int
    pK: int
    ratio: float
    gamma: float
    Cinv: float
    Cest_emp: float
    well_posed: bool


def measure_conditioning(d: int, K: int, N: int, rng: np.random.Generator
                         ) -> ConditioningProbe:
    """Sample N Walsh masks, form the standardized degree-K Gram, and read off
    gamma, Cinv and Cest_emp. Pure design measurement: no response, no signal --
    Assumption 1 concerns the design only (the columns depend on masks/basis, not
    on rho or g_rho), which is exactly why this is reference-free.
    """
    pk = p_K(d, K)
    Z = sample_masks(N, d, rng)
    X = design_matrix(Z, K, intercept=True)       # augmented Gram (report pt 5)
    Xs, _ = standardize_columns(X)
    G = (Xs.T @ Xs) / N
    ratio = pk / N
    try:
        if np.linalg.cond(G) > COND_MAX:
            raise np.linalg.LinAlgError
        Ginv = np.linalg.inv(G)
        gamma = float(np.linalg.eigvalsh(G)[0])       # lambda_min
        Cinv = op_inf_norm(Ginv)
        gpow = gamma ** (-0.5) if gamma > 0 else float("inf")
        Cest_emp = max(gpow, Cinv)
        wp = math.isfinite(Cest_emp)
    except np.linalg.LinAlgError:
        gamma, Cinv, Cest_emp, wp = float("nan"), float("nan"), float("nan"), False
    return ConditioningProbe(d=d, K=K, N=N, pK=pk, ratio=ratio, gamma=gamma,
                             Cinv=Cinv, Cest_emp=Cest_emp, well_posed=wp)


def realized_cest(Z: np.ndarray, K: int) -> float:
    """R1.4 (Path A) -- the forward floor constant Cest read off the ACTUAL
    design that will be used for the fit, rather than a frozen scalar.

    Cest = max{ gamma^{-1/2}, |||Sigma_hat^{-1}|||_inf }  on the standardized
    degree-K Walsh Gram of THIS mask bank. Because Assumption 1 concerns the
    design only (columns depend on masks/basis, not on rho or g_rho), this is a
    pure, model-free, reference-free quantity computable in milliseconds from the
    same X the OLS fit uses. The Tier-1b transfer study (R1.4) shows Cest does
    NOT collapse in pK/N across (d,K), so no single frozen value is correct; the
    honest floor measures Cest per run. If the Gram is ill-conditioned the
    function raises and the run is unresolved; it never falls back to 1.
    """
    X = design_matrix(Z, K, intercept=True)       # augmented Gram (report pt 5)
    Xs, _ = standardize_columns(X)
    N = Xs.shape[0]
    G = (Xs.T @ Xs) / N
    # Report point 9: do NOT fall back to C_FLOOR=1 when the Gram is bad -- that
    # is exactly when C_est is LARGE, so pinning it to the ideal 1 makes the floor
    # anti-conservative.  An ill-conditioned design means the run is unresolved,
    # so we raise (consistent with ols_fit, same COND_MAX); callers treat this as
    # "no certificate for this run".
    if np.linalg.cond(G) > COND_MAX:
        raise np.linalg.LinAlgError(
            "Gram ill-conditioned: design not well-posed, run unresolved "
            "(no certificate; C_est not defined)")
    Ginv = np.linalg.inv(G)
    gamma = float(np.linalg.eigvalsh(G)[0])
    Cinv = op_inf_norm(Ginv)
    gpow = gamma ** (-0.5) if gamma > 0 else float("inf")
    cest = max(gpow, Cinv)
    if not math.isfinite(cest):
        raise np.linalg.LinAlgError(
            "C_est not finite: design not well-posed, run unresolved")
    return cest


def certified_floor_from_design(
    Z: np.ndarray,
    K: int,
    *,
    sigma_obs_ub: float,
    m_ub: float,
    B_pop_ub: float,
    delta: float = None,
    split: int = None,
) -> float:
    """Canonical forward certificate using the realized augmented Gram.

    This is the only function that should be used for a theorem-level certified
    sign decision.  It implements Eq. (6) term-for-term and never uses C_M or
    sigma_eff.
    """
    c_est = realized_cest(Z, K)
    return certified_floor(
        c_est, sigma_obs_ub, m_ub, B_pop_ub,
        d=Z.shape[1], N=Z.shape[0], K=K, delta=delta, split=split,
    )


def planning_floor_from_design(
    Z: np.ndarray,
    s_eff: float,
    K: int,
    *,
    family_wise: bool = True,
    delta: float = None,
    split: int = None,
) -> float:
    """Historical/diagnostic single-scale radius with realized C_est.

    This is NOT a certificate.  It is retained only for planning diagnostics and
    for reproducing old single-scale curves.
    """
    c_est = realized_cest(Z, K)
    return floor_value(
        s_eff, Z.shape[1], Z.shape[0], K,
        family_wise=family_wise, C_floor=c_est, delta=delta, split=split,
    )


def floor_from_design(
    Z: np.ndarray,
    s_eff: float,
    K: int,
    family_wise: bool = True,
    delta: float = None,
    split: int = None,
    sigma_obs: float = None,
    m_hat: float = None,
    B: float = None,
) -> float:
    """Backward-compatible wrapper.

    If (sigma_obs, m_hat, B) are supplied this delegates to the canonical
    two-term certificate.  Otherwise it returns the legacy planning radius.
    New certified code should call ``certified_floor_from_design`` directly.
    """
    if m_hat is not None:
        if sigma_obs is None or B is None:
            raise ValueError(
                "certified mode requires sigma_obs, m_hat and B; "
                "use planning_floor_from_design for the legacy single-scale radius"
            )
        return certified_floor_from_design(
            Z, K, sigma_obs_ub=sigma_obs, m_ub=m_hat, B_pop_ub=B,
            delta=delta, split=split,
        )
    return planning_floor_from_design(
        Z, s_eff, K, family_wise=family_wise, delta=delta, split=split,
    )


# =========================================================================== #
#  THE FLOOR  (forward direction) -- one function, used everywhere
# =========================================================================== #
def sigma_eff(sigma_obs: float, m_hat: float, C_m: float = None) -> float:
    """Empirical planning scale: sigma_obs + C_m * sqrt(max(m_hat, 0)).

    This quantity is used only by the backward budget planner.  It is not a
    confidence radius and must never be used for a certified forward decision.
    """
    C_m = CONSTANTS.C_M if C_m is None else C_m
    return sigma_obs + C_m * math.sqrt(max(m_hat, 0.0))


# --------------------------------------------------------------------------- #
#  R1.5 -- the FULL Bernstein leakage bound (both terms) and its domination
#          regime. Lemma 1 is Bernstein, not purely sub-Gaussian:
#
#     eta_N  <=  sqrt( 2 m log(.) / N )        (sub-Gaussian term)
#             +  (2/3) * B * log(.) / N        (sub-exponential term)
#
#  with m = m>K,rho the mismatch energy and B = ||r>K,rho||_inf its sup-norm.
#  The sub-exponential term is dominated once
#
#     N  >~  (B^2 / m) * log(.)                (domination regime),
#
#  This is an additional regime condition; N >~ pK alone does not imply it
#  because the threshold also depends on B^2/m.  We compute both terms so Tier 1
#  can report the domination diagnostic without using it for validity.
# --------------------------------------------------------------------------- #
def leakage_bound_terms(m: float, B: float, d: int, N: int, K: int = 1,
                        delta: float = None, split: int = None):
    """Return (sub_gaussian_term, sub_exponential_term, log_factor) of the
    Bernstein leakage bound. `B` is the sup-norm ||r>K,rho||_inf of the mismatch
    residual (bounded response => finite). The reported eta bound is their sum.
    """
    L = log_pk_over_delta(d, K, delta, split)
    m = max(m, 0.0)
    sub_g = math.sqrt(2.0 * m * L / N)
    sub_e = (2.0 / 3.0) * max(B, 0.0) * L / N
    return sub_g, sub_e, L


def leakage_domination_N(m: float, B: float, d: int, K: int = 1,
                         delta: float = None, split: int = None) -> float:
    """Smallest N above which the sub-exponential term is <= the sub-Gaussian
    term, i.e. N >= (2 B^2 / (9 m)) * log(.). Below this N the Bernstein tail is
    NOT dominated and C_M cannot absorb it; Tier 1 checks its cells clear this.
    """
    m = max(m, 1e-12)
    L = log_pk_over_delta(d, K, delta, split)
    return (2.0 * B ** 2 / (9.0 * m)) * L


def floor_value(s_eff: float, d: int, N: int, K: int = 1,
                family_wise: bool = True, C_floor: float = None,
                delta: float = None, split: int = None) -> float:
    """floor(N, rho) = C_floor * sigma_eff * sqrt(2 log(2*SPLIT*pK/delta) / N)
    (family-wise), or the single pre-registered-coordinate variant with
    z_{1-alpha}.

    R1.2: the log factor is the two-sided union-bounded log_pk_over_delta().
    With delta = 1/pK and split = 1, L = 2 log pK + log 2 and the radius is
    sqrt(2 L / N); the outer factor 2 under the square root is not removed.

    DEPRECATED FOR CERTIFICATION.  This single-term form folds the whole leakage
    into  C_floor * C_m sqrt(m) * sqrt(2 L / N)  via sigma_eff, which is only an
    upper bound on the leading Lemma-1 term when C_m >= 1 (see the C_M note).  For
    a certified decision use certified_floor()/certified_floor_from_design(), which carry
    the two Bernstein terms explicitly.  This function is retained for the
    BACKWARD budget rule and for the synthetic collapse curves, where sigma_eff
    with the calibrated C_m is the intended planning quantity, and for the
    legacy planning/synthetic diagnostics where this single-term form is intended.
    """
    C_floor = CONSTANTS.C_FLOOR if C_floor is None else C_floor
    if family_wise:
        L = log_pk_over_delta(d, K, delta, split)
        return C_floor * s_eff * math.sqrt(2.0 * L / N)
    return C_floor * s_eff * CONSTANTS.Z_ALPHA / math.sqrt(N)


def certified_floor(C_est: float, sigma_obs: float, m_hat: float, B: float,
                     d: int, N: int, K: int = 1, delta: float = None,
                     split: int = None) -> float:
    """Theorem-1 two-term certified radius.

    floor = C_est * [
        sigma_obs * sqrt(2 L / N)
        + sqrt(2 m L / N)
        + (2/3) B L / N
    ],
    L = log(2 * split * pK / delta).

    ``sigma_obs``, ``m_hat`` and ``B`` are interpreted as VALID UPPER BOUNDS
    for the query-noise sub-Gaussian scale, mismatch energy, and population
    mismatch-residual sup-norm, respectively.  C_M does not appear here.
    """
    vals = {
        "C_est": C_est,
        "sigma_obs": sigma_obs,
        "m_hat": m_hat,
        "B": B,
    }
    for name, value in vals.items():
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be a finite non-negative bound")
    if N <= 0:
        raise ValueError("N must be positive")
    L = log_pk_over_delta(d, K, delta, split)
    noise_radius = float(sigma_obs) * math.sqrt(2.0 * L / N)
    leak_subg = math.sqrt(2.0 * float(m_hat) * L / N)
    leak_sube = (2.0 / 3.0) * float(B) * L / N
    return float(C_est) * (noise_radius + leak_subg + leak_sube)


def population_residual_sup_bound(output_abs_bound: float, d: int,
                                  K: int = 1) -> float:
    """A generic deterministic B_pop bound for a uniformly bounded response.

    If |g(z)| <= M under the uniform Walsh measure, Parseval gives
    ||beta_{<=K}||_2 <= ||g||_2 <= M.  Hence, pointwise,
        |g_{<=K}(z)| <= sqrt(pK) * M
    and therefore
        ||r_{>K}||_inf <= M * (1 + sqrt(pK)).

    For a probability-valued black box, M=1 is valid.  This bound can be loose,
    but unlike a sample residual maximum it is a genuine population sup bound.
    """
    M = float(output_abs_bound)
    if not math.isfinite(M) or M < 0:
        raise ValueError("output_abs_bound must be finite and non-negative")
    return M * (1.0 + math.sqrt(p_K(d, K)))


def certified_set(beta: np.ndarray, fl: float):
    """Forward rule: certify coordinate S iff |beta_hat_S| > floor. Returns
    (index set, sign vector)."""
    idx = np.where(np.abs(beta) > fl)[0]
    return set(idx.tolist()), np.sign(beta)


# =========================================================================== #
#  THE BUDGET RULE  (backward direction) -- Eq. 8, with the budget constant
# =========================================================================== #
def predict_budget(s_eff: float, beta_min: float, d: int, K: int = 1,
                   family_wise: bool = True, C_budget: float = None,
                   delta: float = None, split: int = None) -> int:
    """Calibrated leading-order backward PLANNING rule.

        N_pred = ceil( C_budget^2 * sigma_eff_plan^2
                       * 2 log(2*split*pK/delta) / beta_min^2 )

    This is not the algebraic inverse of the certified two-term floor: it omits
    the lower-order B/N term and cannot know the realized C_est before a design
    exists.  After drawing the design, callers must recompute the actual
    two-term radius before making any certified decision.
    """
    C_budget = CONSTANTS.C_BUDGET if C_budget is None else C_budget
    norm = (2.0 * log_pk_over_delta(d, K, delta, split) if family_wise
            else CONSTANTS.Z_ALPHA ** 2)
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
                family_wise: bool = True, rng: np.random.Generator = None,
                sigma_obs: float = None, m_hat: float = None,
                B: float = None) -> BudgetPlan:
    """Full backward plan: predict N with the empirical C_budget planning
    constant, clamp to feasibility, and report the realized floor and its ratio.

    R1.4 (Path A): the realized floor at N_run measures Cest from a sampled design
    of that size.  Blocking fix: when the mismatch breakdown (sigma_obs, m_hat, B)
    is supplied, the realized floor is the honest TWO-TERM certified floor, so the
    reported realized_floor/beta_min is the real post-design radius rather than the
    absorbed single-term one.  The PREDICTION step still uses the empirical
    planning quantity s_eff (with the calibrated C_m and C_budget), which is the
    intended backward-direction heuristic; the certificate itself is re-measured
    from the design.  A fresh rng is drawn if none is supplied; the design is
    model-free so this adds no query cost.
    """
    N_pred = predict_budget(s_eff, beta_min, d, K, family_wise)
    feas = feasibility_floor(d, K)
    N_run = max(N_pred, feas)
    rng = np.random.default_rng() if rng is None else rng
    Zdesign = sample_masks(N_run, d, rng)
    if m_hat is not None:
        if sigma_obs is None or B is None:
            raise ValueError("post-run certified check needs sigma_obs, m_hat and B")
        fl = certified_floor_from_design(
            Zdesign, K, sigma_obs_ub=sigma_obs, m_ub=m_hat,
            B_pop_ub=B,
        )
    else:
        fl = planning_floor_from_design(
            Zdesign, s_eff, K, family_wise=family_wise,
        )
    return BudgetPlan(N_pred=N_pred, N_run=N_run, realized_floor=fl,
                      ratio=fl / beta_min, clamped=(N_pred < feas))


# =========================================================================== #
#  PILOT ESTIMATION
#  - planning point estimate: cross-fitted difference of variances
#  - certified mismatch-energy UCB: honest train/validation split, no subtraction
# =========================================================================== #
@dataclass
class MismatchEstimate:
    """Planning-only mismatch estimate and diagnostics.

    m_raw     : cross-fitted held-out MSE minus sigma_obs^2 (may be negative)
    m_hat     : max(m_raw, 0), used only in the backward planning scale
    negative  : whether clipping was active
    resid_var : held-out mean squared residual
    B_hat     : sample max |residual|, diagnostic only -- NOT a valid B_pop bound
    """
    m_raw: float
    m_hat: float
    negative: bool
    resid_var: float
    B_hat: float


def _held_out_residual(Z, y, K, cross_fit=True):
    """Return out-of-sample residuals for the planning point estimate.

    With cross_fit=True, every observation is predicted by a model fitted without
    that observation.  This is appropriate for the empirical planning proxy.
    The certified UCB below uses a simpler one-way honest split so its validation
    q_t are i.i.d. conditional on the fitted pilot.
    """
    if cross_fit:
        n = Z.shape[0]
        half = n // 2
        if half <= p_K(Z.shape[1], K):
            raise np.linalg.LinAlgError(
                "pilot fold too small for dense OLS; increase pilot size"
            )
        resid = np.empty(n)
        for tr, te in [(slice(0, half), slice(half, n)),
                       (slice(half, n), slice(0, half))]:
            beta, b0, _ = ols_fit(Z[tr], y[tr], K)
            yhat = b0 + design_matrix(Z[te], K) @ beta
            resid[te] = y[te] - yhat
        return resid
    beta, b0, _ = ols_fit(Z, y, K)
    yhat = b0 + design_matrix(Z, K) @ beta
    return y - yhat


def _honest_validation_residual(Z: np.ndarray, y: np.ndarray, K: int,
                                validation_fraction: float = 0.5):
    """Fit once on a training split and return residuals on an independent split.

    The row order is already i.i.d. from the mask sampler, so a deterministic
    prefix/suffix split preserves independence.  Conditional on the fitted pilot,
    the validation residual squares are i.i.d.; this is the setting used by the
    empirical-Bernstein UCB in Appendix C.1.

    Returns (resid_val, beta_train, intercept_train, n_train, n_val).
    """
    n = int(Z.shape[0])
    if n != int(np.asarray(y).shape[0]):
        raise ValueError("Z and y must have the same number of rows")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must lie in (0,1)")
    pk = p_K(Z.shape[1], K)
    # Keep at least pk+1 training rows and at least 2 validation rows.
    n_val = max(2, int(round(n * validation_fraction)))
    n_train = n - n_val
    if n_train <= pk:
        n_train = pk + 1
        n_val = n - n_train
    if n_train <= pk or n_val < 2:
        raise np.linalg.LinAlgError(
            f"pilot N={n} too small for honest split at pK={pk}"
        )

    beta, b0, _ = ols_fit(Z[:n_train], y[:n_train], K)
    Xv = design_matrix(Z[n_train:], K)
    yhat = b0 + Xv @ beta
    resid = np.asarray(y[n_train:], dtype=float) - yhat
    return resid, beta, b0, n_train, n_val


def estimate_mismatch_detail(Z: np.ndarray, y: np.ndarray, K: int,
                             sigma_obs: float,
                             cross_fit: bool = True) -> MismatchEstimate:
    """Cross-fitted point estimate used only by the BACKWARD planning model."""
    resid = _held_out_residual(Z, y, K, cross_fit)
    mse = float((resid ** 2).mean())
    m_raw = mse - float(sigma_obs) ** 2
    m_hat = max(m_raw, 0.0)
    B_hat = float(np.max(np.abs(resid))) if resid.size else 0.0
    return MismatchEstimate(
        m_raw=m_raw, m_hat=m_hat, negative=(m_raw < 0.0),
        resid_var=mse, B_hat=B_hat,
    )


def estimate_mismatch_from_residual(Z: np.ndarray, y: np.ndarray, K: int,
                                    sigma_obs: float,
                                    cross_fit: bool = True) -> float:
    """Backward-compatible scalar wrapper for the planning point estimate."""
    return estimate_mismatch_detail(Z, y, K, sigma_obs, cross_fit).m_hat


def pilot_N0(d: int, K: int = 1) -> int:
    """Pilot size N0 = max(500, 6 pK), enough for two dense half-sample fits."""
    return max(500, 6 * p_K(d, K))


@dataclass
class MismatchEnergyUCB:
    """One-sided upper-confidence bound for mismatch energy.

    q_bar             : mean held-out squared residual
    m_ucb             : empirical-Bernstein upper bound on m_{>K,rho}
    var_q             : sample variance of q_t
    R_val             : known/conditional bound on |validation residual|
    bernstein_var     : sqrt-variance margin term
    bernstein_range   : bounded-range margin term
    delta_pilot       : failure probability allocated to this pilot event
    n_train, n_val    : honest split sizes
    max_abs_resid     : observed validation maximum (diagnostic only)
    """
    q_bar: float
    m_ucb: float
    var_q: float
    R_val: float
    bernstein_var: float
    bernstein_range: float
    delta_pilot: float
    n_train: int
    n_val: int
    max_abs_resid: float


def mismatch_energy_ucb(
    Z: np.ndarray,
    y: np.ndarray,
    K: int,
    *,
    d: int = None,
    delta_pilot: float = None,
    validation_fraction: float = 0.5,
    output_abs_bound: float = None,
    R_val: float = None,
) -> MismatchEnergyUCB:
    """Empirical-Bernstein UCB for m_{>K,rho}, matching Appendix C.1.

    The pilot is fit on one subset and evaluated on an independent validation
    subset.  With q_t = (y_t - ghat(z_t))^2,

        E[q_t | ghat] >= m_{>K,rho}

    because the population degree-K projection minimizes L2 error, and query
    noise is conditionally mean-zero.  We therefore upper-bound E[q_t] directly:

        m_ucb = q_bar
              + sqrt(2 V_q log(2/delta_pilot) / n_val)
              + (7/3) R_val^2 log(2/delta_pilot) / (n_val - 1).

    IMPORTANT:
      * no estimated noise variance is subtracted;
      * R_val bounds the VALIDATION residual, not the theorem's B_pop;
      * the theorem still separately requires sigma_obs_ub and B_pop_ub;
      * if R_val is not passed explicitly, output_abs_bound must be supplied.
        Conditional on the fitted pilot and |y| <= M,
          R_val = M + |intercept| + sum_j |beta_j|
        is valid because Walsh features are +/-1.

    The experimental default total delta is 1/pK; this pilot receives delta/4.
    """
    d = Z.shape[1] if d is None else int(d)
    pk = p_K(d, K)
    if delta_pilot is None:
        delta_total = 1.0 / pk
        delta_pilot = delta_total / CONSTANTS.DELTA_SPLIT_UCB
    delta_pilot = float(delta_pilot)
    if not (0.0 < delta_pilot < 1.0):
        raise ValueError("delta_pilot must lie in (0,1)")

    resid, beta, b0, n_train, n_val = _honest_validation_residual(
        np.asarray(Z, dtype=float), np.asarray(y, dtype=float), K,
        validation_fraction=validation_fraction,
    )

    if R_val is None:
        if output_abs_bound is None:
            raise ValueError(
                "certified mismatch UCB needs either R_val or output_abs_bound; "
                "a sample residual maximum is not a valid a-priori range"
            )
        M = float(output_abs_bound)
        if not math.isfinite(M) or M < 0:
            raise ValueError("output_abs_bound must be finite and non-negative")
        # Conditional on the training fit, this is deterministic.
        R_val = M + abs(float(b0)) + float(np.abs(beta).sum())
    R_val = float(R_val)
    if not math.isfinite(R_val) or R_val < 0:
        raise ValueError("R_val must be finite and non-negative")

    max_abs = float(np.max(np.abs(resid))) if resid.size else 0.0
    # A supplied/derived bound should dominate the realized validation residuals.
    # If it does not, refusing certification is safer than silently widening from
    # the sample maximum after looking at the validation data.
    tol = 1e-10 * max(1.0, R_val)
    if max_abs > R_val + tol:
        raise ValueError(
            f"R_val={R_val:.6g} is violated by validation residual "
            f"{max_abs:.6g}; provide a valid larger bound"
        )

    q = resid ** 2
    q_bar = float(q.mean())
    var_q = float(q.var(ddof=1))
    Lp = math.log(2.0 / delta_pilot)
    var_term = math.sqrt(2.0 * var_q * Lp / n_val)
    range_term = (7.0 / 3.0) * (R_val ** 2) * Lp / (n_val - 1)
    m_ucb = q_bar + var_term + range_term

    return MismatchEnergyUCB(
        q_bar=q_bar,
        m_ucb=float(m_ucb),
        var_q=var_q,
        R_val=R_val,
        bernstein_var=var_term,
        bernstein_range=range_term,
        delta_pilot=delta_pilot,
        n_train=n_train,
        n_val=n_val,
        max_abs_resid=max_abs,
    )


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
                    fl_exact: float = None, tol: float = 1e-9):
    """Score EVERY run-certified coordinate against the exact projection.

    A run-certified coordinate whose exact coefficient is numerically zero is a
    false certification, not something to remove from the denominator.  This is
    the direct test of the forward claim: every coordinate the run certifies must
    have a non-zero exact coefficient with the same sign.

    ``fl_exact`` is accepted only for signature compatibility and is not used.
    Returns (n_false, n_scored).
    """
    run_cert = np.abs(beta_run) > fl
    exact_nonzero = np.abs(beta_exact) > tol
    sign_match = np.sign(beta_run) == np.sign(beta_exact)
    false = run_cert & (~exact_nonzero | ~sign_match)
    return int(false.sum()), int(run_cert.sum())


@dataclass
class PowerMiss:
    """R1.9 -- enumeration POWER against the exact projection.

    A certificate that resolves almost nothing is trivially free of false signs,
    so zero-false-sign is only meaningful alongside how much the run RECOVERS of
    what is truly resolvable. Ground truth is the exact-cube beta. We report TWO
    distinct quantities that are easy to conflate but mean different things.

    (1) GUARANTEE check -- the Theorem-1 factor-two margin. The margin is stated
        at the RUN's own floor: |beta| > 2*fl_run  =>  the run must certify with
        the right sign. So the honest guarantee denominator is coords with
        |beta_exact| > 2*fl_run, and a miss there is a genuine guarantee gap.
        (Using the exact-cube floor here would be WRONG: fl_exact << fl_run at a
        deployment budget, so a coord can exceed 2*fl_exact yet sit below fl_run,
        which Theorem 1 never promised the run would resolve.)
          n_margin        : # coords with |beta_exact| > 2*fl_run
          n_margin_recovered / n_margin_miss : recovered / missed of those
          -> n_margin_miss MUST be 0 for the guarantee to hold.

    (2) RESOLUTION recovery -- the honest finite-budget resolution cost. The full
        cube (huge N) resolves more than a deployment-budget run; the gap is not
        an error but exactly the "below-floor = unresolved, not zero" message.
          n_cube_resolvable : # coords the EXACT fit certifies (|beta_exact|>fl_exact)
          n_cube_recovered / n_cube_miss : how many the run also / does not certify
          -> n_cube_miss > 0 is EXPECTED (the run's floor is coarser); this is a
             resolution statement, NOT a guarantee violation.
    """
    n_margin: int
    n_margin_recovered: int
    n_margin_miss: int
    n_cube_resolvable: int
    n_cube_recovered: int
    n_cube_miss: int


def power_miss_stats(beta_run: np.ndarray, beta_exact: np.ndarray,
                     fl_run: float, fl_exact: float) -> PowerMiss:
    """R1.9 -- power against the exact projection, split into the Theorem-1
    GUARANTEE check (threshold at the run's own floor, must have zero misses) and
    the RESOLUTION recovery (what the full cube resolves that the smaller-budget
    run does not -- an expected finite-budget cost, not a violation).

    See PowerMiss for why the guarantee threshold is 2*fl_run and NOT 2*fl_exact:
    the factor-two margin of Theorem 1 is relative to the floor at the budget the
    run was actually executed at.
    """
    run_cert = np.abs(beta_run) > fl_run
    # (1) Theorem-1 guarantee stratum: margin at the RUN floor.
    margin = np.abs(beta_exact) > 2.0 * fl_run
    margin_rec = int((margin & run_cert).sum())
    n_margin = int(margin.sum())
    # (2) Resolution recovery relative to the full-cube floor.
    cube_res = np.abs(beta_exact) > fl_exact
    cube_rec = int((cube_res & run_cert).sum())
    n_cube = int(cube_res.sum())
    return PowerMiss(
        n_margin=n_margin, n_margin_recovered=margin_rec,
        n_margin_miss=n_margin - margin_rec,
        n_cube_resolvable=n_cube, n_cube_recovered=cube_rec,
        n_cube_miss=n_cube - cube_rec)


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
    floor_first_old: float = None   # R1.4: old frozen-C_FLOOR floor (first rung)
    floor_last_old: float = None    # R1.4: old frozen-C_FLOOR floor (last rung)
    cest_first: float = None        # R1.4: realized Cest at first rung
    cest_last: float = None         # R1.4: realized Cest at last rung


def sweep_prefix_ladder(Zbank: np.ndarray, ybank: np.ndarray, N_list,
                        s_eff: float, d: int, K: int,
                        family_wise: bool = True, sigma_obs: float = None,
                        m_hat: float = None, B: float = None) -> DiagnosticTrace:
    """Apply the single-budget theorem at each rung of a prefix-nested ladder.

    R1.4 (Path A): the floor at each rung uses Cest MEASURED from that rung's
    realized design prefix Zbank[:N] (floor_from_design), not the frozen C_FLOOR.
    We also record the old frozen-C_FLOOR floor at the first/last rung so drivers
    can report the before/after floor inflation.

    Blocking fix: when (sigma_obs, m_hat, B) are supplied the per-rung floor is
    the honest TWO-TERM certified floor; otherwise the legacy single-scalar
    s_eff floor is used (synthetic callers).

    Primary output: sign_flips among coordinates certified at both rungs.  With
    unknown beta this is a stability/consistency diagnostic, not a direct
    correctness test.  count_monotone / set_nested are additional workflow
    diagnostics.
    """
    tr = DiagnosticTrace()
    prev_set = prev_beta = None
    prev_count = -1
    for N in N_list:
        Zc = Zbank[:N]
        beta, _, _ = ols_fit(Zc, ybank[:N], K)
        if m_hat is not None:
            if sigma_obs is None or B is None:
                raise ValueError("two-term prefix floor needs sigma_obs, m_hat and B")
            fl = certified_floor_from_design(
                Zc, K, sigma_obs_ub=sigma_obs, m_ub=m_hat, B_pop_ub=B,
            )
        else:
            fl = planning_floor_from_design(
                Zc, s_eff, K, family_wise=family_wise,
            )
        cur_set, _ = certified_set(beta, fl)
        if tr.floor_first is None:
            tr.floor_first, tr.cert_first = fl, len(cur_set)
            tr.floor_first_old = floor_value(s_eff, d, N, K, family_wise)
            tr.cest_first = realized_cest(Zc, K)
        tr.floor_last, tr.cert_last = fl, len(cur_set)
        tr.floor_last_old = floor_value(s_eff, d, N, K, family_wise)
        tr.cest_last = realized_cest(Zc, K)
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

# =========================================================================== #
#  TIER 2b -- INDEPENDENT-RESEEDING STABILITY AUDIT
# =========================================================================== #
@dataclass
class ReseedResult:
    """Outcome for one probe.

    The cross-seed majority is only a stability reference, not ground truth.
    Empty-certified-set runs are INCLUDED in the run denominator: such a run has
    zero disagreement by definition.

    `target_1_over_pK` is retained only as a nominal reference scale for tables;
    the disagreement rates are NOT direct tests of Theorem 1's coverage.
    """
    pK: int
    target_1_over_pK: float
    n_cert_checks: int
    n_viol: int
    viol_rate: float
    n_runs: int
    n_run_disagree: int
    run_disagree_rate: float
    jaccard_above: float
    jaccard_band: float
    n_above: float
    n_band: float
    well_posed: bool

    # Backward-compatible read-only aliases for older driver code.
    @property
    def n_run_fail(self):
        return self.n_run_disagree

    @property
    def run_fail_rate(self):
        return self.run_disagree_rate


def _mean_pairwise_jaccard(sets):
    """Mean Jaccard over unordered pairs, avoiding vacuous all-empty = 1 reports."""
    m = len(sets)
    if m < 2 or all(len(s) == 0 for s in sets):
        return float("nan")
    tot, npair = 0.0, 0
    for i in range(m):
        for j in range(i + 1, m):
            a, b = sets[i], sets[j]
            union = a | b
            # Within a non-vacuous stratum, two empty sets genuinely agree.
            tot += 1.0 if not union else len(a & b) / len(union)
            npair += 1
    return tot / npair if npair else float("nan")


def reseed_audit(
    query_fn,
    d: int,
    N: int,
    R: int,
    K: int,
    *,
    sigma_obs_ub: float,
    m_bound: float,
    B_pop_bound: float,
    seed0: int = 0,
    p_keep: float = None,
    split: int = None,
):
    """Independent-reseeding stability audit using the canonical two-term floor.

    Each seed draws an independent mask bank, fits dense OLS, measures C_est from
    that seed's augmented Gram, and certifies with

        certified_floor_from_design(
            sigma_obs_ub, m_bound, B_pop_bound, split=...
        ).

    The nuisance quantities are held fixed across seeds because the pilot is run
    once, exactly as in deployment.  For a theorem-valid UCB row, callers should
    pass m_bound=m_ucb, a valid sigma_obs_ub, a valid B_pop_bound, and split=4.

    For plain point-estimate rows the same routine can be used as an empirical
    stability diagnostic, but the caller must not label the plug-in quantities as
    theorem-valid upper bounds unless they actually are.

    Cross-seed majority disagreement is STABILITY only; correctness is tested in
    the exact-beta enumeration tier.
    """
    pk = p_K(d, K)
    nominal_ref = 1.0 / pk
    pcount = len(feature_subsets(d, K))

    if N <= pk:
        return ReseedResult(
            pk, nominal_ref, 0, 0, float("nan"), 0, 0, float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan"),
            well_posed=False,
        )

    betas, floors, cert_sets = [], [], []
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        Z = sample_masks(N, d, rng, p_keep)
        y = query_fn(Z)
        try:
            beta, _, _ = ols_fit(Z, y, K)
            fl = certified_floor_from_design(
                Z, K,
                sigma_obs_ub=sigma_obs_ub,
                m_ub=m_bound,
                B_pop_ub=B_pop_bound,
                split=split,
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        cset, _ = certified_set(beta, fl)
        betas.append(beta)
        floors.append(fl)
        cert_sets.append(cset)

    R_ok = len(betas)
    if R_ok < 2:
        return ReseedResult(
            pk, nominal_ref, 0, 0, float("nan"), R_ok, 0, float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan"),
            well_posed=False,
        )

    beta_mat = np.stack(betas, axis=0)
    signs = np.sign(beta_mat)
    med_floor = float(np.median(floors))

    # Majority sign over seeds that certified a coordinate. This is a stability
    # proxy only and must not be interpreted as sgn(beta_true).
    ref_sign = np.zeros(pcount)
    for c in range(pcount):
        cert_mask = np.array([c in cs for cs in cert_sets])
        src = signs[cert_mask, c] if cert_mask.any() else signs[:, c]
        s = src.sum()
        ref_sign[c] = 1.0 if s >= 0 else -1.0

    n_cert_checks = 0
    n_viol = 0
    n_run_disagree = 0
    # IMPORTANT: every well-posed draw is in the denominator, including draws
    # with an empty certified set (which have zero disagreement).
    n_runs = R_ok

    for r in range(R_ok):
        run_has_disagree = False
        for c in cert_sets[r]:
            n_cert_checks += 1
            if signs[r, c] != ref_sign[c]:
                n_viol += 1
                run_has_disagree = True
        if run_has_disagree:
            n_run_disagree += 1

    viol_rate = (n_viol / n_cert_checks) if n_cert_checks else float("nan")
    run_disagree_rate = n_run_disagree / n_runs

    med_mag = np.median(np.abs(beta_mat), axis=0)
    above_coords = set(np.where(med_mag > 2.0 * med_floor)[0].tolist())
    band_coords = set(np.where(
        (med_mag >= med_floor) & (med_mag <= 2.0 * med_floor)
    )[0].tolist())

    above_sets = [cs & above_coords for cs in cert_sets]
    band_sets = [cs & band_coords for cs in cert_sets]
    jac_above = _mean_pairwise_jaccard(above_sets)
    jac_band = _mean_pairwise_jaccard(band_sets)
    n_above = float(np.mean([len(s) for s in above_sets]))
    n_band = float(np.mean([len(s) for s in band_sets]))

    return ReseedResult(
        pK=pk,
        target_1_over_pK=nominal_ref,
        n_cert_checks=n_cert_checks,
        n_viol=n_viol,
        viol_rate=viol_rate,
        n_runs=n_runs,
        n_run_disagree=n_run_disagree,
        run_disagree_rate=run_disagree_rate,
        jaccard_above=jac_above,
        jaccard_band=jac_band,
        n_above=n_above,
        n_band=n_band,
        well_posed=True,
    )

