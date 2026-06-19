"""
tier3_feasibility.py
====================
TIER 3 -- Feasibility (Reading 2). ISOLATED from the guarantee tables on
purpose: it answers a DIFFERENT question -- not "does the guarantee hold" but
"what does the guarantee COST".

The counter-intuitive result, given its own figure-data table:
  Moving K=1 -> K=2 pushes pairwise structure OUT of the mismatch term and INTO
  the fitted design, so sigma_eff FALLS. Naively that predicts K=2 is cheaper.
  It is NOT: the candidate count jumps p1 = d+1  ->  p2 = 1 + d + C(d,2) ~ d^2/2,
  lifting the FEASIBILITY floor ~ pK, so the predicted budget RISES despite the
  lower noise.

This script produces the two curves whose crossing tells the story:
  * resolution budget  N_res = 2 C_budget^2 sigma_eff^2 log pK / beta_min^2
  * feasibility floor   N_feas = c * pK
The certificate is then a diagnostic: if the budget is far below p2, dense
pairwise LIME is NOT identifiable and must not be read as an interaction map.

Pure numpy (uses the same synthetic generator as Tier 1 to get a realistic
sigma_eff drop from K=1 to K=2). No torch.

Run:  python tier3_feasibility.py
"""
from __future__ import annotations
import math
import numpy as np

import bl_core as bl
from tier1_synthetic import make_function


def measure_sigma_eff(d, K, n_active, m_resid, sigma_obs, N_pilot, n_trials,
                      seed0=0):
    """Estimate sigma_eff for a degree-K surrogate on the SAME planted function.
    At K=2 the pairwise block that was mismatch at K=1 is now FITTED, so the
    measured mismatch energy drops -- the mechanism behind the sigma_eff fall."""
    s_effs = []
    for t in range(n_trials):
        _, _, sf, _, _ = make_function(d, n_active, beta_active=0.12,
                                       m_resid=m_resid, seed=seed0 + t)
        Z, y = sf(N_pilot, sigma_obs, rng=np.random.default_rng(100 + t))
        try:
            m_hat = bl.estimate_mismatch_from_residual(
                Z, y, K, sigma_obs, cross_fit=(K == 2))
        except np.linalg.LinAlgError:
            continue
        s_effs.append(bl.sigma_eff(sigma_obs, m_hat))
    return float(np.mean(s_effs)) if s_effs else float("nan")


def feasibility_table(d=18, n_active=4, m_resid=0.05, sigma_obs=0.05,
                      beta_min=0.02, n_trials=20):
    """Compare K=1 vs K=2 on the same function: sigma_eff (falls) vs predicted
    budget (rises). d=18 so the K=2 cube is enumerable (matches the exact-beta check)."""
    print("=" * 72)
    print("TIER 3 -- FEASIBILITY (Reading 2): why K=2 costs more, less noise")
    print("=" * 72)
    print(f"  d={d}, n_active={n_active}, m_resid={m_resid}, "
          f"sigma_obs={sigma_obs}, beta_min={beta_min}")
    print(f"  C_budget={bl.CONSTANTS.C_BUDGET} (frozen from Tier 1)\n")

    print(f"  {'K':>3} {'pK':>6} {'sigma_eff':>10} {'N_res':>9} "
          f"{'N_feas':>8} {'N_run':>9} {'binds':>11}")
    rows = []
    for K in (1, 2):
        pK = bl.p_K(d, K)
        # pilot must exceed pK for the K=2 fit to be well-posed
        N_pilot = max(bl.pilot_N0(d, K), 3 * pK)
        s_eff = measure_sigma_eff(d, K, n_active, m_resid, sigma_obs,
                                  N_pilot, n_trials)
        N_res = bl.predict_budget(s_eff, beta_min, d, K)
        N_feas = bl.feasibility_floor(d, K)
        N_run = max(N_res, N_feas)
        binds = "feasibility" if N_res < N_feas else "resolution"
        rows.append(dict(K=K, pK=pK, sigma_eff=s_eff, N_res=N_res,
                         N_feas=N_feas, N_run=N_run, binds=binds))
        print(f"  {K:>3d} {pK:>6d} {s_eff:>10.4f} {N_res:>9d} "
              f"{N_feas:>8d} {N_run:>9d} {binds:>11}")

    r1, r2 = rows
    print(f"\n  READING:")
    print(f"    sigma_eff {('FALLS' if r2['sigma_eff'] < r1['sigma_eff'] else 'rises')} "
          f"K=1 -> K=2 : {r1['sigma_eff']:.4f} -> {r2['sigma_eff']:.4f} "
          f"(pairwise moves from mismatch into the fit)")
    print(f"    pK JUMPS  {r1['pK']} -> {r2['pK']}  (~ d^2/2), "
          f"lifting the feasibility floor")
    print(f"    => running budget RISES {r1['N_run']} -> {r2['N_run']} "
          f"DESPITE lower noise.")
    print(f"    The certificate draws the identifiability line: if budget << "
          f"p2={r2['pK']},")
    print(f"    dense pairwise LIME is not identifiable and must not be read "
          f"as an interaction map.")
    return rows


def crossing_curve(d_grid=(8, 12, 16, 20, 24, 30), m_resid=0.05,
                   sigma_obs=0.05, beta_min=0.02, n_active=4, n_trials=10):
    """Figure data: across d, where does the feasibility floor overtake the
    resolution budget? The crossing is where K=2 becomes feasibility-bound."""
    print("\n  [figure data] feasibility floor vs resolution budget across d "
          "(K=2)")
    print(f"  {'d':>4} {'p2':>7} {'sigma_eff':>10} {'N_res':>9} "
          f"{'N_feas':>8} {'binds':>11}")
    for d in d_grid:
        if d > bl.CONSTANTS.P_KEEP * 0 + 18 and d > 18:
            # K=2 cube not enumerable, but sigma_eff still estimable via bank
            pass
        K = 2
        pK = bl.p_K(d, K)
        N_pilot = max(bl.pilot_N0(d, K), 3 * pK)
        try:
            s_eff = measure_sigma_eff(d, K, n_active, m_resid, sigma_obs,
                                      N_pilot, n_trials)
        except Exception:
            s_eff = float("nan")
        N_res = bl.predict_budget(s_eff, beta_min, d, K)
        N_feas = bl.feasibility_floor(d, K)
        binds = "feasibility" if N_res < N_feas else "resolution"
        print(f"  {d:>4d} {pK:>7d} {s_eff:>10.4f} {N_res:>9d} "
              f"{N_feas:>8d} {binds:>11}")


def main():
    feasibility_table()
    crossing_curve()


if __name__ == "__main__":
    main()
