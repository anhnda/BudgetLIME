"""
lime_image.py
=============
Dense full-feature least-squares LIME-style surrogate on an image classifier
(K=1 additive: d grid cells, p1 = d+1), instrumented to verify the operational
claims of Section 5.2 of "Finite-Budget Certification for LIME-Style Local
Surrogates" ACROSS A (backbone x reference) GRID:

  (A) Monotone recovery in N. NESTED masks: one fixed bank, larger N adds
      queries to the same design, so certified sets nest. As N grows the floor
      drops and the certified set (|beta_hat| > floor) grows; signs stay fixed.
  (B) Forward budget prediction. Pilot -> sigma_eff (Eq. 11) -> predict N
      (Eq. 9) -> run -> realized floor ~ beta_min.

WHY A GRID, NOT AN ABLATION.
The reference (OFF-cell fill) is NOT a hyperparameter the certificate
optimizes; it is part of the estimand. g_rho is the masked response under
reference rho, so two fills define two different coefficient vectors, hence two
different floors. Likewise the backbone IS the black box being explained. Both
enter the floor through one scalar, the mismatch energy m_hat>K,rho, via
    sigma_eff = sigma_obs + Cm * sqrt(m_hat)   (Eq. 11)
    floor(N)  = Cest * sigma_eff * sqrt(2 log p1 / N)   (Eq. 8, family-wise).
The experiment does NOT ask "which fill is best". It shows the floor LAW (A and
B) holds in every cell of the grid, while m_hat and sigma_eff vary across cells
-- the evidence that the reference and backbone are genuine estimand inputs.
The numpy OLS core is byte-identical in every cell; only the black box and the
OFF-cell fill change.

References for images are constant OFF-cell fills:
    white : ones
    black : zeros
    mean  : per-image mean color (the standard LIME default)
All keep the +-1 Walsh design intact (the fill changes y, not Z), so Cest ~ 1.

Deterministic classifier => sigma_obs = 0 (paper Sec 4.3): same image+mask
gives the same logit, so the floor is driven entirely by mismatch energy.

Backbones (torchvision, ImageNet logits):
    resnet50, resnet18, vit_b_16

USAGE (nothing heavy runs on import):
    # full grid over default backbones x references:
    python lime_image.py --image path.jpg --grid 7
    # restrict:
    python lime_image.py --image path.jpg --backbones resnet50,vit_b_16 \
        --references mean,black --claim A --grid 7
    python lime_image.py --image path.jpg --claim B --grid 7 --beta_min 0.5
"""
from __future__ import annotations
import argparse
import math
import numpy as np

# --------------------------------------------------------------------------- #
#  Calibration constants. Cm is fixed by the synthetic leakage-scaling
#  experiment. Cest=1 for the orthonormal +-1 design; grid masks are +-1
#  centered, so Cest ~ 1 once N >~ p1.
# --------------------------------------------------------------------------- #
DEFAULT_CM = 1.0
DEFAULT_CEST = 1.0
Z_ALPHA = 1.96            # single-cell two-sided 95% (paper's z_{1-alpha})

IMAGE_REFERENCES = ("white", "black", "mean")
DEFAULT_BACKBONES = ("resnet50", "resnet18", "vit_b_16")


# =========================================================================== #
#  Model wrapper (the ONLY torch-dependent part)
# =========================================================================== #
class ImageClassifier:
    """Query-only black box: image + binary cell-mask -> class logit.

    Masking fills OFF cells with a reference. Evaluated in batches.
    sigma_obs is 0 for a deterministic forward pass.
    """

    def __init__(self, backbone="resnet50", device=None):
        import torch
        import torchvision.models as tvm
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ctor = {
            "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
            "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
            "vit_b_16": (tvm.vit_b_16, tvm.ViT_B_16_Weights.IMAGENET1K_V1),
        }[backbone]
        weights = ctor[1]
        self.model = ctor[0](weights=weights).eval().to(self.device)
        self.preprocess = weights.transforms()
        self.backbone = backbone

    def close(self):
        """Free the model so the next backbone has room (one model in RAM)."""
        try:
            del self.model
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass

    def load_image(self, path, size=224):
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((size, size))
        return np.asarray(img).astype(np.float32) / 255.0   # HxWx3 in [0,1]

    def make_reference(self, img, kind="mean"):
        """Constant OFF-cell fill. Supported: white | black | mean."""
        if kind == "mean":
            return np.ones_like(img) * img.reshape(-1, 3).mean(0)
        if kind == "black":
            return np.zeros_like(img)
        if kind == "white":
            return np.ones_like(img)
        if kind == "gray":
            return np.ones_like(img) * 0.5
        if kind.startswith("blur"):
            from scipy.ndimage import gaussian_filter
            s = float(kind[4:]) if len(kind) > 4 else 8.0
            return np.stack([gaussian_filter(img[..., c], s)
                             for c in range(3)], axis=-1)
        raise ValueError(f"unknown reference {kind} (white|black|mean)")

    def _cell_slices(self, H, W, grid):
        ys = np.linspace(0, H, grid + 1).astype(int)
        xs = np.linspace(0, W, grid + 1).astype(int)
        slices = []
        for i in range(grid):
            for j in range(grid):
                slices.append((slice(ys[i], ys[i + 1]),
                               slice(xs[j], xs[j + 1])))
        return slices   # length d = grid*grid

    def target_class(self, img):
        import torch
        with torch.no_grad():
            x = self._to_tensor(img[None])
            logit = self.model(x)[0]
            return int(logit.argmax().item())

    def _to_tensor(self, imgs):
        import torch
        from PIL import Image
        ts = []
        for im in imgs:
            pil = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
            ts.append(self.preprocess(pil))
        return torch.stack(ts).to(self.device)

    def query(self, img, ref, slices, Z, target, batch=64):
        """Z: N x d binary masks. Returns y: N logits for `target`."""
        import torch
        H, W, _ = img.shape
        N, d = Z.shape
        cell_map = np.full((H, W), -1, dtype=int)
        for c, (sy, sx) in enumerate(slices):
            cell_map[sy, sx] = c
        y = np.empty(N, dtype=np.float64)
        for b0 in range(0, N, batch):
            b1 = min(b0 + batch, N)
            imgs = np.empty((b1 - b0, H, W, 3), dtype=np.float32)
            for k, t in enumerate(range(b0, b1)):
                on = Z[t][cell_map]
                imgs[k] = np.where(on[..., None], img, ref)
            with torch.no_grad():
                logits = self.model(self._to_tensor(imgs))
                y[b0:b1] = logits[:, target].double().cpu().numpy()
        return y


# =========================================================================== #
#  Surrogate math  (pure numpy; identical OLS core across every grid cell)
# =========================================================================== #
def centered_design(Z):
    return 2.0 * (Z - 0.5)


def standardize_columns(X):
    scale = np.sqrt((X ** 2).mean(axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    return X / scale, scale


def ols_fit(Z, y):
    """Column-standardized dense OLS with intercept. Raises if N <= p1."""
    X = centered_design(Z)
    N, d = X.shape
    if N <= d + 1:
        raise np.linalg.LinAlgError(f"N={N} <= p1={d+1}: dense fit not well-posed")
    Xs, scale = standardize_columns(X)
    y_mean = y.mean()
    yc = y - y_mean
    G = (Xs.T @ Xs) / N
    if np.linalg.cond(G) > 1e8:
        raise np.linalg.LinAlgError("Gram ill-conditioned (N too small)")
    Ginv = np.linalg.inv(G)
    beta_std = Ginv @ (Xs.T @ yc) / N
    beta = beta_std / scale
    return beta, y_mean, np.diag(Ginv)


def p1(d):
    return d + 1


def sample_masks(N, d, rng):
    return (rng.random((N, d)) > 0.5).astype(float)


def nested_mask_bank(N_max, d, seed):
    """One fixed bank; a budget-N fit uses the first N rows, so certified sets
    across N are genuinely NESTED (larger N only ADDS queries)."""
    rng = np.random.default_rng(seed)
    return (rng.random((N_max, d)) > 0.5).astype(float)


def estimate_sigma_obs(clf, img, ref, slices, target, n_repeat=8, n_masks=16,
                       rng=None):
    rng = rng or np.random.default_rng(0)
    d = len(slices)
    Z = sample_masks(n_masks, d, rng)
    cols = [clf.query(img, ref, slices, Z, target) for _ in range(n_repeat)]
    return float(np.stack(cols, 0).std(axis=0).mean())


def estimate_mismatch(clf, img, ref, slices, target, N0, sigma_obs, rng,
                      cross_fit=True):
    """Eq. (11): m_hat = held-out residual variance minus sigma_obs^2.
    Upper-biased proxy (conservative). cross_fit removes in-sample inflation."""
    d = len(slices)
    Z = sample_masks(max(N0, 3 * p1(d)), d, rng)
    y = clf.query(img, ref, slices, Z, target)
    if cross_fit:
        n = Z.shape[0]
        half = n // 2
        resid = np.empty(n)
        for tr, te in [(slice(0, half), slice(half, n)),
                       (slice(half, n), slice(0, half))]:
            beta, b0, _ = ols_fit(Z[tr], y[tr])
            yhat = b0 + centered_design(Z[te]) @ beta
            resid[te] = y[te] - yhat
        mse = float((resid ** 2).mean())
    else:
        beta, b0, _ = ols_fit(Z, y)
        yhat = b0 + centered_design(Z) @ beta
        mse = float(((y - yhat) ** 2).mean())
    return max(mse - sigma_obs ** 2, 0.0)


def floor_value(Cest, sigma_eff, d, N, family_wise=True):
    if family_wise:
        return Cest * sigma_eff * math.sqrt(2.0 * math.log(p1(d)) / N)
    return Cest * sigma_eff * Z_ALPHA / math.sqrt(N)


def fit_surrogate(clf, img, ref, slices, target, N, rng):
    d = len(slices)
    Z = sample_masks(N, d, rng)
    y = clf.query(img, ref, slices, Z, target)
    return ols_fit(Z, y)


# =========================================================================== #
#  Claim A : monotone recovery in N   -> returns a result dict (no printing)
# =========================================================================== #
def claim_A(clf, img, ref, slices, target, N_list, sigma_obs, Cm, Cest,
            family_wise, seed=0):
    d = len(slices)
    N_list = [n for n in N_list if n > p1(d)]
    if len(N_list) < 2:
        return None
    N0 = max(500, 6 * p1(d))
    m_hat = estimate_mismatch(clf, img, ref, slices, target, N0=N0,
                              sigma_obs=sigma_obs,
                              rng=np.random.default_rng(seed + 12345))
    sigma_eff = sigma_obs + Cm * math.sqrt(m_hat)

    N_max = max(N_list)
    Zbank = nested_mask_bank(N_max, d, seed + 777)
    ybank = clf.query(img, ref, slices, Zbank, target)

    prev_set, prev_beta, prev_count = None, None, -1
    count_monotone, set_monotone, sign_flips = True, True, 0
    floor_first = floor_last = cert_first = cert_last = None
    for N in N_list:
        beta, _, _ = ols_fit(Zbank[:N], ybank[:N])
        fl = floor_value(Cest, sigma_eff, d, N, family_wise)
        cur_set = set(np.where(np.abs(beta) > fl)[0].tolist())
        if floor_first is None:
            floor_first, cert_first = fl, len(cur_set)
        floor_last, cert_last = fl, len(cur_set)
        if len(cur_set) < prev_count:
            count_monotone = False
        if prev_set is not None and len(prev_set - cur_set) > 1:
            set_monotone = False
        if prev_beta is not None:
            inter = list(prev_set & cur_set)
            sign_flips += sum(np.sign(beta[i]) != np.sign(prev_beta[i])
                              for i in inter) if inter else 0
        prev_set, prev_beta, prev_count = cur_set, beta, len(cur_set)

    return dict(d=d, p1=p1(d), m_hat=m_hat, sigma_eff=sigma_eff,
                count_monotone=count_monotone, set_nested=set_monotone,
                sign_flips=int(sign_flips),
                floor_first=floor_first, floor_last=floor_last,
                cert_first=cert_first, cert_last=cert_last)


# =========================================================================== #
#  Claim B : forward budget prediction   -> returns a result dict
# =========================================================================== #
def claim_B(clf, img, ref, slices, target, beta_min, sigma_obs, Cm, Cest,
            family_wise, N0=None, seed=0):
    d = len(slices)
    N0 = N0 or max(500, 6 * p1(d))
    rng = np.random.default_rng(seed)
    m_hat = estimate_mismatch(clf, img, ref, slices, target, N0, sigma_obs, rng)
    sigma_eff = sigma_obs + Cm * math.sqrt(m_hat)
    norm = 2.0 * math.log(p1(d)) if family_wise else Z_ALPHA ** 2
    N_pred = int(math.ceil(Cest ** 2 * sigma_eff ** 2 * norm / beta_min ** 2))
    feas_floor = 3 * p1(d)
    feasible = N_pred >= feas_floor
    N_run = max(N_pred, feas_floor)
    beta, _, _ = fit_surrogate(clf, img, ref, slices, target, N_run,
                               np.random.default_rng(seed + 1))
    fl = floor_value(Cest, sigma_eff, d, N_run, family_wise)
    ratio = fl / beta_min
    return dict(d=d, p1=p1(d), m_hat=m_hat, sigma_eff=sigma_eff,
                N_pred=N_pred, N_run=N_run, realized_floor=fl, ratio=ratio,
                feasible=feasible, lands=bool(0.7 <= ratio <= 1.4),
                certified=int(np.sum(np.abs(beta) > fl)))


# =========================================================================== #
#  Grid driver:  backbone (outer, one model in RAM) x reference (inner, cheap)
# =========================================================================== #
def run_grid(backbones, references, image_path, grid, beta_min, sigma_obs_arg,
             Cm, Cest, family_wise, claim, N_list_A, seed=0):
    rows_A, rows_B = [], []
    for bk in backbones:
        try:
            clf = ImageClassifier(bk)
        except Exception as e:
            print(f"[skip backbone {bk}] could not load: {e}")
            continue
        try:
            img = clf.load_image(image_path)
            slices = clf._cell_slices(img.shape[0], img.shape[1], grid)
            d = len(slices)
            target = clf.target_class(img)
            print(f"\n--- backbone={bk} | d={d} cells | p1={p1(d)} | "
                  f"class={target}")
            for ref_kind in references:
                try:
                    ref = clf.make_reference(img, ref_kind)
                except Exception as e:
                    print(f"    [skip ref {ref_kind}] {e}")
                    continue
                if sigma_obs_arg < 0:
                    sigma_obs = 0.0   # deterministic forward pass
                else:
                    sigma_obs = sigma_obs_arg
                if claim in ("A", "all"):
                    rA = claim_A(clf, img, ref, slices, target, N_list_A,
                                 sigma_obs, Cm, Cest, family_wise)
                    if rA:
                        rA.update(backbone=bk, reference=ref_kind)
                        rows_A.append(rA)
                if claim in ("B", "all"):
                    rB = claim_B(clf, img, ref, slices, target, beta_min,
                                 sigma_obs, Cm, Cest, family_wise)
                    if rB:
                        rB.update(backbone=bk, reference=ref_kind)
                        rows_B.append(rB)
        finally:
            clf.close()
    return rows_A, rows_B


def print_grid_A(rows, family_wise):
    if not rows:
        print("\n[Claim A] no identifiable (backbone, reference) cells.")
        return
    print(f"\n[Claim A] monotone recovery in N  (K=1, "
          f"floor={'family-wise' if family_wise else 'single-cell'})")
    print("  the LAW (right columns) should hold in every cell; m_hat and "
          "sigma_eff (left) vary -- reference & backbone are estimand inputs.")
    h = (f"  {'backbone':>9} {'ref':>6} {'d':>3} {'m_hat':>8} {'sig_eff':>8} "
         f"{'cnt-mono':>9} {'nested':>7} {'flips':>6} {'#cert lo->hi':>13}")
    print(h)
    print("  " + "-" * (len(h) - 2))
    for r in rows:
        print(f"  {r['backbone']:>9} {r['reference']:>6} {r['d']:>3d} "
              f"{r['m_hat']:>8.4f} {r['sigma_eff']:>8.4f} "
              f"{('PASS' if r['count_monotone'] else 'CHECK'):>9} "
              f"{('PASS' if r['set_nested'] else 'CHECK'):>7} "
              f"{r['sign_flips']:>6d} "
              f"{str(r['cert_first'])+'->'+str(r['cert_last']):>13}")
    n = len(rows)
    print("  " + "-" * (len(h) - 2))
    print(f"  cells: {n} | count-monotone {sum(r['count_monotone'] for r in rows)}/{n}"
          f" | set-nested {sum(r['set_nested'] for r in rows)}/{n} | "
          f"total sign flips {sum(r['sign_flips'] for r in rows)}")


def print_grid_B(rows, beta_min, family_wise):
    if not rows:
        print("\n[Claim B] no identifiable (backbone, reference) cells.")
        return
    print(f"\n[Claim B] forward budget prediction  (K=1, beta_min={beta_min}, "
          f"floor={'family-wise' if family_wise else 'single-cell'})")
    h = (f"  {'backbone':>9} {'ref':>6} {'d':>3} {'m_hat':>8} {'sig_eff':>8} "
         f"{'N_pred':>7} {'N_run':>7} {'realized':>9} {'ratio':>6} {'lands':>6}")
    print(h)
    print("  " + "-" * (len(h) - 2))
    for r in rows:
        tag = "PASS" if r['lands'] else ("INFEAS" if not r['feasible']
                                         else "CHECK")
        print(f"  {r['backbone']:>9} {r['reference']:>6} {r['d']:>3d} "
              f"{r['m_hat']:>8.4f} {r['sigma_eff']:>8.4f} "
              f"{r['N_pred']:>7d} {r['N_run']:>7d} "
              f"{r['realized_floor']:>9.4f} {r['ratio']:>6.3f} {tag:>6}")
    n = len(rows)
    feas = [r for r in rows if r['feasible']]
    print("  " + "-" * (len(h) - 2))
    print(f"  cells: {n} | feasible {len(feas)}/{n} | budget rule lands "
          f"{sum(r['lands'] for r in feas)}/{len(feas) if feas else 0} "
          f"(realized floor within [0.7,1.4]x beta_min)")


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=False, help="image to explain")
    ap.add_argument("--claim", choices=["A", "B", "all"], default="all")
    ap.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES),
                    help="comma list: resnet50,resnet18,vit_b_16")
    ap.add_argument("--references", default=",".join(IMAGE_REFERENCES),
                    help="comma list of OFF-cell fills: white,black,mean")
    ap.add_argument("--grid", type=int, default=7, help="cells per side; d=grid^2")
    ap.add_argument("--beta_min", type=float, default=0.5, help="target for B")
    ap.add_argument("--sigma_obs", type=float, default=-1.0,
                    help="-1 => deterministic (0); else fixed value")
    ap.add_argument("--Cm", type=float, default=DEFAULT_CM)
    ap.add_argument("--Cest", type=float, default=DEFAULT_CEST)
    ap.add_argument("--single_cell", action="store_true",
                    help="use single-cell z_alpha floor instead of family-wise")
    args = ap.parse_args()

    if not args.image:
        print("provide --image")
        return

    family_wise = not args.single_cell
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    references = [r.strip() for r in args.references.split(",") if r.strip()]
    N_list_A = [256, 512, 1000, 2000, 4000]

    print("=" * 78)
    print(f"Image LIME (backbone x reference) grid | K=1 | grid={args.grid} "
          f"(d={args.grid**2}) | "
          f"floor={'family-wise' if family_wise else 'single-cell'}")
    print(f"backbones : {backbones}")
    print(f"references: {references}  (OFF-cell fills; part of the estimand, "
          f"not ranked)")
    print("=" * 78)

    rows_A, rows_B = run_grid(
        backbones, references, args.image, args.grid, args.beta_min,
        args.sigma_obs, args.Cm, args.Cest, family_wise, args.claim, N_list_A)

    if args.claim in ("A", "all"):
        print_grid_A(rows_A, family_wise)
    if args.claim in ("B", "all"):
        print_grid_B(rows_B, family_wise)


if __name__ == "__main__":
    main()