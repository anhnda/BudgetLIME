"""
draw_certified_figure.py
=========================
Render "Figure 1"-style panels: the Original image (grayscale) followed by one
grid panel per query budget N, showing the CERTIFIED cells of a K=1 LIME-style
dense-OLS surrogate.

Reuses the verified machinery from lime_image.py (no re-implementation):
    nested_mask_bank -> ols_fit -> floor_value (family-wise, Eq. 8)
so the certified sets are REAL (not schematic) and genuinely nested in N.

Per cell c at budget N:
    beta_c                          = OLS coefficient (column-standardized core)
    floor(N) = Cest*sigma_eff*sqrt(2 log p1 / N)
    certified  iff |beta_c| > floor(N)
        warm (orange)  if beta_c > 0
        cool (blue)    if beta_c < 0
        light gray     otherwise (UNRESOLVED at this budget, NOT zero)
Color alpha encodes |beta_c| (stronger coefficient -> more saturated), so larger
N panels show the floor dropping and faint cells lighting up -- the central
thesis that absence from the certified set is not evidence of absence.

sigma_eff is estimated once (pilot -> Eq. 11) and held fixed across the ladder,
exactly as in Claim A, so floor(N) is the only thing that moves.

USAGE (nothing heavy runs on import; torch only touched when run):
    python draw_certified_figure.py --image cat.jpg
    python draw_certified_figure.py --image cat.jpg --backbone resnet50 \
        --reference mean --grid 7 --N_ladder 256,512,2000,4000 \
        --out figure1.png
"""
from __future__ import annotations
import argparse
import math
import numpy as np

import lime_image as li


# --------------------------------------------------------------------------- #
def build_panels(clf, img, ref, slices, target, N_list, sigma_obs,
                 Cm, Cest, family_wise, seed=0):
    """Return (sigma_eff, m_hat, [(N, floor, beta) ...]) using ONE nested bank
    so certified sets nest across N (Claim A semantics)."""
    d = len(slices)
    p1 = li.p1(d)
    N_list = sorted(n for n in N_list if n > p1)
    if len(N_list) < 1:
        raise ValueError(f"all N <= p1={p1}; pick larger budgets or smaller grid")

    # pilot -> sigma_eff (Eq. 11), fixed across the ladder
    N0 = max(500, 6 * p1)
    m_hat = li.estimate_mismatch(
        clf, img, ref, slices, target, N0=N0, sigma_obs=sigma_obs,
        rng=np.random.default_rng(seed + 12345),
    )
    sigma_eff = sigma_obs + Cm * math.sqrt(m_hat)

    N_max = max(N_list)
    Zbank = li.nested_mask_bank(N_max, d, seed + 777)
    ybank = clf.query(img, ref, slices, Zbank, target)

    panels = []
    for N in N_list:
        beta, _, _ = li.ols_fit(Zbank[:N], ybank[:N])
        fl = li.floor_value(Cest, sigma_eff, d, N, family_wise)
        panels.append((N, fl, beta))
    return sigma_eff, m_hat, panels


def beta_to_grid(beta, grid):
    """d=grid*grid coefficients in row-major cell order -> (grid, grid)."""
    return np.asarray(beta, float).reshape(grid, grid)


def panel_rgba(beta_grid, floor, grid, warm, cool, unresolved,
               gamma=0.6, alpha_floor=0.25):
    """Color array (grid, grid, 4). Certified cells colored by sign; alpha
    scales with |beta| (relative to panel max). Unresolved -> flat light gray."""
    H = W = grid
    out = np.empty((H, W, 4), float)
    out[:] = unresolved  # light gray, full alpha
    certified = np.abs(beta_grid) > floor
    if certified.any():
        mx = np.abs(beta_grid[certified]).max()
        mx = mx if mx > 0 else 1.0
        for i in range(H):
            for j in range(W):
                if not certified[i, j]:
                    continue
                b = beta_grid[i, j]
                base = warm if b > 0 else cool
                strength = (abs(b) / mx) ** gamma
                a = alpha_floor + (1.0 - alpha_floor) * strength
                out[i, j, :3] = base[:3]
                out[i, j, 3] = a
    return out


def draw_grid(ax, rgba, grid, edge="#bfbfbf", lw=0.8):
    """Imshow an RGBA cell array with crisp cell borders, axes off."""
    ax.imshow(rgba, interpolation="nearest", aspect="equal",
              extent=[0, grid, grid, 0])  # row 0 on top
    ax.set_xticks(np.arange(grid + 1))
    ax.set_yticks(np.arange(grid + 1))
    ax.grid(True, color=edge, linewidth=lw)
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_edgecolor(edge)
        s.set_linewidth(lw)


def render_figure(img, grid, panels, sigma_eff, m_hat, out_path,
                  backbone, reference, dpi=200, overlay=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    warm = np.array([0.95, 0.55, 0.18, 1.0])   # orange
    cool = np.array([0.36, 0.36, 0.86, 1.0])   # blue
    unresolved = np.array([0.92, 0.92, 0.93, 1.0])

    n = len(panels) + 1  # +1 for Original
    fig, axes = plt.subplots(1, n, figsize=(2.05 * n, 2.4))
    if n == 1:
        axes = [axes]

    def _show_image(ax):
        ax.imshow(np.clip(img, 0, 1), interpolation="nearest", aspect="equal",
                  extent=[0, grid, grid, 0])

    def _grid_lines(ax):
        ax.set_xticks(np.arange(grid + 1))
        ax.set_yticks(np.arange(grid + 1))
        ax.grid(True, color="#bfbfbf", linewidth=0.8)
        ax.tick_params(length=0, labelbottom=False, labelleft=False)
        for s in ax.spines.values():
            s.set_edgecolor("#bfbfbf"); s.set_linewidth(0.8)

    # --- Original panel ---
    ax0 = axes[0]
    if overlay:
        _show_image(ax0)                          # real image
    else:
        gray = img @ np.array([0.299, 0.587, 0.114])
        ax0.imshow(gray, cmap="gray", interpolation="nearest", aspect="equal",
                   extent=[0, grid, grid, 0])      # grayscale (paper style)
    _grid_lines(ax0)
    ax0.set_title("Original", fontsize=11)

    # --- certified panels ---
    for k, (N, fl, beta) in enumerate(panels, start=1):
        ax = axes[k]
        bg = beta_to_grid(beta, grid)
        rgba = panel_rgba(bg, fl, grid, warm, cool, unresolved)
        if overlay:
            # real image UNDER a semi-transparent overlay; unresolved cells
            # become fully transparent so the image shows through.
            _show_image(ax)
            is_unres = np.all(np.isclose(rgba[..., :3], unresolved[:3]), axis=-1)
            rgba[is_unres, 3] = 0.0
        ax.imshow(rgba, interpolation="nearest", aspect="equal",
                  extent=[0, grid, grid, 0])
        _grid_lines(ax)
        label = "LIME, $N{=}%d$" % N if k == 1 else "$N{=}%d$" % N
        ax.set_title(label, fontsize=11)

    # --- legend ---
    unres_handle = (Patch(facecolor="white", edgecolor="#bfbfbf",
                          label="unresolved (image shown)") if overlay
                    else Patch(facecolor=unresolved[:3], edgecolor="#bfbfbf",
                               label="unresolved (not zero)"))
    handles = [
        Patch(facecolor=warm[:3], label="certified $+$"),
        Patch(facecolor=cool[:3], label="certified $-$"),
        unres_handle,
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    # caption = (f"Certified cells under increasing query budget "
    #            f"({backbone}, {reference} reference, {grid}$\\times${grid} grid). "
    #            f"$\\sigma_{{\\mathrm{{eff}}}}={sigma_eff:.3f}$, "
    #            f"$\\hat m={m_hat:.3f}$. As $N$ increases the floor drops and "
    #            f"more cells certify; gray = unresolved at the current budget, "
    #            f"not zero.")
    # fig.text(0.5, -0.13, caption, ha="center", va="top", fontsize=9, wrap=True)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.18,
                        wspace=0.12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--backbone", default="resnet50",
                    choices=list(li.DEFAULT_BACKBONES))
    ap.add_argument("--reference", default="mean",
                    help="OFF-cell fill: white|black|mean|gray|blurS")
    ap.add_argument("--grid", type=int, default=7, help="cells/side; d=grid^2")
    ap.add_argument("--N_ladder", default="256,512,2000,4000")
    ap.add_argument("--sigma_obs", type=float, default=-1.0,
                    help="-1 => deterministic (0); else fixed value")
    ap.add_argument("--Cm", type=float, default=li.DEFAULT_CM)
    ap.add_argument("--Cest", type=float, default=li.DEFAULT_CEST)
    ap.add_argument("--single_cell", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figure_certified.png")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--overlay", action="store_true",
                    help="overlay certified cells on the real image "
                         "(default: solid grid panels, paper style)")
    args = ap.parse_args()

    family_wise = not args.single_cell
    N_list = [int(x) for x in args.N_ladder.split(",") if x.strip()]

    clf = li.ImageClassifier(args.backbone)
    try:
        img = clf.load_image(args.image)
        slices = clf._cell_slices(img.shape[0], img.shape[1], args.grid)
        target = clf.target_class(img)
        ref = clf.make_reference(img, args.reference)
        sigma_obs = 0.0 if args.sigma_obs < 0 else args.sigma_obs

        sigma_eff, m_hat, panels = build_panels(
            clf, img, ref, slices, target, N_list, sigma_obs,
            args.Cm, args.Cest, family_wise, seed=args.seed)

        out = render_figure(img, args.grid, panels, sigma_eff, m_hat,
                            args.out, args.backbone, args.reference, args.dpi,
                            overlay=args.overlay)
        print(f"class={target} | sigma_eff={sigma_eff:.4f} | m_hat={m_hat:.4f}")
        for N, fl, beta in panels:
            ncert = int(np.sum(np.abs(beta) > fl))
            print(f"  N={N:>5}  floor={fl:.4f}  certified={ncert}/{len(beta)}")
        print(f"wrote {out}")
    finally:
        clf.close()


if __name__ == "__main__":
    main()