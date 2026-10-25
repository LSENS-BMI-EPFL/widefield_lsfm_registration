"""Atlas boundaries at layer 5 instead of the surface (for L5 mice). Standalone: imports, does not modify, the pipeline.

For every visible pixel of the fitted surface projection, march along the viewing ray (the same ray the camera
integrates along) and take the label of the first layer-5 voxel. Area borders at L5 differ from the surface ones
where the cortex is tilted against the view (lateral S1): the ray crosses into neighbouring columns on its way down.
Pixels whose ray never meets L5 (fissure walls, brain edge) take the nearest L5 label.

Usage: python l5_boundaries.py PB105 [transform.json]
Writes figures/<brain>_L5/: e/f/g/h panels as in make_figure.py but with L5 borders, a surface-vs-L5 comparison,
atlas_area_labels_widefield_L5.npy, l5_depth_um.npy (depth of L5 below the surface along the ray) and stats.json.
"""
import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
import register as R  # noqa: E402
import surface as S  # noqa: E402
import validate as V  # noqa: E402
import make_figure as MF  # noqa: E402

MAX_DEPTH_VOX = 60          # search L5 down to ~1.5 mm along the ray
STEP_VOX = 0.5
L5 = {i for i, nm in V.NAMES.items() if 315 in V.PATH.get(i, []) and nm.endswith("5")}


def l5_projection(brain, roll, pitch):
    """Displayed-orientation maps (like surface.project): layer-5 label and its depth below the surface (voxels)."""
    R.moving_for_view(brain, roll, pitch)
    prep = R._G[("prep", brain)]
    proj = S.project(prep, roll, pitch)
    z = np.rot90(proj["z"], -1)                                    # back to the pre-rot90 grid used inside project()
    shape = np.array(prep["norm"].shape)
    c = shape / 2.0
    H, W = int(shape[1] * 1.15), int(shape[2] * 1.15)
    oy, ox = (H - shape[1]) // 2, (W - shape[2]) // 2
    Rm = S.rotation(roll, pitch)
    yy, xx = np.nonzero(np.isfinite(z))
    zs = z[yy, xx]
    anno, ids = prep["anno"], prep["ids"]
    scale = np.array(anno.shape) / shape
    is_l5 = np.isin(ids, list(L5))                                 # per anno index
    lab = np.zeros(len(zs), np.int64)
    dep = np.full(len(zs), np.nan, np.float32)
    todo = np.ones(len(zs), bool)
    for d in np.arange(0, MAX_DEPTH_VOX + STEP_VOX, STEP_VOX):
        k = np.nonzero(todo)[0]
        if not len(k):
            break
        q = np.stack([zs[k] + d, yy[k] - oy, xx[k] - ox], 1)
        p = np.round(((q - c) @ Rm + c) * scale).astype(int)
        inb = np.all((p >= 0) & (p < np.array(anno.shape)), 1)
        li = np.zeros(len(k), np.int64)
        li[inb] = anno[p[inb, 0], p[inb, 1], p[inb, 2]]
        hit = is_l5[li] & (li > 0)
        lab[k[hit]], dep[k[hit]] = ids[li[hit]], d
        todo[k[hit]] = False
    lmap = np.zeros(z.shape, np.int64)
    dmap = np.full(z.shape, np.nan, np.float32)
    lmap[yy, xx], dmap[yy, xx] = lab, dep
    return proj, np.rot90(lmap), np.rot90(dmap)


def to_widefield(L, A):
    u, inv = np.unique(L, return_inverse=True)
    return u[R.warp(inv.reshape(L.shape).astype(np.float32), np.array(A), interp=cv2.INTER_NEAREST).astype(int)]


def bfd_stats(L, d):
    bfd = np.isin(L, V.BFD)
    py, px = np.unravel_index(np.argmax(d), d.shape)
    return dict(peak_region=V.NAMES.get(V.area_id(int(L[py, px])), str(L[py, px])),
                top1pct_in_bfd=float(bfd[d >= np.percentile(d, 99)].mean()),
                halfmax_in_bfd=float(bfd[d >= 0.5 * d.max()].mean()))


def main(brain, transform=None):
    out = os.path.join(C.FIGURES, brain + "_L5")
    os.makedirs(out, exist_ok=True)
    t = json.load(open(transform or f"{C.RESULTS}/{brain}_transform.json"))
    roll, pitch, A = t["roll_deg"], t["pitch_deg"], np.array(t["A_projection_to_widefield"])

    proj, l5, l5_depth = l5_projection(brain, roll, pitch)
    fx = R.fixed_for(brain)
    raw = fx["raw"]

    # Surface labels and outline exactly as make_figure.py (outline: the a2 cortex surface).
    Ls = to_widefield(proj["label"], A)
    Las = MF.area_map(Ls)
    region = MF.figure_region(brain, roll, pitch, A)             # outline: a2 cortex surface joined with the window
    Las_d = MF.display_labels(Las, region)                       # figures: no slivers/islands
    rim = region & ~ndi.binary_erosion(region)
    ed_s = (MF.edges_of(Las) & region) | rim

    L5w = to_widefield(l5, A)
    Dw = R.warp(np.nan_to_num(l5_depth, nan=-1), A, interp=cv2.INTER_NEAREST)
    missing = region & (L5w == 0)
    _, (iy, ix) = ndi.distance_transform_edt(L5w == 0, return_indices=True)
    L5f = np.where(region, L5w[iy, ix], 0)
    La5 = MF.area_map(L5f)
    ed5 = (MF.edges_of(La5) & region) | rim
    np.save(f"{out}/atlas_area_labels_widefield_L5.npy", La5)
    np.save(f"{out}/l5_depth_um.npy", np.where(region & (Dw >= 0), Dw * V.VOX_UM, np.nan).astype(np.float32))

    # Figures: area borders as pixel edges inside the outline, the window outline as a smooth line (make_figure).
    # The stats below use ed_s / ed5.
    inside = ndi.binary_erosion(region)
    d5 = MF.edges_of(MF.display_labels(La5, region)) & inside
    ds = MF.edges_of(Las_d) & inside
    outline = MF.outline_lines(region)
    panels, lines = {}, {}
    g = MF.nrm(np.log(raw), 0.5, 99.8)
    rgb = np.dstack([g] * 3)
    rgb[d5] = MF.CYAN
    panels["e_atlas_L5_on_widefield"] = rgb
    lines["e_atlas_L5_on_widefield"] = [(outline, MF.CYAN, MF.LW)]
    cmp_ = np.dstack([g * 0.8] * 3)
    cmp_[ds] = [1.0, 0.75, 0.0]                                    # surface (amber)
    cmp_[d5] = MF.CYAN                                             # L5 (cyan)
    cmp_[ds & d5] = 1.0                                            # both (white)
    panels["e2_surface_vs_L5_borders"] = cmp_
    lines["e2_surface_vs_L5_borders"] = [(outline, "k", MF.LW)]

    d = V.hotspot(brain)
    v = np.percentile(np.abs(d), 99.5)
    f = plt.get_cmap("RdBu_r")((np.clip(d / v, -1, 1) + 1) / 2)[..., :3]
    f[d5] = 0
    panels["f_stim_minus_nostim_atlas_L5"] = f
    lines["f_stim_minus_nostim_atlas_L5"] = [(outline, "k", MF.LW)]

    lgr = np.log(np.maximum(raw, 1))
    flat = lgr - ndi.gaussian_filter(lgr, 10)
    box = MF.crop_box(region, pad=3)
    gv = MF.nrm(flat, 1, 99.5, region)
    g_img = np.dstack([gv] * 3)
    g_img[~region] = 1
    g_img[d5] = MF.CYAN
    ref = MF.view_lines(outline, box, rot=True)
    panels["g_vessels_atlas_L5_reference_view"] = np.rot90(g_img[box])
    lines["g_vessels_atlas_L5_reference_view"] = [(ref, "k", MF.LW)]
    lo, hi = np.percentile(d[region], [2, 99.8])
    h_img = plt.get_cmap("inferno")(np.clip((d - lo) / (hi - lo), 0, 1))[..., :3]
    ves = MF.nrm(-flat, 80, 99.5, region)[..., None] * 0.7
    h_img = h_img * (1 - ves) + 0.95 * ves
    h_img[~region] = 1
    h_img[d5] = 0.55
    panels["h_activity_vessels_atlas_L5_reference_view"] = np.rot90(h_img[box])
    lines["h_activity_vessels_atlas_L5_reference_view"] = [(ref, "k", MF.LW)]
    for name, img in panels.items():
        MF.save(img, f"{out}/{name}", lines=lines[name])

    # Border displacement: distance from each L5 border pixel to the nearest surface border (interior only).
    inner_s, inner5 = ed_s & ~rim, ed5 & ~rim
    shift_px = ndi.distance_transform_edt(~inner_s)[inner5]
    depth = Dw[region & (Dw >= 0)] * V.VOX_UM
    stats = dict(roll=roll, pitch=pitch,
                 l5_missing_frac_in_window=float(missing.sum() / region.sum()),
                 l5_depth_um_median=float(np.median(depth)), l5_depth_um_p5_p95=np.percentile(depth, [5, 95]).tolist(),
                 # widefield px size from the fitted affine (projection px = VOX_UM)
                 border_shift_um_median=float(np.median(shift_px) * V.VOX_UM / np.sqrt(abs(np.linalg.det(A[:, :2])))),
                 border_shift_px_median=float(np.median(shift_px)), border_shift_px_p90=float(np.percentile(shift_px, 90)),
                 surface=bfd_stats(Ls, d), L5=bfd_stats(L5f, d))
    json.dump(stats, open(f"{out}/stats.json", "w"), indent=1)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, k in zip(axs, ["e2_surface_vs_L5_borders", "e_atlas_L5_on_widefield", "f_stim_minus_nostim_atlas_L5",
                           "h_activity_vessels_atlas_L5_reference_view"]):
        ax.imshow(panels[k]); MF.draw_lines(ax, [(P, c, lw * 0.6) for P, c, lw in lines[k]])
        ax.set_title(k.replace("_", " "), fontsize=9); ax.axis("off")
    axs[0].set_title("borders: surface (amber), L5 (cyan), both (white)", fontsize=9)
    fig.suptitle(f"{brain}: L5 borders | wS1 peak in {stats['L5']['peak_region']} "
                 f"(surface: {stats['surface']['peak_region']}), half-max in bfd {stats['L5']['halfmax_in_bfd']:.2f} "
                 f"(surface {stats['surface']['halfmax_in_bfd']:.2f})", fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{out}/overview.png", dpi=120)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "PB105", *sys.argv[2:3])
