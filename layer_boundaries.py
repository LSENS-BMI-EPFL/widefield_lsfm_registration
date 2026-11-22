"""Atlas borders at a chosen cortical layer instead of the surface (config.LAYER: surface, l2/3, l4, l5, l6).

For every visible pixel of the fitted surface projection, follow the viewing ray (the ray the camera integrates along)
and take the area of the first voxel at that layer (layers.layer_projection). Area borders at depth differ from the
surface ones where the cortex is tilted against the view (lateral S1): the ray crosses into neighbouring columns on
its way down. Pixels whose ray never meets the layer (fissure walls, brain edge) take the nearest label.
The panels have the same crop, outline and size as the surface panels of make_figure.py.

Usage: python layer_boundaries.py PB105 [--layer l5] [--transform results/PB105_transform.json]
Writes figures/<brain>_<L>/ (L = surface, L2-3, L4, L5, L6): e/f/g/h panels as in make_figure.py with the layer's
borders, a surface-vs-layer comparison (e2), atlas_area_labels_widefield_<L>.npy, <L>_depth_um.npy (depth of the
layer below the surface along the ray), stats.json; and results/<brain>_atlas_labels_widefield_<L>.npy (structure id
at the layer for every widefield pixel, 0 where the layer was not found).
"""
import os
import sys
import json
import argparse
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
import enhance  # noqa: E402
import layers  # noqa: E402

AMBER = (1.0, 0.75, 0.0)


def to_widefield(L, A):
    u, inv = np.unique(L, return_inverse=True)
    return u[R.warp(inv.reshape(L.shape).astype(np.float32), np.array(A), interp=cv2.INTER_NEAREST).astype(int)]


def bfd_stats(L, d, win):
    """Response in the window only (as validate.evaluate)."""
    bfd = np.isin(L, V.BFD)
    py, px = np.unravel_index(np.argmax(np.where(win, d, -np.inf)), d.shape)
    return dict(peak_region=V.NAMES.get(V.area_id(int(L[py, px])), str(L[py, px])),
                top1pct_in_bfd=float(bfd[win & (d >= np.percentile(d[win], 99))].mean()),
                halfmax_in_bfd=float(bfd[win & (d >= 0.5 * d[win].max())].mean()))


def main(brain, layer, transform=None):
    layer = layers.check(layer)
    tag = layers.TAG[layer]
    out = os.path.join(C.FIGURES, f"{brain}_{tag}")
    os.makedirs(out, exist_ok=True)
    t = json.load(open(transform or f"{C.RESULTS}/{brain}_transform.json"))
    roll, pitch, A = t["roll_deg"], t["pitch_deg"], np.array(t["A_projection_to_widefield"])

    proj, lab, depth, exact = layers.layer_projection(brain, roll, pitch, layer)

    # Surface labels and outline exactly as make_figure.py (outline: the a2 cortex surface).
    Ls = to_widefield(proj["label"], A)
    Las = MF.area_map(Ls)
    region = MF.figure_region(brain, roll, pitch, A)
    Las_d = MF.display_labels(Las, region)                       # figures: no slivers/islands
    rim = region & ~ndi.binary_erosion(region)
    ed_s = (MF.edges_of(Las) & region) | rim

    Lw = to_widefield(lab, A)
    np.save(f"{C.RESULTS}/{brain}_atlas_labels_widefield_{tag}.npy", Lw.astype(np.int64))
    Dw = R.warp(np.nan_to_num(depth, nan=-1), A, interp=cv2.INTER_NEAREST)
    Ew = R.warp(exact.astype(np.float32), A, interp=cv2.INTER_NEAREST) > 0.5
    missing = region & (Lw == 0)
    _, (iy, ix) = ndi.distance_transform_edt(Lw == 0, return_indices=True)
    Lf = np.where(region, Lw[iy, ix], 0)
    La = MF.area_map(Lf)
    ed = (MF.edges_of(La) & region) | rim
    La_d = MF.display_labels(La, region)
    np.save(f"{out}/atlas_area_labels_widefield_{tag}.npy", La_d)
    np.save(f"{out}/{tag}_depth_um.npy", np.where(region & (Dw >= 0), Dw * S.VOX_UM, np.nan).astype(np.float32))

    panels = MF.widefield_panels(brain, La_d, region, suffix=f"_{tag}")
    if layer != "surface":
        g = MF.nrm(np.log(R.fixed_for(brain)["raw"]), 0.5, 99.8)
        panels[f"e2_surface_vs_{tag}_borders"] = (
            np.dstack([g * 0.8] * 3), None,
            [(MF.border_lines(Las_d, region), AMBER, MF.LW), (MF.border_lines(La_d, region), MF.CYAN, MF.LW),
             (MF.outline_lines(region), "k", MF.LW)])
    for name, (img, _, lines) in panels.items():
        MF.save(img, f"{out}/{name}", lines=lines)
    MF.activity_colorbar(f"{out}/f_colorbar")

    # Border displacement: distance from each layer border pixel to the nearest surface border (interior only).
    d = V.hotspot(brain)
    win = enhance.window_mask(brain, d.shape, display=True)
    inner_s, inner = ed_s & ~rim, ed & ~rim
    shift_px = ndi.distance_transform_edt(~inner_s)[inner] if inner.any() else np.zeros(1)
    dep_um = Dw[region & (Dw >= 0)] * S.VOX_UM
    found = region & (Lw > 0)
    stats = dict(layer=layer, roll=roll, pitch=pitch,
                 missing_frac_in_outline=float(missing.sum() / region.sum()),
                 # pixels whose area has no such layer (e.g. no L4 in motor cortex): area taken from the next deeper layer
                 from_deeper_layer_frac=float((found & ~Ew).sum() / max(found.sum(), 1)),
                 depth_um_median=float(np.median(dep_um)), depth_um_p5_p95=np.percentile(dep_um, [5, 95]).tolist(),
                 # widefield px size from the fitted affine (projection px = VOX_UM)
                 border_shift_um_median=float(np.median(shift_px) * S.VOX_UM / np.sqrt(abs(np.linalg.det(A[:, :2])))),
                 border_shift_px_median=float(np.median(shift_px)), border_shift_px_p90=float(np.percentile(shift_px, 90)),
                 activity_clim_dff=list(V.activity_clim()),
                 surface=bfd_stats(Ls, d, win), layer_borders=bfd_stats(Lf, d, win))
    json.dump(stats, open(f"{out}/stats.json", "w"), indent=1)

    keys = ([f"e2_surface_vs_{tag}_borders"] if layer != "surface" else []) + \
        [f"e_atlas_{tag}_on_widefield", f"f_stim_response_atlas_{tag}", f"h_activity_vessels_atlas_{tag}_reference_view"]
    fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5))
    for ax, k in zip(np.atleast_1d(axs), keys):
        img, _, lines = panels[k]
        ax.imshow(img); MF.draw_lines(ax, [(P, c, lw * 0.6) for P, c, lw in lines])
        ax.set_title(k.replace("_", " "), fontsize=9); ax.axis("off")
    if layer != "surface":
        np.atleast_1d(axs)[0].set_title(f"borders: surface (amber), {tag} (cyan)", fontsize=9)
    fig.suptitle(f"{brain}: {tag} borders | wS1 peak in {stats['layer_borders']['peak_region']} "
                 f"(surface: {stats['surface']['peak_region']}), half-max in bfd "
                 f"{stats['layer_borders']['halfmax_in_bfd']:.2f} (surface {stats['surface']['halfmax_in_bfd']:.2f})",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{out}/overview.png", dpi=120)
    plt.close(fig)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("brain")
    ap.add_argument("--layer", default=C.LAYER, help=f"one of {layers.LAYERS} (default: config.LAYER)")
    ap.add_argument("--transform", default=None)
    a = ap.parse_args()
    main(a.brain, a.layer, a.transform)
