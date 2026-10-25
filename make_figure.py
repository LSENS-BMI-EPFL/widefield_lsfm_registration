"""Panels for the vessel-registration methods figure (iDISCO/LSFM -> widefield -> atlas).

Usage: python make_figure.py PB105 [transform.json] [tag]
Writes figures/<brain><tag>/<panel>.png (300 dpi) + .pdf, and a composite overview.png.
Each panel is saved without titles so it can be assembled/labelled in Illustrator/Inkscape.
"""
import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import ndimage as ndi
from skimage.filters import frangi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
import register as R  # noqa: E402
import validate as V  # noqa: E402
import enhance  # noqa: E402
import surface as S  # noqa: E402

MAGENTA = np.array([0.90, 0.20, 0.80])
GREEN = np.array([0.20, 0.75, 0.30])
CYAN = (0.0, 0.85, 0.95)
LW = 0.8          # border line width (pt); one image pixel is 0.72 pt in the saved panels
LOSS_RANGE = 40  # +/- px translation around the optimum for the loss landscape
LOSS_STEP = 1


def nrm(x, lo=1, hi=99, mask=None):
    v = x[mask] if mask is not None else x[np.isfinite(x)]
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    return np.clip((np.nan_to_num(x, nan=a) - a) / (b - a), 0, 1)


def tint(x, color):
    """Grayscale [0,1] -> colored on black."""
    return np.clip(x[..., None] * color[None, None], 0, 1)


def edges_of(L):
    return (ndi.maximum_filter(L, 2) != ndi.minimum_filter(L, 2)) & (L > 0)


def outline_lines(region, sigma=1.0):
    """Outline of a (smooth) region as sub-pixel polylines (x, y), to draw over a panel instead of its pixel rim."""
    from skimage.measure import find_contours
    padded = np.pad(region.astype(float), 2)                       # closed contours at the image border
    return [c[:, ::-1] - 2 for c in find_contours(ndi.gaussian_filter(padded, sigma), 0.5)]


def hemisphere_map(brain, roll, pitch):
    """Displayed-orientation map (like surface.project's label) of brainreg's hemisphere label (1 left, 2 right) of the
    surface: the shallowest non-zero value at the same depths as the atlas label."""
    import brain_axes
    proj, _ = R.moving_for_view(brain, roll, pitch)
    prep = R._G[("prep", brain)]
    hem = brain_axes.hemisphere_volume(brain)
    z = np.rot90(proj["z"], -1)                                           # back to the grid used inside project()
    shape = np.array(prep["norm"].shape)
    c = shape / 2.0
    H, W = int(shape[1] * 1.15), int(shape[2] * 1.15)
    oy, ox = (H - shape[1]) // 2, (W - shape[2]) // 2
    Rm = S.rotation(roll, pitch)
    yy, xx = np.nonzero(np.isfinite(z))
    scale = np.array(hem.shape) / shape
    out = np.zeros(len(yy), np.uint8)
    for d in S.LABEL_DEPTHS[::-1]:
        q = np.stack([z[yy, xx] + d, yy - oy, xx - ox], 1)
        p = np.round(((q - c) @ Rm + c) * scale).astype(int)
        inb = np.all((p >= 0) & (p < np.array(hem.shape)), 1)
        v = np.zeros(len(yy), np.uint8)
        v[inb] = hem[p[inb, 0], p[inb, 1], p[inb, 2]]
        out = np.where(v > 0, v, out)
    m = np.zeros(z.shape, np.uint8)
    m[yy, xx] = out
    return np.rot90(m)


def figure_region(brain, roll, pitch, A):
    """Outline region for the widefield figures: the imaged hemisphere's cortex surface (isocortex labels of the
    projection, cut at the midline with brainreg's hemisphere labels), mapped into the widefield image."""
    proj, _ = R.moving_for_view(brain, roll, pitch)
    hem = hemisphere_map(brain, roll, pitch)
    side = C.HEMISPHERE.get(brain)
    if side is None:                                                      # the hemisphere under the window
        win = enhance.window_mask(brain, display=True)
        hw = cv2.warpAffine(hem, np.asarray(A, np.float32), win.shape[::-1], flags=cv2.INTER_NEAREST)
        side = {1: "left", 2: "right"}[int(np.bincount(hw[win], minlength=3)[1:].argmax() + 1)]
    m = np.isin(proj["label"], enhance.ISOCORTEX) & (hem == {"left": 1, "right": 2}[side])
    m = R.warp(m.astype(np.float32), np.asarray(A)) > 0.5
    return enhance._largest(ndi.binary_fill_holes(ndi.binary_opening(m, iterations=2)))


def display_labels(La, region, min_px=30):
    """Area map for figures, inside region: thin unlabelled gaps and islands smaller than min_px take the label of the
    nearest area."""
    keep = La > 0
    for k in np.unique(La[region]):
        lab, n = ndi.label(La == k)
        sizes = np.bincount(lab.ravel())
        keep &= ~((sizes[lab] < min_px) & (lab > 0))
    _, (iy, ix) = ndi.distance_transform_edt(~keep, return_indices=True)
    return np.where(region, La[iy, ix], 0)


def view_lines(lines, box, rot=False):
    """Polylines (x, y) into a panel cropped to box (row, col slices) and, with rot, rotated like np.rot90."""
    r0, c0 = box[0].start, box[1].start
    w = box[1].stop - c0
    out = [P - [c0, r0] for P in lines]
    return [np.stack([P[:, 1], w - 1 - P[:, 0]], 1) for P in out] if rot else out


def draw_lines(ax, lines):
    """lines: [(polylines, color, linewidth in pt)]."""
    for polys, color, lw in lines or ():
        for P in polys:
            ax.plot(P[:, 0], P[:, 1], color=color, lw=lw, solid_joinstyle="round", solid_capstyle="round")


def crop_box(m, pad=6):
    ys, xs = np.nonzero(m)
    return slice(max(ys.min() - pad, 0), ys.max() + pad + 1), slice(max(xs.min() - pad, 0), xs.max() + pad + 1)


def area_map(L, size=7):
    """Collapse layer-level labels to cortical areas, then mode-filter to remove speckle at the boundaries."""
    ids, inv = np.unique(L, return_inverse=True)
    A = np.array([V.area_id(i) if i else 0 for i in ids])[inv.reshape(L.shape)]
    u, idx = np.unique(A, return_inverse=True)
    idx = idx.reshape(A.shape)
    mode = lambda w: np.bincount(w.astype(int)).argmax()
    return u[ndi.generic_filter(idx, mode, size=size).astype(int)]


def save(img, path, cmap=None, vmin=None, vmax=None, lines=None):
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    draw_lines(ax, lines)
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
    ax.axis("off")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300)
    plt.close(fig)


def main(brain, transform=None, tag=""):
    out = os.path.join(C.FIGURES, brain + tag)
    os.makedirs(out, exist_ok=True)
    t = json.load(open(transform or f"{C.RESULTS}/{brain}_transform.json"))
    roll, pitch, A = t["roll_deg"], t["pitch_deg"], np.array(t["A_projection_to_widefield"])

    proj, e = R.moving_for_view(brain, roll, pitch)
    fx = R.fixed_for(brain)
    M = e["ridge"] * e["mask"]
    panels = {}

    # a1: raw LSFM vessel channel, dorsal MIP (same rot90 orientation as the projection)
    vol = np.load(f"{C.CACHE}/{brain}_lsfm25.npy")
    mip = np.rot90(vol.max(0).astype(np.float32))
    bm = np.rot90(np.load(f"{C.CACHE}/{brain}_norm_f16.npy", mmap_mode="r").max(0) > 0)
    bm = ndi.binary_erosion(bm, iterations=3)
    panels["a1_lsfm_dorsal_mip"] = (np.where(bm, nrm(mip, 1, 99.5, bm), 0)[crop_box(bm)], "gray")
    # a2: cleaned surface layer (13-75 um below the pial surface, isocortex, artifacts filled), at the fitted
    # view. Vessels are dark in the iDISCO channel -> inverted so vessels are bright on black.
    valid = e["valid"] & (ndi.distance_transform_edt(e["valid"]) > 4)      # trim the dark mask rim
    box = crop_box(valid, pad=4)
    panels["a2_surface_vessels_depth_stacked"] = (np.where(valid, 1 - nrm(e["flat"], 0.5, 99.5, valid), 0)[box], "gray")
    # a3: segmented surface-vessel network (what enters the fit)
    a3 = tint(np.clip(0.35 + 0.65 * e["vesselness"], 0, 1) * e["seg"], MAGENTA)
    a3[valid & ~e["seg"]] = 0.12                                   # faint cortex outline for context
    panels["a3_lsfm_vessel_projection"] = (a3[box], None)

    # b1: raw widefield; b2: Frangi response; b3: vessel mask used for fitting
    raw = fx["raw"]
    panels["b1_widefield_raw"] = (nrm(np.log(raw), 0.5, 99.8), "gray")
    lg = np.log(np.maximum(raw, 1))
    fr = frangi(lg - ndi.gaussian_filter(lg, 10), sigmas=enhance.LARGE["sigmas"], black_ridges=True,
                alpha=0.5, beta=0.5, gamma=None)
    panels["b2_widefield_frangi"] = (nrm(fr, 1, 99.5), "afmhot")
    panels["b3_widefield_vessel_mask"] = (tint(nrm(R.blur(fx["ridge"], 0.6), 0, 99.5), GREEN), None)

    # c: overlay in widefield space (LSFM magenta, widefield green, overlap white)
    Mw = R.warp(M, A)
    wr = np.clip(R.blur(fx["ridge"], 0.8) * 1.6, 0, 1)
    mr = np.clip(R.blur(Mw, 0.8) * 1.6, 0, 1)
    panels["c_vessel_alignment"] = (np.clip(tint(mr, MAGENTA) + tint(wr, GREEN), 0, 1), None)

    # e: atlas boundaries (warped label map) on the widefield image, inside the window
    ev = V.evaluate(brain, roll, pitch, A)
    L = ev.pop("L")
    La = area_map(L)
    region = figure_region(brain, roll, pitch, A)                # outline: a2 cortex surface joined with the window
    La = display_labels(La, region)                               # no slivers or tiny islands
    ed = edges_of(La) & ndi.binary_erosion(region)               # area borders: pixel edges inside the outline
    outline = outline_lines(region)                               # outline: smooth line drawn over the panel
    np.save(f"{out}/atlas_area_labels_widefield.npy", La)
    g = nrm(np.log(raw), 0.5, 99.8)
    rgb = np.dstack([g] * 3)
    rgb[ed] = CYAN
    panels["e_atlas_on_widefield"] = (rgb, None, [(outline, CYAN, LW)])

    # f: stim - no stim (1.00-1.15 s), atlas edges in black
    d = V.hotspot(brain)
    v = np.percentile(np.abs(d), 99.5)
    cm = plt.get_cmap("RdBu_r")((np.clip(d / v, -1, 1) + 1) / 2)[..., :3]
    cm[ed] = 0
    panels["f_stim_minus_nostim_atlas"] = (cm, None, [(outline, "k", LW)])

    # g/h: reference-style views (rot90: medial right, lateral left), cropped to the outline region.
    # g: atlas on widefield vessels; h: stim map (inferno) with widefield vessels and atlas.
    lgr = np.log(np.maximum(raw, 1))
    flat = lgr - ndi.gaussian_filter(lgr, 10)                      # vessels dark, illumination removed
    box = crop_box(region, pad=3)
    gv = nrm(flat, 1, 99.5, region)
    g_img = np.dstack([gv] * 3)
    g_img[~region] = 1
    g_img[ed] = CYAN
    ref = view_lines(outline, box, rot=True)
    panels["g_vessels_atlas_reference_view"] = (np.rot90(g_img[box]), None, [(ref, "k", LW)])
    lo, hi = np.percentile(d[region], [2, 99.8])
    h_img = plt.get_cmap("inferno")(np.clip((d - lo) / (hi - lo), 0, 1))[..., :3]
    ves = nrm(-flat, 80, 99.5, region)[..., None] * 0.7
    h_img = h_img * (1 - ves) + 0.95 * ves
    h_img[~region] = 1
    h_img[ed] = 0.55
    panels["h_activity_vessels_atlas_reference_view"] = (np.rot90(h_img[box]), None, [(ref, "k", LW)])

    panels = {k: v if len(v) == 3 else (*v, None) for k, v in panels.items()}
    for name, (img, cmap, lines) in panels.items():
        save(img, f"{out}/{name}", cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None, lines=lines)
    # colorbar for f, in dF/F units of the trial averages
    fig, ax = plt.subplots(figsize=(0.5, 2))
    fig.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(-v, v), "RdBu_r"), cax=ax)
    ax.tick_params(labelsize=7)
    fig.savefig(f"{out}/f_colorbar.pdf", bbox_inches="tight"); fig.savefig(f"{out}/f_colorbar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # d: loss landscape = 1 - masked NCC over translation offsets around the optimum (same score as the fit)
    F, Fm, Mb = R.blur(fx["ridge"], 1.2), fx["mask"], R.blur(M, 1.2)
    offs = np.arange(-LOSS_RANGE, LOSS_RANGE + 1, LOSS_STEP)
    loss = np.zeros((len(offs), len(offs)))
    for i, dy in enumerate(offs):
        for j, dx in enumerate(offs):
            Ad = A.copy()
            Ad[:, 2] += (dx, dy)
            loss[i, j] = 1 - R.score(F, Fm, Mb, Ad)
    np.savez(f"{out}/d_loss_landscape.npz", dx=offs, dy=offs, loss=loss)
    iy, ix = np.unravel_index(np.argmin(loss), loss.shape)
    DX, DY = np.meshgrid(offs, offs)
    fig = plt.figure(figsize=(4.5, 3.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(DX, DY, loss, cmap="Blues_r", linewidth=0, antialiased=True, rcount=len(offs), ccount=len(offs))
    ax.set_xlabel("dx (px)", fontsize=8); ax.set_ylabel("dy (px)", fontsize=8); ax.set_zlabel("1 − NCC", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=28, azim=-60)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    fig.savefig(f"{out}/d_loss_landscape.pdf", bbox_inches="tight"); fig.savefig(f"{out}/d_loss_landscape.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # composite preview
    order = ["a1_lsfm_dorsal_mip", "a2_surface_vessels_depth_stacked", "a3_lsfm_vessel_projection", "c_vessel_alignment",
             "e_atlas_on_widefield", "b1_widefield_raw", "b2_widefield_frangi", "b3_widefield_vessel_mask",
             None, "f_stim_minus_nostim_atlas"]
    fig, axs = plt.subplots(2, 5, figsize=(17, 7))
    for ax, k in zip(axs.ravel(), order):
        if k is None:
            ax.remove()
            ax = fig.add_subplot(2, 5, 9, projection="3d")
            ax.plot_surface(DX, DY, loss, cmap="Blues_r", linewidth=0)
            ax.set_title(f"d  loss (1−NCC), min at dx {offs[ix]}, dy {offs[iy]}", fontsize=9)
            ax.tick_params(labelsize=6)
            continue
        img, cmap, lines = panels[k]
        ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        draw_lines(ax, [(P, c, lw * 0.6) for P, c, lw in lines or ()])
        ax.set_title(k.replace("_", " "), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{brain}: roll {roll}°, pitch {pitch}°, vessel NCC {R.score(F, Fm, Mb, A):.2f}, "
                 f"wS1 peak in {ev['peak_region']}", fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{out}/overview.png", dpi=120)
    stats = dict(loss_min_offset_px=[int(offs[ix]), int(offs[iy])], loss_min=float(loss.min()),
                 loss_at_zero=float(loss[LOSS_RANGE // LOSS_STEP, LOSS_RANGE // LOSS_STEP]),
                 f_colorbar_abs_max=float(v), **{k: ev[k] for k in ("peak_region", "halfmax_in_bfd", "top1pct_in_bfd")})
    json.dump(stats, open(f"{out}/stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "PB105", *sys.argv[2:4])
