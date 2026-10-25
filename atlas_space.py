"""Widefield -> standard Perens LSFM atlas space (perens_lsfm_mouse_20um, BrainGlobe 'asr', 20 um).

Chain for each widefield pixel:
  widefield (x, y) --inverse fitted affine--> surface projection pixel --surface depth + view rotation-->
  point in the iDISCO volume (cache grid, 25.3 um) --rescale--> registered-atlas grid (20 um, original
  orientation, flipped like cache/<brain>_anno20.npy) --axis swap/flip--> brainreg's downsampled grid ('asr')
  --deformation_field_{0,1,2}.tiff--> Perens atlas coordinates (mm).
Checks: deformation fields reproduce registered_atlas.tiff labels exactly; the axis mapping
reg[a, s, r] == orig[O0-1-s, O1-1-a, r] holds exactly (both verified 2026-10-21).

Usage: python atlas_space.py PB105 PB104
Writes results/<brain>_widefield_to_perens_mm.npy (H, W, 3; NaN outside the shown window),
results/<brain>_perens_dorsal.npz (data resampled onto the atlas dorsal view) and figures/atlas_space/.
"""
import os
import sys
import json
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.interpolate import griddata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
import register as R  # noqa: E402
import surface as S  # noqa: E402
import enhance as E  # noqa: E402
import validate as V  # noqa: E402

RES_MM = 0.020
SAMPLE_DEPTH_VOX = 2.0     # map the point 2 voxels (~50 um) below the visible surface, i.e. inside cortex


def load_atlas():
    meta = json.load(open(f"{C.PERENS_ATLAS}/metadata.json"))
    return dict(annotation=np.load(f"{C.PERENS_ATLAS}/annotation.npy", mmap_mode="r"),
                hemispheres=np.load(f"{C.PERENS_ATLAS}/hemispheres.npy", mmap_mode="r"), meta=meta)


def widefield_to_atlas_mm(brain, transform, region):
    """Perens coordinates (mm, axes a/s/r) for every widefield pixel in `region`; NaN elsewhere."""
    t = json.load(open(transform))
    roll, pitch, A = t["roll_deg"], t["pitch_deg"], np.array(t["A_projection_to_widefield"])
    R.moving_for_view(brain, roll, pitch)
    prep = R._G[("prep", brain)]
    proj = S.project(prep, roll, pitch)
    z = proj["z"]
    shape = np.array(prep["norm"].shape)
    c = shape / 2.0
    H, W = int(shape[1] * 1.15), int(shape[2] * 1.15)
    oy, ox = (H - shape[1]) // 2, (W - shape[2]) // 2
    Hd = z.shape[0]
    wy, wx = np.nonzero(region)
    Ai = np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]
    xd, yd = Ai[:, :2] @ np.vstack([wx, wy]) + Ai[:, 2:3]                       # projection (display) coords
    zs = ndi.map_coordinates(np.nan_to_num(z, nan=np.nan), [yd, xd], order=1, cval=np.nan)
    ok = np.isfinite(zs)
    # display (row=yd, col=xd) -> pre-rot90 (y, x) = (col, Hd-1-row) -> volume coords (cache grid)
    q = np.stack([zs + SAMPLE_DEPTH_VOX, xd - oy, (Hd - 1 - yd) - ox], 1)
    p = (q - c) @ S.rotation(roll, pitch) + c
    anno_shape = np.array(np.load(f"{C.CACHE}/{brain}_anno20.npy", mmap_mode="r").shape)
    p20 = p * anno_shape / shape                                                  # anno20 grid (= orig[::-1])
    orig = tifffile.memmap(f"{C.DATA}/{brain}/registration/registered_atlas_original_orientation.tiff")
    O0, O1 = orig.shape[:2]
    # anno20 index (i0, i1, i2) = orig (O0-1-i0, i1, i2); reg[a, s, r] = orig[O0-1-s, O1-1-a, r]
    reg_idx = np.stack([O1 - 1 - p20[:, 1], p20[:, 0], p20[:, 2]], 0)
    mm = np.full((len(wx), 3), np.nan)
    for k in range(3):
        F = tifffile.memmap(f"{C.DATA}/{brain}/registration/deformation_field_{k}.tiff")
        mm[ok, k] = ndi.map_coordinates(np.asarray(F), reg_idx[:, ok], order=1, cval=np.nan)
    out = np.full(region.shape + (3,), np.nan, np.float32)
    out[wy, wx] = mm
    return out


def dorsal_view(atlas):
    """Top-down (superior) view of the atlas: first labelled voxel along s for each (a, r) column."""
    ann = np.asarray(atlas["annotation"])
    hit = ann > 0
    top = np.where(hit.any(1), hit.argmax(1), -1)                                 # (a, r)
    a, r = np.nonzero(top >= 0)
    lab = np.zeros(top.shape, np.int64)
    # label a little below the surface (layer 1 is thin; ~60 um down gives the area reliably)
    s = np.minimum(top[a, r] + 3, ann.shape[1] - 1)
    lab[a, r] = ann[a, s, r]
    hemi = np.zeros(top.shape, np.uint8)
    hemi[a, r] = np.asarray(atlas["hemispheres"])[a, s, r]
    return lab, top, hemi


def area_map(lab):
    ids, inv = np.unique(lab, return_inverse=True)
    return np.array([V.area_id(i) if i else 0 for i in ids])[inv.reshape(lab.shape)]


def clean_areas(areas, sigma=1.5):
    """Remove speckle: each pixel takes the area with the largest smoothed share in its neighbourhood."""
    u, inv = np.unique(areas, return_inverse=True)
    inv = inv.reshape(areas.shape)
    best, arg = np.full(areas.shape, -1.0), np.zeros(areas.shape, int)
    for k in range(len(u)):
        sm = ndi.gaussian_filter((inv == k).astype(np.float32), sigma)
        upd = sm > best
        best[upd], arg[upd] = sm[upd], k
    return u[arg]


def to_dorsal(values, mm, region, shape, step=1):
    """Resample widefield values onto the dorsal grid (a, r) at the mapped atlas positions."""
    ok = region & np.isfinite(mm[..., 0])
    pts = mm[ok][:, [0, 2]] / RES_MM                                              # (a, r) in atlas voxels
    grid_a, grid_r = np.mgrid[0:shape[0], 0:shape[1]]
    img = griddata(pts[::step], values[ok][::step], (grid_a, grid_r), method="linear")
    # footprint: where mapped points fall (dilated, holes filled)
    foot = np.zeros(shape, bool)
    ia, ir = np.round(pts).astype(int).T
    inb = (ia >= 0) & (ia < shape[0]) & (ir >= 0) & (ir < shape[1])
    foot[ia[inb], ir[inb]] = True
    foot = ndi.binary_fill_holes(ndi.binary_closing(ndi.binary_dilation(foot, iterations=1), iterations=3))
    return np.where(foot, img, np.nan), foot


def main(brains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_fig = f"{C.FIGURES}/atlas_space"
    os.makedirs(out_fig, exist_ok=True)
    atlas = load_atlas()
    lab, top, hemi = dorsal_view(atlas)
    # view from above with anterior up: the animal's left must appear on the viewer's left. The atlas r axis
    # runs right -> left, so flip it for display (data are mapped in atlas coordinates and flipped with it).
    lab, hemi = lab[:, ::-1], hemi[:, ::-1]
    areas = clean_areas(area_map(lab))
    bfd = np.isin(areas, V.BFD) | np.isin(areas, [V.area_id(i) for i in V.BFD])
    edges = (ndi.maximum_filter(areas, 2) != ndi.minimum_filter(areas, 2)) & (areas > 0)
    brain_dorsal = lab > 0
    shape = lab.shape
    res = {}
    for b in brains:
        transform = f"{C.RESULTS}/{b}_transform.json"
        fx = R.fixed_for(b)
        region = E.window_mask(b, fx["raw"].shape, display=True)
        mm = widefield_to_atlas_mm(b, transform, region)
        np.save(f"{C.RESULTS}/{b}_widefield_to_perens_mm.npy", mm)
        # check: area via deformation fields vs area assigned by the vessel pipeline (projected labels)
        vox = np.round(mm / RES_MM).astype(float)
        ok = region & np.all(np.isfinite(vox), -1)
        vi = vox[ok].astype(int)
        inb = np.all((vi >= 0) & (vi < np.array(atlas["annotation"].shape)), 1)
        lab_def = np.zeros(len(vi), np.int64)
        lab_def[inb] = np.asarray(atlas["annotation"])[vi[inb, 0], vi[inb, 1], vi[inb, 2]]
        L = np.load(f"{C.RESULTS}/{b}_atlas_labels_widefield.npy")[ok]
        a_def, a_pipe = area_map(lab_def), area_map(L)
        both = (a_def > 0) & (a_pipe > 0)
        agree = float((a_def == a_pipe)[both].mean())
        hemis = np.asarray(atlas["hemispheres"])[vi[inb, 0], vi[inb, 1], vi[inb, 2]]
        side = int(np.bincount(hemis).argmax())
        print(f"[{b}] mapped {ok.sum()} px; area agreement deformation-field vs vessel pipeline: {agree:.1%}; "
              f"hemisphere value {side} ({'left' if side == atlas['meta']['left_hemisphere_value'] else 'right'})")
        raw = np.log(np.maximum(fx["raw"], 1))
        flat = raw - ndi.gaussian_filter(raw, 10)
        d = V.hotspot(b)
        mm_disp = mm.copy()
        mm_disp[..., 2] = (shape[1] - 1) * RES_MM - mm[..., 2]                   # same left/right flip as the map
        stim, foot = to_dorsal(d, mm_disp, region, shape)
        ves, _ = to_dorsal(flat, mm_disp, region, shape)
        np.savez(f"{C.RESULTS}/{b}_perens_dorsal.npz", stim_minus_nostim=stim, vessels_flatfield=ves, footprint=foot,
                 atlas_dorsal_labels=lab, atlas_dorsal_areas=areas, note="dorsal view of perens_lsfm_mouse_20um, rows = a (anterior up), columns = r flipped so the "
                      "animal's left is on the left (column c = atlas r index 460 - c), 20 um px")
        res[b] = dict(stim=stim, ves=ves, foot=foot, agree=agree)

    # crop to the imaged hemisphere region (union of footprints, padded)
    foot_all = np.any([r["foot"] for r in res.values()], 0)
    ys, xs = np.nonzero(foot_all | (ndi.binary_dilation(foot_all, iterations=60) & brain_dorsal))
    box = (slice(max(ys.min() - 5, 0), ys.max() + 6), slice(max(xs.min() - 5, 0), xs.max() + 6))
    def base():
        img = np.ones(shape + (3,))
        img[brain_dorsal] = 0.93
        return img
    ext = None
    n = len(brains)
    fig, ax = plt.subplots(2, n + 1, figsize=(5.2 * (n + 1), 10))
    ax = np.atleast_2d(ax)
    for j, b in enumerate(brains):
        r = res[b]
        for i, (key, cmap, title) in enumerate((("stim", "inferno", "stim − no stim"), ("ves", "gray", "widefield vessels"))):
            img = base()
            v = r[key]
            m = np.isfinite(v)
            lo, hi = np.nanpercentile(v[m], [2, 99.8]) if key == "stim" else np.nanpercentile(v[m], [1, 99.5])
            col = plt.get_cmap(cmap)(np.clip((np.nan_to_num(v) - lo) / (hi - lo), 0, 1))[..., :3]
            img[m] = col[m]
            img[edges] = [0.35, 0.35, 0.35] if key == "stim" else [0.0, 0.75, 0.9]
            img[bfd & ~ndi.binary_erosion(bfd, iterations=2)] = [0.2, 0.85, 0.3]          # SSp-bfd
            ax[i, j].imshow(img[box], interpolation="nearest")
            ax[i, j].set_title(f"{b}: {title} (Perens dorsal view)\narea agreement with pipeline {r['agree']:.0%}", fontsize=10)
            ax[i, j].axis("off")
            # single-panel export
            h, w = img[box].shape[:2]
            f1 = plt.figure(figsize=(w / 100, h / 100)); a1 = f1.add_axes([0, 0, 1, 1]); a1.imshow(img[box], interpolation="nearest"); a1.axis("off")
            for ext in ("png", "pdf"):
                f1.savefig(f"{out_fig}/{b}_{'stim' if key == 'stim' else 'vessels'}_perens_dorsal.{ext}", dpi=300)
            plt.close(f1)
    # all brains: wS1 half-max contours on the same standard map
    img = base()
    img[edges] = [0.55, 0.55, 0.55]
    img[bfd & ~ndi.binary_erosion(bfd, iterations=2)] = [0.2, 0.85, 0.3]
    a = ax[0, n]
    a.imshow(img[box], interpolation="nearest")
    colors = ["#e7298a", "#1b9e77", "#7570b3", "#d95f02"]
    for k, b in enumerate(brains):
        v = res[b]["stim"]
        hm = np.nan_to_num(v) >= 0.5 * np.nanmax(v)
        a.contour(hm[box], [0.5], colors=colors[k % len(colors)], linewidths=2)
        a.plot([], [], color=colors[k % len(colors)], lw=2, label=b)
    a.legend(loc="lower left", fontsize=9)
    a.set_title("whisker response half-max, all mice\n(standard Perens dorsal view; SSp-bfd green)", fontsize=10)
    a.axis("off")
    ax[1, n].axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_fig}/overview_perens_dorsal.png", dpi=110)
    h, w = img[box].shape[:2]
    f1 = plt.figure(figsize=(w / 100, h / 100)); a1 = f1.add_axes([0, 0, 1, 1]); a1.imshow(img[box], interpolation="nearest")
    for k, b in enumerate(brains):
        v = res[b]["stim"]; a1.contour((np.nan_to_num(v) >= 0.5 * np.nanmax(v))[box], [0.5], colors=colors[k % len(colors)], linewidths=1.5)
    a1.axis("off")
    for ext in ("png", "pdf"):
        f1.savefig(f"{out_fig}/all_mice_whisker_halfmax_perens_dorsal.{ext}", dpi=300)


if __name__ == "__main__":
    main(sys.argv[1:] or ["PB105", "PB104"])
