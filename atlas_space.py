"""Widefield -> standard Perens LSFM atlas space (perens_lsfm_mouse_20um, BrainGlobe 'asr', 20 um).

Chain for each widefield pixel:
  widefield (x, y) --inverse fitted affine--> surface projection pixel --surface depth + view rotation-->
  point in the iDISCO volume (cache grid = the registered-atlas grid, 20 um, original orientation, flipped like
  cache/<brain>_anno20.npy) --axis swap/flip--> brainreg's downsampled grid ('asr')
  --deformation_field_{0,1,2}.tiff--> Perens atlas coordinates (mm).
The depth depends on the layer (--layer): "surface" maps the point 50 um below the visible surface (as before); a
deeper layer maps the point where the pixel's camera ray first reaches that layer, in the imaged hemisphere
(layers.layer_projection): the same point that gives the pixel its area in layer_boundaries.py.
Checks: deformation fields reproduce registered_atlas.tiff labels exactly; the axis mapping
reg[a, s, r] == orig[O0-1-s, O1-1-a, r] holds exactly (both verified 2026-10-21).

Two views of the atlas:
  dorsal   straight from above (anterior up, the animal's left on the left), as before
  rotated  the same, with the atlas brain turned by the imaging-angle prior (config.ATLAS_VIEW_TILT, 30 deg lateral
           toward the imaged hemisphere), so lateral cortex (barrel field) is seen face-on as in the widefield image.
Atlas borders: "surface" = the areas just below the surface (as before); a layer = the areas where each view ray
reaches that layer. The brain's outline (silhouette in the view) is drawn as well.

Usage: python atlas_space.py PB105 PB104 [--layer l5]
The mapped region is the outline of the widefield figures (make_figure.figure_region: the imaged hemisphere's cortex
in the image, also where it extends beyond the window), not only the window; the window is used for the checks and the
half-max contours.
Writes results/<brain>_widefield_to_perens_mm[_<L>].npy (H, W, 3; NaN outside the outline region),
results/<brain>_perens_<view>[_<L>].npz (data resampled onto the view) and figures/atlas_space[_<L>]/
(no suffix for the surface).
"""
import os
import sys
import json
import argparse
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
import layers  # noqa: E402

RES_MM = 0.020
SURFACE_LABEL_VOX = 3      # atlas views, "surface": label 3 voxels (60 um) below the surface (layer 1 is thin)


def load_atlas():
    meta = json.load(open(f"{C.PERENS_ATLAS}/metadata.json"))
    return dict(annotation=np.load(f"{C.PERENS_ATLAS}/annotation.npy", mmap_mode="r"),
                hemispheres=np.load(f"{C.PERENS_ATLAS}/hemispheres.npy", mmap_mode="r"), meta=meta)


def widefield_to_atlas_mm(brain, transform, region, layer="surface"):
    """Perens coordinates (mm, axes a/s/r) for every widefield pixel in `region` at the layer; NaN elsewhere."""
    t = json.load(open(transform))
    roll, pitch, A = t["roll_deg"], t["pitch_deg"], np.array(t["A_projection_to_widefield"])
    proj, _, depth, _ = layers.layer_projection(brain, roll, pitch, layer)
    prep = R._G[("prep", brain)]
    z = proj["z"]
    shape = np.array(prep["norm"].shape)
    c = shape / 2.0
    H, W = int(shape[1] * 1.15), int(shape[2] * 1.15)
    oy, ox = (H - shape[1]) // 2, (W - shape[2]) // 2
    Hd = z.shape[0]
    wy, wx = np.nonzero(region)
    Ai = np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]
    xd, yd = Ai[:, :2] @ np.vstack([wx, wy]) + Ai[:, 2:3]                       # projection (display) coords
    zs = ndi.map_coordinates(z, [yd, xd], order=1, cval=np.nan)
    ds = ndi.map_coordinates(depth, [yd, xd], order=0, cval=np.nan)            # layer depth along the ray (voxels)
    ok = np.isfinite(zs) & np.isfinite(ds)
    # display (row=yd, col=xd) -> pre-rot90 (y, x) = (col, Hd-1-row) -> volume coords (cache grid)
    q = np.stack([zs + ds, xd - oy, (Hd - 1 - yd) - ox], 1)
    p = (q - c) @ S.rotation(roll, pitch) + c
    anno_shape = np.array(np.load(f"{C.CACHE}/{brain}_anno20.npy", mmap_mode="r").shape)
    p20 = p * anno_shape / shape                                                  # anno20 grid (= orig[::-1]); same grid
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


def view_basis(lat, ap, side):
    """Viewing direction u (into the brain) and image axes e1 (rows, down = posterior) and e2 (columns), in atlas
    voxel axes (a, s, r). (0, 0) is the dorsal view; lat tilts the camera toward the imaged hemisphere's lateral side,
    ap > 0 toward anterior (as brain_axes.anatomical). Columns run so that the animal's left is on the viewer's left
    (r runs right -> left, so the dorsal view has e2 = -r)."""
    L, P = np.radians(lat), np.radians(ap)
    sgn = -1.0 if side == "left" else 1.0         # left hemisphere is at high r: the camera there looks toward -r
    u = np.array([np.sin(P), np.cos(P) * np.cos(L), sgn * np.cos(P) * np.sin(L)])
    e1 = np.array([1.0, 0, 0]) - u[0] * u
    e1 /= np.linalg.norm(e1)
    return u, e1, np.cross(u, e1)


def atlas_view(atlas, layer, lat=0.0, ap=0.0, side="left"):
    """The atlas seen along u (view_basis), orthographic, 1 px = 1 voxel. For each pixel the ray enters the brain at
    the first labelled voxel; the label is taken at the layer below it (rule of layers.layer_projection) or, for
    "surface", 60 um below it. Returns labels, hemisphere values, brain mask and the view geometry."""
    ann = np.asarray(atlas["annotation"])
    hem = np.asarray(atlas["hemispheres"])
    shape = np.array(ann.shape)
    u, e1, e2 = view_basis(lat, ap, side)
    c = (shape - 1) / 2.0
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * (shape - 1) - c
    r0, c0, d0 = (np.floor((corners @ e).min()) for e in (e1, e2, u))
    r1, c1, d1 = (np.ceil((corners @ e).max()) for e in (e1, e2, u))
    rows, cols = np.arange(r0, r1 + 1), np.arange(c0, c1 + 1)
    RR, CC = np.meshgrid(rows, cols, indexing="ij")
    base = c + RR[..., None] * e1 + CC[..., None] * e2                           # (H, W, 3) on the plane d = 0
    Hs, Ws = RR.shape
    flat = base.reshape(-1, 3)
    n = len(flat)
    sids = np.array(sorted(int(k) for k in atlas["meta"]["structures"]))
    rk_s = layers.rank_of(sids)
    target = None if layer == "surface" else layers.TARGET[layer]
    top = np.full(n, np.nan)
    lab = np.zeros(n, np.int64)
    hv = np.zeros(n, np.uint8)
    todo = np.arange(n)
    for d in np.arange(d0, d1 + 1):
        if not len(todo):
            break
        pt = np.round(flat[todo] + d * u).astype(int)
        inb = np.all((pt >= 0) & (pt < shape), 1)
        li = np.zeros(len(todo), np.int64)
        li[inb] = ann[pt[inb, 0], pt[inb, 1], pt[inb, 2]]
        new = np.isnan(top[todo]) & (li > 0)
        top[todo[new]] = d
        started = np.isfinite(top[todo])
        if target is None:
            hit = started & (d >= top[todo] + SURFACE_LABEL_VOX) & (li > 0)
        else:
            pos = np.clip(np.searchsorted(sids, li), 0, len(sids) - 1)
            rk = np.where(sids[pos] == li, rk_s[pos], -1)
            hit = started & (rk >= target)
        hit &= inb
        lab[todo[hit]] = li[hit]
        hv[todo[hit]] = hem[pt[hit, 0], pt[hit, 1], pt[hit, 2]]
        gone = hit | (started & (d - top[todo] > layers.MAX_DEPTH_UM / (RES_MM * 1000)))
        todo = todo[~gone]
    geo = dict(u=u, e1=e1, e2=e2, c=c, r0=r0, c0=c0, lat=lat, ap=ap, side=side)
    return lab.reshape(Hs, Ws), hv.reshape(Hs, Ws), np.isfinite(top).reshape(Hs, Ws), geo


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


def to_view(values, mm, region, shape, geo, step=1):
    """Resample widefield values onto a view grid at the in-plane position of each pixel's mapped atlas point."""
    ok = region & np.isfinite(mm[..., 0])
    p = mm[ok] / RES_MM - geo["c"]                                                # atlas voxels, from the centre
    pts = np.stack([p @ geo["e1"] - geo["r0"], p @ geo["e2"] - geo["c0"]], 1)
    grid_a, grid_r = np.mgrid[0:shape[0], 0:shape[1]]
    img = griddata(pts[::step], values[ok][::step], (grid_a, grid_r), method="linear")
    # footprint: where mapped points fall (dilated, holes filled)
    foot = np.zeros(shape, bool)
    ia, ir = np.round(pts).astype(int).T
    inb = (ia >= 0) & (ia < shape[0]) & (ir >= 0) & (ir < shape[1])
    foot[ia[inb], ir[inb]] = True
    foot = ndi.binary_fill_holes(ndi.binary_closing(ndi.binary_dilation(foot, iterations=1), iterations=3))
    return np.where(foot, img, np.nan), foot


def halfmax(r):
    """Half-maximum region of the stim response, measured inside the window (the image edges carry artifacts)."""
    v = np.where(r["win"], np.nan_to_num(r["stim"], nan=-np.inf), -np.inf)
    return r["win"] & (v >= 0.5 * v.max())


def main(brains, layer="surface"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_figure as MF
    layer = layers.check(layer)
    tag = layers.TAG[layer]
    sfx = "" if layer == "surface" else f"_{tag}"
    out_fig = f"{C.FIGURES}/atlas_space{sfx}"
    os.makedirs(out_fig, exist_ok=True)
    atlas = load_atlas()
    clim = V.activity_clim()
    sides = [C.HEMISPHERE.get(b) or "left" for b in brains]
    side = max(set(sides), key=sides.count)
    views = {"dorsal": (0.0, 0.0), "rotated": tuple(C.ATLAS_VIEW_TILT)}

    mm_all, agree = {}, {}
    for b in brains:
        transform = f"{C.RESULTS}/{b}_transform.json"
        fx = R.fixed_for(b)
        win = E.window_mask(b, fx["raw"].shape, display=True)
        t = json.load(open(transform))
        # mapped: the outline region of the widefield figures (the imaged hemisphere's cortex in the image, beyond the
        # window where the cortex extends under bone/cement), not only the window
        region = MF.figure_region(b, t["roll_deg"], t["pitch_deg"], np.array(t["A_projection_to_widefield"]))
        mm = widefield_to_atlas_mm(b, transform, region, layer)
        np.save(f"{C.RESULTS}/{b}_widefield_to_perens_mm{sfx}.npy", mm)
        # check: area via deformation fields vs area assigned by the vessel pipeline at the same layer
        vox = np.round(mm / RES_MM).astype(float)
        ok = win & np.all(np.isfinite(vox), -1)            # checks inside the window
        vi = vox[ok].astype(int)
        inb = np.all((vi >= 0) & (vi < np.array(atlas["annotation"].shape)), 1)
        lab_def = np.zeros(len(vi), np.int64)
        lab_def[inb] = np.asarray(atlas["annotation"])[vi[inb, 0], vi[inb, 1], vi[inb, 2]]
        lab_path = f"{C.RESULTS}/{b}_atlas_labels_widefield{sfx}.npy"
        if os.path.exists(lab_path):
            L = np.load(lab_path)[ok]
            a_def, a_pipe = area_map(lab_def), area_map(L)
            both = (a_def > 0) & (a_pipe > 0)
            agree[b] = float((a_def == a_pipe)[both].mean())
        else:
            agree[b] = float("nan")
            print(f"[{b}] {lab_path} missing (finalize.py / layer_boundaries.py): no area check")
        hemis = np.asarray(atlas["hemispheres"])[vi[inb, 0], vi[inb, 1], vi[inb, 2]]
        hside = int(np.bincount(hemis).argmax())
        print(f"[{b}] {tag}: mapped {np.isfinite(mm[..., 0]).sum()} px ({ok.sum()} in the window); area agreement deformation-field vs vessel pipeline: {agree[b]:.1%}; "
              f"hemisphere value {hside} ({'left' if hside == atlas['meta']['left_hemisphere_value'] else 'right'})")
        mm_all[b] = (mm, region, win)

    colors = ["#e7298a", "#1b9e77", "#7570b3", "#d95f02"]
    n = len(brains)
    fig, ax = plt.subplots(2 * len(views), n + 1, figsize=(5.2 * (n + 1), 10 * len(views)))
    ax = np.atleast_2d(ax)
    for vi_, (vname, (lat, apt)) in enumerate(views.items()):
        lab, hemi, brain_v, geo = atlas_view(atlas, layer, lat, apt, side)
        areas = clean_areas(area_map(lab))
        bfd = np.isin(areas, V.BFD) | np.isin(areas, [V.area_id(i) for i in V.BFD])
        edges = (ndi.maximum_filter(areas, 2) != ndi.minimum_filter(areas, 2)) & (areas > 0)
        edges |= brain_v & ~ndi.binary_erosion(brain_v)                           # brain outline
        shape = lab.shape
        res = {}
        for b in brains:
            mm, region, win = mm_all[b]
            fx = R.fixed_for(b)
            raw = np.log(np.maximum(fx["raw"], 1))
            flat = raw - ndi.gaussian_filter(raw, 10)
            stim, foot = to_view(V.hotspot(b), mm, region, shape, geo)
            ves, _ = to_view(flat, mm, region, shape, geo)
            wv, _ = to_view(win.astype(np.float32), mm, region, shape, geo)
            in_win = np.nan_to_num(wv) > 0.5                 # the window in the view (half-max is measured there)
            name = f"{C.RESULTS}/{b}_perens_{vname}{sfx}.npz"
            np.savez(name, stim_response=stim, vessels_flatfield=ves, footprint=foot, window=in_win, atlas_labels=lab,
                     atlas_areas=areas, layer=layer, view_tilt_lat_ap_deg=[lat, apt], side=side,
                     view_u=geo["u"], view_e1=geo["e1"], view_e2=geo["e2"], view_center_vox=geo["c"],
                     view_origin_rc=[geo["r0"], geo["c0"]],
                     note="orthographic view of perens_lsfm_mouse_20um ('asr' voxel axes, 20 um px) along view_u; "
                          "pixel (row, col) = ((p - center) . e1 - origin_r, (p - center) . e2 - origin_c) for an atlas "
                          "voxel position p; rows anterior up, the animal's left on the left")
            res[b] = dict(stim=stim, ves=ves, foot=foot, win=in_win)

        foot_all = np.any([r["foot"] for r in res.values()], 0)
        ys, xs = np.nonzero(foot_all | (ndi.binary_dilation(foot_all, iterations=60) & brain_v))
        box = (slice(max(ys.min() - 5, 0), ys.max() + 6), slice(max(xs.min() - 5, 0), xs.max() + 6))

        def base():
            img = np.ones(shape + (3,))
            img[brain_v] = 0.93
            return img
        title_v = "dorsal view" if vname == "dorsal" else f"rotated view ({lat:g}° lat, {apt:g}° AP)"
        for j, b in enumerate(brains):
            r = res[b]
            for i, (key, title) in enumerate((("stim", "stim response, baseline-corrected"), ("ves", "widefield vessels"))):
                img = base()
                v = r[key]
                m = np.isfinite(v)
                if key == "stim":
                    col = MF.activity_rgb(np.nan_to_num(v), clim)
                else:
                    lo, hi = np.nanpercentile(v[m], [1, 99.5])
                    col = plt.get_cmap("gray")(np.clip((np.nan_to_num(v) - lo) / (hi - lo), 0, 1))[..., :3]
                img[m] = col[m]
                img[edges] = MF.ACT_BORDER if key == "stim" else [0.0, 0.75, 0.9]
                img[bfd & ~ndi.binary_erosion(bfd, iterations=2)] = MF.BFD_GREEN          # SSp-bfd
                a = ax[2 * vi_ + i, j]
                a.imshow(img[box], interpolation="nearest")
                a.set_title(f"{b}: {title}\n{tag}, Perens {title_v}; area agreement {agree[b]:.0%}", fontsize=10)
                a.axis("off")
                h, w = img[box].shape[:2]
                f1 = plt.figure(figsize=(w / 100, h / 100)); a1 = f1.add_axes([0, 0, 1, 1])
                a1.imshow(img[box], interpolation="nearest"); a1.axis("off")
                for ext in ("png", "pdf"):
                    f1.savefig(f"{out_fig}/{b}_{'stim' if key == 'stim' else 'vessels'}_perens_{vname}.{ext}", dpi=300)
                plt.close(f1)
        # all brains: wS1 half-max contours on the same standard map
        img = base()
        img[edges] = [0.55, 0.55, 0.55]
        img[bfd & ~ndi.binary_erosion(bfd, iterations=2)] = MF.BFD_GREEN
        a = ax[2 * vi_, n]
        a.imshow(img[box], interpolation="nearest")
        for k, b in enumerate(brains):
            hm = halfmax(res[b])
            a.contour(hm[box], [0.5], colors=colors[k % len(colors)], linewidths=2)
            a.plot([], [], color=colors[k % len(colors)], lw=2, label=b)
        a.legend(loc="lower left", fontsize=9)
        a.set_title(f"whisker response half-max, all mice\n{tag}, Perens {title_v}; SSp-bfd green", fontsize=10)
        a.axis("off")
        ax[2 * vi_ + 1, n].axis("off")
        h, w = img[box].shape[:2]
        f1 = plt.figure(figsize=(w / 100, h / 100)); a1 = f1.add_axes([0, 0, 1, 1])
        a1.imshow(img[box], interpolation="nearest")
        for k, b in enumerate(brains):
            a1.contour(halfmax(res[b])[box], [0.5], colors=colors[k % len(colors)], linewidths=1.5)
        a1.axis("off")
        for ext in ("png", "pdf"):
            f1.savefig(f"{out_fig}/all_mice_whisker_halfmax_perens_{vname}.{ext}", dpi=300)
        plt.close(f1)
    plt.tight_layout()
    plt.savefig(f"{out_fig}/overview_perens.png", dpi=90)
    plt.close(fig)
    MF.activity_colorbar(f"{out_fig}/stim_perens_colorbar", clim)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("brains", nargs="*", default=["PB105", "PB104"])
    ap.add_argument("--layer", default="surface", help=f"one of {layers.LAYERS}")
    a = ap.parse_args()
    main(a.brains, a.layer)
