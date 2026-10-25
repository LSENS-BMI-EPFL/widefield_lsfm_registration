"""Vessel enhancement for both modalities (dark vessels -> bright ridges), plus masks."""
import os
import json
import numpy as np
import cv2
import tifffile
from scipy import ndimage as ndi
from skimage.filters import frangi, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, skeletonize

import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS = json.load(open(f"{HERE}/atlas_ids.json"))
ISOCORTEX = np.array(ATLAS["isocortex"])

WINDOW, WINDOW_DISPLAY = C.WINDOW, C.WINDOW_DISPLAY   # per-brain polygons, set in config.py


LARGE = dict(sigmas=(2.0, 3.0, 4.0), min_len=25, keep_frac=0.12)


def _ridges(img, mask, erode=4, sigmas=None, min_len=None, keep_frac=None):
    """Frangi (dark vessels). With min_len, keep only long connected vessel segments (large pial vessels)."""
    sigmas = sigmas or LARGE["sigmas"]
    min_len = LARGE["min_len"] if min_len is None else min_len
    keep_frac = keep_frac or LARGE["keep_frac"]
    # Filter the whole image, then mask (filling outside the mask first creates edge ridges).
    r = frangi(img, sigmas=sigmas, black_ridges=True, alpha=0.5, beta=0.5, gamma=None)
    mask = ndi.binary_erosion(mask, iterations=erode)
    r = np.where(mask, r, 0)
    hi = np.percentile(r[mask], 99.5)
    r = np.clip(r / hi, 0, 1).astype(np.float32)
    if min_len:
        thr = np.percentile(r[mask], 100 * (1 - keep_frac))
        seg = r > thr
        lab, n = ndi.label(seg, structure=np.ones((3, 3)))
        if n:
            # component extent ~ length: max of bounding-box diagonal
            sl = ndi.find_objects(lab)
            ext = np.array([0] + [np.hypot(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in sl])
            keep = ext[lab] >= min_len
            r = np.where(ndi.binary_dilation(keep, iterations=2), r, 0)
    return r


def window_mask(brain, shape=(250, 320), display=False):
    """Cortex under the window. display=True: the whole visible window (figures, atlas mapping); else the fitting
    region. config.WINDOW_MODE "auto": detected (auto_window): for display its convex outline, for fitting the detected
    cortex shrunk by WINDOW_FIT_MARGIN_PX; "manual": the hand-drawn polygons WINDOW / WINDOW_DISPLAY."""
    if C.WINDOW_MODE == "auto":
        if display:
            return auto_window(brain, convex=True)
        return ndi.binary_erosion(auto_window(brain), iterations=C.WINDOW_FIT_MARGIN_PX)
    poly = WINDOW_DISPLAY.get(brain, WINDOW[brain]) if display else WINDOW[brain]
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m.astype(bool)


def _largest(m):
    lab, n = ndi.label(m)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n else m


def signal_correlation(brain):
    """Per-pixel correlation of the trial-averaged dF/F (stim and no-stim, concatenated) with the cortex's shared time
    course. Living cortex under the window shares it; bone, cement and the window rim do not. The reference starts
    as the image centre and is then re-estimated from the best-correlated 10% of pixels."""
    X = np.concatenate([np.load(f"{C.DATA}/{brain}/wf_data/{k}_trial_avg.npy") for k in ("stim", "no_stim")])
    X = (X - X.mean(0)).astype(np.float32)
    H, W = X.shape[1:]
    norm = np.sqrt((X ** 2).sum(0)) + 1e-12
    ref = np.median(X[:, H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20].reshape(len(X), -1), 1)
    for _ in range(2):
        c = np.tensordot(ref, X, 1) / (norm * np.sqrt((ref ** 2).sum()))
        ref = np.median(X[:, c >= np.percentile(c, 90)], 1)
    return ndi.gaussian_filter(c, 3)


AUTO = dict(corr=0.3, valley_size=31, valley_depth=0.08, valley_width=3, seed_erode=8, smooth=3,
            close_px=16, outline_sigma=8, convex_sigma=8)


def _convex_outline(m, sigma):
    """Convex hull of the region, resampled along its length and smoothed (rounded corners, still convex): one clean
    curve for the window as displayed (a cranial window is convex)."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    H = cv2.convexHull(max(cs, key=cv2.contourArea))[:, 0].astype(np.float64)
    Hc = np.vstack([H, H[:1]])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(Hc, axis=0).T))]
    t = np.arange(0, s[-1], 0.5)
    P = np.stack([np.interp(t, s, Hc[:, k]) for k in (0, 1)], 1)
    P = ndi.gaussian_filter1d(P, sigma / 0.5, axis=0, mode="wrap")
    out = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(out, [np.round(P * 8).astype(np.int32)], 1, shift=3)     # sub-pixel vertices
    return out.astype(bool)


def _smooth_outline(m, close_px, sigma):
    """Closing with a disk (fills notches cut by large vessels), then the outer contour smoothed along its length
    (Gaussian, sigma in contour px) and filled again: a clean outline instead of a pixel staircase."""
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
    pad = close_px + 1                                                    # so the closing works at the image border
    mp = cv2.morphologyEx(np.pad(m, pad).astype(np.uint8), cv2.MORPH_CLOSE, disk)[pad:-pad, pad:-pad]
    cs, _ = cv2.findContours(mp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    P = max(cs, key=cv2.contourArea)[:, 0].astype(np.float64)
    P = ndi.gaussian_filter1d(P, sigma, axis=0, mode="wrap")
    out = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(out, [np.round(P).astype(np.int32)], 1)
    return out.astype(bool)


def auto_window(brain, convex=False):
    """Visible cortex under the window, from the data: pixels correlated with the cortex signal (signal_correlation),
    split into compartments along wide dark valleys of the widefield image (sagittal sinus, window rim; narrower
    vessels are ignored), keeping the compartment with the best-correlated pixels. convex=True: its convex outline
    (_convex_outline), for display. Cached in CACHE."""
    from skimage.segmentation import watershed
    path = f"{C.CACHE}/{brain}_window_auto.npz"
    if os.path.exists(path):
        return np.load(path)["convex" if convex else "cortex"]
    p = AUTO
    c = signal_correlation(brain)
    raw = tifffile.imread(f"{C.DATA}/{brain}/{brain}_wf_image.tif").astype(np.float32)
    lg = ndi.gaussian_filter(np.log(np.maximum(raw, 1)), 2)
    depth = ndi.black_tophat(lg, size=p["valley_size"])                   # dark valleys narrower than valley_size
    valley = ndi.binary_opening(depth > p["valley_depth"], iterations=p["valley_width"])
    cortex = c > p["corr"]
    seeds, _ = ndi.label(ndi.binary_erosion(cortex & ~valley, iterations=p["seed_erode"]))
    comp = watershed(depth, seeds, mask=cortex)                            # boundaries on the valley ridges
    core = _largest(c > np.percentile(c, 95))
    m = ndi.binary_fill_holes(comp == np.bincount(comp[core & (comp > 0)]).argmax())
    m = _largest(ndi.binary_opening(m, iterations=p["smooth"]))
    m = _smooth_outline(m, p["close_px"], p["outline_sigma"])
    os.makedirs(C.CACHE, exist_ok=True)
    hull = _convex_outline(m, p["convex_sigma"])
    np.savez(path, cortex=m, convex=hull)
    return hull if convex else m


def widefield(brain):
    raw = tifffile.imread(f"{C.DATA}/{brain}/{brain}_wf_image.tif").astype(np.float32)
    lg = np.log(np.maximum(raw, 1))
    mask = window_mask(brain, raw.shape)
    # Flat-field inside the window only (normalized convolution), so edges don't leak in.
    num = ndi.gaussian_filter(np.where(mask, lg, 0), 10)
    den = ndi.gaussian_filter(mask.astype(np.float32), 10)
    flat = lg - num / np.maximum(den, 1e-3)
    return dict(raw=raw, ridge=_ridges(flat, mask), mask=ndi.binary_erosion(mask, iterations=3))


def _masked_smooth(x, m, s):
    return ndi.gaussian_filter(np.where(m, x, 0), s) / np.maximum(ndi.gaussian_filter(m.astype(np.float32), s), 1e-3)


NETWORK = dict(sigmas=(1.0, 1.5, 2.0, 3.0, 4.0), lo=0.06, hi=0.2, min_size=40, max_slope=1.5, edge_px=5,
               edge_keep_px=7, valley_vox=2.5)


def idisco(proj):
    """Surface-vessel network of the iDISCO projection (vessels are dark).

    1. Isocortex, minus large-scale steep surface (brain edges, fissure walls), eroded by edge_px.
    2. Log + flat-field; bright debris blobs, round dark bubbles and interior mask holes (with their rims) are
       set to the neutral tissue level so they create no false ridges.
    3. Multi-scale Frangi (25-100 um), hysteresis threshold (a pixel is kept if weakly vessel-like and
       connected to a strongly vessel-like one), small pieces removed.
    ridge = vesselness on the segmented network, used for the fit; seg/skel/flat for figures."""
    p = NETWORK
    v, z = proj["vessel"], proj["z"]
    cortex = ndi.binary_opening(np.isin(proj["label"], ISOCORTEX) & np.isfinite(v), iterations=2)
    zf = np.nan_to_num(z, nan=np.nanmax(z))
    gy, gx = np.gradient(ndi.gaussian_filter(zf, 5))
    # deep, large surface valleys = fissures (interhemispheric, transverse), not vessels
    valley = remove_small_objects((zf - ndi.gaussian_filter(zf, 8)) > p["valley_vox"], 200)
    keep = cortex & (np.hypot(gx, gy) < p["max_slope"]) & ~ndi.binary_dilation(valley, iterations=4)
    m = ndi.binary_erosion(keep, iterations=p["edge_px"])
    lg = np.log(np.maximum(np.nan_to_num(v, nan=1.0), 1e-3))
    flat = np.where(m, lg - _masked_smooth(lg, m, 10), 0)
    med = np.median(flat[m])
    mad = np.median(np.abs(flat[m] - med)) * 1.4826
    blobs = ndi.binary_dilation(m & (flat > med + 4 * mad), iterations=2)          # bright debris
    lab, _ = ndi.label(m & (flat < med - 6 * mad))
    for k, sl in enumerate(ndi.find_objects(lab), 1):                              # round dark bubbles
        comp = ndi.binary_fill_holes(lab[sl] == k)
        h, w = comp.shape
        if comp.sum() > 30 and comp.sum() / (h * w) > 0.45 and max(h, w) / max(min(h, w), 1) < 2.5:
            blobs[sl] |= ndi.binary_dilation(comp, iterations=3)
    holes = ndi.binary_fill_holes(m) & ~m
    blobs |= holes | (ndi.binary_dilation(holes, iterations=3) & m)                # holes + their dark rims
    flat = np.where(blobs, 0, flat)                                                 # neutral tissue level
    valid = m | holes
    flat = np.where(valid, flat, 0)                                                 # neutral outside
    vs = frangi(flat, sigmas=p["sigmas"], black_ridges=True, alpha=0.5, beta=0.5, gamma=None)
    inner = ndi.binary_erosion(ndi.binary_fill_holes(m), iterations=p["edge_px"]) & ~ndi.binary_dilation(blobs, iterations=2)
    vs = np.where(inner, vs, 0)
    vs = np.clip(vs / np.percentile(vs[inner], 99.5), 0, 1)
    seg = remove_small_objects(apply_hysteresis_threshold(vs, p["lo"], p["hi"]), p["min_size"])
    # drop pieces that run along the edge of the cortex (fissure / silhouette walls, not vessels)
    lab, n = ndi.label(seg, structure=np.ones((3, 3)))
    if n:
        med_edge = np.array(ndi.median(ndi.distance_transform_edt(valid), lab, np.arange(1, n + 1)))
        seg = np.isin(lab, np.arange(1, n + 1)[med_edge >= p["edge_keep_px"]])
    ridge = vs * ndi.binary_dilation(seg, iterations=1)
    ridge = np.clip(ridge / np.percentile(ridge[ridge > 0], 95), 0, 1).astype(np.float32)
    return dict(ridge=ridge, mask=inner, seg=seg, skel=skeletonize(seg), flat=flat, valid=valid,
                vesselness=vs, filled=blobs & valid)
