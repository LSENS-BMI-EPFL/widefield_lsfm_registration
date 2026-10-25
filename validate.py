"""Post-hoc control: does the whisker-stim hotspot land in the barrel field (SSp-bfd)?

Not used for fitting. For each candidate (roll, pitch, A), warp the projected atlas label map into
widefield space and measure where the stim - no-stim hotspot falls.
"""
import os
import sys
import json
import numpy as np
import cv2
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
import register as R  # noqa: E402

ATLAS = json.load(open(f"{HERE}/atlas_ids.json"))
BFD = np.array(ATLAS["bfd"])
NAMES = {int(k): v for k, v in ATLAS["names"].items()}
PATH = {int(k): v for k, v in ATLAS["path"].items()}
WIN = slice(100, 116)       # 1.00-1.15 s, early response (100 Hz)
VOX_UM = 25.32


def area_id(label):
    """Collapse layer-level IDs to their cortical area (parent of layers), e.g. SSp-bfd4 -> SSp-bfd."""
    p = PATH.get(int(label))
    if not p:
        return int(label)
    if 315 in p and len(p) > p.index(315) + 2:
        # isocortex path: ..., 315, <region group>, <area>, <layer>  -> stop at area (one above layer)
        nm = NAMES.get(p[-1], "")
        return p[-2] if any(ch.isdigit() for ch in nm[-3:]) else p[-1]
    return int(label)


def hotspot(brain):
    s = np.load(f"{C.DATA}/{brain}/wf_data/stim_trial_avg.npy")
    n = np.load(f"{C.DATA}/{brain}/wf_data/no_stim_trial_avg.npy")
    d = ndi.gaussian_filter((s - n)[WIN].mean(0), 3)
    return d


def evaluate(brain, roll, pitch, A):
    proj, _ = R.moving_for_view(brain, roll, pitch)
    lab = proj["label"].astype(np.float32)   # ids fit in float32 exactly? use index map instead
    ids, inv = np.unique(proj["label"], return_inverse=True)
    idx = inv.reshape(proj["label"].shape).astype(np.float32)
    L = ids[R.warp(idx, np.array(A), interp=cv2.INTER_NEAREST).astype(int)]
    d = hotspot(brain)
    py, px = np.unravel_index(np.argmax(d), d.shape)
    bfd = np.isin(L, BFD)
    top = d >= np.percentile(d, 99)
    half = d >= 0.5 * d.max()
    # Distance peak -> barrel field, measured in iDISCO voxels (um), via inverse transform
    Ai = cv2.invertAffineTransform(np.array(A, np.float32))
    pm = Ai[:, :2] @ np.array([px, py]) + Ai[:, 2]
    bfd_mov = np.isin(proj["label"], BFD)
    dist_px = ndi.distance_transform_edt(~bfd_mov)
    inside_d = ndi.distance_transform_edt(bfd_mov)
    yy, xx = int(round(pm[1])), int(round(pm[0]))
    ok = 0 <= yy < dist_px.shape[0] and 0 <= xx < dist_px.shape[1]
    dist_um = float(dist_px[yy, xx] * VOX_UM) if ok else float("nan")
    depth_um = float(inside_d[yy, xx] * VOX_UM) if ok else float("nan")
    # centroid offset hotspot(half-max) vs bfd, in moving frame
    hy, hx = np.nonzero(half)
    hm = (Ai[:, :2] @ np.vstack([hx, hy])).T + Ai[:, 2]
    by, bx = np.nonzero(bfd_mov)
    off = (hm.mean(0) - np.array([bx.mean(), by.mean()])) * VOX_UM
    peak_label = int(L[py, px])
    return dict(peak_xy=[int(px), int(py)], peak_region=NAMES.get(area_id(peak_label), str(peak_label)),
                peak_in_bfd=bool(bfd[py, px]), peak_dist_to_bfd_um=dist_um, peak_depth_inside_bfd_um=depth_um,
                top1pct_in_bfd=float(bfd[top].mean()), halfmax_in_bfd=float(bfd[half].mean()),
                halfmax_centroid_minus_bfd_centroid_um=[float(off[0]), float(off[1])], L=L)
