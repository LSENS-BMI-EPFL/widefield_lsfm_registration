"""Cortical layer selection (config.LAYER) and the layer projection along the camera rays.

Layers: "surface", "l2/3", "l4", "l5", "l6". For a deeper layer, every visible pixel of the fitted surface projection
follows its viewing ray (the ray the camera integrates along) from the surface down and takes the area of the first
isocortex voxel at that layer or deeper, in the imaged hemisphere (config.HEMISPHERE). Taking "or deeper" means
areas that lack the layer (no L4 in motor, frontal, cingulate cortex...) give the area of their next deeper layer
instead of being skipped; the share of such pixels is reported. "surface" is the pipeline's own surface label
(shallowest atlas label 50-300 um below the surface).
"""
import re
import numpy as np

import config as C
import surface as S
import register as R
import validate as V

LAYERS = ("surface", "l2/3", "l4", "l5", "l6")
TAG = {"surface": "surface", "l2/3": "L2-3", "l4": "L4", "l5": "L5", "l6": "L6"}
RANK = {"1": 0, "2/3": 1, "2": 1, "3": 1, "4": 2, "5": 3, "6a": 4, "6b": 4}
TARGET = {"l2/3": 1, "l4": 2, "l5": 3, "l6": 4}
MAX_DEPTH_UM = 1500.0        # search the layer down to 1.5 mm along the ray
STEP_UM = 10.0
SURFACE_SAMPLE_UM = 50.0     # "surface": map the point 50 um below the visible surface (inside cortex)


def check(layer):
    layer = layer.lower()
    if layer not in LAYERS:
        raise ValueError(f"layer {layer!r}: use one of {LAYERS}")
    return layer


def rank_of(ids):
    """Layer rank per structure id (0 = L1 ... 4 = L6a/b); -1 for anything that is not an isocortex layer."""
    out = np.full(len(ids), -1, np.int8)
    for n, i in enumerate(ids):
        i = int(i)
        if 315 not in V.PATH.get(i, []):
            continue
        m = re.search(r"(2/3|6a|6b|1|2|3|4|5)$", V.NAMES.get(i, ""))
        if m and V.area_id(i) != i:
            out[n] = RANK[m.group(1)]
    return out


def layer_projection(brain, roll, pitch, layer):
    """Displayed-orientation maps (like surface.project) for one layer: structure id at the layer, the depth of that
    point below the surface along the ray (voxels; NaN where not found), and whether the area has the layer itself
    (False: taken from a deeper layer). Also returns the surface projection."""
    R.moving_for_view(brain, roll, pitch)
    prep = R._G[("prep", brain)]
    proj = S.project(prep, roll, pitch)
    if layer == "surface":
        lab = proj["label"]
        dep = np.where(proj["visible"], S.vox(SURFACE_SAMPLE_UM), np.nan).astype(np.float32)
        return proj, lab, dep, np.ones(lab.shape, bool)
    t = TARGET[layer]
    z = np.rot90(proj["z"], -1)                                    # back to the pre-rot90 grid used inside project()
    shape = np.array(prep["norm"].shape)
    c = shape / 2.0
    H, W = int(shape[1] * 1.15), int(shape[2] * 1.15)
    oy, ox = (H - shape[1]) // 2, (W - shape[2]) // 2
    Rm = S.rotation(roll, pitch)
    yy, xx = np.nonzero(np.isfinite(z))
    zs = z[yy, xx]
    anno, ids = prep["anno"], prep["ids"]
    rk = rank_of(ids)                                              # per anno index
    # the layer must be in the imaged hemisphere (if not set: where the ray enters the brain): at the medial edge,
    # tilted rays cross the fissure and would otherwise reach the layer in the other hemisphere
    import brain_axes
    hem = brain_axes.hemisphere_volume(brain)
    assert hem.shape == anno.shape
    hem0 = np.zeros(len(zs), np.uint8)
    # imaged hemisphere (config.HEMISPHERE; 1 left, 2 right): rays that enter the other hemisphere (window edge over
    # the sinus) get no layer point
    side = {"left": 1, "right": 2}.get(C.HEMISPHERE.get(brain))
    lab = np.zeros(len(zs), np.int64)
    dep = np.full(len(zs), np.nan, np.float32)
    exact = np.zeros(len(zs), bool)
    todo = np.ones(len(zs), bool)
    step = S.vox(STEP_UM)
    for d in np.arange(0, S.vox(MAX_DEPTH_UM) + step, step):
        k = np.nonzero(todo)[0]
        if not len(k):
            break
        q = np.stack([zs[k] + d, yy[k] - oy, xx[k] - ox], 1)
        p = np.round((q - c) @ Rm + c).astype(int)
        inb = np.all((p >= 0) & (p < np.array(anno.shape)), 1)
        li = np.zeros(len(k), np.int64)
        li[inb] = anno[p[inb, 0], p[inb, 1], p[inb, 2]]
        hv = np.zeros(len(k), np.uint8)
        hv[inb] = hem[p[inb, 0], p[inb, 1], p[inb, 2]]
        first = (hem0[k] == 0) & (hv > 0)
        hem0[k[first]] = hv[first]
        hit = (rk[li] >= t) & (li > 0) & (hv == (side if side else hem0[k]))
        lab[k[hit]], dep[k[hit]], exact[k[hit]] = ids[li[hit]], d, rk[li[hit]] == t
        todo[k[hit]] = False
    lmap = np.zeros(z.shape, np.int64)
    dmap = np.full(z.shape, np.nan, np.float32)
    emap = np.zeros(z.shape, bool)
    lmap[yy, xx], dmap[yy, xx], emap[yy, xx] = lab, dep, exact
    return proj, np.rot90(lmap), np.rot90(dmap), np.rot90(emap)
