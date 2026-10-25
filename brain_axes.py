"""Anatomical axes of the iDISCO brain, and conversion of projection view angles (code roll/pitch, which are
relative to the LSFM volume axes) into anatomical tilts.

ML axis: mean left->right vector between the centroids of bilateral atlas structures. AP: brain long axis,
orthogonalized to ML, pointing anterior (toward the olfactory bulb). DV = ML x AP, pointing dorsal.
Lateral tilt > 0: the window hemisphere's lateral side faces the camera. AP tilt > 0: the camera axis leans
toward anterior (the anterior cortex faces the camera more).
Usage: python brain_axes.py PB105 results/PB105/init.json
"""
import os
import sys
import json
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
import surface as S  # noqa: E402
import register as R  # noqa: E402


def unsigned_axes(brain):
    """ML, AP, DV axes and the ML pair spread; ML's sign (which hemisphere is under the window) is not set."""
    anno = np.load(f"{C.CACHE}/{brain}_anno20.npy")[::2, ::2, ::2]
    ids, idx = np.unique(anno, return_inverse=True)
    idx = idx.reshape(anno.shape).astype(np.int32)
    cnt = np.bincount(idx.ravel())
    boxes = ndi.find_objects(idx)                       # boxes[i - 1] bounds index i
    vecs = []
    for i in np.argsort(cnt[1:])[::-1][:150] + 1:       # 150 largest structures (index 0 = background)
        sl = boxes[i - 1]
        lab, n = ndi.label(idx[sl] == i)
        if n < 2:
            continue
        sz = np.bincount(lab.ravel())[1:]
        o = np.argsort(sz)[::-1]
        if sz[o[1]] < 0.6 * sz[o[0]]:                   # need two similar halves (left / right)
            continue
        c = [np.argwhere(lab == o[k] + 1).mean(0) for k in (0, 1)]
        v = c[0] - c[1]
        vecs.append(v / np.linalg.norm(v))
    V = np.array(vecs)
    V *= np.sign(V @ V[0])[:, None]
    ML = V.mean(0) / np.linalg.norm(V.mean(0))
    pts = np.argwhere(anno > 0).astype(float)
    cen = pts.mean(0)
    AP = np.linalg.eigh(np.cov((pts - cen).T))[1][:, -1]
    AP -= (AP @ ML) * ML
    AP /= np.linalg.norm(AP)
    atlas = json.load(open(f"{HERE}/atlas_ids.json"))
    mob = [int(k) for k, p in atlas["path"].items() if 507 in p]          # main olfactory bulb
    AP *= np.sign((np.argwhere(np.isin(anno, mob)).mean(0) - cen) @ AP)
    DV = np.cross(ML, AP)
    DV *= np.sign(DV @ np.array([-1.0, 0, 0]))      # viewer at low z sees the dorsal surface
    spread = np.median(np.degrees(np.arccos(np.clip(V @ ML, -1, 1))))
    return dict(ML=ML, AP=AP, DV=DV, ml_pair_spread_deg=spread), anno, cen


def hemisphere_volume(brain):
    """brainreg's registered_hemispheres.tiff (1 = left, 2 = right) in the sample orientation of the cache (20 um),
    oriented like downsampled.tiff in load_cache.py."""
    import tifffile
    reg = tifffile.imread(f"{C.DATA}/{brain}/registration/registered_hemispheres.tiff")
    return np.transpose(reg[::-1, ::-1, :], (1, 0, 2))[::-1]


def hemisphere_sign(brain, ML):
    """+1 if ML points into the imaged hemisphere (config.HEMISPHERE), -1 if not, None if it is not set."""
    side = C.HEMISPHERE.get(brain)
    if side is None:
        return None
    hem = hemisphere_volume(brain)
    pts = np.argwhere(hem > 0)
    on = hem[tuple(pts.T)] == {"left": 1, "right": 2}[side]
    proj = (pts - pts.mean(0)) @ ML
    return 1 if proj[on].mean() > 0 else -1


def axes(brain, A_ref=None, view_ref=None):
    """A_ref/view_ref: any transform that places the window, used to find which hemisphere is imaged."""
    path = f"{C.RESULTS}/{brain}/brain_axes.json"
    if os.path.exists(path):
        return {k: np.array(v) for k, v in json.load(open(path)).items()}
    ax, anno, cen = unsigned_axes(brain)
    ML = ax["ML"]
    # Window hemisphere: 3D surface point under the window centre.
    fx = R.fixed_for(brain)
    wy, wx = np.argwhere(fx["mask"]).mean(0)
    Ai = np.linalg.inv(np.vstack([A_ref, [0, 0, 1]]))[:2]
    px, py = Ai[:, :2] @ [wx, wy] + Ai[:, 2]
    prep = S.prepare(brain)
    proj = S.project(prep, *view_ref)
    shape = np.array(prep["norm"].shape)
    H, W = int(shape[1] * 1.15), int(shape[2] * 1.15)
    oy, ox = (H - shape[1]) // 2, (W - shape[2]) // 2
    r0, c0 = int(round(py)), int(round(px))
    orig = np.unravel_index(np.rot90(np.arange(H * W).reshape(H, W))[r0, c0], (H, W))
    q = np.array([proj["z"][r0, c0], orig[0] - oy, orig[1] - ox])
    p3 = ((q - shape / 2) @ S.rotation(*view_ref) + shape / 2) * np.array(anno.shape) / shape
    out = dict(ax, ML=ML * np.sign((p3 - cen) @ ML))
    json.dump({k: np.asarray(v).tolist() for k, v in out.items()}, open(path, "w"), indent=1)
    return out


def anatomical(ax, roll, pitch):
    """(lateral tilt, AP tilt) in degrees for a code (roll, pitch) view. ax['ML'] points to the window side."""
    up = -S.rotation(roll, pitch)[0]
    up_ = up - (up @ ax["AP"]) * ax["AP"]
    return (float(np.degrees(np.arctan2(up_ @ ax["ML"], up_ @ ax["DV"]))),
            float(np.degrees(np.arcsin(np.clip(up @ ax["AP"], -1, 1)))))


def code_angles(ax, lat, ap):
    """Inverse of anatomical(): code (roll, pitch) whose view has the given anatomical tilts (deg)."""
    from scipy.optimize import minimize
    la, aa = np.radians(lat), np.radians(ap)
    u = np.sin(aa) * ax["AP"] + np.cos(aa) * (np.cos(la) * ax["DV"] + np.sin(la) * ax["ML"])
    f = lambda q: float(((-S.rotation(*q)[0]) - u) @ ((-S.rotation(*q)[0]) - u))
    grid = min(((r, p) for r in range(-90, 91, 10) for p in range(-90, 91, 10)), key=f)   # either hemisphere
    best = min((minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-12)) for x0 in
                ([0, 0], [lat - 30, -ap], [lat - 30, ap], grid)), key=lambda r: r.fun)
    assert best.fun < 1e-6, (lat, ap, best.fun)
    return float(best.x[0]), float(best.x[1])


if __name__ == "__main__":
    t = json.load(open(sys.argv[2]))
    A, view = np.array(t["A_projection_to_widefield"]), (t["roll_deg"], t["pitch_deg"])
    ax = axes(sys.argv[1], A, view)
    lat, ap = anatomical(ax, *view)
    print(f"code roll {view[0]}, pitch {view[1]} -> anatomical lateral tilt {lat:.1f} deg, AP tilt {ap:.1f} deg")
