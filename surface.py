"""Surface-following projection of the iDISCO volume for an arbitrary viewing direction.

prepare(brain): brain mask -> boundary points; intensity normalized by local tissue level.
project(prep, roll, pitch): for every pixel of a view rotated by (roll, pitch), find the
    visible surface (z-buffer of boundary points, smoothed to remove voxel terracing) and
    sample the volume with trilinear interpolation at continuous depths below it.
Output grid: 1 px = 1 voxel = 25.3 um, rotated 90 deg CCW like the pipeline's aligned_bv.
"""
import os
import numpy as np
from scipy import ndimage as ndi
import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
VOX_MM = 0.02532
DEPTHS = np.arange(0.5, 3.01, 0.25)     # voxels below surface for vessel signal (~13-75 um; vessels fade below ~100 um)
VESSEL_Z_SIGMA = 3.0                    # extra surface smoothing for vessel sampling (removes voxel-terrace rings)
LABEL_DEPTHS = np.arange(2, 12.1, 1.0)  # voxels below surface for the cortical label (~50-300 um)


def prepare(brain):
    cache_pts = f"{C.CACHE}/{brain}_boundary.npy"
    cache_norm = f"{C.CACHE}/{brain}_norm_f16.npy"
    if not (os.path.exists(cache_pts) and os.path.exists(cache_norm)):
        v = np.load(f"{C.CACHE}/{brain}_lsfm25.npy").astype(np.float32)
        frame = v > 0
        anno_idx = np.load(f"{C.CACHE}/{brain}_anno20_idx.npy", mmap_mode="r")
        zf = np.array(v.shape) / np.array(anno_idx.shape)
        inside = ndi.zoom(np.asarray(anno_idx), zf, order=0)[: v.shape[0], : v.shape[1], : v.shape[2]] > 0
        sm = ndi.gaussian_filter(v, 1.5)
        t_in, t_out = np.median(sm[inside]), np.median(sm[frame & ~inside])
        mask = (sm > 0.5 * (t_in + t_out)) & frame
        mask = ndi.binary_opening(mask, iterations=2)
        lab, _ = ndi.label(mask)
        mask = ndi.binary_fill_holes(lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1))
        depth = ndi.distance_transform_edt(mask)
        w = ndi.gaussian_filter(np.where(mask, v, 0), 6)
        wn = ndi.gaussian_filter(mask.astype(np.float32), 6)
        norm = np.where(mask, v / np.maximum(w / np.maximum(wn, 1e-3), 1), 0)
        # written via temp files + rename, so a reader never sees a half-written cache
        for path, arr in ((cache_norm, norm.astype(np.float16)), (cache_pts, np.argwhere(mask & (depth <= 1.5)).astype(np.float32))):
            np.save(path + ".tmp.npy", arr)
            os.replace(path + ".tmp.npy", path)
    return dict(pts=np.load(cache_pts), norm=np.load(cache_norm).astype(np.float32),
                anno=np.load(f"{C.CACHE}/{brain}_anno20_idx.npy", mmap_mode="r"),
                ids=np.load(f"{C.CACHE}/{brain}_anno20_ids.npy"))


def rotation(roll, pitch):
    """Rotation of (z, y, x) coords. roll: about AP (axis1, tilts z-x), pitch: about ML (axis2, tilts z-y)."""
    r, p = np.radians(roll), np.radians(pitch)
    Rroll = np.array([[np.cos(r), 0, -np.sin(r)], [0, 1, 0], [np.sin(r), 0, np.cos(r)]])
    Rpitch = np.array([[np.cos(p), -np.sin(p), 0], [np.sin(p), np.cos(p), 0], [0, 0, 1]])
    return Rpitch @ Rroll


def _nan_smooth(z, s):
    ok = np.isfinite(z)
    num = ndi.gaussian_filter(np.where(ok, z, 0), s)
    den = ndi.gaussian_filter(ok.astype(np.float32), s)
    return np.where(ok, num / np.maximum(den, 1e-6), np.nan)


def project(prep, roll=0.0, pitch=0.0):
    shape = np.array(prep["norm"].shape)
    c = shape / 2.0
    R = rotation(roll, pitch)
    q = (prep["pts"] - c) @ R.T + c
    H, W = int(shape[1] * 1.15), int(shape[2] * 1.15)
    oy, ox = (H - shape[1]) // 2, (W - shape[2]) // 2
    iy = np.round(q[:, 1]).astype(int) + oy
    ix = np.round(q[:, 2]).astype(int) + ox
    ok = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
    zbuf = np.full(H * W, np.inf, np.float32)
    np.minimum.at(zbuf, iy[ok] * W + ix[ok], q[ok, 0])
    z = zbuf.reshape(H, W)
    z[~np.isfinite(z)] = np.nan
    visible = np.isfinite(z)
    # Fill single-pixel holes, then smooth away voxel terracing.
    z = _nan_smooth(z, 1.0)
    z = np.where(visible | np.isfinite(z), z, np.nan)
    z = _nan_smooth(z, 1.5)
    visible = np.isfinite(z)

    yy, xx = np.nonzero(visible)
    zs = z[visible]
    zs_v = _nan_smooth(z, VESSEL_Z_SIGMA)[visible]

    def to_vol(d, zz=zs):
        qq = np.stack([zz + d, yy - oy, xx - ox], 1).astype(np.float32)
        return (qq - c) @ R + c                     # inverse rotation (R orthonormal)

    acc = np.zeros(len(zs), np.float32)
    for d in DEPTHS:
        acc += ndi.map_coordinates(prep["norm"], to_vol(d, zs_v).T, order=1, mode="constant", cval=0)
    vmap = np.full((H, W), np.nan, np.float32)
    vmap[visible] = acc / len(DEPTHS)

    # Label: shallowest non-zero atlas label within LABEL_DEPTHS below the surface.
    anno = prep["anno"]
    scale = np.array(anno.shape) / shape
    lab = np.zeros(len(zs), np.int64)
    for d in LABEL_DEPTHS[::-1]:
        p = np.round(to_vol(d) * scale).astype(int)
        inb = np.all((p >= 0) & (p < np.array(anno.shape)), 1)
        li = np.zeros(len(zs), np.int64)
        li[inb] = anno[p[inb, 0], p[inb, 1], p[inb, 2]]
        lab = np.where(li > 0, prep["ids"][li], lab)
    lmap = np.zeros((H, W), np.int64)
    lmap[visible] = lab
    return dict(vessel=np.rot90(vmap), label=np.rot90(lmap), visible=np.rot90(visible), z=np.rot90(z))


def transfer_affine(prep, A_ref, view_ref, view, fixed_mask, step=4):
    """Carry a projection->widefield affine fitted at view_ref over to another view: surface points under the
    widefield window are lifted to 3D at view_ref, re-projected at view, and the affine is re-fitted to the
    same widefield positions (accounts for the shift, rotation and foreshortening caused by the new view)."""
    ref = project(prep, *view_ref)
    z = ref["z"]
    shape = np.array(prep["norm"].shape)
    c = shape / 2.0
    H, W = int(shape[1] * 1.15), int(shape[2] * 1.15)
    oy, ox = (H - shape[1]) // 2, (W - shape[2]) // 2
    Hd = z.shape[0]                                         # displayed rows = pre-rot90 columns
    wy, wx = np.nonzero(fixed_mask[::step, ::step])
    wy, wx = wy * step, wx * step
    Ai = np.linalg.inv(np.vstack([A_ref, [0, 0, 1]]))[:2]
    xd, yd = Ai[:, :2] @ np.vstack([wx, wy]) + Ai[:, 2:3]
    r, col = np.round(yd).astype(int), np.round(xd).astype(int)
    ok = (r >= 0) & (r < z.shape[0]) & (col >= 0) & (col < z.shape[1])
    r, col, wx, wy = r[ok], col[ok], wx[ok], wy[ok]
    zz = z[r, col]
    ok = np.isfinite(zz)
    r, col, wx, wy, zz = r[ok], col[ok], wx[ok], wy[ok], zz[ok]
    # displayed (r, col) -> pre-rot90 (y, x) = (col, Hd-1-r); -> volume coords at view_ref
    q = np.stack([zz, col - oy, (Hd - 1 - r) - ox], 1)
    p3 = (q - c) @ rotation(*view_ref) + c
    q2 = (p3 - c) @ rotation(*view).T + c                    # same 3D points seen from the new view
    yp, xp = q2[:, 1] + oy, q2[:, 2] + ox
    X = np.stack([yp, Hd - 1 - xp, np.ones(len(yp))], 1)     # displayed (x=col, y=row) at the new view
    sol, *_ = np.linalg.lstsq(X, np.stack([wx, wy], 1), rcond=None)
    return sol.T                                            # 2x3
