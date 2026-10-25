"""Shared helpers for the vessel fit: vessel maps for a view, masked NCC score, 2D affine parametrization.

The wS1 response is NOT used for fitting; it is evaluated afterwards (validate.py).
"""
import os
import sys

import numpy as np
import cv2
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import surface  # noqa: E402
import enhance  # noqa: E402

FIX_SHAPE = (250, 320)
COARSE_SIGMA = 2.0
FINE_SIGMAS = (2.0, 1.2)

_G = {}


def blur(x, s):
    return ndi.gaussian_filter(x, s) if s > 0 else x


def moving_for_view(brain, roll, pitch):
    prep = _G.get(("prep", brain))
    if prep is None:
        prep = _G[("prep", brain)] = surface.prepare(brain)
    proj = surface.project(prep, roll, pitch)
    e = enhance.idisco(proj)
    return proj, e


def fixed_for(brain):
    f = _G.get(("fix", brain))
    if f is None:
        f = _G[("fix", brain)] = enhance.widefield(brain)
    return f


def warp(img, A, interp=cv2.INTER_LINEAR):
    return cv2.warpAffine(img.astype(np.float32), A.astype(np.float32), FIX_SHAPE[::-1], flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


MIN_COVERAGE = 0.85


def score(F, Fm, M, A, Mm=None):
    """Masked NCC over the widefield window; moving outside its own mask counts as 'no vessel'.
    With Mm (moving cortex mask), penalize transforms where the iDISCO cortex does not cover the
    window (the window shows cortex everywhere; otherwise one lucky vessel can score highly)."""
    W = warp(M, A)[Fm]
    f = F[Fm]
    f = f - f.mean()
    W = W - W.mean()
    d = np.sqrt((f * f).sum() * (W * W).sum())
    ncc = float((f * W).sum() / d) if d > 0 else 0.0
    if Mm is not None:
        cov = float((warp(Mm.astype(np.float32), A) > 0.5)[Fm].mean())
        ncc -= 2.0 * max(0.0, MIN_COVERAGE - cov)
    return ncc


def coverage(Fm, Mm, A):
    return float((warp(Mm.astype(np.float32), A) > 0.5)[Fm].mean())


def base_affine(theta, s, cm, cc):
    """Rotate by theta (deg) and scale s about moving center cm, put cm at cc."""
    A = cv2.getRotationMatrix2D((float(cm[0]), float(cm[1])), theta, s)
    A[:, 2] += np.array(cc) - np.array(cm)
    return A


def compose(P, A):
    """P (2x3) applied after A (2x3)."""
    P3 = np.vstack([P, [0, 0, 1]])
    A3 = np.vstack([A, [0, 0, 1]])
    return (P3 @ A3)[:2]


def perturb(p, mode):
    """Affine perturbation about the fixed-image center. p = tx, ty, theta, log sx, log sy, shear."""
    tx, ty, th, lsx, lsy, k = p
    if mode == "similarity":
        lsy, k = lsx, 0.0
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    L = np.array([[c, -s], [s, c]]) @ np.diag([np.exp(lsx), np.exp(lsy)]) @ np.array([[1, k], [0, 1]])
    cf = np.array([FIX_SHAPE[1] / 2, FIX_SHAPE[0] / 2])
    t = cf - L @ cf + np.array([tx, ty])
    return np.hstack([L, t[:, None]])


def describe(A):
    L = np.array(A)[:, :2]
    U, S, Vt = np.linalg.svd(L)
    rot = np.degrees(np.arctan2(L[1, 0], L[0, 0]))
    # Decompose L = R(theta) * [[sx, k*sx... ]] via QR-like: L = R @ U_upper
    q, r = np.linalg.qr(L)
    sgn = np.sign(np.diag(r))
    q, r = q * sgn, (r.T * sgn).T
    return dict(rotation_deg=float(np.degrees(np.arctan2(q[1, 0], q[0, 0]))), scale_x=float(r[0, 0]),
                scale_y=float(r[1, 1]), shear=float(r[0, 1] / r[1, 1]), singular=S.tolist())
