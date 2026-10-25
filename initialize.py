"""Starting transform from the data alone: global vessel search over hemisphere, view, rotation, scale and translation.

The fit in refine_local.py is local, so it needs a starting transform. This finds one from the LSFM surface vessels and
the widefield vessels only (no manual alignment, no other brain):
  1. Views: both hemispheres (sign of the brain's ML axis) x a coarse grid of anatomical tilts (brain_axes.py).
  2. Coarse: for each view, every in-plane rotation (full circle) x scale (bounded, SCALE_RANGE); translation by masked template matching,
     several peaks per rotation/scale. Score = masked vessel NCC with the window-coverage penalty (register.score).
  3. Refine: the best coarse candidates of every view are refined (Powell, 2D affine) on the same objective as the main
     fit: vessel NCC - imaging-angle prior.
The best candidate is written as the start for refine_local.py; its placement also tells brain_axes.py which hemisphere
is under the window.
Only the hemisphere in config.HEMISPHERE is searched when it is set; the scale bound comes from config.WF_PIXEL_UM.
Usage: python initialize.py PB105 [--lat 15:45:5] [--ap -15:15:5] [--view-prior lat0:ap0:sl:sa]
       (defaults: config.PRIORS)
Writes results/<brain>/init.json.
"""
import os
import sys
import json
import argparse
from multiprocessing import Pool

import numpy as np
import cv2
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
import register as R  # noqa: E402
import refine_local as RL  # noqa: E402
import brain_axes  # noqa: E402
import surface  # noqa: E402

THETAS = np.arange(-180, 180, 6)            # in-plane rotation (deg), full circle


def scale_range():
    """Bounds on the overall scale, projection px (LSFM, surface.VOX_MM) -> widefield px. The camera magnification is
    fixed, so with a calibrated widefield pixel only the clearing shrinkage is free (config.CLEARING_SCALE).
    Unbounded, the fit inflates the scale (~1.45) and matches a few large vessels of the wrong hemisphere."""
    if C.WF_PIXEL_UM is None:
        return tuple(C.SCALE_RANGE_UNCALIBRATED)
    lsfm_um = surface.VOX_MM * 1000
    return tuple(lsfm_um / (c * C.WF_PIXEL_UM) for c in sorted(C.CLEARING_SCALE, reverse=True))


SCALE_RANGE = scale_range()
SCALES = np.arange(SCALE_RANGE[0], SCALE_RANGE[1] + 1e-6, 0.05)
RL.SCALE_RANGE = SCALE_RANGE                # bound in the refinement too (refine_local.penalty)
PEAKS = 5                                   # translation peaks tried per rotation/scale
KEEP = 4                                    # distinct coarse candidates refined per view


def coarse(job):
    brain, view = job["brain"], job["view"]
    proj, e = R.moving_for_view(brain, *view)
    fx = R.fixed_for(brain)
    M, Mm = e["ridge"] * e["mask"], e["mask"]
    F, Fm = R.blur(fx["ridge"], R.COARSE_SIGMA), fx["mask"]
    Mb = R.blur(M, R.COARSE_SIGMA)
    H, W = M.shape
    cm = (W / 2, H / 2)
    CH, CW = H + R.FIX_SHAPE[0], W + R.FIX_SHAPE[1]
    cc = (CW / 2, CH / 2)
    cands = []
    for th in THETAS:
        for s in SCALES:
            A0 = R.base_affine(float(th), float(s), cm, cc)
            canvas = cv2.warpAffine(Mb.astype(np.float32), A0, (CW, CH))
            r = cv2.matchTemplate(canvas, F.astype(np.float32), cv2.TM_CCOEFF_NORMED, mask=Fm.astype(np.float32))
            r = np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)
            for _ in range(PEAKS):
                _, v, _, (x, y) = cv2.minMaxLoc(r)
                if v <= -1:
                    break
                A = A0.copy()
                A[:, 2] -= (x, y)
                cands.append((R.score(F, Fm, Mb, A, Mm), float(th), float(s), A))
                r[max(y - 6, 0):y + 7, max(x - 6, 0):x + 7] = -1
    cands.sort(key=lambda c: -c[0])
    keep = []
    for c in cands:                                  # distinct placements: rotation or window-centre position differs
        ctr = c[3][:, :2] @ np.array(cm) + c[3][:, 2]
        if all(abs((c[1] - k[1] + 180) % 360 - 180) > 10 or np.hypot(*(ctr - k[4])) > 15 for k in keep):
            keep.append((*c, ctr))
        if len(keep) == KEEP:
            break
    return [dict(job, coarse=k[0], theta=k[1], scale=k[2], A=k[3].tolist()) for k in keep]


def refine(c):
    brain, view = c["brain"], c["view"]
    proj, e = R.moving_for_view(brain, *view)
    fx = R.fixed_for(brain)
    M, Mm, F, Fm = e["ridge"] * e["mask"], e["mask"], fx["ridge"], fx["mask"]
    A = np.array(c["A"])
    p = np.zeros(6)
    for sig in R.FINE_SIGMAS:
        Fb, Mb = R.blur(F, sig), R.blur(M, sig)
        def fun(q):
            Aq = R.compose(R.perturb(q, "affine"), A)
            return -R.score(Fb, Fm, Mb, Aq, Mm) + RL.penalty(q, "affine", A)
        p = minimize(fun, p, method="Powell", options=dict(xtol=1e-3, ftol=1e-5, maxiter=4000)).x
    Af = R.compose(R.perturb(p, "affine"), A)
    ncc = R.score(R.blur(F, 1.2), Fm, R.blur(M, 1.2), Af, Mm)
    return dict(c, A=Af.tolist(), score=ncc, coverage=R.coverage(Fm, Mm, Af),
                objective=ncc - c["prior_penalty"], decomposition=R.describe(Af))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brain")
    ap.add_argument("--lat", default="15:45:5", help="anatomical lateral tilt grid lo:hi:step (deg)")
    ap.add_argument("--ap", default="-15:15:5", help="anatomical AP tilt grid lo:hi:step (deg)")
    ap.add_argument("--view-prior", default=RL.PRIOR_DEFAULT, help="lat0:ap0:sigma_lat:sigma_ap (deg)")
    ap.add_argument("--view-prior-weight", type=float, default=C.PRIORS["weight"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    b = args.brain
    os.makedirs(f"{C.RESULTS}/{b}", exist_ok=True)
    frng = lambda t: np.arange(float(t.split(":")[0]), float(t.split(":")[1]) + 1e-6, float(t.split(":")[2]))
    l0, a0, sl, sa = (float(x) for x in args.view_prior.split(":"))
    ax0, _, _ = brain_axes.unsigned_axes(b)
    known = brain_axes.hemisphere_sign(b, ax0["ML"])
    jobs = []
    for sign in ((known,) if known else (1, -1)):
        ax = dict(ax0, ML=sign * ax0["ML"])
        for lat in frng(args.lat):
            for apt in frng(args.ap):
                pen = args.view_prior_weight * (((lat - l0) / sl) ** 2 + ((apt - a0) / sa) ** 2)
                jobs.append(dict(brain=b, ml_sign=sign, lat=float(lat), ap=float(apt),
                                 view=brain_axes.code_angles(ax, lat, apt), prior_penalty=float(pen)))
    R.moving_for_view(b, 0, 0)                       # load the volume once; workers share it (fork)
    R.fixed_for(b)
    print(f"[{b}] coarse: {len(jobs)} views x {len(THETAS)} rotations x {len(SCALES)} scales", flush=True)
    with Pool(min(args.workers, len(jobs))) as pool:
        cands = [c for cs in pool.map(coarse, jobs) for c in cs]
        print(f"[{b}] refine: {len(cands)} candidates", flush=True)
        ref = pool.map(refine, cands)
    ref.sort(key=lambda r: -r["objective"])
    best = ref[0]
    for r in ref[:10]:
        d = r["decomposition"]
        print(f"   obj {r['objective']:.4f} ncc {r['score']:.3f}  ML sign {r['ml_sign']:+d} "
              f"lat {r['lat']:4.0f} ap {r['ap']:4.0f}  rot {d['rotation_deg']:6.1f} sx {d['scale_x']:.3f} sy {d['scale_y']:.3f} "
              f"(coarse theta {r['theta']:.0f}, scale {r['scale']:.2f}, ncc {r['coarse']:.3f})")
    out = dict(brain=b, roll_deg=best["view"][0], pitch_deg=best["view"][1], A_projection_to_widefield=best["A"],
               ml_sign=best["ml_sign"], anatomical_lateral_tilt_deg=best["lat"], anatomical_ap_tilt_deg=best["ap"],
               vessel_ncc=best["score"], objective=best["objective"],
               decomposition=best["decomposition"], view_prior=args.view_prior,
               scale_range=SCALE_RANGE, hemisphere=None if known is None else C.HEMISPHERE[b],
               candidates=[{k: r[k] for k in ("ml_sign", "lat", "ap", "score", "coverage", "prior_penalty",
                                               "objective", "coarse", "theta", "scale", "decomposition")} for r in ref])
    json.dump(out, open(f"{C.RESULTS}/{b}/init.json", "w"), indent=1)


if __name__ == "__main__":
    main()
