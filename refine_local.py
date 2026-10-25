"""Local registration around a starting transform, over 3D view angles and the 2D affine.

Usage: python refine_local.py PB105 --init results/PB105/init.json [--lat=24:36:2] [--ap=-20:20:2]
The objective is vessel NCC - imaging-angle prior (config.PRIORS).
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
import brain_axes  # noqa: E402
import surface  # noqa: E402

TRANS_WIN = 25   # px around the initial translation (None = free)
ANISO_LIM = 0.15


SCALE_RANGE = None   # (lo, hi) bounds on overall scale sqrt(|det|) of the final transform (set by initialize.py)
PRIOR_DEFAULT = "{lat:g}:{ap:g}:{sigma_lat:g}:{sigma_ap:g}".format(**C.PRIORS)


def penalty(p, mode, A=None):
    pen = 0.0
    if mode == "affine":
        pen += max(0.0, abs(p[3] - p[4]) - ANISO_LIM) + max(0.0, abs(p[5]) - ANISO_LIM)
    if SCALE_RANGE is not None and A is not None:
        s = np.sqrt(abs(np.linalg.det(R.compose(R.perturb(p, mode), A)[:, :2])))
        pen += max(0.0, np.log(SCALE_RANGE[0] / s)) + max(0.0, np.log(s / SCALE_RANGE[1]))
    return 5.0 * pen


def local_search(M, F, Fm, A_init, trans_win, thetas, scales, sig=2.0, Mm=None):
    """Search rotation/scale perturbations of A_init (about the fixed center); translation by masked NCC."""
    Mb, Fb = R.blur(M, sig), R.blur(F, sig)
    H, W = M.shape
    cm = np.array([W / 2, H / 2])
    # where does the moving center land under A_init?
    c_fix = np.array(A_init)[:, :2] @ cm + np.array(A_init)[:, 2]
    L0 = np.array(A_init)[:, :2]
    CH, CW = H + R.FIX_SHAPE[0], W + R.FIX_SHAPE[1]
    cc = np.array([CW / 2, CH / 2])
    best = (-np.inf, None)
    for th in thetas:
        for s in scales:
            c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
            L = s * np.array([[c, -sn], [sn, c]]) @ L0
            A0 = np.hstack([L, (cc - L @ cm)[:, None]])        # moving center -> canvas center
            canvas = cv2.warpAffine(Mb.astype(np.float32), A0, (CW, CH))
            r = cv2.matchTemplate(canvas, Fb.astype(np.float32), cv2.TM_CCOEFF_NORMED, mask=Fm.astype(np.float32))
            r = np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)
            if trans_win is not None:
                # template top-left (x,y) => moving center lands at cc - (x,y) in fixed coords
                ex, ey = (cc - c_fix).astype(int)
                win = np.full_like(r, -1)
                y0, y1 = max(ey - trans_win, 0), min(ey + trans_win + 1, r.shape[0])
                x0, x1 = max(ex - trans_win, 0), min(ex + trans_win + 1, r.shape[1])
                win[y0:y1, x0:x1] = r[y0:y1, x0:x1]
                r = win
            _, v, _, (x, y) = cv2.minMaxLoc(r)
            A = A0.copy()
            A[:, 2] -= (x, y)
            sc = R.score(Fb, Fm, Mb, A, Mm)
            if sc > best[0]:
                best = (sc, A)
            if trans_win is None:
                # free translation: NCC peak may be a degenerate placement; also try the best
                # placements that satisfy coverage, from the top-k response peaks
                rr = r.copy()
                for _ in range(15):
                    _, v2, _, (x2, y2) = cv2.minMaxLoc(rr)
                    if v2 <= -1:
                        break
                    A2 = A0.copy()
                    A2[:, 2] -= (x2, y2)
                    sc2 = R.score(Fb, Fm, Mb, A2, Mm)
                    if sc2 > best[0]:
                        best = (sc2, A2)
                    rr[max(y2 - 6, 0):y2 + 7, max(x2 - 6, 0):x2 + 7] = -1
    return best


def fit_view(args):
    brain, roll, pitch, A_init, trans_win, thetas, scales = args[:7]
    view_ref = args[7] if len(args) > 7 else None
    proj, e = R.moving_for_view(brain, roll, pitch)
    fx = R.fixed_for(brain)
    if view_ref is not None and tuple(view_ref) != (roll, pitch):
        # start from the reference transform carried over to this view in 3D (see surface.transfer_affine)
        A_init = surface.transfer_affine(R._G[("prep", brain)], np.array(A_init), view_ref, (roll, pitch), fx["mask"])
    M = e["ridge"] * e["mask"]
    F, Fm = fx["ridge"], fx["mask"]
    Mm = e["mask"]
    sc0, A = local_search(M, F, Fm, A_init, trans_win, thetas, scales, Mm=Mm)
    out = dict(roll=roll, pitch=pitch, coarse=sc0)
    for mode in ("similarity", "affine"):
        p = np.zeros(6)
        for sig in R.FINE_SIGMAS:
            Fb, Mb = R.blur(F, sig), R.blur(M, sig)
            def fun(q):
                Aq = R.compose(R.perturb(q, mode), A)
                return -R.score(Fb, Fm, Mb, Aq, Mm) + penalty(q, mode, A)
            p = minimize(fun, p, method="Powell", options=dict(xtol=1e-3, ftol=1e-5, maxiter=4000)).x
        Af = R.compose(R.perturb(p, mode), A)
        out[mode] = dict(score=R.score(R.blur(F, 1.2), Fm, R.blur(M, 1.2), Af),
                         coverage=R.coverage(Fm, Mm, Af),
                         score_s2=R.score(R.blur(F, 2.0), Fm, R.blur(M, 2.0), Af),
                         A=Af.tolist(), decomposition=R.describe(Af))
        out[mode]["objective"] = out[mode]["score"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brain")
    ap.add_argument("--init", required=True, help="json with roll_deg, pitch_deg, A_projection_to_widefield "
                                                   "(initialize.py output, or a previous <brain>_transform.json)")
    ap.add_argument("--lat", default="24:36:2", help="anatomical lateral tilt grid lo:hi:step (deg); needs "
                                                      "results/<brain>/brain_axes.json")
    ap.add_argument("--ap", default="-20:20:2", help="anatomical AP tilt grid lo:hi:step (deg)")
    ap.add_argument("--view-prior", default=PRIOR_DEFAULT,
                    help="lat0:ap0:sigma_lat:sigma_ap (deg): expected imaging angle; each view's objective is reduced "
                         "by w * (((lat-lat0)/sigma_lat)^2 + ((ap-ap0)/sigma_ap)^2)")
    ap.add_argument("--view-prior-weight", type=float, default=C.PRIORS["weight"], help="w above, in NCC units")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    b = args.brain
    out = os.path.join(C.RESULTS, b)
    os.makedirs(out, exist_ok=True)

    t0 = json.load(open(args.init))
    A_init, view_ref = t0["A_projection_to_widefield"], (t0["roll_deg"], t0["pitch_deg"])
    trans_win, thetas, scales = TRANS_WIN, np.arange(-8, 8.1, 2), np.arange(0.92, 1.081, 0.02)

    frng = lambda t: np.arange(float(t.split(":")[0]), float(t.split(":")[1]) + 1e-6, float(t.split(":")[2]))
    axes_ = brain_axes.axes(b)
    rows, cols = list(frng(args.lat)), list(frng(args.ap))
    grid_views = {(la, aa): brain_axes.code_angles(axes_, la, aa) for la in rows for aa in cols}
    row_name, col_name = "lateral tilt", "AP tilt"
    keys = list(grid_views)
    l0, a0, sl, sa = (float(x) for x in args.view_prior.split(":"))
    prior = dict(lat0=l0, ap0=a0, sigma_lat=sl, sigma_ap=sa, weight=args.view_prior_weight)
    pen = {k: args.view_prior_weight * (((k[0] - l0) / sl) ** 2 + ((k[1] - a0) / sa) ** 2) for k in keys}
    jobs = [(b, *grid_views[k], A_init, trans_win, thetas, scales, view_ref) for k in keys]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(fit_view, jobs)
        # Pass 2: re-fit every view starting from the best pass-1 solution (carried over in 3D); keep the
        # better start per view, so neighbouring views are not stuck in a worse local optimum.
        b1 = res[int(np.argmax([r["affine"]["objective"] - pen[k] for k, r in zip(keys, res)]))]
        jobs2 = [(b, *grid_views[k], b1["affine"]["A"], trans_win, thetas, scales, (b1["roll"], b1["pitch"]))
                 for k in keys]
        res2 = pool.map(fit_view, jobs2)
        for r1, r2 in zip(res, res2):
            for m in ("similarity", "affine"):
                if r2[m]["objective"] > r1[m]["objective"]:
                    r1[m] = r2[m]
    for k, r in zip(keys, res):
        r["grid"] = list(k)
        for m in ("similarity", "affine"):        # expected imaging angle (soft prior), constant per view
            r[m]["view_prior_penalty"] = pen[k]
            r[m]["objective"] -= pen[k]
    # Choose the view by the objective averaged over each view and its grid neighbours (3x3), so an isolated
    # lucky fit does not decide the view; report that view's own fit.
    look = {tuple(r["grid"]): r for r in res}
    def smoothed(r, m):
        i, j = rows.index(r["grid"][0]), cols.index(r["grid"][1])
        nb = [look[(rows[a], cols[c])][m]["objective"] for a in range(max(i - 1, 0), min(i + 2, len(rows)))
              for c in range(max(j - 1, 0), min(j + 2, len(cols)))]
        return float(np.mean(nb))
    for r in res:
        for m in ("similarity", "affine"):
            r[m]["objective_smoothed"] = smoothed(r, m)
    best = {m: max(res, key=lambda r: r[m]["objective_smoothed"]) for m in ("similarity", "affine")}
    summary = {m: dict(roll=best[m]["roll"], pitch=best[m]["pitch"], grid=best[m]["grid"], **best[m][m]) for m in best}
    json.dump(dict(views=res, best=summary, init=A_init, view_prior=prior, grid_axes=[row_name, col_name],
                   grid_rows=rows, grid_cols=cols), open(f"{out}/local.json", "w"), indent=1)
    for m in ("similarity", "affine"):
        s = summary[m]
        print(f"[{b}] {m:10s} {row_name}={s['grid'][0]:g} {col_name}={s['grid'][1]:g} (code roll={s['roll']:.1f} "
              f"pitch={s['pitch']:.1f}) score={s['score']:.4f} (s2 {s['score_s2']:.4f}) "
              f"rot={s['decomposition']['rotation_deg']:.1f} sx={s['decomposition']['scale_x']:.3f} "
              f"sy={s['decomposition']['scale_y']:.3f} shear={s['decomposition']['shear']:.3f} coverage={s['coverage']:.2f} "
              f"prior_penalty={s['view_prior_penalty']:.3f}")
    # landscape over views
    look = {tuple(r["grid"]): r for r in res}
    print(f"[{b}] affine vessel NCC over views (rows {row_name} {[f'{v:g}' for v in rows]}, cols {col_name} "
          f"{[f'{v:g}' for v in cols]}):")
    for ro in rows:
        print(f"   {ro:6g}: " + " ".join(f"{look[(ro, co)]['affine']['score']:.3f}" for co in cols))


if __name__ == "__main__":
    main()
