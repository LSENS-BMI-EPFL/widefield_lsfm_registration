"""Write the final per-brain outputs from a refine_local.py result.

Usage: python finalize.py PB105 results/PB105/local.json [affine|similarity]
Writes results/<brain>_transform.json (transform, view in code and anatomical angles, scores, wS1 control)
and results/<brain>_atlas_labels_widefield.{npy,tif} (Allen IDs per widefield pixel).
"""
import os
import sys
import json
import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
import validate as V  # noqa: E402
import register as R  # noqa: E402
import brain_axes as BA  # noqa: E402


def main(brain, local_json, mode="affine"):
    fit = json.load(open(local_json))
    best = fit["best"][mode]
    roll, pitch, A = best["roll"], best["pitch"], np.array(best["A"])
    _, e = R.moving_for_view(brain, roll, pitch)
    fx = R.fixed_for(brain)
    ncc = R.score(R.blur(fx["ridge"], 1.2), fx["mask"], R.blur(e["ridge"] * e["mask"], 1.2), A)
    ev = V.evaluate(brain, roll, pitch, A)
    L = ev.pop("L")
    lat, ap = BA.anatomical(BA.axes(brain, A, (roll, pitch)), roll, pitch)
    rec = dict(
        brain=brain, mode=mode, source=os.path.relpath(local_json, C.RESULTS),
        roll_deg=roll, pitch_deg=pitch,
        anatomical_lateral_tilt_deg=lat, anatomical_ap_tilt_deg=ap,
        A_projection_to_widefield=A.tolist(), decomposition=best["decomposition"],
        vessel_ncc=ncc, window_coverage=best["coverage"],
        widefield_frame="<brain>_wf_image.tif",
        projection="surface.project(prepare(brain), roll, pitch): 25.3 um px, rot90 CCW; roll/pitch are relative "
                   "to the LSFM volume axes, anatomical_* are relative to the brain's own axes (brain_axes.py)",
        wS1_control=ev)
    json.dump(rec, open(f"{C.RESULTS}/{brain}_transform.json", "w"), indent=1)
    np.save(f"{C.RESULTS}/{brain}_atlas_labels_widefield.npy", L.astype(np.int64))
    tifffile.imwrite(f"{C.RESULTS}/{brain}_atlas_labels_widefield.tif", L.astype(np.uint32))
    print(json.dumps({k: rec[k] for k in ("roll_deg", "pitch_deg", "anatomical_lateral_tilt_deg", "anatomical_ap_tilt_deg",
                                          "vessel_ncc")}, indent=1))
    print("scale x/y:", round(best["decomposition"]["scale_x"], 3), round(best["decomposition"]["scale_y"], 3))
    print("wS1:", {k: ev[k] for k in ("peak_region", "halfmax_in_bfd", "top1pct_in_bfd")})


if __name__ == "__main__":
    main(*sys.argv[1:])
