"""Paths and per-brain settings. Edit this file before the first run; nothing else holds a path.

Inputs per brain (under DATA/<brain>/):
  registration/            brainreg output for the LSFM vessel channel (downsampled.tiff,
                           registered_atlas_original_orientation.tiff, deformation_field_{0,1,2}.tiff, brainreg.json)
  <brain>_wf_image.tif     widefield reference image of the cranial window (vessels dark)
  wf_data/{stim,no_stim}_trial_avg.npy   trial-averaged dF/F movies (frames x H x W), validation only
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATA = os.environ.get("WLR_DATA", "/raid0/student_projects/data")
PERENS_ATLAS = os.environ.get("WLR_ATLAS", "/raid0/lmsmith/brainglobe/perens_lsfm_mouse_20um_export")  # BrainGlobe export
CACHE = os.environ.get("WLR_CACHE", "/raid0/lmsmith/widefield_lsfm_registration_cache")              # ~2 GB per brain
RESULTS = os.environ.get("WLR_RESULTS", os.path.join(HERE, "results"))                                # override for test runs
FIGURES = os.path.join(HERE, "figures")

# Window masks. "auto": detected from the widefield image and trial averages (enhance.auto_window); the fitting mask is
# the detected window shrunk by WINDOW_FIT_MARGIN_PX. "manual": the hand-drawn polygons below.
WINDOW_MODE = "auto"
WINDOW_FIT_MARGIN_PX = 3
# Manual mode only. Cranial-window ROI drawn on the widefield image (x, y), excluding bone/cement and window edges. Used for fitting.
WINDOW = {
    "PB104": [(40, 60), (100, 45), (170, 52), (220, 62), (255, 72), (282, 110), (296, 145), (292, 180),
              (278, 198), (60, 200), (42, 170), (36, 110)],
    "PB105": [(30, 72), (80, 56), (150, 50), (205, 60), (238, 82), (266, 118), (282, 158), (284, 192),
              (262, 204), (60, 204), (32, 172)],
}
# Full visible window (cortex up to the bone/cement edge and the sinus), for display and atlas mapping only.
WINDOW_DISPLAY = {
    "PB104": [(38, 5), (95, 5), (102, 40), (150, 48), (200, 52), (240, 62), (262, 80), (285, 110), (298, 145),
              (292, 185), (282, 200), (60, 202), (42, 180), (35, 110)],
    "PB105": [(4, 3), (192, 3), (205, 28), (225, 52), (255, 72), (285, 96), (305, 116), (308, 150),
              (305, 200), (290, 208), (40, 208), (10, 195), (4, 120)],
}
# Imaging priors (same rig for every brain). Command-line flags of initialize.py / refine_local.py override them.
PRIORS = dict(
    lat=30.0, ap=0.0,             # expected anatomical tilt of the camera (deg): lateral (window side toward camera), AP
    sigma_lat=5.0, sigma_ap=5.0,  # each view pays weight * (deviation / sigma)^2
    weight=0.01,                  # NCC units, so a 1-sigma deviation costs 0.01 NCC
)
# Imaged hemisphere ("left" / "right", the animal's own side; None = search both). Read against brainreg's
# registered_hemispheres.tiff, so the orientation given to brainreg must be correct (not mirrored).
HEMISPHERE = {
    "PB104": "left",
    "PB105": "left",
}
# Widefield pixel size at the sample (um): camera pixel x binning / magnification. None = not calibrated, then the
# fitted scale is bounded by SCALE_RANGE_UNCALIBRATED. The LSFM side is known from brainreg.json (2 x 12.66 um).
WF_PIXEL_UM = None
# Linear size of the cleared brain relative to in vivo; with WF_PIXEL_UM it sets the scale bound (initialize.py).
CLEARING_SCALE = (0.75, 1.05)
SCALE_RANGE_UNCALIBRATED = (0.8, 1.2)

