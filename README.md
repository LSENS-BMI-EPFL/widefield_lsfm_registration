# Widefield ↔ LSFM registration

Maps in vivo widefield imaging of mouse dorsal cortex onto a brain atlas, using the same animal's brain after
clearing and light-sheet imaging (iDISCO + LSFM). The surface blood vessels are visible in both modalities and serve as
landmarks. For every widefield pixel, the result gives an atlas area label and a position in a standard atlas space:
the Perens LSFM mouse atlas, which shares area IDs with the Allen CCF.

```
LSFM volume ──(brainreg)──► atlas registered to this brain
     │
     └─ surface vessels ──(this code: 3D view + 2D affine)──► widefield image
                                                                 │
             atlas labels per widefield pixel  ◄─────────────────┘
             widefield pixel → Perens atlas coordinates (mm)
```

Only vessels are used for the fit. A sensory response (here whisker stimulation) is used afterwards as a check: it
should fall in the matching area (here the barrel field, SSp-bfd).

## Installation

Python 3.9 and the packages in `requirements.txt`:

```bash
conda create -n wlr -c conda-forge python=3.9 pip
conda activate wlr
pip install -r requirements.txt
```

The Perens LSFM atlas (`perens_lsfm_mouse_20um`) comes from [BrainGlobe](https://brainglobe.info). The code reads it
as a folder with `annotation.npy`, `hemispheres.npy` and `metadata.json`.

## Configuration

All paths and settings are in `config.py`:

| Setting | What |
|---|---|
| `DATA` | Input folder, one subfolder per brain (see Inputs) |
| `PERENS_ATLAS` | Perens atlas folder (above) |
| `CACHE` | Working cache, about 2 GB per brain |
| `RESULTS`, `FIGURES` | Output folders (default: `results/`, `figures/` next to the code) |
| `HEMISPHERE` | Imaged hemisphere per brain (`"left"` / `"right"`, the animal's own side); `None` searches both |
| `PRIORS` | Expected imaging tilt of the camera relative to the brain (default 30° lateral / 0° AP, σ 5°) and its weight |
| `WF_PIXEL_UM` | Widefield pixel size at the sample (µm). Bounds the fitted scale together with `CLEARING_SCALE`; `None` uses a scale range of 0.8–1.2 |
| `WINDOW_MODE` | `"auto"`: cranial-window masks detected from the data; `"manual"`: hand-drawn polygons `WINDOW` (fitting) and `WINDOW_DISPLAY` (display, atlas mapping) |
| `WINDOW_FIT_MARGIN_PX` | Auto mode: the fitting mask is the detected window shrunk by this margin |

`DATA`, `PERENS_ATLAS`, `CACHE` and `RESULTS` can also be set with the environment variables `WLR_DATA`, `WLR_ATLAS`,
`WLR_CACHE` and `WLR_RESULTS`. A new brain needs only an entry in `HEMISPHERE` (and polygons in manual mode).

## Inputs (per brain, in `DATA/<brain>/`)

| File | Content |
|---|---|
| `registration/` | brainreg output for the LSFM vessel channel, run with `--save-original-orientation`: `downsampled.tiff`, `registered_atlas_original_orientation.tiff`, `registered_hemispheres.tiff`, `deformation_field_{0,1,2}.tiff`, `brainreg.json` |
| `<brain>_wf_image.tif` | Widefield reference image of the cranial window (vessels dark), in the same pixel frame as the functional data |
| `wf_data/{stim,no_stim}_trial_avg.npy` | Trial-averaged ΔF/F movies (frames × H × W, 100 Hz). Used for the window masks and the validation, not for the fit |

`atlas_ids.json` holds the atlas structure names, hierarchy paths, and the isocortex and SSp-bfd IDs.

## Running

```bash
./run_pipeline.sh <brain>                       # e.g. ./run_pipeline.sh PB105
PYTHON=/path/to/python ./run_pipeline.sh <brain>  # with a specific interpreter
```

The view searches run 100–150 fits in parallel (one process per CPU core). A brain takes about 6 min on 112 cores.

| Step | Script | What it does |
|---|---|---|
| 1. Cache | `load_cache.py` | Loads the vessel channel (brainreg `downsampled.tiff`, resampled to a 25.3 µm grid) and the registered atlas; builds the brain mask and the normalized volume. Skipped if the cache exists. |
| 2. Start | `initialize.py` | Global search for a starting transform, from the data alone |
| 3. Brain axes | `brain_axes.py` | The brain's left–right, anterior–posterior and dorsal axes inside the scan |
| 4. Fit | `refine_local.py` | Fits a 2D affine for each view on a grid of anatomical tilts and picks the view |
| 5. Final outputs | `finalize.py` | Transform, atlas label per widefield pixel, sensory-response check |
| 6. Figure panels | `make_figure.py` | Figure panels (300 dpi PNG + PDF) and `overview.png` |
| 7. Standard space | `atlas_space.py` | Each widefield pixel → Perens atlas coordinates; data on the atlas dorsal view |
| 8. Layer-5 borders | `l5_boundaries.py` | Area borders at layer 5 instead of the surface, for layer-5 recordings |

`atlas_space.py` accepts several brains (`python atlas_space.py PB105 PB104`) and then also overlays all of them on the
standard dorsal view.

Supporting modules: `surface.py` (surface projection for any view), `enhance.py` (vessel maps and window masks),
`register.py` (score and affine helpers), `validate.py` (sensory-response check).

## Outputs

| File | Content |
|---|---|
| `results/<brain>_transform.json` | Fitted view (volume angles and anatomical tilts), 2×3 affine (projection → widefield), vessel score, sensory-response check |
| `results/<brain>_atlas_labels_widefield.{npy,tif}` | Atlas structure ID (layer level) for every widefield pixel |
| `results/<brain>_widefield_to_perens_mm.npy` | (H, W, 3) Perens coordinates (mm, axes a/s/r) for every widefield pixel; NaN outside the window. Maps any widefield frame into standard space. |
| `results/<brain>_perens_dorsal.npz` | Sensory response and vessel image resampled onto the atlas dorsal view |
| `results/<brain>/init.json` | Starting transform and every candidate of the global search |
| `results/<brain>/local.json` | Fit for every view of the grid |
| `results/<brain>/brain_axes.json` | Brain axes |
| `figures/<brain>/` | a1 LSFM dorsal projection, a2 surface layer, a3 LSFM vessel network, b1–b3 widefield raw / vesselness / vessel mask, c vessel overlay, d loss landscape, e atlas on widefield, f response with atlas (+ colorbar), g/h atlas on vessels / activity in a reference orientation; `stats.json` |
| `figures/<brain>_L5/` | Panels e–h with layer-5 borders, surface vs layer-5 borders (e2), `l5_depth_um.npy`, `stats.json` |
| `figures/atlas_space/` | Response and vessel maps on the Perens dorsal view; response half-max contours of all brains |

## Method

### Surface projection (`surface.py`)

- **Mask and normalization.** The brain mask comes from the smoothed volume, thresholded halfway between the median
  intensity inside and outside the registered atlas, then cleaned morphologically. Intensities are divided by the local
  tissue level (Gaussian, σ = 6 voxels).
- **Projection.** For a given view, the first surface along each ray is found and smoothed. The image is the mean
  intensity 13–75 µm below it; surface vessels fade below ~100 µm.
- **Labels.** Each pixel takes the shallowest atlas label 50–300 µm below the surface.

Views are given as lateral and AP tilt relative to the brain's own axes:
- **ML:** mean left→right vector between centroids of bilateral atlas structures.
- **AP:** principal axis, pointing to the olfactory bulb.
- **DV:** their cross product.

### Window masks (`enhance.auto_window`)

1. **Cortex signal.** Each pixel's trial-averaged ΔF/F is correlated with the cortex's shared time course (first the
   image centre, then the best-correlated 10% of pixels). Cortex under the window shares it; bone, cement and the
   window rim do not. Cortex = correlation > 0.3.
2. **Cut lines.** Wide dark valleys of the widefield image (black top-hat, 31 px; narrow vessels removed by opening)
   mark the sagittal sinus and the window rim. A watershed on the valley depth splits the cortex along them; the part
   holding the best-correlated pixels is the window.
3. **Outline.** Notches cut by large vessels are closed (disk, 16 px) and the outline is smoothed along its length
   (Gaussian, σ 8 px).
4. **Masks.** Shrunk by 3 px, the region is the fitting mask. Its convex outline (hull, smoothed) is the window used
   for atlas mapping.

### Vessel maps (`enhance.py`)

**LSFM:**
1. **Mask.** Isocortex surface only, without steep edges or fissures.
2. **Artifacts.** Log image, flat-fielded. Debris blobs, bubbles and holes are set to the neutral tissue level.
3. **Vesselness.** Frangi filter, vessel widths 25–100 µm.
4. **Network.** Hysteresis threshold (0.06 / 0.2). Small pieces and pieces running along the mask edge are removed.

**Widefield.** Log image, flat-fielded inside the fitting mask, then Frangi-filtered (σ 2–4 px). Only long, large pial
vessels are kept: the top 12% of the response, in segments at least 25 px long.

### Starting transform (`initialize.py`)

The fit is local, so it needs a start. The start comes from the vessels alone, without manual alignment or another
brain.

1. **Views.** The imaged hemisphere × lateral tilt 15–45° × AP tilt −15…+15°, in 5° steps.
2. **Coarse search.** For each view, every in-plane rotation (full circle, 6° steps) × overall scale (bounded, below).
   The translation comes from masked template matching, 5 peaks each.
3. **Refinement.** The 4 best distinct placements of each view are refined (Powell, 2D affine) on the same objective as
   the fit.

The scale bound is needed: the camera magnification is fixed, so the bound only allows for tissue shrinkage and
calibration error. Without it, the fit can inflate the LSFM map until a few large vessels of the opposite hemisphere
line up.

### Fit (`refine_local.py`)

The transform from projection to widefield is a 2D affine.

**Objective:**
- Masked normalized cross-correlation (NCC) of the vessel maps inside the fitting mask.
- Transforms where the LSFM cortex covers less than 85% of the window are penalized.
- Scale anisotropy and shear are softly limited to 0.15.
- **Imaging-angle prior.** 0.01 × [((lat − 30°)/5°)² + (AP/5°)²], so a 5° deviation costs 0.01 NCC.

**Search** over a grid of views (lateral tilt 24–36°, AP tilt −20…+20°, 2° steps):
1. **Start.** The starting transform is carried to every view in 3D.
2. **First pass.** Local search: ±8° rotation, scale ±8%, translation ±25 px; then Powell refinement, coarse to fine.
3. **Second pass.** Every view is re-fitted from the best first-pass solution, keeping the better result.
4. **Choice.** The view with the best objective averaged over its 3 × 3 grid neighbourhood is chosen, so a single
   lucky fit does not decide it.

### Atlas mapping (`finalize.py`, `atlas_space.py`)

**Widefield labels.** The projected labels are warped into the widefield image (nearest neighbour). For display, they
are collapsed to cortical areas with a 7 × 7 mode filter.

**Standard space.** Each widefield pixel is mapped to the LSFM point 50 µm below the surface (inverse affine + view),
then through the brainreg deformation fields into Perens coordinates.

**Figures.** The outline in the widefield figures is the imaged hemisphere's cortex (isocortex labels of the
projection, cut at the midline with brainreg's hemisphere labels), mapped into the widefield image. It can extend
beyond the window.

## Limitations

- The vessels constrain the view only loosely; within that range, the imaging-angle prior decides the view.
- The starting search relies on the scale bound. Set `WF_PIXEL_UM` for your microscope.
- The automatic window detection was developed on two brains. Check the window outline in each new brain's figures,
  and fall back to `WINDOW_MODE = "manual"` if it fails.
- The 2D affine absorbs anisotropic clearing shrinkage only for the imaged surface.
- Perens coordinates are not Allen CCF coordinates; comparing with CCF data needs a Perens → CCF transform. The area
  IDs are shared.

## Credits

Lana Smith (LSENS, EPFL), building on an earlier pipeline by Clara Delandre, Paola Biocchi and Zhiyan Liao.
