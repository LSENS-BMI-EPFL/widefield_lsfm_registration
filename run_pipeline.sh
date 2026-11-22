#!/bin/bash
# Full registration for one brain, from the brainreg output, the widefield image and the trial averages only:
# cache -> global start -> brain axes -> local fit (vessels) -> final outputs -> figure panels
# -> borders at the chosen layer (config.LAYER) -> standard atlas space (surface and that layer).
# Usage: ./run_pipeline.sh PB105          (paths, priors and per-brain settings: config.py)
set -euo pipefail
B=$1
P=${PYTHON:-python}
cd "$(dirname "$0")"

# Settings (priors, hemisphere, widefield pixel size, window masks) are in config.py; the local fit's view grid
# (lateral tilt 24-36 deg, AP tilt -20..+20 deg, 2 deg steps) is refine_local.py's default.

RES=$($P -c "import config; print(config.RESULTS)")
mkdir -p $RES/$B logs
[ -f "$($P -c "import surface; print(surface.cache_path('$B', 'lsfm'))")" ] || $P load_cache.py $B
echo "== start (global search)"
$P initialize.py $B
echo "== brain axes"
rm -f $RES/$B/brain_axes.json                 # recomputed: the window hemisphere comes from the new start
$P brain_axes.py $B $RES/$B/init.json
echo "== fit"
$P refine_local.py $B --init $RES/$B/init.json
echo "== finalize"
$P finalize.py $B $RES/$B/local.json affine
echo "== figure panels"
$P make_figure.py $B
LAYER=$($P -c "import config; print(config.LAYER)")
echo "== borders at layer $LAYER"
$P layer_boundaries.py $B --layer "$LAYER"
echo "== standard atlas space"
ALL=$(ls $RES/*_transform.json | xargs -n1 basename | sed 's/_transform.json//')   # every brain with results,
$P atlas_space.py $ALL --layer surface                                              # so the all-brains overlay is complete
[ "$LAYER" = surface ] || $P atlas_space.py $ALL --layer "$LAYER"
