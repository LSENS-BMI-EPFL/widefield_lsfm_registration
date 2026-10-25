#!/bin/bash
# Full registration for one brain, from the brainreg output, the widefield image and the trial averages only:
# cache -> global start -> brain axes -> local fit (vessels) -> final outputs -> figure panels
# -> standard atlas space -> layer-5 borders.
# Usage: ./run_pipeline.sh PB105          (paths, priors and per-brain settings: config.py)
set -euo pipefail
B=$1
P=${PYTHON:-python}
cd "$(dirname "$0")"

# Settings (priors, hemisphere, widefield pixel size, window masks) are in config.py; the local fit's view grid
# (lateral tilt 24-36 deg, AP tilt -20..+20 deg, 2 deg steps) is refine_local.py's default.

RES=$($P -c "import config; print(config.RESULTS)")
mkdir -p $RES/$B logs
[ -f "$($P -c 'import config; print(config.CACHE)')/${B}_lsfm25.npy" ] || $P load_cache.py $B
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
echo "== standard atlas space"
$P atlas_space.py $B
echo "== layer-5 borders"
$P l5_boundaries.py $B
