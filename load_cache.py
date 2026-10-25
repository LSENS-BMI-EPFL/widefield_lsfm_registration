"""Cache the LSFM vessel channel and the registered atlas from the brainreg output folder.

Vessel channel: brainreg's downsampled.tiff (the registered channel, 20 um, brainreg 'asr' orientation), put back into
the original sample orientation and resampled to the 25.3 um working grid (all voxel-based settings assume it).
Atlas: registered_atlas_original_orientation.tiff (20 um), flipped along axis 0 like the vessel volume.
Usage: python load_cache.py PB105
"""
import sys, os, json, numpy as np, tifffile
from scipy import ndimage as ndi
import config as C
VOX_UM = 25.3254   # working grid (2x the 12.6627 um brainreg input voxels)
b = sys.argv[1]
reg_dir = f"{C.DATA}/{b}/registration"
os.makedirs(C.CACHE, exist_ok=True)
src_um = float(json.load(open(f"{reg_dir}/brainreg.json"))["voxel_sizes"][0])
atlas_um = 20.0
anno = tifffile.imread(f"{reg_dir}/registered_atlas_original_orientation.tiff")
O0, O1 = anno.shape[:2]
# reg[a, s, r] = orig[O0-1-s, O1-1-a, r]  ->  orig[i, j, k] = reg[O1-1-j, O0-1-i, k]
reg = tifffile.imread(f"{reg_dir}/downsampled.tiff").astype(np.float32)
assert reg.shape == (O1, O0, anno.shape[2]), (reg.shape, anno.shape)
v20 = np.transpose(reg[::-1, ::-1, :], (1, 0, 2))[::-1]          # original orientation, axis 0 flipped (= anno[::-1])
# voxel centres: new i at (i + 0.5) * VOX_UM  ->  20 um index (i + 0.5) * VOX_UM / 20 - 0.5
shape = tuple(int(round(n * atlas_um / VOX_UM)) for n in v20.shape)
f = VOX_UM / atlas_um
vol = ndi.affine_transform(v20, np.diag([f] * 3), offset=[0.5 * f - 0.5] * 3, output_shape=shape, order=1, mode="nearest")
np.save(f"{C.CACHE}/{b}_lsfm25.npy", np.clip(vol, 0, 65535).astype(np.uint16))
anno = anno[::-1]
np.save(f"{C.CACHE}/{b}_anno20.npy", anno)
# Compact index map + id table (surface.py reads these): anno == ids[idx]
ids, idx = np.unique(anno, return_inverse=True)
np.save(f"{C.CACHE}/{b}_anno20_ids.npy", ids.astype(np.int64))
np.save(f"{C.CACHE}/{b}_anno20_idx.npy", idx.reshape(anno.shape).astype(np.uint16))
print(b, f"brainreg input {src_um} um; vol ({VOX_UM} um)", vol.shape, "anno (20 um)", anno.shape,
      "vol extent mm", np.round(np.array(vol.shape) * VOX_UM / 1000, 2), "anno extent mm", np.array(anno.shape) * 0.02)
# Surface cache (boundary points + normalized volume) is derived from the volume: rebuild it here, once, before the
# fit starts many workers that would otherwise all build it at the same time.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surface  # noqa: E402
for f in (f"{C.CACHE}/{b}_boundary.npy", f"{C.CACHE}/{b}_norm_f16.npy"):
    if os.path.exists(f):
        os.remove(f)
surface.prepare(b)
