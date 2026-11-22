"""Cache the LSFM vessel channel and the registered atlas from the brainreg output folder.

Vessel channel: brainreg's downsampled.tiff (the registered channel, already on the atlas grid: 20 um, brainreg 'asr'
orientation), put back into the original sample orientation. It is not resampled: the volume and the registered atlas
share one grid (surface.VOX_UM = atlas resolution).
Atlas: registered_atlas_original_orientation.tiff (same grid), flipped along axis 0 like the vessel volume.
Usage: python load_cache.py PB105
"""
import sys, os, json, numpy as np, tifffile
import config as C
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surface  # noqa: E402

b = sys.argv[1]
reg_dir = f"{C.DATA}/{b}/registration"
os.makedirs(C.CACHE, exist_ok=True)
src_um = float(json.load(open(f"{reg_dir}/brainreg.json"))["voxel_sizes"][0])
anno = tifffile.imread(f"{reg_dir}/registered_atlas_original_orientation.tiff")
O0, O1 = anno.shape[:2]
# reg[a, s, r] = orig[O0-1-s, O1-1-a, r]  ->  orig[i, j, k] = reg[O1-1-j, O0-1-i, k]
reg = tifffile.imread(f"{reg_dir}/downsampled.tiff")
assert reg.shape == (O1, O0, anno.shape[2]), (reg.shape, anno.shape)
vol = np.transpose(reg[::-1, ::-1, :], (1, 0, 2))[::-1]          # original orientation, axis 0 flipped (= anno[::-1])
anno = anno[::-1]
assert vol.shape == anno.shape
np.save(surface.cache_path(b, "lsfm"), np.clip(vol, 0, 65535).astype(np.uint16))
np.save(f"{C.CACHE}/{b}_anno20.npy", anno)
# Compact index map + id table (surface.py reads these): anno == ids[idx]
ids, idx = np.unique(anno, return_inverse=True)
np.save(f"{C.CACHE}/{b}_anno20_ids.npy", ids.astype(np.int64))
np.save(f"{C.CACHE}/{b}_anno20_idx.npy", idx.reshape(anno.shape).astype(np.uint16))
print(b, f"brainreg input {src_um} um; volume and atlas on the atlas grid ({surface.VOX_UM:g} um)", vol.shape,
      "extent mm", np.round(np.array(vol.shape) * surface.VOX_UM / 1000, 2))
# Surface cache (boundary points + normalized volume) is derived from the volume: rebuild it here, once, before the
# fit starts many workers that would otherwise all build it at the same time.
for f in (surface.cache_path(b, "boundary"), surface.cache_path(b, "norm_f16")):
    if os.path.exists(f):
        os.remove(f)
surface.prepare(b)
