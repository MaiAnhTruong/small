"""GO/NO-GO signal test on REAL mip-NeRF360 sparse-12 data (CPU, no rasterizer, no training).

Decides whether VS-Depth's covisibility gate can possibly help:
  On sparse-12, does covisibility have EXPLOITABLE SPATIAL STRUCTURE, or is it flat (every region seen by
  ~the same #views -> gate == uniform -> no possible gain)?

Covisibility the METHOD actually uses = GEOMETRIC FRUSTUM covisibility: #views whose frustum contains a
surface point (VS-Depth refines this with rendered-depth occlusion). NOTE: SfM *track length* (#views that
MATCH a feature) is always degenerate in sparse SfM (2-3) and is NOT what the gate uses -> we measure
frustum covisibility of the SfM 3D points across the 12 views. Structured -> PROCEED.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                  read_points3D_binary, qvec2rotmat)
from utils.graphics_utils import getWorld2View2

BASE = r"D:/All for one/data/_3dgs_splits_lightning_build/mipnerf360_sparse12"
SCENES = ["bicycle", "garden", "room", "stump", "counter"]

def view_W2C_intr(e, intr):
    R = np.transpose(qvec2rotmat(e.qvec)); T = np.array(e.tvec)
    W2C = getWorld2View2(R, T)
    ii = intr[e.camera_id]
    f = ii.params[0]; W, H = ii.width, ii.height
    return W2C, f, W, H

def analyze(scene):
    sp = os.path.join(BASE, scene, "hold8_train12_sparsegs_triangulate", "train", "sparse", "0")
    extr = read_extrinsics_binary(os.path.join(sp, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(sp, "cameras.bin"))
    xyz = read_points3D_binary(os.path.join(sp, "points3D.bin"))[0]            # [N,3]
    views = list(extr.values()); n_views = len(views)
    Xh = np.concatenate([xyz, np.ones((len(xyz), 1))], 1)                       # [N,4]

    # GEOMETRIC FRUSTUM covisibility: for each 3D point, #views whose frustum contains it (z>0, in-frame)
    cov = np.zeros(len(xyz), dtype=np.int32)
    for e in views:
        W2C, f, W, H = view_W2C_intr(e, intr)
        cam = (W2C @ Xh.T).T
        z = cam[:, 2]
        u = f * cam[:, 0] / np.clip(z, 1e-6, None) + W / 2.0
        v = f * cam[:, 1] / np.clip(z, 1e-6, None) + H / 2.0
        cov += ((z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).astype(np.int32)

    print(f"\n--- {scene}: {n_views} views, {len(xyz)} SfM points ---")
    print(f"  FRUSTUM covisibility: min={cov.min()} max={cov.max()} mean={cov.mean():.2f} std={cov.std():.2f}")
    for k in (2, 4, 6, 8, 10):
        print(f"    frac(covis >= {k}) = {(cov >= k).mean():.2f}")
    low = float((cov <= 3).mean()); high = float((cov >= 6).mean())

    # spatial: in view 0, correlate each point's covisibility with its radial image position
    e = views[0]; W2C, f, W, H = view_W2C_intr(e, intr); cx, cy = W / 2.0, H / 2.0
    cam = (W2C @ Xh.T).T; z = cam[:, 2]
    u = f * cam[:, 0] / np.clip(z, 1e-6, None) + cx; v = f * cam[:, 1] / np.clip(z, 1e-6, None) + cy
    infr = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    r = np.sqrt(((u[infr] - cx) / W) ** 2 + ((v[infr] - cy) / H) ** 2)
    corr = np.corrcoef(r, cov[infr])[0, 1] if infr.sum() > 10 else float("nan")
    print(f"  spatial: corr(radial_dist, covis) = {corr:+.3f}  (negative => periphery less-covisible = structure)")

    structured = (cov.max() - cov.min() >= 3) and (cov.std() >= 0.7) and (0.05 < low < 0.97)
    print(f"  => low-covis(<=3) frac={low:.2f}, high-covis(>=6) frac={high:.2f}, "
          f"STRUCTURED={'YES' if structured else 'NO'}")
    return structured

print("=" * 72)
print("VS-Depth GO/NO-GO: does covisibility have exploitable structure on sparse-12 (real data)?")
print("=" * 72)
res = []
for s in SCENES:
    try:
        res.append(analyze(s))
    except Exception as ex:
        print(f"  {s}: SKIP ({type(ex).__name__}: {ex})")

print("\n" + "=" * 72)
n = sum(1 for r in res if r)
print(f"VERDICT: {n}/{len(res)} scenes show STRUCTURED covisibility.")
if res and n >= max(1, len(res) - 1):
    print("=> SIGNAL PRESENT: gate has real structure to exploit. PROCEED to full impl + Kaggle multi-seed.")
else:
    print("=> SIGNAL WEAK/FLAT: gate ~ uniform on this data. STOP & DISCUSS before building further.")
