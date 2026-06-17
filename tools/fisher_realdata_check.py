"""GO/NO-GO for the v2 FISHER contribution on REAL data (CPU, no rendering/GPU).

For each SfM point on real mip360 scenes, compute the two signals the gate could use:
  count(p) = #views observing p              (CoMapGS uses this)
  H(p)     = Sum_j vis_j * (|grad I_j| * f / z_j^2)^2   (Fisher info; ours -- adds texture+geometry)
Question: do they DISAGREE, and is there a real 'CoMapGS blind spot' = points seen by MANY views but with
LOW Fisher (textureless/far) that count over-trusts but Fisher correctly flags as needing depth?
  substantial blind-spot fraction + count!=H  => Fisher reallocates meaningfully => PROCEED/commit.
  count ~ H (Spearman~1, blind spot ~0)        => contribution marginal here => STOP & discuss.
"""
import os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                  read_points3D_binary, qvec2rotmat)
from utils.graphics_utils import getWorld2View2

BASE = r"D:/All for one/data/_3dgs_splits_lightning_build/mipnerf360_sparse12"
SCENES = ["garden", "room", "counter", "bicycle", "stump"]
MAXDIM = 600

def grad_mag(path):
    im = Image.open(path).convert("L")
    s = MAXDIM / max(im.size)
    if s < 1: im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))))
    a = np.asarray(im, np.float32) / 255.0
    gx = np.zeros_like(a); gy = np.zeros_like(a)
    gx[:, 1:] = a[:, 1:] - a[:, :-1]; gy[1:, :] = a[1:, :] - a[:-1, :]
    return np.abs(gx) + np.abs(gy)                      # [h,w] downscaled

def analyze(scene):
    sp = os.path.join(BASE, scene, "hold8_train12_sparsegs_triangulate", "train")
    s0 = os.path.join(sp, "sparse", "0")
    extr = read_extrinsics_binary(os.path.join(s0, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(s0, "cameras.bin"))
    xyz = read_points3D_binary(os.path.join(s0, "points3D.bin"))[0]
    Xh = np.concatenate([xyz, np.ones((len(xyz), 1))], 1)
    N = len(xyz); count = np.zeros(N); Hf = np.zeros(N)
    for e in extr.values():
        ii = intr[e.camera_id]; f = ii.params[0]; W, Hh = ii.width, ii.height
        g = grad_mag(os.path.join(sp, "images", e.name)); gh, gw = g.shape
        W2C = getWorld2View2(np.transpose(qvec2rotmat(e.qvec)), np.array(e.tvec))
        cam = (W2C @ Xh.T).T; z = cam[:, 2]
        u = f * cam[:, 0] / np.clip(z, 1e-6, None) + W / 2.0
        v = f * cam[:, 1] / np.clip(z, 1e-6, None) + Hh / 2.0
        infr = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < Hh)
        count += infr
        gu = np.clip((u * gw / W).astype(int), 0, gw - 1)
        gv = np.clip((v * gh / Hh).astype(int), 0, gh - 1)
        gval = g[gv, gu]                                 # |grad I_j| at projection
        sens = gval * f / np.clip(z, 1e-6, None) ** 2    # photometric depth-sensitivity
        Hf += infr * sens ** 2

    keep = count >= 1
    count, Hf = count[keep], Hf[keep]
    ra = np.argsort(np.argsort(count)); rb = np.argsort(np.argsort(Hf))
    sp_corr = np.corrcoef(ra, rb)[0, 1]
    hi_cov = count >= np.median(count)                   # well-covisible (CoMapGS: 'enough')
    lo_H = Hf <= np.quantile(Hf, 0.33)                   # low Fisher (under-constrained truth)
    blind = (hi_cov & lo_H).mean()                       # CoMapGS blind spot
    print(f"\n--- {scene}: {len(count)} pts ---")
    print(f"  Spearman(count, Fisher-H) = {sp_corr:+.3f}   (1.0 => identical signal; lower => Fisher differs)")
    print(f"  CoMapGS blind spot (high-covis BUT low-Fisher) = {blind:.1%} of points")
    return sp_corr, blind

print("=" * 74)
print("v2 FISHER GO/NO-GO on REAL data: does Fisher-H differ from view-count, and is the blind spot real?")
print("=" * 74)
res = []
for s in SCENES:
    try: res.append((s, *analyze(s)))
    except Exception as ex: print(f"  {s}: SKIP ({type(ex).__name__}: {ex})")

print("\n" + "=" * 74)
corrs = np.array([r[1] for r in res]); blinds = np.array([r[2] for r in res])
print(f"mean Spearman(count,H) = {corrs.mean():.3f}   mean blind-spot = {blinds.mean():.1%}")
if corrs.mean() < 0.9 and blinds.mean() > 0.12:
    print("=> FISHER REALLOCATES MEANINGFULLY: count is a poor proxy + real blind spot. PROCEED/commit.")
else:
    print("=> count ~ Fisher on this data: contribution marginal. STOP & DISCUSS.")
