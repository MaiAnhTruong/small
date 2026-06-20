"""Unit tests for FRGD core (CPU): unproject geometry, cross-view refinement reduces error,
candidate selection respects (under-recon ∧ low-texture ∧ reliable) + cap, color/position correctness."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graphics_utils import getWorld2View2
from utils.covisibility import project_points
from utils.frgd import unproject_pixels, refine_depth_maps, generate_frgd_points

PASS = []
def chk(n, c, d=""):
    PASS.append(bool(c)); print(f"[{'ok' if c else 'FAIL'}] {n}  {d}")

class FakeCam:
    def __init__(self, R, T, fov, W, H):
        wvt = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float()
        self.world_view_transform = wvt
        self.camera_center = torch.inverse(wvt)[3, :3]
        self.FoVx = fov; self.FoVy = fov; self.image_width = W; self.image_height = H

I = np.eye(3, dtype=np.float32); fov = 1.0; W = H = 32

# ---------- (1) unproject ∘ project round-trip ----------
cam = FakeCam(I, np.zeros(3, np.float32), fov, W, H)
u = torch.rand(50) * (W - 1); v = torch.rand(50) * (H - 1); z = torch.rand(50) * 3 + 2
X = unproject_pixels(cam, u, v, z)
u2, v2, z2 = project_points(X, cam)
chk("unproject->project round-trip", (u - u2).abs().max() < 1e-2 and (z - z2).abs().max() < 1e-3,
    f"err u={float((u-u2).abs().max()):.1e} z={float((z-z2).abs().max()):.1e}")

# ---------- (2) refinement reduces error: 3 views of plane z=5, per-view mono noise ----------
cams = [FakeCam(I, np.array([t, 0, 0], np.float32), fov, W, H) for t in (0.0, -0.3, 0.3)]
true = torch.full((H, W), 5.0)
torch.manual_seed(0)
# each view's mono = true + independent bias field (random per view) -> fusion should cancel
monos = [true + 0.6 * torch.randn(1, 1, H, W).repeat(1, 1, 1, 1)[0, 0] for _ in range(3)]
D_ref, REL = refine_depth_maps(cams, monos, tau=0.05, max_dim=32)
cc = (slice(H//2-6, H//2+6), slice(W//2-6, W//2+6))                 # central in-frame region
err_mono = (monos[0][cc] - 5.0).abs().mean().item()
err_ref = (D_ref[0][cc] - 5.0).abs().mean().item()
chk("refinement reduces mono error (multi-view fusion)", err_ref < err_mono,
    f"mono={err_mono:.3f} -> refined={err_ref:.3f}")
chk("reliability in (0,1], high where views agree", float(REL[0].max()) <= 1.0 + 1e-5 and float(REL[0][cc].mean()) > 0.3)

# ---------- (3) candidate selection: under-recon ∧ low-texture ∧ reliable ----------
img = torch.rand(3, H, W)
D = torch.full((H, W), 5.0)
rel = torch.ones(H, W)            # all reliable
grad = torch.ones(H, W)          # high texture everywhere...
grad[:, :W//2] = 0.01            # ...except left half = low texture
hole = torch.zeros(H, W)
hole[:H//2, :] = 1.0             # top half = under-reconstructed
# candidates = low-texture (left) ∧ hole (top) ∧ reliable = top-left quadrant
xyz, rgb = generate_frgd_points(cam, img, D, rel, grad, hole, tex_thr=0.05, hole_thr=0.5, rel_thr=0.5)
u, v, z = project_points(xyz, cam)
chk("points only in top-left quadrant (low-tex ∧ hole)", xyz.shape[0] > 0 and float(u.max()) < W/2 and float(v.max()) < H/2,
    f"n={xyz.shape[0]} u<{float(u.max()):.0f} v<{float(v.max()):.0f}")
chk("points placed at D_ref depth", (z - 5.0).abs().max() < 1e-2)
chk("color sampled from image", rgb.shape == xyz.shape and rgb.min() >= 0)

# reliability gate: if rel low -> no points
xyz2, _ = generate_frgd_points(cam, img, D, torch.zeros(H, W), grad, hole, rel_thr=0.5)
chk("rel<thr -> no points (reliability gate)", xyz2.shape[0] == 0)
# no holes -> no points (do not densify well-reconstructed regions)
xyz3, _ = generate_frgd_points(cam, img, D, rel, grad, torch.zeros(H, W), hole_thr=0.5)
chk("no holes -> no points (non-destructive, only fill gaps)", xyz3.shape[0] == 0)
# cap respected
hole_all = torch.ones(H, W); grad_lo = torch.zeros(H, W)
xyz4, _ = generate_frgd_points(cam, img, D, rel, grad_lo, hole_all, max_points=100)
chk("cap respected", xyz4.shape[0] <= 100, f"n={xyz4.shape[0]}")

print("\n" + "=" * 60)
print(f"RESULT: {sum(PASS)}/{len(PASS)} passed")
print("ALL FRGD-CORE TESTS PASSED" if all(PASS) else "SOME FAILED")
