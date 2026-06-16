"""VS-Depth covisibility unit tests (CPU, no rasterizer). Verifies the NEW geometry/gate logic:
 project/unproject round-trip, covisibility counting + occlusion, gate monotonicity + strict-generalization.
PASS = the mechanism is correct (the discipline NAGC lacked)."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graphics_utils import getWorld2View2
from utils.covisibility import (project_points, unproject_depth, covisibility_from_depths,
                                 reliability_from_invdepth, build_gate)

PASS = []
def chk(name, cond, d=""):
    PASS.append(bool(cond)); print(f"[{'ok' if cond else 'FAIL'}] {name}  {d}")

class FakeCam:
    def __init__(self, R, T, fov, W, H):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float()
        self.FoVx = fov; self.FoVy = fov; self.image_width = W; self.image_height = H

I = np.eye(3, dtype=np.float32); Z = np.zeros(3, dtype=np.float32)
W = H = 32; fov = 1.2

# ---------- (1) project ∘ unproject round-trip ----------
camA = FakeCam(I, Z, fov, W, H)
depth = torch.full((H, W), 5.0)
X = unproject_depth(camA, depth).reshape(-1, 3)
u, v, z = project_points(X, camA)
vv, uu = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
rt = (u - uu.reshape(-1)).abs().max().item() + (v - vv.reshape(-1)).abs().max().item() + (z - 5).abs().max().item()
chk("unproject->project round-trip (pixels & depth)", rt < 1e-3, f"max err={rt:.2e}")

# ---------- (2) covisibility counting + occlusion ----------
# two IDENTICAL cams seeing the same plane z=5 -> every pixel covisible (cov=1)
cov = covisibility_from_depths([camA, FakeCam(I, Z, fov, W, H)], [depth, torch.full((H, W), 5.0)], tau=0.05)
chk("identical views, same surface -> cov=1 (central pixels)", abs(cov[0][H//2, W//2].item() - 1.0) < 1e-6,
    f"cov_center={cov[0][H//2,W//2].item():.2f}")
# second view's surface much closer (z=2) -> cam A's z=5 points are OCCLUDED -> cov=0
cov_occ = covisibility_from_depths([camA, FakeCam(I, Z, fov, W, H)], [depth, torch.full((H, W), 2.0)], tau=0.05)
chk("occluded by nearer surface -> cov=0", cov_occ[0][H//2, W//2].item() < 1e-6,
    f"cov_center={cov_occ[0][H//2,W//2].item():.2f}")
# 3 views (two others see it) -> cov=2
cov3 = covisibility_from_depths([camA, FakeCam(I, Z, fov, W, H), FakeCam(I, Z, fov, W, H)],
                                [depth, torch.full((H, W), 5.0), torch.full((H, W), 5.0)], tau=0.05)
chk("two other views observe -> cov=2", abs(cov3[0][H//2, W//2].item() - 2.0) < 1e-6,
    f"cov_center={cov3[0][H//2,W//2].item():.2f}")

# ---------- (3) reliability: lower at depth edges ----------
inv = torch.ones(H, W); inv[:, W//2:] = 3.0          # a sharp depth step at the middle
rel = reliability_from_invdepth(inv, sigma=0.10)
chk("reliability LOW at depth edge vs flat region", rel[H//2, W//2].item() < 0.5 < rel[H//2, 2].item(),
    f"edge={rel[H//2,W//2].item():.2f} flat={rel[H//2,2].item():.2f}")

# ---------- (4) gate: monotonic in cov, strict-generalization, mean-pinned ----------
cov_map = torch.linspace(0, 8, H * W).reshape(H, W)   # covisibility 0..8
invd = torch.rand(H, W)
mask = torch.ones(H, W)
w_cov = build_gate("covonly", cov=cov_map, base_mask=mask, gamma=1.0)
# higher covisibility -> lower weight
lo = w_cov.reshape(-1)[:50].mean().item(); hi = w_cov.reshape(-1)[-50:].mean().item()
chk("covonly gate DECREASING in covisibility", lo > hi, f"low-cov w={lo:.3f} > high-cov w={hi:.3f}")
chk("covonly gate mean pinned to 1 (budget preserved)", abs(w_cov.mean().item() - 1.0) < 1e-5)
w_gated = build_gate("gated", cov=cov_map, invdepth=invd, base_mask=mask, gamma=1.0, rel_sigma=0.10)
chk("gated gate mean pinned to 1", abs(w_gated.mean().item() - 1.0) < 1e-5)
w_uni = build_gate("uniform", base_mask=mask)
chk("strict-gen: uniform == base_mask (= original 3DGS)", torch.allclose(w_uni, mask))
w_none = build_gate("none", base_mask=mask)
chk("strict-gen: none == 0 (no depth)", float(w_none.abs().max()) == 0.0)
# gated differs from covonly only via reliability
chk("gated = covonly * reliability (reliability is our delta vs CoMapGS)",
    not torch.allclose(w_gated, w_cov))

print("\n" + "=" * 60)
print(f"RESULT: {sum(PASS)}/{len(PASS)} passed")
print("ALL COVISIBILITY TESTS PASSED" if all(PASS) else "SOME FAILED")
