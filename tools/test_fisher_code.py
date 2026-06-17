"""Unit tests for the v2 Fisher-gate CODE (CPU, no rasterizer): image_grad_mag, fisher_from_depths,
fisher_gate, compute_fisher_gates. Verifies the implementation matches the proven design
(test_fisher_gate.py is the math proof; this checks the actual functions in utils/covisibility.py)."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graphics_utils import getWorld2View2
from utils.covisibility import (image_grad_mag, fisher_from_depths, fisher_gate, compute_fisher_gates)

PASS = []
def chk(n, c, d=""):
    PASS.append(bool(c)); print(f"[{'ok' if c else 'FAIL'}] {n}  {d}")

class FakeCam:
    def __init__(self, R, T, fov, W, H):
        wvt = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float()
        self.world_view_transform = wvt
        self.camera_center = torch.inverse(wvt)[3, :3]
        self.FoVx = fov; self.FoVy = fov; self.image_width = W; self.image_height = H

I = np.eye(3, dtype=np.float32)
W = H = 40; fov = 1.0

# ---------- (1) image_grad_mag: high at edges, ~0 on flat ----------
img = torch.ones(3, H, W) * 0.5; img[:, :, W // 2:] = 0.9          # vertical edge at center
gm = image_grad_mag(img)
chk("image_grad_mag: high at edge, ~0 on flat", gm[H // 2, W // 2].item() > 0.1 and gm[H // 2, 5].item() < 1e-6,
    f"edge={gm[H//2,W//2].item():.2f} flat={gm[H//2,5].item():.2e}")

# ---------- (2) fisher_from_depths: textured neighbor -> high H; textureless -> low H ----------
camA = FakeCam(I, np.zeros(3, np.float32), fov, W, H)
camB = FakeCam(I, np.array([-0.3, 0, 0], np.float32), fov, W, H)   # nonzero baseline
base = (camA.camera_center - camB.camera_center).norm().item()
depth = torch.full((H, W), 5.0)
grad_hi = torch.ones(H, W)            # textured neighbor
grad_lo = torch.full((H, W), 0.01)    # textureless neighbor
# ref=camA; only neighbor=camB. H_A depends on grad sampled in camB.
Hs_hi, _ = fisher_from_depths([camA, camB], [depth, depth], [torch.zeros(H, W), grad_hi], [None, None], tau=0.05)
Hs_lo, _ = fisher_from_depths([camA, camB], [depth, depth], [torch.zeros(H, W), grad_lo], [None, None], tau=0.05)
cc = (slice(H // 2 - 5, H // 2 + 5), slice(W // 2 - 5, W // 2 + 5))   # central, in-frame region
Hhi, Hlo = Hs_hi[0][cc].mean().item(), Hs_lo[0][cc].mean().item()
chk("fisher H: textured neighbor >> textureless neighbor (baseline>0)", Hhi > 100 * max(Hlo, 1e-20),
    f"H_textured={Hhi:.3e}  H_textureless={Hlo:.3e}  (baseline={base:.2f})")
chk("fisher H ~ 0 when neighbor textureless (no constraint despite covisible)", Hlo < 1e-3 * Hhi)

# ---------- (3) fisher_gate: DECREASING in H, mean-pinned, strict-gen ----------
Hmap = torch.linspace(0.05, 5.0, H * W).reshape(H, W)
mask = torch.ones(H, W)
w = fisher_gate(Hmap, None, mask, c=0.5, cap=8.0)
lo = w.reshape(-1)[:80].mean().item(); hi = w.reshape(-1)[-80:].mean().item()
chk("fisher_gate DECREASING in H (low-H gets MORE depth)", lo > hi, f"low-H w={lo:.3f} > high-H w={hi:.3f}")
chk("fisher_gate mean pinned to 1 (budget kept)", abs(w.mean().item() - 1.0) < 1e-5)
# delta: higher mono-depth error -> less weight
wd = fisher_gate(torch.ones(H, W), torch.linspace(0.1, 3.0, H * W).reshape(H, W), mask, c=0.5)
chk("fisher_gate DECREASING in delta (less weight where mono-depth unreliable)",
    wd.reshape(-1)[:80].mean() > wd.reshape(-1)[-80:].mean())

# ---------- (4) the discriminating case in CODE: textured-covisible vs textureless-covisible ----------
# region split: left half neighbor textured, right half textureless -> H low on right -> gate HIGH on right
gradB = torch.ones(H, W); gradB[:, W // 2:] = 0.02
Hs, _ = fisher_from_depths([camA, camB], [depth, depth], [torch.zeros(H, W), gradB], [None, None], tau=0.05)
g = fisher_gate(Hs[0], None, torch.ones(H, W), c=0.5)
left = g[cc[0], slice(W // 2 - 10, W // 2 - 2)].mean().item()   # textured side
right = g[cc[0], slice(W // 2 + 2, W // 2 + 10)].mean().item()  # textureless side
chk("CODE: textureless-covisible region gets MORE depth weight than textured (fixes CoMapGS blind spot)",
    right > left, f"textureless w={right:.3f} > textured w={left:.3f}")

# ---------- (5) compute_fisher_gates end-to-end ----------
gates = compute_fisher_gates([camA, camB], [depth, depth], [torch.zeros(H, W), torch.ones(H, W)],
                             [torch.ones(H, W), torch.ones(H, W)], [torch.ones(H, W), torch.ones(H, W)],
                             c=0.5, max_dim=32, cap=8.0)
chk("compute_fisher_gates returns full-res gates, mean~1, finite", len(gates) == 2
    and gates[0].shape == (H, W) and abs(gates[0].mean().item() - 1.0) < 0.05 and torch.isfinite(gates[0]).all())

print("\n" + "=" * 60)
print(f"RESULT: {sum(PASS)}/{len(PASS)} passed")
print("ALL FISHER-CODE TESTS PASSED" if all(PASS) else "SOME FAILED")
