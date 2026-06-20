"""BDVR (v5) tests.
  CPU (always): (1) compute_supp_weight phi-logic on a synthetic plane scene (floater vs surface vs behind),
                (2) hard anchor protection, (3) min_views gating, (4) L_supp gradient sign (suppression
                lowers opacity of flagged points only).
  GPU (if cuda): (5) supp_weight buffer stays consistent through add_frgd_points + prune_points.
Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python tools/test_bdvr.py
"""
import os, sys, math
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.bdvr import compute_supp_weight, nearest_anchor_dist

PASS = []
def chk(n, c, d=""):
    PASS.append(bool(c)); print(f"[{'ok' if c else 'FAIL'}] {n}  {d}")

def make_cam(W=64, H=64):
    # world_view_transform = identity (row conv: cam_h = world_h @ wvt) -> camera at origin looking +z.
    return SimpleNamespace(world_view_transform=torch.eye(4), FoVx=math.pi / 2, FoVy=math.pi / 2,
                           image_width=W, image_height=H)   # FoV=90deg -> fx=fy=32, cx=cy=32

# ---------- synthetic plane scene: surface at z=5 in every view ----------
cams = [make_cam() for _ in range(3)]
depth_plane = [torch.full((64, 64), 5.0) for _ in cams]
#               surface     floater(front) behind      floater-on-anchor
xyz = torch.tensor([[0., 0., 5.], [0.5, 0.5, 3.], [0.5, -0.5, 7.], [0., 0., 3.]])
anchors = torch.tensor([[0., 0., 5.], [0., 0., 3.]])     # one on surface, one at the would-be floater
r_s = 0.2

phi = compute_supp_weight(xyz, cams, depth_plane, anchors, tau=0.05, r_s=r_s, min_views=2)
print("phi =", [round(float(x), 3) for x in phi])
chk("surface point not flagged (anchor)", phi[0] < 0.01, f"phi={float(phi[0]):.3f}")
chk("free-space floater FLAGGED",         phi[1] > 0.80, f"phi={float(phi[1]):.3f}")
chk("behind-surface point not flagged",   phi[2] < 0.01, f"phi={float(phi[2]):.3f}")
chk("floater near anchor PROTECTED",      phi[3] < 0.01, f"phi={float(phi[3]):.3f}")

# ---------- anchor protection: drop the close anchor -> the same floater is now flagged ----------
phi2 = compute_supp_weight(xyz, cams, depth_plane, anchors[:1], tau=0.05, r_s=r_s, min_views=2)
chk("without close anchor the floater-at-(0,0,3) is flagged", phi2[3] > 0.80, f"phi={float(phi2[3]):.3f}")

# ---------- min_views gating: require more views than exist -> OCC forced 0 -> nothing flagged ----------
phi3 = compute_supp_weight(xyz, cams, depth_plane, anchors[:1], tau=0.05, r_s=r_s, min_views=5)
chk("min_views>#views => no flags", float(phi3.max()) < 1e-6, f"max={float(phi3.max()):.3f}")

# ---------- nearest_anchor_dist correctness ----------
d = nearest_anchor_dist(torch.tensor([[0., 0., 0.], [3., 0., 0.]]), torch.tensor([[1., 0., 0.]]))
chk("nearest_anchor_dist", torch.allclose(d, torch.tensor([1.0, 2.0]), atol=1e-5), f"{d.tolist()}")

# ---------- L_supp gradient sign: suppression pushes opacity DOWN only for flagged (phi>0) points ----------
op = torch.zeros(4, requires_grad=True)                    # logit 0 -> opacity 0.5
phi_w = torch.tensor([0.0, 1.0, 0.0, 0.5])
L = 0.01 * (phi_w * torch.sigmoid(op)).sum()
L.backward()
g = op.grad
# gradient descent does op -= lr*g; g>0 => opacity decreases (suppressed); g==0 => untouched
chk("grad>0 for flagged (opacity will drop)", g[1] > 0 and g[3] > 0, f"g={[round(float(x),5) for x in g]}")
chk("grad==0 for unflagged (opacity kept)",  abs(float(g[0])) < 1e-9 and abs(float(g[2])) < 1e-9)

# ---------- GPU: buffer bookkeeping through add_frgd_points + prune_points ----------
if torch.cuda.is_available():
    import numpy as np
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import BasicPointCloud
    N0 = 200
    pcd = BasicPointCloud(points=np.random.randn(N0, 3).astype(np.float32),
                          colors=np.random.rand(N0, 3).astype(np.float32),
                          normals=np.zeros((N0, 3), np.float32))
    cam_infos = [SimpleNamespace(image_name=f"i{i}") for i in range(3)]
    opt = SimpleNamespace(percent_dense=0.01, position_lr_init=1.6e-4, position_lr_final=1.6e-6,
                          position_lr_delay_mult=0.01, position_lr_max_steps=30000, feature_lr=2.5e-3,
                          opacity_lr=0.025, scaling_lr=5e-3, rotation_lr=1e-3, exposure_lr_init=1e-2,
                          exposure_lr_final=1e-3, exposure_lr_delay_steps=0, exposure_lr_delay_mult=0.0,
                          iterations=30000)
    g = GaussianModel(3, "default"); g.create_from_pcd(pcd, cam_infos, 1.0); g.training_setup(opt)
    chk("supp_weight init length == N0", g.supp_weight.shape[0] == N0)
    g.supp_weight = torch.arange(N0, device="cuda", dtype=torch.float32)        # mark identity
    M = 50
    g.add_frgd_points(torch.randn(M, 3).cuda(), torch.rand(M, 3).cuda())
    ok_len = g.supp_weight.shape[0] == N0 + M
    ok_grace = bool((g.supp_weight[N0:] == 0).all())                            # new points get phi=0 (grace)
    ok_keep = bool((g.supp_weight[:N0] == torch.arange(N0, device="cuda")).all())  # old phi preserved
    chk("add_frgd_points: supp_weight extended + grace + preserved", ok_len and ok_grace and ok_keep,
        f"len={g.supp_weight.shape[0]} grace={ok_grace} keep={ok_keep}")
    mask = torch.zeros(N0 + M, dtype=torch.bool, device="cuda"); mask[::3] = True  # prune every 3rd
    g.tmp_radii = torch.zeros(N0 + M, device="cuda")
    kept = (~mask)
    expected = g.supp_weight[kept].clone()
    g.prune_points(mask)
    chk("prune_points: supp_weight masked consistently",
        g.supp_weight.shape[0] == int(kept.sum()) and bool((g.supp_weight == expected).all()),
        f"len={g.supp_weight.shape[0]}")
else:
    print("[skip] cuda not available -> skipped GPU buffer test (run on Kaggle/GPU)")

print("\n" + "=" * 60)
print(f"RESULT: {sum(PASS)}/{len(PASS)} passed")
print("ALL BDVR TESTS PASSED" if all(PASS) else "SOME FAILED")
sys.exit(0 if all(PASS) else 1)
