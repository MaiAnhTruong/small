"""CGD IMPLEMENTATION tests (the code path, not the theory which is in test_cgd.py).
  CPU: generate_frgd_points(return_rel=True) returns per-candidate rel (= confidence) consistent with the
       input rel map, and o_init=0.1*rel lands in (0,0.1]; return_rel=False keeps the 2-tuple (back-compat).
  GPU (if cuda): add_frgd_points(opacities=...) writes exactly those opacities; no opacities -> 0.1 (back-compat).
Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python tools/test_cgd_impl.py
"""
import os, sys, math
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.frgd import generate_frgd_points

PASS = []
def chk(n, c, d=""):
    PASS.append(bool(c)); print(f"[{'ok' if c else 'FAIL'}] {n}  {d}")

# ---------- (1) generate_frgd_points return_rel (CPU) ----------
H = W = 16
cam = SimpleNamespace(world_view_transform=torch.eye(4), FoVx=math.pi/2, FoVy=math.pi/2,
                      image_width=W, image_height=H, camera_center=torch.zeros(3))
image = torch.rand(3, H, W)
D_ref = torch.full((H, W), 5.0)
rel = torch.linspace(0.0, 1.0, H * W).reshape(H, W)          # known confidence map
grad = torch.zeros(H, W)                                     # passes grad<tex_thr
hole = torch.ones(H, W)                                      # passes hole>hole_thr
mask = torch.ones(H, W)
rel_thr = 0.1

out3 = generate_frgd_points(cam, image, D_ref, rel, grad, hole, base_mask=mask,
                            tex_thr=0.05, hole_thr=0.15, rel_thr=rel_thr, max_points=10**9, return_rel=True)
chk("return_rel=True -> 3-tuple", len(out3) == 3)
xyz, rgb, rel_sel = out3
n_expect = int((rel > rel_thr).sum())
chk("kept count == (rel>rel_thr)", xyz.shape[0] == n_expect and rel_sel.shape[0] == n_expect, f"{xyz.shape[0]} vs {n_expect}")
chk("all kept rel > rel_thr", bool((rel_sel > rel_thr).all()), f"min={float(rel_sel.min()):.3f}")
chk("rel_sel is exactly the rel of kept pixels",
    torch.allclose(torch.sort(rel_sel).values, torch.sort(rel[rel > rel_thr]).values, atol=1e-6))
o_init = (0.1 * rel_sel).clamp(1e-4, 0.99)                   # Eq 4.1 (as in frgd_step)
chk("o_init = 0.1*rel in (0, 0.1]", bool((o_init > 0).all() and (o_init <= 0.1 + 1e-6).all()),
    f"range=({float(o_init.min()):.4f},{float(o_init.max()):.4f})")

out2 = generate_frgd_points(cam, image, D_ref, rel, grad, hole, base_mask=mask,
                            tex_thr=0.05, hole_thr=0.15, rel_thr=rel_thr, max_points=10**9)
chk("return_rel=False -> 2-tuple (back-compat)", len(out2) == 2)
e = generate_frgd_points(cam, image, D_ref, torch.zeros(H, W), grad, hole, base_mask=mask,
                         tex_thr=0.05, hole_thr=0.15, rel_thr=0.5, max_points=10, return_rel=True)
chk("empty candidate set -> 3 empty tensors", len(e) == 3 and e[0].shape[0] == 0 and e[2].shape[0] == 0)

# ---------- (2) add_frgd_points opacities (GPU) ----------
if torch.cuda.is_available():
    import numpy as np
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import BasicPointCloud
    N0 = 200
    pcd = BasicPointCloud(points=np.random.randn(N0, 3).astype(np.float32),
                          colors=np.random.rand(N0, 3).astype(np.float32), normals=np.zeros((N0, 3), np.float32))
    cam_infos = [SimpleNamespace(image_name=f"i{i}") for i in range(3)]
    opt = SimpleNamespace(percent_dense=0.01, position_lr_init=1.6e-4, position_lr_final=1.6e-6,
                          position_lr_delay_mult=0.01, position_lr_max_steps=30000, feature_lr=2.5e-3,
                          opacity_lr=0.025, scaling_lr=5e-3, rotation_lr=1e-3, exposure_lr_init=1e-2,
                          exposure_lr_final=1e-3, exposure_lr_delay_steps=0, exposure_lr_delay_mult=0.0, iterations=30000)
    g = GaussianModel(3, "default"); g.create_from_pcd(pcd, cam_infos, 1.0); g.training_setup(opt)
    M = 64
    op = (torch.rand(M).cuda() * 0.1).clamp(1e-4, 0.99)      # confidence-scaled opacities (0,0.1]
    g.add_frgd_points(torch.randn(M, 3).cuda(), torch.rand(M, 3).cuda(), opacities=op)
    chk("add_frgd_points(opacities): get_opacity[N0:] == opacities", torch.allclose(g.get_opacity[N0:].squeeze(1), op, atol=1e-5),
        f"max|d|={ (g.get_opacity[N0:].squeeze(1)-op).abs().max():.2e}")

    g2 = GaussianModel(3, "default"); g2.create_from_pcd(pcd, cam_infos, 1.0); g2.training_setup(opt)
    g2.add_frgd_points(torch.randn(5, 3).cuda(), torch.rand(5, 3).cuda())   # no opacities -> 0.1
    chk("back-compat: no opacities -> 0.1", torch.allclose(g2.get_opacity[N0:], torch.full((5, 1), 0.1, device="cuda"), atol=1e-5))
else:
    print("[skip] cuda not available -> GPU add_frgd_points(opacities) test skipped")

print("\n" + "="*60)
print(f"RESULT: {sum(PASS)}/{len(PASS)} passed")
print("ALL CGD IMPL TESTS PASSED" if all(PASS) else "SOME FAILED")
sys.exit(0 if all(PASS) else 1)
