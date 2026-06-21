"""FRGD-G IMPLEMENTATION tests (the code, not the theory). CUDA-only: utils.general_utils.build_rotation
allocates on cuda, so verification runs on GPU (theory T1/T2/T3 are covered CPU-only by test_frgd_g.py).
  (1) matrix_to_quaternion round-trips through build_rotation (incl. tr<=0 / 180-deg branches).
  (2) frgd_g_shape: anisotropic scales sig_lat=c_f*z/f, sig_n=beta*sig_lat; orthonormal frame, thin axis=view ray.
  (3) add_frgd_points(scales,quats) writes exactly those scales/rotations + optimizer stays consistent.
  (4) backward-compat: add_frgd_points without scales == isotropic distCUDA2.
Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python tools/test_frgd_g_impl.py
"""
import os, sys, math
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.frgd_g import matrix_to_quaternion, frgd_g_shape
from utils.general_utils import build_rotation

PASS = []
def chk(n, c, d=""):
    PASS.append(bool(c)); print(f"[{'ok' if c else 'FAIL'}] {n}  {d}")

if not torch.cuda.is_available():
    print("[skip] FRGD-G impl test needs cuda (build_rotation is cuda-only); theory proven by test_frgd_g.py")
    sys.exit(0)
DEV = "cuda"

# ---------- (1) matrix_to_quaternion round-trip ----------
torch.manual_seed(0)
q_rand = torch.nn.functional.normalize(torch.randn(500, 4, device=DEV), dim=1)
R = build_rotation(q_rand)
R_rec = build_rotation(matrix_to_quaternion(R))
chk("build_rotation(matrix_to_quaternion(R)) == R", torch.allclose(R_rec, R, atol=1e-5),
    f"max|dR|={ (R_rec-R).abs().max():.2e}")
R180 = torch.stack([torch.diag(torch.tensor([1., -1., -1.])),
                    torch.diag(torch.tensor([-1., 1., -1.])),
                    torch.diag(torch.tensor([-1., -1., 1.]))], 0).to(DEV)
chk("round-trip on tr<=0 (180-deg) rotations",
    torch.allclose(build_rotation(matrix_to_quaternion(R180)), R180, atol=1e-5))

# ---------- (2) frgd_g_shape ----------
W = 800; f = W / (2 * math.tan((math.pi / 2) / 2))         # FoVx=90deg -> f=400
cam = SimpleNamespace(world_view_transform=torch.eye(4, device=DEV), FoVx=math.pi/2, FoVy=math.pi/2,
                      image_width=W, image_height=W, camera_center=torch.zeros(3, device=DEV))
zdep = 4.0
xyz = torch.tensor([[0.5, -0.3, zdep], [1.0, 0.7, zdep], [-0.2, 0.4, zdep]], device=DEV)
c_f, beta = 1.0, 0.25
scales, quats = frgd_g_shape(xyz, cam, c_f=c_f, beta=beta, sigma_max_frac=0.1, extent=1.0)
sig_lat = c_f * zdep / f
chk("lateral scale == c_f*z/f", torch.allclose(scales[:, :2], torch.full((3, 2), sig_lat, device=DEV), atol=1e-6),
    f"sig_lat={sig_lat:.5f} got={scales[0,0].item():.5f}")
chk("along-ray scale == beta*sig_lat", torch.allclose(scales[:, 2], torch.full((3,), beta*sig_lat, device=DEV), atol=1e-6))
chk("along-ray is the THIN axis", bool((scales[:, 2] < scales[:, 0]).all()))
chk("quats are unit", torch.allclose(quats.norm(dim=1), torch.ones(3, device=DEV), atol=1e-6))
Rg = build_rotation(quats)                                 # columns = [t1, t2, ray]
ray = torch.nn.functional.normalize(xyz - cam.camera_center[None], dim=1)
chk("frame column 2 == view ray (thin axis along ray)", torch.allclose(Rg[:, :, 2], ray, atol=1e-5),
    f"max|dcol2|={ (Rg[:,:,2]-ray).abs().max():.2e}")
chk("frame orthonormal (R^T R == I)",
    torch.allclose(Rg.transpose(1, 2) @ Rg, torch.eye(3, device=DEV).expand(3, 3, 3), atol=1e-5))

# ---------- (3,4) add_frgd_points new path + backward-compat ----------
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
def fake_step(gm):
    L = (gm.get_xyz.sum()+gm._features_dc.sum()+gm._features_rest.sum()+gm._opacity.sum()+gm._scaling.sum()+gm._rotation.sum())
    L.backward(); gm.optimizer.step(); gm.optimizer.zero_grad(set_to_none=True)
fake_step(g)

M = 64
nx = torch.randn(M, 3).cuda(); nc = torch.rand(M, 3).cuda()
sc = torch.rand(M, 3).cuda() * 0.02 + 0.001
qu = torch.nn.functional.normalize(torch.randn(M, 4).cuda(), dim=1)
g.add_frgd_points(nx, nc, scales=sc, quats=qu)
chk("add_frgd_points(scales,quats): exp(_scaling)==scales", torch.allclose(g.get_scaling[N0:], sc, atol=1e-5),
    f"max|d|={ (g.get_scaling[N0:]-sc).abs().max():.2e}")
chk("add_frgd_points(scales,quats): rotations written",
    torch.allclose(torch.nn.functional.normalize(g._rotation[N0:], dim=1), qu, atol=1e-5))
try:
    fake_step(g); fake_step(g); ok_step = True
except Exception as e:
    ok_step = False; print("  step err:", e)
chk("optimizer.step() works after FRGD-G add", ok_step)

g2 = GaussianModel(3, "default"); g2.create_from_pcd(pcd, cam_infos, 1.0); g2.training_setup(opt)
n2 = g2.add_frgd_points(torch.randn(10, 3).cuda(), torch.rand(10, 3).cuda())   # no scales -> isotropic
iso = g2.get_scaling[N0:]
chk("backward-compat (no scales): isotropic (3 axes equal)",
    torch.allclose(iso[:, 0], iso[:, 1], atol=1e-6) and torch.allclose(iso[:, 1], iso[:, 2], atol=1e-6) and n2 == 10)

print("\n" + "="*60)
print(f"RESULT: {sum(PASS)}/{len(PASS)} passed")
print("ALL FRGD-G IMPL TESTS PASSED" if all(PASS) else "SOME FAILED")
sys.exit(0 if all(PASS) else 1)
