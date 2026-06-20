"""Local GPU test of the SENSITIVE part: GaussianModel.add_frgd_points keeps every tensor + Adam state
consistent, and training continues without shape errors. Needs cuda + simple_knn (same as scene-load)."""
import os, sys
import numpy as np
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import BasicPointCloud

PASS = []
def chk(n, c, d=""):
    PASS.append(bool(c)); print(f"[{'ok' if c else 'FAIL'}] {n}  {d}")

assert torch.cuda.is_available(), "need cuda (simple_knn/distCUDA2)"

N0 = 300
pts = np.random.randn(N0, 3).astype(np.float32)
pcd = BasicPointCloud(points=pts, colors=np.random.rand(N0, 3).astype(np.float32), normals=np.zeros((N0, 3), np.float32))
cam_infos = [SimpleNamespace(image_name=f"img{i}") for i in range(3)]
opt = SimpleNamespace(percent_dense=0.01, position_lr_init=1.6e-4, position_lr_final=1.6e-6,
                      position_lr_delay_mult=0.01, position_lr_max_steps=30000, feature_lr=2.5e-3,
                      opacity_lr=0.025, scaling_lr=5e-3, rotation_lr=1e-3, exposure_lr_init=1e-2,
                      exposure_lr_final=1e-3, exposure_lr_delay_steps=0, exposure_lr_delay_mult=0.0,
                      iterations=30000)

g = GaussianModel(3, "default")
g.create_from_pcd(pcd, cam_infos, 1.0)
g.training_setup(opt)

def fake_step():
    loss = (g.get_xyz.sum() + g._features_dc.sum() + g._features_rest.sum()
            + g._opacity.sum() + g._scaling.sum() + g._rotation.sum())
    loss.backward(); g.optimizer.step(); g.optimizer.zero_grad(set_to_none=True)

fake_step()                                   # populate Adam state (exp_avg/exp_avg_sq)
n_before = g.get_xyz.shape[0]
chk("init count", n_before == N0, f"{n_before}")

M = 120
new_xyz = torch.randn(M, 3).cuda(); new_rgb = torch.rand(M, 3).cuda()
added = g.add_frgd_points(new_xyz, new_rgb)
n_after = g.get_xyz.shape[0]
chk("added M points", added == M and n_after == N0 + M, f"after={n_after}")

# every parameter tensor grew consistently
ok_t = all(t.shape[0] == n_after for t in [g._xyz, g._features_dc, g._features_rest, g._opacity, g._scaling, g._rotation])
chk("all param tensors length == N0+M", ok_t)
chk("accum/denom/max_radii2D resized", g.xyz_gradient_accum.shape[0] == n_after and g.max_radii2D.shape[0] == n_after)

# Adam state extended consistently for every group
ok_state = True
for grp in g.optimizer.param_groups:
    st = g.optimizer.state.get(grp["params"][0], None)
    if st is not None:
        if st["exp_avg"].shape[0] != n_after or st["exp_avg_sq"].shape[0] != n_after:
            ok_state = False
chk("Adam exp_avg/exp_avg_sq extended to N0+M", ok_state)

# new points are at the given positions
chk("new xyz placed correctly", torch.allclose(g.get_xyz[N0:], new_xyz, atol=1e-5))

# training continues without shape error after surgery
try:
    fake_step(); fake_step(); train_ok = True
except Exception as e:
    train_ok = False; print("   step error:", e)
chk("optimizer.step() works after add (no shape mismatch)", train_ok)

# adding 0 points is a no-op
a0 = g.add_frgd_points(torch.zeros(0, 3).cuda(), torch.zeros(0, 3).cuda())
chk("empty add = no-op", a0 == 0 and g.get_xyz.shape[0] == n_after)

print("\n" + "=" * 60)
print(f"RESULT: {sum(PASS)}/{len(PASS)} passed")
print("ALL add_frgd_points TESTS PASSED" if all(PASS) else "SOME FAILED")
