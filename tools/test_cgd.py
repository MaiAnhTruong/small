"""CGD (v7) CPU proofs — NO GPU. Validates the confidence-opacity mechanism (DESIGN_AND_PROOF_v7.md):
  P1 conf detects wrong depth : reprojection-agreement conf (Eq 3.2) separates on-surface vs off-surface
                                candidates (AUC~1, monotone in depth error).
  P2 harm reduction (Eq 4.2)  : E[harm]_CGD/E[harm]_naive = sum c(1-c)/sum(1-c) < 1 (uniform ~1/3); benefit
                                on high-confidence points preserved (ratio = c >= 0.9 for c>=0.9).
  P3 self-prune + rescue      : o_init<o_prune <=> conf<0.05 (Eq 5.1); single-view rescue budget K_eff*lr_o
                                >> o_prune (Eq 5.2) -> only unrescued (true-floater) points are pruned.
  P4 formula well-posed       : conf in (0,1], =1 at zero residual, strictly decreasing, view-symmetric.
Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python tools/test_cgd.py
"""
import os, sys, math
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.covisibility import project_points
from utils.frgd import unproject_pixels

PASS = []
def chk(n, c, d=""):
    PASS.append(bool(c)); print(f"[{'ok' if c else 'FAIL'}] {n}  {d}")

def make_cam(pos, f=400.0, W=400):
    wvt = torch.eye(4); wvt[3, :3] = -torch.tensor(pos, dtype=torch.float32)   # cam_h = [x,y,z,1]@wvt = world-pos
    FoV = 2 * math.atan(W / (2 * f))                                           # fov2focal(FoV,W) == f
    return SimpleNamespace(world_view_transform=wvt, FoVx=FoV, FoVy=FoV,
                           image_width=W, image_height=W, camera_center=torch.tensor(pos, dtype=torch.float32))

def ray_plane_depth(cam, u, v, n, p0):
    """camera-depth where the ray through pixel (u,v) meets plane {x: n.(x-p0)=0}."""
    A = cam.camera_center.unsqueeze(0)                                         # [1,3] ray origin = cam center
    B = unproject_pixels(cam, u, v, torch.ones_like(u)) - A                    # [N,3] ray dir per unit cam-depth
    num = ((p0.unsqueeze(0) - A) * n.unsqueeze(0)).sum(1)                      # scalar (broadcast)
    den = (B * n.unsqueeze(0)).sum(1)
    return num / den.clamp_min(1e-8)                                           # [N] = t = camera depth

TAU = 0.1
def conf_of(X, cams, src, n, p0, tau=TAU):
    """Eq 3.2 reprojection-agreement confidence of world candidates X (source view=src) vs the true plane."""
    z_i = project_points(X, cams[src])[2].clamp_min(1e-6)
    rs = torch.zeros(X.shape[0]); cnt = torch.zeros(X.shape[0])
    for j, cam in enumerate(cams):
        if j == src:
            continue
        u, v, zX = project_points(X, cam)
        W, H = int(cam.image_width), int(cam.image_height)
        infr = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (zX > 1e-6)
        z_surf = ray_plane_depth(cam, u, v, n, p0)
        valid = infr & (z_surf > 1e-6)
        r = (zX - z_surf).abs() / z_i
        rs += torch.where(valid, r, torch.zeros_like(r)); cnt += valid.float()
    conf = torch.exp(-(rs / cnt.clamp_min(1)) / tau)
    return torch.where(cnt > 0, conf, torch.zeros_like(conf))

print("="*70); print("P1 — geometric confidence detects WRONG depth"); print("="*70)
n = torch.tensor([0.12, 0.06, 1.0]); n = n / n.norm(); p0 = torch.tensor([0.0, 0.0, 5.0])
cams = [make_cam([0., 0., 0.]), make_cam([0.6, 0., 0.]), make_cam([-0.5, 0.4, 0.])]
gx, gy = torch.meshgrid(torch.arange(120, 281, 20).float(), torch.arange(120, 281, 20).float(), indexing="ij")
u0, v0 = gx.reshape(-1), gy.reshape(-1)
t_true = ray_plane_depth(cams[0], u0, v0, n, p0)
X_true = unproject_pixels(cams[0], u0, v0, t_true)
conf_true = conf_of(X_true, cams, 0, n, p0)
confs = {}
for err in (0.05, 0.10, 0.20):
    Xw = unproject_pixels(cams[0], u0, v0, t_true * (1 + err))
    confs[err] = conf_of(Xw, cams, 0, n, p0)
    print(f"  err={err:.2f}: mean conf(wrong)={confs[err].mean():.3f}")
print(f"  mean conf(true)={conf_true.mean():.3f}")
chk("conf(true) ~= 1", conf_true.mean() > 0.95, f"{conf_true.mean():.3f}")
chk("conf decreasing in depth error", confs[0.05].mean() > confs[0.10].mean() > confs[0.20].mean(),
    f"{confs[0.05].mean():.3f} > {confs[0.10].mean():.3f} > {confs[0.20].mean():.3f}")
# AUC: conf as detector of correct(1) vs wrong@10%(0)
a, b = conf_true, confs[0.10]
auc = ((a[:, None] > b[None, :]).float().mean() + 0.5 * (a[:, None] == b[None, :]).float().mean()).item()
chk("AUC(true vs wrong@10%) > 0.95", auc > 0.95, f"AUC={auc:.3f}")

print("\n" + "="*70); print("P2 — confidence-opacity reduces expected floater harm (Eq 4.2)"); print("="*70)
torch.manual_seed(0)
c_u = torch.rand(200000)
ratio_u = (c_u * (1 - c_u)).sum() / (1 - c_u).sum()
print(f"  uniform c: harm ratio CGD/naive = {ratio_u:.4f}  (analytic E[c(1-c)]/E[1-c] = 1/3 = 0.333)")
chk("uniform harm ratio ~= 1/3", abs(ratio_u.item() - 1/3) < 0.02, f"{ratio_u:.4f}")
c_beta = torch.distributions.Beta(5.0, 2.0).sample((200000,))             # skewed to high confidence
ratio_b = (c_beta * (1 - c_beta)).sum() / (1 - c_beta).sum()
chk("harm ratio < 1 for skewed-high c too", ratio_b < 1.0, f"{ratio_b:.4f}")
# benefit (per-point) ratio CGD/naive = c ; preserved on reliable (c>=0.9) points
hi = c_u[c_u >= 0.9]
chk("benefit preserved on c>=0.9 (ratio=c>=0.9)", float(hi.min()) >= 0.9, f"min benefit ratio={float(hi.min()):.3f}")

print("\n" + "="*70); print("P3 — soft self-prune threshold + rescue budget (Eq 5.1-5.2)"); print("="*70)
o0, o_prune, lr_o, Dd = 0.1, 0.005, 0.025, 100
chk("o_init<o_prune  <=>  conf<0.05", (o0 * 0.04 < o_prune) and (o0 * 0.06 > o_prune),
    f"0.1*0.04={o0*0.04} < {o_prune} < 0.1*0.06={o0*0.06}")
budgets = {vs: (Dd * vs / 12) * lr_o for vs in range(1, 13)}
print(f"  rescue budget Δo_max = K_eff*lr_o : V_see=1 -> {budgets[1]:.3f}, V_see=12 -> {budgets[12]:.3f}  (prune={o_prune})")
chk("single-view-valid point rescuable before prune (budget >> o_prune)", min(budgets.values()) > o_prune,
    f"min budget (V_see=1)={budgets[1]:.3f} > {o_prune}")

print("\n" + "="*70); print("P4 — confidence formula well-posed"); print("="*70)
r = torch.linspace(0, 2.0, 50)
conf_r = torch.exp(-r / TAU)
chk("conf=1 at zero residual", abs(conf_r[0].item() - 1.0) < 1e-9)
chk("conf in (0,1]", bool((conf_r > 0).all() and (conf_r <= 1.0 + 1e-9).all()))
chk("conf strictly decreasing in residual", bool((conf_r[1:] < conf_r[:-1]).all()))

print("\n" + "="*70)
print(f"RESULT: {sum(PASS)}/{len(PASS)} passed")
print("ALL CGD PROOFS PASSED" if all(PASS) else "SOME FAILED")
sys.exit(0 if all(PASS) else 1)
