"""FRGD-G (v6) CPU proofs — NO GPU, NO CUDA rasterizer, NO distCUDA2 (emulated on CPU). Validates the three
formulas the method rests on (DESIGN_AND_PROOF_v6.md):
  T1  init-persistence : rho(K,H)=(1-eta*H)^K (GD residual) + Adam budget Delta_l_max=K_eff*lr_s (Eq 4.1-4.2)
                         -> init survives ONLY in sparse+few-iter (late/under-observed points).
  T2  frustum scale    : pixel footprint = z/f (Eq 2.1); emulated distCUDA2 on k-subsampled back-proj = k*(z/f)
                         (Eq 2.2, over-size = subsample factor); image-splat blur L1 grows with scale.
  T3  orientation      : on-surface mass = erf(t/(sqrt2*sigma_n)) (Eq 3.2); disk(beta=.25)~0.95 vs iso~0.38.
Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python tools/test_frgd_g.py
"""
import math
import torch

PASS = []
def chk(n, c, d=""):
    PASS.append(bool(c)); print(f"[{'ok' if c else 'FAIL'}] {n}  {d}")

print("="*70); print("T1 — init-persistence  rho(K,H)=(1-eta*H)^K  +  Adam budget K_eff*lr_s"); print("="*70)
eta = 0.005

# (a) GD on the local quadratic L=0.5*H*(l-l*)^2 : residual ratio must equal (1-eta*H)^K exactly
H, K, lstar, l0 = 10.0, 100, 0.30, 2.00
l = l0
for _ in range(K):
    l = l - eta * H * (l - lstar)                     # GD step
sim_ratio = (l - lstar) / (l0 - lstar)
formula   = (1.0 - eta * H) ** K
chk("GD residual ratio == (1-eta*H)^K", abs(sim_ratio - formula) < 1e-9,
    f"sim={sim_ratio:.3e} formula={formula:.3e}")

rho = lambda K, H: (1.0 - eta * H) ** K
chk("rho decreasing in K (more iters -> init washed)", rho(50, 10.0) > rho(200, 10.0),
    f"rho(50)={rho(50,10.):.3e} > rho(200)={rho(200,10.):.3e}")
chk("rho decreasing in H (more views -> init washed)", rho(100, 5.0) > rho(100, 20.0),
    f"rho(H=5)={rho(100,5.):.3e} > rho(H=20)={rho(100,20.):.3e}")

# (b) Adam correction budget: a Gaussian can move log-scale by at most K_eff*lr_s ; init persists if < Delta_init
lr_s = 5e-3
def budget(t_add, v_see, v_tot, N):
    K_eff = (N - t_add) * (v_see / v_tot)
    return K_eff, K_eff * lr_s
dl_init = math.log(4.0)                               # naive distCUDA2 over-size factor k=4 -> Delta_init=ln4
K_late,  B_late  = budget(9000, 3, 12, 10000)        # sparse, late-added point
K_early, B_early = budget(2000, 3, 12, 10000)        # sparse, early point
K_dense, B_dense = budget(20000, 20, 100, 30000)     # dense, 30k, late point
print(f"  Delta_init=ln4={dl_init:.3f} | budgets: late(K={K_late:.0f})={B_late:.2f}  "
      f"early(K={K_early:.0f})={B_early:.2f}  dense(K={K_dense:.0f})={B_dense:.2f}")
chk("sparse-LATE point: budget < Delta_init  -> init PERSISTS", B_late < dl_init, f"{B_late:.2f} < {dl_init:.2f}")
chk("sparse-early point: budget > Delta_init -> correctable", B_early > dl_init)
chk("dense-30k point:    budget > Delta_init -> correctable", B_dense > dl_init)

print("\n" + "="*70); print("T2 — frustum footprint z/f  vs  distCUDA2 over-size"); print("="*70)
f, z = 500.0, 3.0
W = Hh = 80
cx = cy = W / 2.0
foot = z / f                                          # Eq 2.1 (c_f=1)

# (a) footprint identity: a world lateral step z/f at depth z projects to exactly 1 pixel
du = f * (foot) / z                                   # Δu for Δx_c = z/f
chk("pixel footprint: world (z/f) <-> 1 px", abs(du - 1.0) < 1e-9, f"du={du:.6f} foot={foot:.5f}")

# (b) emulated distCUDA2 (mean 3-NN distance) on back-projected pixels subsampled by k  ==  k*(z/f)
def emulate_distCUDA2_scale(k):
    us = torch.arange(0, W, k).float(); vs = torch.arange(0, Hh, k).float()
    gu, gv = torch.meshgrid(us, vs, indexing="ij")
    xc = (gu.reshape(-1) - cx) / f * z
    yc = (gv.reshape(-1) - cy) / f * z
    P = torch.stack([xc, yc, torch.full_like(xc, z)], 1)         # back-projected world points
    D = torch.cdist(P, P)
    D.fill_diagonal_(float("inf"))
    nn3_sq = (D.topk(3, largest=False).values ** 2).mean(1)      # mean squared dist to 3 NN (= distCUDA2)
    sigma = nn3_sq.sqrt()
    return float(sigma.median())
for k in (1, 4):
    sig = emulate_distCUDA2_scale(k); ratio = sig / foot
    chk(f"distCUDA2 scale at subsample k={k} ~= k*(z/f)", abs(ratio - k) < 0.15 * k,
        f"sigma={sig:.5f} foot={foot:.5f} ratio={ratio:.3f} (expect ~{k})")

# (c) image-space splat == blur by s_img; over-size -> larger blur -> larger L1 on a textured image
def gblur(img, sigma):
    r = max(1, int(3 * sigma)); xs = torch.arange(-r, r + 1).float()
    ker = torch.exp(-(xs ** 2) / (2 * sigma * sigma)); ker = (ker / ker.sum()).view(1, 1, -1)
    x = img[None, None]
    x = torch.nn.functional.conv2d(x, ker.view(1, 1, 1, -1), padding=(0, r))
    x = torch.nn.functional.conv2d(x, ker.view(1, 1, -1, 1), padding=(r, 0))
    return x[0, 0]
torch.manual_seed(0)
img = (torch.rand(64, 64) > 0.5).float()              # high-frequency texture (worst case for blur)
L1_foot = (gblur(img, 1.0) - img).abs().mean().item() # s_img = c_f = 1 px (footprint)
L1_over = (gblur(img, 4.0) - img).abs().mean().item() # s_img = k*c_f = 4 px (distCUDA2 over-size)
chk("over-sized splat has higher render L1 (blur)", L1_over > L1_foot > 0,
    f"L1_foot(1px)={L1_foot:.4f}  L1_over(4px)={L1_over:.4f}")

print("\n" + "="*70); print("T3 — surface disk vs isotropic blob: off-surface (haze) mass"); print("="*70)
sig_lat = 1.0; beta = 0.25; t = sig_lat / 2.0
def M_on_analytic(sigma_n): return math.erf(t / (math.sqrt(2.0) * sigma_n))
on_iso_a  = M_on_analytic(sig_lat)                    # isotropic: sigma_n = sigma_lat
on_disk_a = M_on_analytic(beta * sig_lat)             # disk:      sigma_n = beta*sigma_lat
# numeric: sample the normal-axis coordinate of each Gaussian, measure mass within |d|<t
torch.manual_seed(0); Npts = 400000
d_iso  = torch.randn(Npts) * sig_lat
d_disk = torch.randn(Npts) * (beta * sig_lat)
on_iso_n  = float((d_iso.abs()  < t).float().mean())
on_disk_n = float((d_disk.abs() < t).float().mean())
print(f"  on-surface mass  iso: analytic={on_iso_a:.3f} numeric={on_iso_n:.3f} | "
      f"disk: analytic={on_disk_a:.3f} numeric={on_disk_n:.3f}")
chk("analytic == numeric (erf formula correct)",
    abs(on_iso_a - on_iso_n) < 0.01 and abs(on_disk_a - on_disk_n) < 0.01)
chk("disk concentrates mass on surface (on>0.9) vs isotropic (on<0.45)",
    on_disk_a > 0.90 and on_iso_a < 0.45, f"disk={on_disk_a:.3f} iso={on_iso_a:.3f}")
off_ratio = (1 - on_iso_a) / (1 - on_disk_a)
chk("off-surface haze cut >= 10x by the disk", off_ratio >= 10.0, f"off_iso/off_disk={off_ratio:.1f}x")

print("\n" + "="*70)
print(f"RESULT: {sum(PASS)}/{len(PASS)} passed")
print("ALL FRGD-G PROOFS PASSED" if all(PASS) else "SOME FAILED")
import sys; sys.exit(0 if all(PASS) else 1)
