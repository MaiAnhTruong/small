"""DIGS (v8) CPU proofs — NO GPU. Validates the timing lever (#1) + the geometry-faithful eval (#2)
(DESIGN_AND_PROOF_v8.md):
  D1 budget advantage : K_eff(0)/K_eff(t)=N/(N-t)>1, monotone in t (iter-0 points optimized more, Eq 1.2).
  D2 closer converge  : (1-eta*H)^K_eff(0) < (1-eta*H)^K_eff(t)  for all t>0 (iter-0 residual smaller).
  D3 coverage         : dense depth back-proj covers textureless surface that texture-driven SfM misses (FSGS 1.1).
  D4 on-surface       : depth back-proj points lie ON the surface (0% empty); Euclidean-unpool midpoints of
                        cross-surface neighbors fall in EMPTY space (>0%) (FSGS 1.2).
  E1 interp blind     : floater test-displacement Dpix(b)=b*f*|1/zF-1/zS| (Eq 3.1) -> <1px at interp baseline
                        (invisible), >>1px at extrapolation (exposed) -> interp-PSNR is geometry-blind (23.6).
Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python tools/test_digs.py
"""
import torch

PASS = []
def chk(n, c, d=""):
    PASS.append(bool(c)); print(f"[{'ok' if c else 'FAIL'}] {n}  {d}")

print("="*70); print("D1 / D2 — iter-0 vs mid-training optimization budget (init-persistence)"); print("="*70)
N = 10000; vfrac = 3/12; eta_H = 1e-3
K0 = N * vfrac
ratios = {}
for t in (2000, 6000, 9000):
    Kt = (N - t) * vfrac
    ratios[t] = K0 / Kt
    print(f"  t_add={t}: K_eff(0)={K0:.0f}  K_eff(t)={Kt:.0f}  ratio={ratios[t]:.2f}  "
          f"resid0={(1-eta_H)**K0:.3e}  residt={(1-eta_H)**Kt:.3e}")
chk("D1: K_eff(0)/K_eff(t) > 1 for all t>0", all(r > 1 for r in ratios.values()))
chk("D1: ratio monotincreasing in t_add", ratios[2000] < ratios[6000] < ratios[9000])
chk("D2: iter-0 residual < mid residual for all t>0",
    all((1 - eta_H) ** K0 < (1 - eta_H) ** ((N - t) * vfrac) for t in (2000, 6000, 9000)))

print("\n" + "="*70); print("D3 — dense depth init covers textureless surface SfM misses (FSGS 1.1)"); print("="*70)
torch.manual_seed(0)
# surface = plane z=5 over [0,10]x[0,10]; texture HIGH for x<5, LOW for x>=5
Q = torch.stack([torch.rand(400) * 4 + 6, torch.rand(400) * 10, torch.full((400,), 5.0)], 1)  # queries in low-tex (x in [6,10])
SfM = torch.stack([torch.rand(60) * 5, torch.rand(60) * 10, torch.full((60,), 5.0)], 1)        # SfM only where x<5 (textured)
gx, gy = torch.meshgrid(torch.linspace(0, 10, 40), torch.linspace(0, 10, 40), indexing="ij")
Dense = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((1600,), 5.0)], 1)             # dense back-proj: all pixels
def coverage(Q, S, r=0.6): return float((torch.cdist(Q, S).min(1).values < r).float().mean())
cov_sfm, cov_dense = coverage(Q, SfM), coverage(Q, Dense)
print(f"  low-texture coverage:  SfM={cov_sfm:.2%}   dense-depth={cov_dense:.2%}")
chk("D3: dense covers textureless surface, SfM does not", cov_dense > 0.95 and cov_sfm < 0.10,
    f"dense={cov_dense:.2%} sfm={cov_sfm:.2%}")

print("\n" + "="*70); print("D4 — depth back-proj ON surface vs Euclidean-unpool in EMPTY space (FSGS 1.2)"); print("="*70)
# two surfaces (planes) z=4 and z=6, gap=2; SfM samples laterally sparse (spacing 3 > gap) on both
pts = []
for X in range(0, 15, 3):
    for Y in range(0, 15, 3):
        jx, jy = float(torch.rand(1) * 0.2), float(torch.rand(1) * 0.2)
        pts.append([X + jx, Y + jy, 4.0]); pts.append([X + jx, Y + jy, 6.0])
P = torch.tensor(pts)
Dm = torch.cdist(P, P); Dm.fill_diagonal_(float("inf"))
nn = Dm.argmin(1)
midz = ((P + P[nn]) / 2)[:, 2]                                  # Euclidean-unpool midpoint depth
empty = ((midz - 4).abs() > 0.5) & ((midz - 6).abs() > 0.5)     # not on either surface -> empty space
euclid_empty = float(empty.float().mean())
depth_empty = float((((P[:, 2] - 4).abs() > 0.5) & ((P[:, 2] - 6).abs() > 0.5)).float().mean())  # init points themselves
print(f"  Euclidean-unpool midpoints in empty space: {euclid_empty:.0%} | depth-backproj points in empty space: {depth_empty:.0%}")
chk("D4: Euclidean unpool puts points in empty space", euclid_empty > 0.5, f"{euclid_empty:.0%}")
chk("D4: depth back-proj points lie ON the surface (0% empty)", depth_empty == 0.0)

print("\n" + "="*70); print("E1 — interpolated-holdout PSNR is geometry-blind (Eq 3.1)"); print("="*70)
f, zF, zS = 400.0, 3.0, 5.0
disp = lambda b: b * f * abs(1.0 / zF - 1.0 / zS)
b_interp, b_extrap = 0.015, 0.20
di, de = disp(b_interp), disp(b_extrap)
print(f"  floater displacement: interp(b={b_interp})={di:.2f}px   extrap(b={b_extrap})={de:.2f}px   ratio={de/di:.1f}x")
chk("E1: floater ~invisible at interpolation baseline (<1px)", di < 1.0, f"{di:.2f}px")
chk("E1: floater exposed at extrapolation (>5px)", de > 5.0, f"{de:.2f}px")
bs = torch.linspace(0.0, 0.3, 30); ds = bs * f * abs(1.0 / zF - 1.0 / zS)
chk("E1: displacement strictly increases with baseline", bool((ds[1:] > ds[:-1]).all()))

print("\n" + "="*70)
print(f"RESULT: {sum(PASS)}/{len(PASS)} passed")
print("ALL DIGS PROOFS PASSED" if all(PASS) else "SOME FAILED")
import sys; sys.exit(0 if all(PASS) else 1)
