# DESIGN_AND_PROOF v6 — FRGD-G: Geometry-correct densification for sparse-view, few-iteration 3DGS

**Status:** design + proof only. NO method code is changed until the CPU gates (§7) pass.
**Base:** vanilla depth-regularized 3DGS (Final1606) + FRGD (v4, implemented, proven +0.302 PSNR over uniform).
**One line:** FRGD picks *where* to add Gaussians (refined depth, holes). FRGD-G fixes *what shape* each added
Gaussian has — **lateral scale = pixel-frustum footprint `z/f`, flattened along the view ray, oriented to the
local surface** — instead of the isotropic `distCUDA2`-kNN blob. The novelty is not the placement (that ties
naive, measured) but the **initial geometry**, and the reason it matters is a regime fact we prove:
**in sparse-view + few-iteration training, optimization cannot correct a bad init, so the init persists into
the final render.**

---

## 0. Why this axis (what the four previous results force)

| direction | axis touched | measured outcome |
|---|---|---|
| depth-LOSS gating (v1/v2) | reweight existing | DEAD (zero-sum) |
| FRGD densify (v4) | **add capacity** | **+0.302 PSNR (real)** |
| FRGD-refine vs naive | **placement** of added pts | tie on PSNR; +SSIM/+LPIPS only |
| BDVR suppression (v5) | remove capacity | over-broad / confounded, no positive |

Two hard conclusions: (i) the **only** positive lever is **densification**; (ii) being clever about the
**position** of added points ties naive (in textureless holes the back-projected position is ~identical
whether refined or raw — proven by the frgd≈rawdensify PSNR tie). Therefore FRGD-G **does not touch position**.
It touches the **scale and orientation** — degrees of freedom that naive sets to a *constant heuristic*
(`distCUDA2` kNN + isotropic) and that are **provably wrong** for hole-filling, and — crucially — that
**survive** to the output in our regime.

---

## 1. Notation

- Camera `c`: focal `f` (px), image `W×H`, intrinsics `u = f·x_c/z + c_x`, `v = f·y_c/z + c_y` (covisibility.py).
- Densified Gaussian `i` seeded for pixel `p=(u,v)` at refined metric depth `z`: world center `μ_i` (back-proj),
  covariance `Σ_i = R_i diag(σ_{1}^2,σ_{2}^2,σ_{3}^2) R_i^T` (3DGS parameterization: `_scaling=log σ`, `_rotation=quat`).
- `n_i` = unit surface normal at `p` (from refined-depth gradient; camera-facing fallback).
- `lr_s` = scaling learning rate (`scaling_lr = 5e-3`).

---

## 2. The pixel-frustum footprint (exact)  — basis of T2

A pixel is a sample of the image function; the world region it integrates at depth `z` is its **frustum
footprint**. From `u = f·x_c/z + c_x`:
```
∂u/∂x_c = f/z      ⇒      Δx_c (one pixel, Δu=1) = z/f.
```
So **one pixel spans `z/f` in world space laterally at depth `z`** (likewise `z/f` vertically for square px).
A Gaussian that represents the surface seen through that pixel should have **lateral world std**
```
σ_lat = c_f · (z/f),     c_f ∈ [0.5, 1]   (overlap constant; default c_f = 1, tunable)         (Eq. 2.1)
```
This is **linear in `z`** and **per-pixel exact**. It is the unique scale at which adjacent per-pixel
Gaussians tile the surface with ~1σ overlap.

**What `distCUDA2` does instead.** 3DGS init (and `add_frgd_points`) sets `σ = sqrt(mean 3-NN squared
distance)` over the *added point set*. If the back-projected hole pixels are sampled at native resolution
(every pixel), their world spacing is exactly `z/f` and `distCUDA2 ≈ z/f` — *correct*. **But FRGD subsamples**
(per-view `max_points` cap, global `frgd_max_per_step` cap, `randperm`, voxel dedup). At subsample factor `k`
the world spacing becomes `k·z/f`, hence
```
σ_distCUDA2 ≈ k · (z/f) = k · σ_lat        (over-size by the subsample factor k ≥ 1)            (Eq. 2.2)
```
and additionally `distCUDA2` is **isotropic** and **depth-blind** (a point's nearest neighbor may belong to a
different view/depth, inflating `σ` further). So naive scale is **correct only at k=1**, and over-sizes
otherwise. FRGD-G's `z/f` is correct **for every point regardless of sampling**.

**Render consequence.** A Gaussian splat of image-space std `s_img = σ_lat·f/z = c_f` pixels blurs the image
by ~`s_img`. Over-sizing by `k` ⇒ image blur `k·c_f` px ⇒ monotonically higher reconstruction L1 on any
non-constant (textured/edge) region. *(T2c.)*

---

## 3. Orientation / anisotropy: surface disk vs isotropic blob — basis of T3

A real surface element is ~2D. Represent it by a **flattened disk**: two large tangent axes
`σ_1=σ_2=σ_lat` (Eq. 2.1) and one small normal axis
```
σ_3 = β·σ_lat,   β ∈ (0,1],   default β = 0.25   (flattened, NOT degenerate — robust to normal error)  (Eq. 3.1)
```
with `R_i = [t_1, t_2, n_i]` (tangent, tangent, normal). The isotropic blob uses `σ_1=σ_2=σ_3=σ`.

**Off-surface (haze) mass — closed form.** For a Gaussian, the signed distance to the surface plane (normal
`n`) is `d = n·(x−μ) ~ N(0, σ_n^2)`, where `σ_n = σ` (isotropic) or `σ_n = σ_3 = β σ_lat` (disk). The mass
within a thin surface slab `|d| < t` is
```
M_on(t) = erf( t / (√2 · σ_n) ).                                                                 (Eq. 3.2)
```
For slab half-thickness `t = σ_lat/2`:
- isotropic (`σ_n=σ_lat`): `M_on = erf(1/(2√2)) ≈ erf(0.354) ≈ 0.382`  → **~62% of the mass is OFF-surface**.
- disk (`σ_n=0.25 σ_lat`): `M_on = erf(1/(0.5√2)) = erf(1.414) ≈ 0.954`  → **~5% off-surface**.

Off-surface mass is exactly what renders as **wrong-depth haze / floaters in novel views** (it sits in front
of/behind the true surface from other cameras). The disk cuts it from ~62% to ~5% **at equal lateral
extent** — a strict, always-on improvement that does not depend on sampling density. *(This is why FRGD-refine
already won SSIM/LPIPS: smoother/less-hazy geometry. FRGD-G makes it explicit and stronger.)*

---

## 4. The init-persistence principle (why init matters HERE) — basis of T1

A densified Gaussian's final shape = init + optimization. We quantify how much init survives.

**Mechanism (linear/quadratic model).** Let `ℓ = log σ` (a scale coordinate). Near the optimum the
photometric loss is locally quadratic: `L(ℓ) = ½ H (ℓ−ℓ*)^2`, `H` = curvature (= photometric Fisher info on
scale, ∝ number/informativeness of views that constrain the Gaussian). Gradient descent with rate `η`:
```
ℓ_{K} − ℓ* = (1 − ηH)^K (ℓ_0 − ℓ*)   ⇒   init-dependence  ρ(K,H) = ∂ℓ_K/∂ℓ_0 = (1 − ηH)^K.       (Eq. 4.1)
```
`ρ → 0` only as the **optimization budget `ηHK → ∞`**. `ρ ≈ 1` when budget is small. So init survives exactly
when `H` (few views) **or** `K` (few effective updates) is small — **the sparse-view, few-iteration regime.**

**Effective-update budget (Adam, honest upper bound).** Adam's per-step move in `ℓ` is bounded by `lr_s`
(its update is `lr_s·m̂/√v̂`, `|m̂/√v̂| ≲ 1`). A Gaussian is updated only on iterations whose sampled view sees
it. A point added at iteration `t_add`, alive `N−t_add` iters, visible in `V_see` of `V_tot` views, receives
```
K_eff ≈ (N − t_add) · (V_see / V_tot),     and can move at most   Δℓ_max ≈ K_eff · lr_s.          (Eq. 4.2)
```
If the init scale error is `Δℓ_init = |log(σ_init/σ*)|` (for naive over-size by `k`, `Δℓ_init = ln k`), then
the init **cannot be corrected** whenever `Δℓ_max < Δℓ_init`. With `lr_s = 5e-3`:

| regime | N | t_add | V_see/V_tot | K_eff | Δℓ_max | vs Δℓ_init=ln4≈1.39 |
|---|---|---|---|---|---|---|
| sparse, **late** point | 10000 | 9000 | 3/12 | 250 | **1.25** | **< 1.39 → init PERSISTS** |
| sparse, early point | 10000 | 2000 | 3/12 | 2000 | 10.0 | > 1.39 → correctable |
| dense, late point | 30000 | 20000 | 20/100 | 2000 | 10.0 | > 1.39 → correctable |

This is an **upper bound** on correction (assumes a perfectly consistent gradient); the weak, noisy
photometric signal in low-texture holes makes real correction *smaller*, so persistence is **at least** this
strong. Conclusion: **init geometry survives specifically for the late-added, under-observed points** — and
FRGD adds throughout training in **low-`H` (low-covisibility, textureless) holes**, i.e. exactly the points
whose init is *not* corrected. The anisotropy benefit (§3) applies to **all** added points (it is not an init
that needs "correcting" — a flat disk that gets slightly re-optimized stays flatter than a sphere).

---

## 5. The FRGD-G mechanism (exact change, surgical)

Only `add_frgd_points` changes (and it receives, per point, the camera `f`, depth `z`, and normal `n`). For
each densified point:
```
σ_lat = clamp( c_f · z / f ,  σ_min, σ_max )         # Eq. 2.1, per-point, depth-adaptive
σ_3   = β · σ_lat                                     # Eq. 3.1, flattened (β=0.25)
_scaling_i = log( [σ_lat, σ_lat, σ_3] )              # anisotropic (vs isotropic distCUDA2)
R_i        = orthonormal_basis(n_i)  (3rd col = n_i) # tangent,tangent,normal
_rotation_i= quat(R_i)                               # vs identity quat
_opacity_i = inverse_sigmoid(0.1)                    # UNCHANGED  -> keeps the proven +0.302
```
Normal `n_i`: from the gradient of the refined depth map (`n ∝ unproject(p+du) × unproject(p+dv)`), with a
**camera-facing fallback** `n_i = (μ_i − cam_center)/‖·‖` where the depth gradient is unreliable (high
`|∇z|`, low FRGD `rel`). `β=0.25` keeps the disk non-degenerate so a small normal error cannot open a gap.

**Why low-risk (the property BDVR lacked):** this is **init-only**. Opacity, placement, and the densification
schedule are untouched, so FRGD-G **cannot lose the +0.302** (every point FRGD added, FRGD-G still adds, at
the same place and opacity). Worst case the optimizer overwrites the init → **tie**. It cannot collapse the
scene the way active suppression can. This is the "more certain" property requested.

---

## 6. Novelty positioning (honest)

| prior | what it does | FRGD-G delta |
|---|---|---|
| 3DGS / FSGS / CoMapGS densify | add points, `distCUDA2` isotropic init, rely on 30k-iter optimization to fix shape | **frustum-`z/f` anisotropic disk init**; argues init *cannot* be fixed in sparse-few-iter |
| 2DGS / surfels | flat primitives as the *representation* (whole model) | keep 3D Gaussians; only **densified** points are surfel-initialized, from depth, at hole locations |
| AbsGS / Pixel-GS | better densification *criterion* (where/when) | orthogonal: we fix *shape*, not *where* |

Contribution = (i) **frustum-footprint + surface-disk initialization of depth-densified Gaussians**;
(ii) the **init-persistence analysis** (Eq. 4.1–4.2) showing why init geometry is a real lever in
sparse-view/few-iteration (and why dense-30k work could ignore it); (iii) honest multi-seed, same-depth
ablation. Honest caveat: adjacent to surfel/2DGS init ideas — the delta is the *sparse-few-iter
init-persistence justification* + the *densification-only, opacity-preserving* surgical form.

---

## 7. CPU proof gates (run BEFORE any code; tools/test_frgd_g.py)

- **T1 init-persistence.** (a) Simulate GD on the quadratic; verify residual ratio = `(1−ηH)^K` to ~1e-9 and
  that `ρ` is monotonically decreasing in both `K` and `H`. (b) Compute the Adam budget `Δℓ_max=K_eff·lr_s`
  (Eq. 4.2) for {sparse-late, sparse-early, dense-late} vs `Δℓ_init=ln k`. **PASS:** sparse-late budget
  `< ln k ≤` early/dense budget (init persists exactly in our regime, nowhere else).
- **T2 frustum scale.** (a) footprint identity `z/f` ↔ 1px (numeric). (b) emulated `distCUDA2`
  (mean-3-NN distance) on back-projected pixels subsampled by `k` equals `k·(z/f)` (over-size factor = k).
  (c) image-space splat (= blur by `s_img`) L1 increases with scale ⇒ over-size ⇒ blur. **PASS:** (b) ratio
  ≈ k for k>1 while ≈1 at k=1; (c) L1(oversize) > L1(footprint).
- **T3 orientation.** Closed-form + numeric: on-surface mass `erf(t/(√2 σ_n))` (Eq. 3.2); disk (`β=0.25`)
  on-surface ≈ 0.95 vs isotropic ≈ 0.38 at `t=σ_lat/2`; numeric sampling matches analytic. **PASS:** disk
  on-surface mass ≫ isotropic (off-surface haze cut by ≥10×) and numeric≈analytic.

Only if **T1 ∧ T2 PASS** (T3 is a bonus, near-certain) → implement §5 → controlled eval (§8).

---

## 8. Eval protocol (NO confound — the v5 lesson)

BDVR was judged against an old-session frgd number on *different* depth → uninterpretable. Never again:
1. **Same-depth baseline (required):** run `frgd` (densify_mode=frgd) on the *exact* prepared scene
   (`room_prep`), ≥2 seeds → the honest reference.
2. **FRGD-G:** run on the *same* `room_prep`, *same* seeds, only the init changed → isolates §5.
3. Report mean±std + worst-seed for PSNR/SSIM/LPIPS(vgg) and #Gaussians. (FRGD-G should match #Gaussians of
   frgd — same schedule — so any metric change is shape-only, not capacity.)

---

## 9. Honest expectations & pre-registered outcomes

- **Risk:** low (init-only, cannot lose +0.302; worst case tie). This is the requested "more certain" path.
- **Expected gain:** *modest.* T1 says the scale-correction benefit is concentrated in **late-added /
  under-observed** points (a fraction of all added), so PSNR gain may be small; the **anisotropy/orientation**
  benefit (§3, all points) most likely shows on **SSIM/LPIPS and depth/haze** (consistent with FRGD-refine
  already winning perceptual). Novelty rests on the **shape init + init-persistence theory**, not on a large
  PSNR jump.
- **Pre-registered:**
  1. If T1∨T2 **fail** (init is washed out even at 10k, or `distCUDA2`≈`z/f` always): FRGD-G has no lever →
     do **not** code it; fold "init-persistence does/doesn't hold" into the analysis paper.
  2. If coded and FRGD-G **> frgd** (same-depth, multi-seed) on PSNR or LPIPS significantly: method result.
  3. If FRGD-G **ties** frgd: report as "geometry-correct init does not beat optimizer-fixed init even in
     sparse-few-iter" — a clean negative that *completes* the densification study (placement ties, shape ties,
     suppression fails ⇒ the +0.3 is irreducibly 'just add depth capacity'); strengthens the analysis paper.

No metric reported without ≥2 seeds, mean±std, worst-seed, and the same-depth baseline.
