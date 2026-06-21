# DESIGN_AND_PROOF v8 — DIGS: Depth-Initialized Gaussian Splatting + geometry-faithful evaluation

**Status:** design + proof only. NO method code until the CPU gates (§7) pass + the decisive experiment (§8)
is run. Reuses everything proven: FRGD-G shape (v6, 13/13), CGD confidence-opacity (v7, 11/11), aligned depth.
**Thesis (unifies method #1 and evaluation #2):** the bottleneck of sparse-view 3DGS is the **initialization
geometry**, not how depth is used as supervision/mid-training densification — and the standard interpolated
hold-out PSNR is **blind to geometry**, which is *why* our seven "use-depth-cleverly" levers all tied. DIGS
fixes the initialization (dense, geometry-correct, confidence-gated depth back-projection at iteration 0); the
geometry-faithful evaluation makes the resulting geometry gain *visible*. They are one unit: **method improves
geometry; evaluation reveals it.**

---

## 0. The hidden assumption all 7 tied levers shared

| lever | axis varied | result |
|---|---|---|
| loss gating / refine-placement / shape / confidence / suppression / depth-quality | *how* depth is used | tie / hurt |

Every one kept **two things fixed**: (i) **sparse SfM initialization**, (ii) **capacity added MID-training**
(FRGD at iter ≥2000). We never varied **WHEN** capacity enters, nor **WHAT we measure**. Those are the two
unexplored axes — and our own theory says the first one matters.

---

## 1. Why timing is a real (untested) lever — our own init-persistence theorem

From v6 T1 (proven): a Gaussian added at iteration `t_add` receives
```
K_eff(t_add) ≈ (N − t_add) · (V_see / V_tot)   updates,                                            (Eq. 1.1)
```
and its init-error decays as `(1−ηH)^{K_eff}`. FRGD injects depth points at `t_add ∈ [2000, N]`; the late ones
get few updates → under-optimized → **this is precisely why FRGD-refine/shape/confidence washed out on them.**
Moving the *same* depth points to `t_add = 0` gives
```
K_eff(0) / K_eff(t_add) = N / (N − t_add)  > 1   (1.25× at t_add=2000 … 10× at t_add=9000)          (Eq. 1.2)
```
so iter-0 points are optimized 1.25–10× more → converge far closer to their photometric optimum (D1, D2). **The
densification benefit is realized MORE fully at iter 0, and the proven-correct shape/confidence mechanisms,
neutralized on under-optimized late points, can re-activate on fully-optimized iter-0 points.** Timing is not
"rearranging the same information" — it changes the **optimization budget per point**, a different axis.

**Literature corroboration (raises the prior this works):** FSGS (Unpooling) and NexusGS (flow dense-init)
obtain real gains substantially *from densifying/initializing more*, i.e. adding capacity early. Our +0.302 is
a *mid-training* version of the same lever. DIGS = the iter-0 limit, which (Eq 1.2) should be ≥ that.

---

## 2. DIGS method (reuses v6/v7, injected at iter 0)

At initialization, **augment** the sparse SfM cloud with dense depth back-projection:
```
for each train view i (aligned depth z_i, image I_i):
    sample a pixel grid p (stride s_pix to control density)
    X      = unproject_i(p, z_i(p))                       # back-projected SURFACE point (visibility-verified)
    color  = I_i(p)
    scale,rot = frgd_g_shape(X, cam_i)                    # v6: frustum z/f anisotropic disk, camera-facing
    conf   = multi-view reprojection agreement (v7)        # cross-view consistency
    o_init = 0.1 * conf                                   # v7: low-conf (wrong-depth) born faint -> pruned
voxel-dedup across views;  init cloud = SfM points (anchors) ∪ {X}
then: standard 3DGS training (densify + prune + uniform depth loss) -- unchanged.
```
- **Density (s_pix):** full 12-view back-proj is ~20M points; subsample to a target (~0.5–1M) and let 3DGS prune.
  Because scale uses the **frustum footprint z/f** (v6), the init is **correct at any subsample stride** (the
  thing distCUDA2 gets wrong, T2) — DIGS is exactly the use-case FRGD-G shape was built for.
- **Confidence gating (v7) finally has a clear job:** a dense back-projection contains many wrong-depth points
  (textureless/boundary); `o_init=0.1·conf` makes them faint → 3DGS prune removes the unrescued ones. (At
  iter 0 there is full optimization budget to rescue the right ones — the regime v7 needed.)
- Base 3DGS, depth loss, densify, rasterizer: **untouched.** DIGS is an initialization (a new `--init_mode`),
  composable with `densify_mode` and `gate_mode`.

**Limitations attacked (survey):**
- FSGS §1.1 (sparse init insufficient) → dense depth init directly (D3: covers textureless holes SfM misses).
- FSGS §1.2 (Euclidean unpool → empty space) → we back-project DEPTH; points lie ON the surface (D4).
- our 7-tie ceiling → varies the TIMING axis (Eq 1.2).
- §23.x → geometry-correct + confidence-weighted (reuses proven v6/v7).

---

## 3. The evaluation is part of the method (why #1 needs #2)

DIGS improves *initial geometry*. If the standard metric cannot see geometry, a real DIGS gain would look like
another "tie". **E1 proves interpolated hold-out PSNR is geometry-blind:** a floater consistent with a source
view produces a test-view displacement
```
Δpix(b) = b · f · | 1/z_F − 1/z_S |   ∝ baseline b between test and the view that fit it,               (Eq. 3.1)
```
so at a *near* (interpolated) test view (small `b`, e.g. mip360 hold-8 = every-8th frame) the floater is
~invisible (`Δpix < 1`), while at a *far* (extrapolated) view it is exposed (`Δpix ≫ 1`). Hence PSNR-on-
interpolated-holdout **cannot penalize wrong geometry** — exactly §23.6, and exactly why our geometry-improving
levers tied. The geometry-faithful evaluation therefore reports, in addition to interp PSNR/SSIM/LPIPS:
- **(M1) Extrapolation protocol:** hold out a *contiguous* block (large baseline) instead of every-8th →
  tests geometry, not interpolation. (Same scenes, only the split changes.)
- **(M2) Multi-view geometry consistency:** render depth from each train view, reproject, measure cross-view
  depth agreement (no GT needed) → floaters/haze lower it.
- **(M3) Efficiency:** PSNR vs #Gaussians (DIGS should reach equal quality with fewer points).

This is not metric-shopping: E1 *derives* that the geometry signal lives at large baseline / in consistency,
and is provably absent from small-baseline PSNR. Reporting M1–M3 is the correct measurement, and it is itself a
contribution (operationalizes §23.6).

---

## 4. The unified paper (method + the negative-7 as motivation + geometry eval)

Story: *"In lightweight sparse-view 3DGS, depth helps only as added capacity; supervision/placement/shape/
confidence/suppression/depth-quality are all metric-neutral under interpolated PSNR (we show 7 controlled,
multi-seed ablations). We show this is because (a) the gain is an initialization-geometry effect and (b)
interpolated PSNR is geometry-blind (Eq 3.1). DIGS moves the depth capacity to a dense, geometry-correct,
confidence-gated initialization, and — measured with geometry-faithful protocols — yields the gain that
interpolated PSNR hides."* The 7 negatives become the **motivation/ablation**, not wasted work; DIGS is the
method; M1–M3 is the measurement contribution.

---

## 5. Novelty positioning (honest)

| prior | dense geometry | DIGS delta |
|---|---|---|
| FSGS Gaussian-Unpool | Euclidean midpoints from sparse SfM | **depth back-projection** (on-surface, D4) at iter-0, **frustum-disk shape** + **confidence-opacity** |
| NexusGS | dense flow/epipolar init (heavy, flow-dependent) | mono/aligned-depth back-proj (lighter), + shape/conf + init-persistence theory |
| 3DGS/CoMapGS init | SfM / covisibility points | dense per-pixel depth surfels, confidence-gated, geometry-faithful eval |

Contribution = (i) **geometry-correct, confidence-gated dense DEPTH initialization** (depth-backproj + frustum
disk + conf-opacity, reusing v6/v7); (ii) the **init-persistence account** of *why* iter-0 > mid-densify in
sparse-few-iter (Eq 1.2); (iii) the **geometry-faithful evaluation** (Eq 3.1 + M1–M3) that exposes geometry
gains hidden by interpolated PSNR, with the 7-lever study as evidence. Honest: dense-init is established to
help (FSGS/NexusGS) — so the *risk it works is low*; the novelty is the specific depth-surfel-confidence init
+ the theory + the measurement, i.e. incremental-but-defensible, ACCV-tier.

---

## 6. CPU proof gates (tools/test_digs.py) — run BEFORE code

- **D1 budget advantage (Eq 1.2):** `K_eff(0)/K_eff(t_add)=N/(N−t_add)>1`, monotone increasing in `t_add`.
- **D2 closer convergence:** `(1−ηH)^{K_eff(0)} < (1−ηH)^{K_eff(t_add)}` for all `t_add>0` (iter-0 residual smaller).
- **D3 coverage:** dense depth-backproj covers textureless surface that texture-driven SfM misses (coverage
  dense ≫ SfM in low-texture region). Attacks FSGS §1.1.
- **D4 on-surface vs empty-space:** depth-backproj points lie on the surface (0% in empty space) while
  Euclidean-unpool midpoints fall in empty space for cross-surface neighbors (>0%). Attacks FSGS §1.2.
- **E1 interp-PSNR is geometry-blind (Eq 3.1):** floater displacement ∝ baseline; `Δpix(interp) < 1px`
  (invisible) while `Δpix(extrap) ≫ 1px` (exposed). Justifies M1–M3.

Only if **D1∧D2∧D3∧D4∧E1 PASS** → eligible to code DIGS init + the eval protocols.

---

## 7. Decisive experiment (isolate the TIMING lever; no confound)

- **DIGS vs FRGD, same depth, same shape/conf, #G-matched:** the *only* difference is injection time (iter-0
  dense vs mid-training selective). DIGS > FRGD on interp PSNR → timing is the hidden lever. Either way, also
  report **M1 (extrapolation), M2 (geometry-consistency), M3 (efficiency)** — DIGS should win these even if
  interp PSNR ties (E1).
- Controls: same aligned depth, same seeds (≥2), DIGS subsample stride tuned so final #G ≈ FRGD's.

---

## 8. Honest EV & pre-registered outcomes

- **EV:** highest of all directions tried — dense-init is *literature-confirmed* to help (unlike the 7 novel
  rearrangements), it varies a genuinely new axis (timing), and it reactivates our proven mechanisms at the
  right place. Main uncertainty is novelty-strength (vs FSGS/NexusGS), mitigated by the eval contribution.
- **Pre-registered:**
  1. Any of D1–D4/E1 fails → the timing/eval premise is unsound → do not code; report.
  2. DIGS > FRGD on interp PSNR (multi-seed, #G-matched) → timing is the lever → method result.
  3. DIGS ≈ FRGD on interp PSNR but wins M1/M2/M3 (multi-seed) → geometry gain hidden by interp-PSNR →
     method + measurement result (the §23.6 contribution); this is a *win*, not a tie.
  4. DIGS ≈ FRGD on interp AND M1/M2/M3 → initialization geometry is also not a lever here → the 7+1 negative
     becomes a strong analysis paper (information-limited ceiling, fully mapped).
- No metric without ≥2 seeds, mean±std, worst-seed, #G control, and the same-depth FRGD baseline.
