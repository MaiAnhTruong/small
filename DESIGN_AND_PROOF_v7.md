# DESIGN_AND_PROOF v7 — CGD: Confidence-Guided Densification (attacking the depth-correctness bottleneck)

**Status:** design + proof only. NO method code changes until the CPU gates (§8) pass AND the empirical
depth gate (§9, Experiment-1) is run. Built ON TOP of FRGD (a new `densify_mode="cgd"`, base 3DGS untouched).
**One line:** the 5 prior results + the survey's "oracle gap" all say the bottleneck is **depth correctness,
not the loss/placement formula**. CGD attacks it two ways: **(L1)** densify from a multi-view-consistent
depth instead of raw monocular (empirical lever); **(L2)** give every densified Gaussian an **initial opacity
proportional to a geometric multi-view confidence**, so depth-uncertain points are born faint and self-prune
unless photometric evidence rescues them (novel, provable, safe — soft reversible pruning).

---

## 0. Why this, and an honest yellow flag first

Five experiments converge: depth-LOSS gating = dead; **densification = the only positive lever (+0.302)**;
refined *placement* ties naive; BDVR *suppression* fails; FRGD-G *shape* = perceptual trade-off. All five
**reused the same raw-monocular depth** and got clever about *using* it. The survey says exactly why this
saturates: **§2.5 (Depth-Regularized oracle gap) proves the ceiling is depth correctness**; §23.1/23.2/23.6
say the same. So the un-mined headroom is **making the depth used for densification more correct**.

**Yellow flag (kept front-and-center):** our `frgd ≈ rawdensify` ablation means FRGD's hard reliability
*filter* (`rel>rel_thr`) gave ~no PSNR gain over no-filter. So **reliability used as a placement filter is
empirically neutral here.** CGD must therefore (a) use reliability differently — as a *graded opacity* with
*automatic pruning*, not a keep/drop filter — and (b) lean its metric gain on the **depth-quality lever
(L1)**, which has never been tested. This doc proves L2 is *correct and safe*; §9 is the gate that decides
whether L1 (+L2) actually moves metrics. We do not claim a gain we have not measured.

---

## 1. Notation

- Views `{C_k}`, focal `f`, intrinsics `u=f x_c/z+c_x`. Aligned depth map `z_k(·)`; world point `X=unproj_k(p,z)`.
- Covisible set of pixel `p` in view `i`: `V_p = {j≠i : reproj of X is in-frame in j}`.
- 3DGS low-opacity prune threshold (train.py): `o_prune = 0.005`. Base densify opacity: `o_0 = 0.1`.
- Opacity LR: `lr_o = 0.025`; densification interval `Δ_d = 100` iters.

---

## 2. The two levers of CGD (and where each lives — all in the FRGD add-side)

- **L1 (depth source):** feed FRGD a multi-view-consistent depth (aligned/refined) instead of raw mono.
  *Data/input change* (`-d` + prep), **0 base code**. Tested by Experiment-1 (§9).
- **L2 (confidence-opacity):** compute per-pixel geometric confidence `conf(p)` (§3); set the densified
  Gaussian's **initial opacity `o_init = o_0 · conf`** (§4); rely on the *existing* 3DGS prune for soft,
  evidence-gated removal (§5). *Code:* one new util (`utils/cgd.py`) + `add_frgd_points` opacity arg
  (already parameterized for FRGD-G scales) + `frgd_step` computes/passes it. **Base 3DGS untouched.**
- **L3 (shape):** FRGD-G frustum disk — already built & proven (v6).

`densify_mode="cgd"` = FRGD refined placement + L1 depth + L2 conf-opacity + L3 shape.

---

## 3. Geometric confidence (reprojection agreement) — proven by P1

For pixel `p` in view `i` with depth `z_i(p)`, world point `X=unproj_i(p,z_i(p))`. For each `j∈V_p`, let
`ẑ_j(X)` = camera-depth of `X` in view `j` (= `project_j(X).z`), and `z_j(q)` = view `j`'s own depth at the
reprojected pixel `q=π_j(X)`. Relative disagreement and confidence:
```
r_{ij}(p) = | ẑ_j(X) − z_j(q) | / z_i(p)                                                          (Eq. 3.1)
conf_i(p) = exp( − (1/|V_p|) Σ_{j∈V_p} r_{ij}(p) / τ ) ∈ (0,1],   conf=1 ⟺ perfect agreement     (Eq. 3.2)
```
**Why this is the right signal (not an image proxy):** `conf` is low exactly where `X` is **multi-view
3D-inconsistent** — i.e. a point that *cannot* be a single surface seen by several cameras. That is the
*definition* of a would-be floater. So `conf` detects floater-prone depth **geometrically**, fixing the
image-edge≈depth-edge fallacy shared by FSGS/RDG-GS/DNGaussian (§23.x). It is NOT the monocular spread that
FRGD's `rel` used; it is reprojection-vs-rendered/own-depth residual, and (key) it is used for *opacity*, not
placement (§6). **P1** proves `conf` separates correct from wrong depth (AUC≈1, monotone in error).

*Honest scope:* `conf` cannot flag an error that is *3D-consistent across all views* — but such an error is,
by definition, a consistent surface (and a real one for photometric purposes). Per-view monocular errors are
independent → caught. (Stated as an assumption; tested by P1 with independent per-view perturbations.)

---

## 4. Confidence → initial opacity (the mechanism) — proven by P2

```
o_init(p) = o_0 · conf_i(p)          (o_0 = 0.1)                                                  (Eq. 4.1)
```
**Derivation (expected floater harm).** Treat `conf=c` as `P(point is true surface)`. A densified Gaussian of
opacity `o` contributes, in novel views, benefit `∝ o` if true and harm (wrong-depth/wrong-color mass = a
floater) `∝ o` if false. Over a densified set `{c_i}`:
```
E[harm]_naive = o_0 Σ_i (1−c_i)            (every point at o_0)
E[harm]_CGD   = o_0 Σ_i c_i (1−c_i)        (point i at o_0 c_i)
ratio = E[harm]_CGD / E[harm]_naive = ( Σ c_i(1−c_i) ) / ( Σ (1−c_i) ) ≤ max_i c_i < 1            (Eq. 4.2)
```
The harm reduction `(1−c_i)` weighting means it is **largest exactly for the low-`c` (most-likely-wrong)
points** — the floaters. Meanwhile the benefit on reliable points is **preserved**: for `c≈1`, `o_init≈o_0`
(≤(1−c) relative loss). For a uniform `c`, `ratio = E[c(1−c)]/E[1−c] = (1/6)/(1/2) = 1/3` → **67% expected
floater-harm cut** while high-confidence hole-filling is essentially untouched. **P2** verifies Eq. 4.2 +
benefit preservation numerically over confidence distributions.

This is a **reliability-weighted prior on "this Gaussian is real"**: the initial render becomes the
confidence-weighted estimate, and RGB optimization refines from there. It is *not* the hard `rel>thr` filter
(which tied); it is graded, and it shifts the **burden of proof onto photometric evidence** (§5).

---

## 5. Soft, reversible self-pruning (fixes §23.5) — proven by P3

CGD adds **no pruning code**. It relies on the *existing* 3DGS prune (`o < o_prune = 0.005`). From Eq. 4.1:
```
o_init < o_prune  ⟺  conf < o_prune/o_0 = 0.05.                                                   (Eq. 5.1)
```
So a point with `conf < 0.05` (strongly multi-view-inconsistent ⇒ almost surely a floater) is **born below
the prune line** → removed at the next densify *unless rescued*. **Rescue budget (safety vs §23.5).** A point
photometrically *needed* by even one view gets a consistent upward opacity gradient. In one densify interval
`Δ_d=100`, the number of updates from views that see it is `K_eff = Δ_d · (V_see/V_tot)`; Adam can raise
opacity by up to `Δo ≤ K_eff · lr_o`. For a *single-view-valid* point (`V_see=1, V_tot=12`):
```
Δo_max = (100 · 1/12) · 0.025 ≈ 0.21  ≫  o_prune = 0.005.                                         (Eq. 5.2)
```
⇒ **a faint point that ANY view genuinely needs is pulled above the prune line before the prune fires.** Only
points that **no view wants** (true floaters) stay faint and get pruned. This is the precise sense in which
CGD's removal is **soft, reversible, and evidence-gated**, overcoming the irreversible hard-pruning danger of
SparseGS (§23.5): we never delete; we *start uncertain points faint and let photometric evidence decide.*
**P3** verifies Eq. 5.1–5.2 and that the rescue budget exceeds the prune threshold across `V_see`.

---

## 6. Why CGD ≠ FRGD's reliability filter (the thing that already tied)

| | FRGD `rel` (tied) | CGD `conf` (this) |
|---|---|---|
| signal | mono-depth multi-view *spread* | reprojection depth *residual* (Eq 3.1), on aligned depth (L1) |
| use | **hard placement filter** (keep iff `rel>0.5`, survivors all at `o_0`) | **graded initial opacity** `o_0·conf` + existing-prune auto-removal |
| low-rel/conf point | dropped OR kept at full `o_0` (binary) | kept but **faint**, self-prunes unless rescued (graded, reversible) |
| target | which pixels to seed | floater-harm at init (Eq 4.2) + §23.5-safe removal |

So CGD is not "filter again": it is graded reliability *expressed as opacity*, which (i) reduces floater harm
proportionally (Eq 4.2), (ii) auto-prunes only the unrescued (Eq 5.2), (iii) needs no threshold tuning on
placement. Whether this **graded** form beats the **binary** form on metrics is the §9 same-depth ablation —
the proofs here establish it is *correct and strictly safer*, not that it must win.

---

## 7. Mechanism summary (surgical, FRGD-derived)

```
for each densified pixel p (view i) at refined/aligned depth z:
    X        = unproj_i(p, z)                         # placement: FRGD (unchanged)
    conf     = Eq 3.2 (reprojection agreement)        # NEW (utils/cgd.py)
    o_init   = inverse_sigmoid(0.1 * conf)            # NEW (Eq 4.1) -> add_frgd_points opacity arg
    scale,rot= frgd_g_shape(X, cam_i)                 # FRGD-G (v6, done)
add_frgd_points(X, color, scales, quats, opacities=o_init)
```
Changes: `utils/cgd.py` (conf, reuses `project_points`); `add_frgd_points` gains optional `opacities`
(currently hardcoded `inverse_sigmoid(0.1)`); `frgd_step` computes `conf`→`o_init`. Base 3DGS, depth loss,
densification, rasterizer: **all untouched**. `densify_mode∈{none,frgd,rawdensify,bdvr,frgdg}` unchanged.

---

## 8. CPU proof gates (run BEFORE any code; tools/test_cgd.py)

- **P1 conf detects wrong depth.** Synthetic plane + multi-cam; `conf` for on-surface vs off-surface (wrong)
  candidates. **PASS:** `conf(true)≈1`, monotonically decreasing in depth error, AUC(true vs wrong@10%)>0.95.
- **P2 harm reduction (Eq 4.2).** Over confidence distributions: `E[harm]_CGD/E[harm]_naive < 1`, concentrated
  on low-`c`; benefit on `c≥0.9` preserved (≥0.9×). **PASS:** uniform-`c` ratio ≈ 1/3; high-`c` benefit ≥0.9×.
- **P3 self-prune + rescue (Eq 5.1–5.2).** `o_init<o_prune ⟺ conf<0.05`; rescue budget `K_eff·lr_o > o_prune`
  for `V_see≥1`. **PASS:** threshold map exact; single-view rescue budget ≫ prune line.
- **P4 formula well-posed.** `conf∈(0,1]`, `=1` at zero residual, strictly decreasing, view-symmetric. **PASS.**

Only if **P1∧P2∧P3 PASS** → eligible to code (after §9 gate).

---

## 9. Empirical gate (the metric verdict — no confound, the BDVR/FRGD-G lesson)

- **Experiment-1 (the L1 lever, decisive):** FRGD on **aligned depth** vs **raw-mono depth**, same scene/seeds,
  same #Gaussians control. If aligned **wins clearly** → depth-correctness is the lever → build CGD. If
  aligned ≈ raw → the bottleneck is elsewhere → **stop** (no CGD). *(This is the one experiment that decides
  whether there is real headroom; cheap, 1 paired run.)*
- **Same-depth ablation (the L2 mechanism):** `cgd` vs `frgd` on the *same* depth, same seeds, #G-matched →
  isolates conf-opacity+shape. Required for the method (vs "just better depth") claim.

---

## 10. Honest expectations & pre-registered outcomes

- **Risk:** L2 is low-risk (init-only opacity, no active suppression; proven safe by P3) — worst case `cgd≈frgd`.
  The *gain* is uncertain and most likely rides on **L1 (better depth)**, untested → Experiment-1 gates it.
- **Pre-registered:**
  1. P1∨P2∨P3 fail → mechanism unsound/unsafe → do not code; report.
  2. Experiment-1: aligned ≈ raw → depth-correctness not the lever here → **stop CGD**, pivot to the analysis
     paper (the depth-correctness ceiling is itself the finding).
  3. Experiment-1 aligned > raw, and `cgd > frgd` (same depth, multi-seed, #G-matched) significant on PSNR or
     LPIPS → method result (gain = L1 + the L2 mechanism that makes using better depth safe/principled).
  4. aligned > raw but `cgd ≈ frgd` → "better depth helps; the conf-opacity mechanism is neutral over a hard
     filter" → honest negative that still supports the depth-correctness thesis (analysis paper).
- No metric without ≥2 seeds, mean±std, worst-seed, same-depth baseline, #G control.
