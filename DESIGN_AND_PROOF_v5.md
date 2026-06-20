# DESIGN_AND_PROOF v5 — BDVR: Bidirectional Depth-Verified Refinement

**Status:** design + proof only. NO method code is changed until the GO/NO-GO probe (§9) passes.
**Base:** vanilla depth-regularized 3DGS (Final1606) + FRGD (v4, already implemented & tested).
**One line:** FRGD only *adds* surface where it is *missing* (holes). BDVR adds the dual operation —
*reversibly remove* surface where it is *wrong* (floaters) — with **add** and **remove** driven by the
**same** multi-view geometric consensus (refined depth + SfM anchors). Capacity is *placed* where geometry
says surface IS, and *withdrawn* where geometry says surface IS NOT.

---

## 0. Why we need a NEW axis (the measured wall)

Multi-seed room ablation (n=5):

| Δ (frgd − rawdensify) | value | t | verdict |
|---|---|---|---|
| PSNR  | −0.022 | <1 | **tie** (noise) |
| SSIM  | +0.0042 | ≈3.9 | refine wins (perceptual) |
| LPIPS | −0.0043 | ≈3.3 | refine wins (perceptual) |

Reading: depth **densification** gives the +0.3 PSNR over `uniform`, but the *smart* part of FRGD
(refine + Fisher/texture targeting) only buys a **perceptual** sliver over naive raw-mono densify; on PSNR
it ties. Cause: densification happens in **holes = textureless** regions where refined depth ≈ raw mono
(few views to fuse), so placement is ~identical → PSNR identical.

**Crucial:** both `frgd` and `rawdensify` **only ADD**. Neither removes anything. The dominant remaining
sparse-view artifact — **floaters / background haze** (survey §0, §23.5; SparseGS; FreeNeRF) — is left
completely untouched. Floaters are an **orthogonal** failure mode, so a mechanism that attacks them can
**add gain on top of** the densification baseline instead of re-tuning a saturated axis. This is why BDVR
has a structurally better chance than NAGC / Fisher-gate / FRGD-refine (all of which re-tuned an axis the
baseline already exploited → all tied).

---

## 1. The exact limitation BDVR must overcome (and why prior fixes are unsafe)

A **floater** is a Gaussian whose photometric contribution is **sustained by the training views but is not
supported by multi-view geometric consensus.** In sparse view this is the defining pathology:

- The training reconstruction loss is **under-determined** (few rays). Many geometry configurations fit the
  training views equally well; some put mass on the true surface, some put semi-transparent mass *in front
  of* the surface (a floater) that happens to also fit the few training pixels. The photometric loss alone
  **cannot** break this tie — both are global minima on the training set.
- These floaters have opacity `o_i > 0` that **training loss actively keeps** (they help train pixels), so
  3DGS's own low-opacity prune **never** removes them — yet they **hurt held-out views** (parallax exposes
  them as haze / occluders).

**Why naive removal is dangerous (the limitation of SparseGS-style hard pruning, survey §23.5):**
hard-deleting "inconsistent" Gaussians in sparse view also deletes **legitimately under-observed** geometry
(thin structures, single-view-only valid surface) → irreversible damage. This is exactly the trap NAGC fell
into (a color-variance signal that was really a *texture* detector → it removed detail).

**Therefore BDVR must:** (i) break the train-loss tie with a **geometry prior** (not more photometric
optimization, which restores floaters); (ii) be **soft / reversible** so a false positive is *recovered* by
photometric evidence rather than destroyed; (iii) use a **geometrically valid** signal (depth/SfM
consensus), not a color proxy (the NAGC lesson).

---

## 2. Notation

- Views `{C_k}`, k=1..K, with extrinsics `(R_k, t_k)`, intrinsics `(f_k, c_k)`, image `I_k`.
- Gaussian `i`: center `μ_i ∈ R³`, opacity `o_i ∈ (0,1)`, scale `s_i`, SH color.
- Rendered (median/alpha) depth of view k: `D_k(u)`; the consensus surface seen by the current model.
- SfM points `P = {p_m}` — sparse but **reliable** samples on the true surface (the only externally
  anchored geometry).
- FRGD refined depth `D_ref^k` and reliability `rel^k` (v4) — robust multi-view median fusion of aligned
  mono depth; `rel` = cross-view agreement.
- Scene extent `ρ = cameras_extent`.
- Projection of `μ_i` into view k: pixel `π_k(μ_i)`, camera-space depth `z_{i,k} = (R_k μ_i + t_k)_z`.

---

## 3. The unified principle (FRGD and BDVR are one operator)

Define a **geometric support field** `S(x) ∈ [0,1]` over 3-space: how strongly multi-view consensus +
SfM say "true surface passes near x".

> **FRGD (add):** insert Gaussians where `S` is high but the current model is empty (a hole).
> **BDVR-suppress (remove):** withdraw opacity from Gaussians where `S` is low but the model has mass
> (a floater).

Both are the **same** statement — *match represented capacity to the geometric support field* — applied in
opposite directions. This is the conceptual novelty: **bidirectional, geometry-conditioned capacity
control** under a single consensus signal, instead of an add-only densifier or a stand-alone pruner.

---

## 4. The unsupportedness score φ_i  (the detector)

For Gaussian `i` define two **independent** support signals, then combine conservatively (AND-like, so a
Gaussian is flagged only when **both** geometry channels agree it is unsupported — this is the safety
margin).

**(A) Anchor support** — proximity to reliable surface samples:
```
d_i      = min_{ q ∈ P ∪ Surf_ref } || μ_i − q ||           # nearest reliable surface sample
GS_i     = exp( − d_i² / (2 r_s²) ),     r_s = κ · percent_dense · ρ      # ∈ (0,1], 1 = on surface
```
`Surf_ref` = back-projected refined-depth points (the same D_ref FRGD already computes). `r_s` is the
"surface shell" thickness. **Hard anchor protection:** if `d_i < r_s` (inside the shell of a real sample),
i is never suppressed regardless of other signals.

**(B) Consensus support** — does i lie ON the multi-view rendered surface, or float in front of it?
For each view k where i projects in-frame (`π_k(μ_i)` inside image, `z_{i,k} > 0`):
```
inlier_{i,k} = 1[ | z_{i,k} − D_k(π_k(μ_i)) |  <  τ · D_k(π_k(μ_i)) ]      # i is on the consensus surface
CS_i         = ( Σ_k vis_{i,k} · inlier_{i,k} ) / ( Σ_k vis_{i,k} )         # fraction of views agreeing
```
A genuine surface Gaussian is an inlier in (almost) all covisible views → `CS_i ≈ 1`. A floater that sits
**in front of** the consensus (occluding haze) is an *outlier* (`z_{i,k} < D_k`) in those views →
`CS_i ≈ 0`. (Pure occlusion behind the surface is invisible and harmless — handled because such i is never
the rendered depth, so it is down-weighted by the visibility/alpha weight below.)

**Unsupportedness (combine, conservative):**
```
φ_i = (1 − GS_i) · (1 − CS_i) ∈ [0,1]
```
`φ_i ≈ 1` ⇔ i is **far from every reliable sample AND off the multi-view consensus surface** ⇒ floater.
Either channel saying "supported" (near a sample, or on the surface in enough views) drives φ→0 ⇒ safe.

> **Why this is geometrically valid (not the NAGC trap):** φ uses *depth/position consensus + SfM anchors*,
> never raw color variance. Color variance conflates "floater" with "high texture" (NAGC's fatal bug);
> depth-consensus + anchor distance does not. This is the single most important correction over NAGC.

---

## 5. The suppression operator: a persistent geometric opacity prior (the mechanism)

**Naïve idea (REJECTED):** one-shot opacity decay `o_i ← o_i − η φ_i o_i` then keep optimizing.
*Why it fails:* harmful floaters are exactly the ones the **training loss sustains** (`o_i^* > 0` at the
train optimum). Decay-then-reoptimize converges back to the same `o_i^*` → the floater returns. A one-shot
kick cannot beat a standing photometric pull. (This is the subtle error that would have wasted a run.)

**BDVR mechanism (ADOPTED):** add a **persistent** prior term to the loss that *continuously competes*
with the photometric pull:
```
L_supp = λ_s · Σ_i  φ_i · ω_i · o_i
```
where `ω_i ∈ [0,1]` is the Gaussian's **mean rendered alpha-weight** over training views (so we penalize
*visible* unsupported mass, not invisible/irrelevant Gaussians). Total objective:
```
L = L_photo + λ_depth · L_depth + L_supp
```
Per-Gaussian opacity stationarity:
```
∂L/∂o_i = ∂L_photo/∂o_i + λ_s φ_i ω_i = 0
```
- **True floater** (φ_i ≈ 1): train-loss pull `∂L_photo/∂o_i` is *weak* (it only marginally helps a few
  train pixels), so the constant prior `λ_s φ_i ω_i` dominates → `o_i` driven down → falls below the
  low-opacity prune threshold → **removed permanently** at the next prune. The tie is broken **toward the
  geometry**, which is precisely the missing prior in sparse view.
- **Genuine but flagged Gaussian** (rare false positive, φ_i high but actually needed): `∂L_photo/∂o_i`
  is *strong* (loss rises sharply without it) → photometric term overrides the prior → `o_i` retained.
  **Self-correcting / reversible** — no hard deletion, exactly the property hard-pruning lacks.

So BDVR shifts the **fixed point** of opacity (persistent prior), not just its initialization — this is
why it removes train-sustained floaters that one-shot decay or 3DGS's own prune cannot.

**This is the dual of FRGD:** FRGD *adds* positive-value capacity at high-`S` holes; `L_supp` *withdraws*
capacity at low-`S` mass. Same consensus, opposite sign.

---

## 6. Schedule, cost, and safety guards

- **Schedule:** activate at `supp_start = 2000` (geometry formed, like FRGD). Recompute `D_k`, φ every
  `supp_interval = 1000` iters. **Cost ≈ free:** the all-view depth renders needed for φ are the *same*
  renders FRGD already does each interval → reuse them; per-iter `L_supp` is one masked sum.
- **Anchor protection:** `d_i < r_s` ⇒ φ_i := 0 (never suppress near a real sample).
- **Grace period:** FRGD-added points are exempt from `L_supp` for `grace = 1000` iters (let them settle
  before judging).
- **Rate cap:** clamp `λ_s φ_i ω_i` so opacity cannot drop by more than a fixed fraction per interval
  (avoid mass collapse from a transient bad D_k).
- **Stop early:** disable `L_supp` after `densify_until_iter` (no structural change in the fine-tune phase).

All guards bias toward *under*-suppression (false-negative over false-positive) — consistent with §1: in
sparse view, wrongly keeping a floater costs a little; wrongly deleting real geometry costs a lot.

---

## 7. Theory: the oracle ceiling — the rigorous "ensure it will be good"

We cannot promise a metric win blindly (the info-limited ceiling is real). Instead we make the guarantee
**measurable and conditional**:

**Definition (oracle gain).** Let `M(G)` be a held-out metric (PSNR/SSIM/LPIPS) of model `G`. For the
trained model `G₀` and a floater set `F`, the **oracle gain** is
```
U(F) = M( G₀ with o_i:=0 for i∈F ) − M( G₀ )
```
i.e. the metric change from *perfectly* deleting `F` with zero collateral. The §9 counterfactual probe
**measures `U` directly** over several candidate floater sets.

**Proposition (upper bound).** Any suppression method that withdraws a subset `F̂` of capacity achieves
held-out gain `≤ max_F U(F)`: you cannot beat perfect removal of the best set. BDVR realizes a fraction
```
ΔM_BDVR  ≈  η · U(F*) ,    η = recall(detector) × suppression_completeness ∈ (0,1].
```
With the persistent prior (§5) suppression_completeness is high for clearly-unsupported mass; with the
two-channel φ (§4) recall is conservative-but-real. Empirically expect `η ∈ [0.5, 0.8]`.

**Decision rule (GO/NO-GO):**
```
if max_F U(F) ≥ U_min (e.g. +0.1 PSNR or −0.003 LPIPS):  BDVR has headroom → BUILD.
else (U ≈ 0):                                            no harmful floaters → BDVR is futile → STOP,
                                                          pivot to the analysis paper. (honest)
```
Unlike depth-loss gating (which was **zero-sum by construction** → `U≡0`, why it was dead), floater
removal can have **U>0** because deleting net-negative-value capacity strictly lowers held-out error. The
probe tells us *before any training* whether that headroom exists.

**Predicted gain calculation.** If the probe returns, say, `U = +0.30 PSNR` for the best set, BDVR is
expected to deliver `ΔPSNR ≈ η·U ≈ +0.15…+0.24` on held-out, **on top of** the FRGD densification gain,
plus an LPIPS improvement (floaters hurt perceptual most). If `U ≈ +0.03`, expected `≈ +0.02` → not worth
a method claim. **The probe number is the contract.**

---

## 8. Novelty positioning (honest)

| Prior work | What it does | BDVR difference |
|---|---|---|
| SparseGS / FloaterNoMore | **hard** prune via depth/score heuristic | **soft, persistent, differentiable** opacity prior; false positives **self-recover** via photometric competition (overcomes irreversible-pruning danger, §1/§23.5) |
| DropGaussian / Dropout-GS | **random** opacity dropout (regularizer) | **targeted** by multi-view geometric unsupportedness φ, not random |
| CoMapGS / FSGS / FRGD-add | **add** depth-densified points | **bidirectional**: same consensus drives **add (FRGD) + remove (L_supp)** — one unified support field `S` |
| DNGaussian / depth-reg | opacity/depth loss on **rendered** quantities | per-**Gaussian** geometric prior with **SfM-anchor protection**, not a pixel loss |

Contribution bundle = (i) **bidirectional geometry-conditioned capacity control** (one field, add+remove);
(ii) **soft self-correcting suppression prior** that removes *train-loss-sustained* floaters which 3DGS's
own prune cannot, while being reversible for false positives; (iii) **oracle-gated, multi-seed honest
evaluation** with the U-bound. Honest caveat: this lives in the floater-removal family; the
differentiable-soft + bidirectional-unified + theory/eval rigor is the novelty, not "first to remove
floaters."

---

## 9. GO/NO-GO probe (cheap, no training) — must pass before any code

**Goal:** measure the oracle gain `U` on the existing trained `frgd` model. Counterfactual: delete
candidate floaters (opacity→0), re-render the **held-out** set, compare metrics. (Cell shipped separately;
loads the model via its saved `cfg_args` so the source path is exact.)

Candidate sets sweep `{low opacity} × {far from SfM} × {near-camera}`. Report, per set:
`#removed`, held-out `PSNR/SSIM/LPIPS`, and `ΔPSNR/ΔLPIPS` vs full.

**Interpretation:**
- `max ΔPSNR ≥ +0.1` or `min ΔLPIPS ≤ −0.003` ⇒ harmful floaters exist ⇒ **BUILD BDVR** (§4–6),
  then ablate `frgd` vs `frgd+supp` (isolate exactly `L_supp`) on 5 seeds.
- otherwise ⇒ **no headroom** ⇒ do **not** build; report this negative as part of the analysis paper.

---

## 10. Pre-registered outcomes (lock before running, anti-p-hack)

1. If probe `U ≤ 0`: BDVR abandoned; "no removable floaters at 10k on mip360-sparse12" is a recorded
   negative result.
2. If `U > 0` and `frgd+supp − frgd` is significant on **held-out** PSNR or LPIPS (multi-seed t-test,
   mean±std + worst-seed): BDVR validated → method paper spine = theory (Thm5/6 + §3 unified field +
   §7 oracle bound) with FRGD-add and BDVR-remove as the two instantiations.
3. If `U > 0` but `frgd+supp` ties `frgd` (detector η too low): report as "oracle headroom exists but
   practical detector cannot realize it" — still a publishable, honest finding about the gap between
   floater *identification* and floater *contribution*.

No metric is reported without mean±std over ≥3 seeds and the worst-seed. No selective threshold reporting:
the full candidate sweep table goes in the paper/appendix.
