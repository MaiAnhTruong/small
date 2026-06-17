# VS-Depth — Variance-Stable Covisibility-Gated Depth Supervision for Sparse-View 3DGS

Built on vanilla 3DGS (graphdeco-inria/gaussian-splatting). The base already does **uniform** monocular-depth
regularization: `L_depth = |(invDepth − mono_invdepth) · depth_mask|.mean()`, `depth_mask = ones`.
**VS-Depth replaces the uniform `depth_mask` with a per-pixel gate `w = g_cov(cov) · rel`** that — by a
bias-variance analysis (proven in `tools/test_vsdepth_theory.py`, 21/21) — is the **MSE-optimal allocation**:
put the depth prior where photometric supervision is weak (low covisibility = under-observed = high
seed-variance) and back off where mono-depth is unreliable (depth edges).

## Why (the gap, confirmed by reading the competitors)
- CoMapGS (covisibility, **static** MASt3R), UGOT / Pi-GS (depth-uncertainty) — **none report seed variance**,
  yet their gains (~0.2 dB) are within the ~1.6 dB seed-noise of sparse 3DGS. None combine an **exact, dynamic**
  covisibility gate with depth reliability, and Pi-GS's π³ is "infeasible on consumer hardware".
- VS-Depth: exact geometric covisibility (from poses + rendered depth, **dynamic**) × mono-depth reliability,
  **DepthAnything-V2 only** (T4-friendly), evaluated **multi-seed (mean ± std + worst-case)**.

## `--gate_mode` (the only change vs base; clean controlled ablation)
| mode | depth weight | = which method |
|---|---|---|
| `none` | — (no depth) | vanilla 3DGS, ablation (i) |
| `uniform` | `depth_mask` (ones) | original 3DGS depth-reg, ablation (ii) ≈ FSGS/global depth |
| `covonly` | `g_cov(cov)` | ablation (iii) ≈ CoMapGS covisibility gate |
| `gated` | `g_cov(cov)·rel` | ours v1 (iv) |
| `fisher` | `a*(H,δ)`, `H=Σ vis·(bf/z²)²·\|∇I\|²` | **ours v2** — photometric Fisher info (see `DESIGN_AND_PROOF_v2.md`) |

**v2 (FIG-Depth):** the curvature `H` in the theory = photometric Fisher info = MVS triangulation info, NOT
raw view count. CoMapGS's count drops texture `|∇I|²` + geometry `(bf/z²)²` → wrong proxy of `H`; it is blind
to "textureless-but-covisible" regions (low H = under-constrained, but count high → CoMapGS under-supervises).
`--gate_mode fisher` sets the depth weight to the MSE-optimal `a*(H,δ)`. Proven: `tools/test_fisher_gate.py`
(8/8, −31% MSE vs count = 22% optimal-form + 11% texture/geom signal), `tools/test_fisher_code.py` (8/8).
Real-data check `tools/fisher_realdata_check.py`: on mip360 sparse-12, Spearman(count,H)=0.44 (≠1) and **21.7%**
of points are the CoMapGS blind spot (room 19%, counter 21%, bicycle 33%; garden 2% = texture-rich, fisher≈cov).

`g_cov = 1/(1+cov)^gamma` (decreasing in covisibility); `rel = exp(−|∇ mono_invdepth|/sigma)`; gate mean-pinned
to 1 (same depth budget as uniform). At `gate_mode=uniform`/`none` the code is bit-identical to base.

## Pipeline (Kaggle)
```bash
# 0) build rasterizer/simple-knn/fused-ssim with the standard header-patch recipe (same as before)
# 1) mono depth + scale (TRAIN split only, per scene):
python prepare_depth.py --images_dir <scene>/train/images --out_dir <scene>/train/depths
python utils/make_depth_scale.py --base_dir <scene>/train --depths_dir <scene>/train/depths --model_type bin
# 2) train the 4 ablation arms (multi-seed):
for MODE in none uniform covonly gated; do
  for SEED in 0 1 2; do
    GS_SEED=$SEED python train.py -s <scene> -d depths --eval -r -1 \
        --gate_mode $MODE -m out/<scene>_${MODE}_s${SEED} --test_iterations 30000 --save_iterations 30000 --quiet
    python render.py -m out/<scene>_${MODE}_s${SEED}
    python metrics.py -m out/<scene>_${MODE}_s${SEED}
  done
done
```
Data: lightning split `<scene>/{train,test}/{images, sparse/0}` (auto-detected: `CustomHold8Split`).

## Headline metric (proven framing)
Report **mean ± std and WORST-CASE over seeds**. Expected (and what the theory says): `gated` lowest MSE/error,
`gated < covonly < uniform` in error; all depth modes ≫ `none` in stability. We do **not** claim "lowest std"
(uniform over-damps to lower raw std but worse error — see proof P9).

## Verified locally (CPU, before any GPU)
- `tools/test_vsdepth_theory.py` — bias-variance theory, **21/21 PASS** (V∝1/H, var-reduction, MSE-optimal gate).
- `tools/test_covisibility.py` — projection/occlusion/gate logic, **11/11 PASS**.
- `tools/signal_test_real.py` — real mip360 sparse-12: **4/5 scenes have structured covisibility** (go).
- Final metric verdict requires the Kaggle multi-seed run above (no GPU rasterizer locally).

Design + full proof: `DESIGN_AND_PROOF.md` (in the vsdepth design folder).
