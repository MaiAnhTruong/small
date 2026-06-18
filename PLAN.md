# PLAN.md — VS-Depth → ACCV paper plan (locked before the 15-GPU-h grid)
### 2026-06-17. CHƯA đụng code. Mục tiêu: paper NOVEL, tầm ACCV. Duyệt bản này → mới chạy grid 1 lần.

---

## 0. Thesis một câu
> *Tiến bộ trong sparse-view 3DGS được báo cáo single-run, trong khi phương sai-theo-seed lớn hơn khoảng cách giữa các method. Ta (i) đo điều đó lần đầu (multi-seed), (ii) chứng minh depth-supervision là bộ giảm-phương-sai có trọng số tối-ưu-MSE = **photometric Fisher information** (mà "đếm covisibility" của CoMapGS chỉ là proxy mất-mát), và (iii) đề xuất Fisher-gated depth, đánh giá bằng mean±std + worst-case-seed trên consumer GPU.*

**Working title:** *"How Real Are Sparse-View 3DGS Gains? Seed Variance and a Fisher-Optimal Depth Gate."*

---

## 1. Ba đóng góp (claims)
- **C1 (headline, methodological — NOVEL):** lần đầu đo seed-variance của depth-supervised sparse-view 3DGS; std-seed ≥ gap giữa các method (cả field báo cáo single-run) ⟹ nhiều "improvement" có thể nằm trong nhiễu. Đề xuất giao thức **worst-case-seed**.
- **C2 (theory — ĐÃ CHỨNG MINH CPU):** depth = variance-reducer (`V ∝ 1/H`); trọng số depth tối-ưu-MSE `a*(H,δ)=cH/(2δ²H−c)`; `H` = photometric Fisher info `Σ_j vis_j(b_j f/z²)²|∇I_j|²`; CoMapGS dùng view-count = proxy bỏ texture+geometry. (`test_vsdepth_theory.py` 21/21, `test_fisher_gate.py` 8/8.)
- **C3 (method):** Fisher-gated depth (`gate_mode=fisher`), DAv2-only, đánh giá đa-seed (mean±std + worst-case) so với none / uniform / covonly(≈CoMapGS).

---

## 2. RANH GIỚI CLAIM (trung thực — đọc kỹ, chống overclaim)
| Claim | Bán được | KHÔNG được bán |
|---|---|---|
| C1 | "gap báo cáo của field ⊂ dải nhiễu-seed; ta đo lần đầu cho sparse-depth-3DGS; worst-case là metric đúng hơn" | "mọi method là rác" (chỉ nói *unverified*, không phủ định) |
| C2 | tối-ưu-MSE + Fisher = đại lượng đúng (proof CPU) | "đã cải thiện PSNR" (theory ≠ metric thật) |
| C3 | "depth-gating cải thiện mean so với no-depth ĐÁNG TIN (đa-seed); Fisher **nguyên lý hơn** + ngang/nhỉnh covonly, ăn ở vùng textureless/worst-case" | ❌ "gate của tôi = variance-reducer" (giảm-variance là tính chất của DEPTH nói chung; uniform thậm chí over-damp std thấp hơn). Gate-edge = **MSE/accuracy + texture-aware**, KHÔNG phải variance. |

**Hệ quả khung paper:** đây là **METHOD paper** (gate = hook), seed-study là *động lực + rigor*. KHÔNG đóng khung là benchmark thuần (ACCV main track ít nhận). Paper đứng vững **dù** `fisher ≈ covonly` ở mean, vì đóng góp = C1 (seed-study) + C2 (Fisher theory) + eval trung thực — KHÔNG phụ thuộc một số SOTA.

**Kỳ vọng calibrate:** Framing này cho **một cửa thật**, KHÔNG phải vé chắc. Borderline-but-plausible ACCV nếu làm sạch.

---

## 3. Novelty & positioning (đã lit-check)
| Công trình | Họ làm | Ta KHÁC |
|---|---|---|
| **NerfBaselines** (NeurIPS-DB'24) | consistent eval *protocol* (cài đặt, metric đồng nhất) | ta đo **phương sai-SEED** + significance, riêng cho sparse-depth; + một method |
| **CoMapGS** (CVPR'25) | covisibility-**count** gate + point enhance; single-run | ta dùng **Fisher H** (texture+geom, count là proxy lossy) + **đa-seed** + theory |
| **UGOT / Pi-GS** | depth-uncertainty (của **mạng**) weighting; Pi-GS cần π³ (nặng) | ta dùng **photometric Fisher** (hình học, không phải uncertainty-mạng); **DAv2 consumer GPU** |
| **FSGS / DNGaussian** | depth global/normalized; single-run | ta **gated** + theory + seed-study |
| **FisherRF** (ECCV'24) | Fisher info để **chọn view** | ta dùng Fisher info để **đặt trọng số depth** |
| **Co-Adaptation** 2508.12720 / **Analysis-of-Converged-3DGS** 2602.08909 | dropout-instability / density-variance | KHÔNG phải depth, KHÔNG phải seed-variance-of-gains |

**Khe trống (novel triple):** [first multi-seed reliability study cho sparse-depth-3DGS] × [photometric Fisher làm trọng số depth + a* closed-form] × [consumer-GPU + worst-case eval]. Tổ hợp này chưa bị chiếm. Novelty **modest nhưng thật**.

---

## 4. PROTOCOL GRID (chạy DUY NHẤT 1 lần — khoá, không tinh chỉnh sau)
- **Base:** repo này (3DGS depth-reg), commit ≥ 2ec638c (có `gate_mode=fisher`).
- **Scenes (3, phủ phổ texture):** `garden` (textured, blind-spot 2% → kỳ vọng fisher≈cov), `room` (19%), `counter` (21%). *Loại bicycle/stump (depth căn <50% → ghi rõ lý do loại trong paper).*
- **Configs (4):** `none` / `uniform` / `covonly`(≈CoMapGS) / `fisher`(ours).
- **Seeds (5):** 0,1,2,3,4 (qua `GS_SEED`).
- **Cố định (KHÔNG tune theo scene):** `-r -1` (1600), `iterations=10000`, `position_lr_max_steps=10000`, test = hold-8 (sẵn trong data), depth = DAv2-Base + affine-align-SfM (`compute_depth_params`). Hyperparams gate: `cov_start=2000, cov_interval=1000, cov_max_dim=200, cov_tau=0.05, gate_gamma=1.0, gate_rel_sigma=0.10, fisher_c=0.5, fisher_cap=8.0`.
- **Metrics:** PSNR, SSIM, LPIPS(vgg). Báo **mean, std, worst-case** (minPSNR / minSSIM / maxLPIPS) qua 5 seed.
- **Quy mô:** 4×5×3 = **60 run × ~15’ ≈ 15 GPU-h** (2 phiên Kaggle). Chạy MỘT lần.
- **Bắt buộc:** in `[DEPTH CHECK] {scene}: X/12 aligned` đầu mỗi scene (X≥6 mới tin).

*(Camera-ready sau sẽ mở rộng lên ~7–9 scene mip360 + reproduce CoMapGS thật; grid 3-scene này là để QUYẾT go/stop + có số lõi.)*

---

## 5. PRE-REGISTERED outcomes + luật quyết định (viết TRƯỚC khi chạy — chống p-hack)
**Giả thuyết:**
- **H1:** `depth (uniform/covonly/fisher) ≫ none` ở mean PSNR, vượt std. *(kỳ vọng mạnh)*
- **H2:** gap giữa `uniform/covonly/fisher` NHỎ, có thể < std. *(đây chính là điểm C1)*
- **H3:** `fisher ≥ covonly` ở mean trên textureless (room/counter); ≈ trên garden.
- **H4:** `fisher` cải thiện **worst-case-seed** vs covonly/uniform (ổn định hơn ở seed xấu).
- **H5:** std-seed (đo được) ≥ gap-giữa-config AND ≥ gain per-scene mà CoMapGS/Pi-GS báo cáo (~0.2–0.5 dB).

**Luật quyết định (cam kết trước):**
- **STRONG (viết method paper C1+C2+C3):** H1 đúng **và** (H3 *hoặc* H4 đúng với biên rõ trên room/counter).
- **MEDIUM (paper dựa C1+C2; C3 lùi về "ngang covonly nhưng nguyên lý hơn"):** H1 đúng nhưng fisher≈covonly mọi nơi trong nhiễu. Vẫn nộp được nếu C1 sạch (H5 mạnh).
- **STOP/debug:** H1 SAI (depth không hơn none đáng tin) → setup hỏng (vd config 30k-ở-10k), sửa ARG rồi chạy lại; nếu vẫn → dừng.
- **Anti-p-hack:** KHÔNG tune hyperparam theo scene; báo CẢ garden (bất lợi); std luôn hiển thị; không chọn-lọc seed.

---

## 6. Bảng & hình (grid phải sinh ra đúng các thứ này)
- **T1 (chính):** scene × config → PSNR/SSIM/LPIPS **mean±std** + **worst-case**.
- **T2 (Δ ablation):** Δ(fisher−covonly), Δ(fisher−uniform), Δ(depth−none) ± std, per-scene + trung bình.
- **F1 (HEADLINE C1):** mỗi scene, dải ±std PSNR của từng config, **overlay** độ lớn gain báo cáo của CoMapGS/Pi-GS (~0.2–0.5 dB) → cho thấy gap ⊂ dải nhiễu.
- **F2 (định tính):** bản đồ gate **count vs Fisher** trên vùng textureless + render RGB/depth → ít floater nơi Fisher bơm depth.
- **T3 (phụ lục):** tóm tắt CPU proofs (21/21, 8/8) + blind-spot thật per-scene (room 19%, counter 21%, garden 2%; TB 21.7%).

---

## 7. SKELETON paper (ACCV)
1. **Introduction** — sparse 3DGS overfit; depth giúp; *nhưng tiến bộ có thật không?* → đặt vấn đề seed-instability + giới thiệu Fisher-gate. Liệt kê 3 đóng góp.
2. **Related Work** — sparse-view GS (FSGS/DNGaussian/DropGaussian/CoR-GS); depth-gating (CoMapGS/UGOT/Pi-GS); Fisher/uncertainty (FisherRF); eval/reproducibility (NerfBaselines). Nêu khe (mục 3).
3. **Preliminaries** — 3DGS render, depth loss (L1-inv + affine-SfM), sparse setting + hold-8.
4. **Method**
   - 4.1 Depth as a variance reducer: `V∝1/H`, trọng số `a*(H,δ)` (C2).
   - 4.2 `H` = photometric Fisher info; covisibility-count là proxy mất texture+geometry; ca "textureless-but-covisible".
   - 4.3 Fisher-gated depth: thuật toán (recompute định kỳ trong `compute_gates`), strict-gen về uniform.
5. **A Reliability Protocol for Sparse-View 3DGS** — multi-seed, mean±std + worst-case; lập luận significance (C1).
6. **Experiments** — setup (mục 4); T1/T2; **F1** (seed-noise vs reported gains); **F2**; ablations (`fisher_c`, `cap`, `cov_interval`); blind-spot analysis (T3).
7. **Discussion & Limitations** (trung thực) — gains-within-noise; quy kết depth-vs-gate; phụ thuộc scene (textured→fisher≈cov); SfM-sparse fragility (bicycle); DAv2 bias.
8. **Conclusion.**

---

## 8. RỦI RO & KILL-CRITERIA
- **C1 bị scoop** — thấp (đã lit-check; NerfBaselines = protocol, khác). *Mitigate:* định vị rõ; nếu phát hiện trùng → C1 lùi, dựa C2+C3.
- **fisher ≈ covonly** — trung bình-cao. *Mitigate:* khung MEDIUM (mục 5); worst-case + textureless-subset có thể cứu; nếu hoà tuyệt đối → paper dựa C1+C2, C3 = "principled, on-par".
- **depth không hơn none** — thấp. *Kill/Debug:* sửa config (densify_until/opacity-reset cho 10k, hoặc chạy 30k). Nếu vẫn → DỪNG.
- **Venue-fit (benchmark-y)** — *Mitigate:* đóng khung method (gate hook), không benchmark thuần.
- **Compute** — 15 GPU-h, chặn được; chạy 2 phiên.

---

## 9. Compute & thứ tự thực hiện
1. (NOW) Bạn duyệt PLAN.md này.
2. Chạy grid 60-run (2 phiên) → `summary.json` (mean±std + worst-case per scene×config).
3. Đối chiếu mục 5 → chọn STRONG / MEDIUM / STOP.
4. Nếu STRONG/MEDIUM → viết theo skeleton (mục 7) + mở rộng scenes cho camera-ready.

---

## 10. Checklist trước nộp ACCV
- [ ] ≥5–7 scene mip360 (mở rộng từ 3) + ghi rõ scene bị loại & lý do (depth-align).
- [ ] error bars (±std) ở MỌI bảng; worst-case-seed.
- [ ] reproduce CoMapGS thật (hoặc lập luận covonly = proxy trung thực) trên cùng protocol.
- [ ] F1 (seed-noise vs reported-gains) — hình headline.
- [ ] CPU proofs + real-data blind-spot ở phụ lục; code release (repo small.git).
- [ ] Limitations trung thực (mục 7.7).

*Nguồn đã kiểm: 3DGS (Kerbl'23); FSGS (ECCV'24); DNGaussian (CVPR'24); CoR-GS (ECCV'24); DropGaussian (CVPR'25); CoMapGS (CVPR'25); UGOT (2405.19657); Pi-GS (2602.03327); FisherRF (ECCV'24); NerfBaselines (2406.17345); Co-Adaptation (2508.12720); Analysis-of-Converged-3DGS (2602.08909); MVS triangulation-uncertainty.*
