# VS-Depth v4 — FRGD: Fisher-Reliability-Guided Densification
### Thiết kế + chứng minh v4 (2026-06-20). CHƯA sửa code. Duyệt + verify CPU (T4,T6) xong mới code.

> Đo được (room, 5 seed): `none 18.27 → uniform 19.18 → covonly 19.13 → fisher 19.05`. Depth giúp lớn (+0.9 dB)
> nhưng **gating depth-LOSS thua uniform** (zero-sum). v4: chuyển depth từ **LOSS (zero-sum, đã chết)** sang
> **DENSIFICATION (non-zero-sum, lever đã chứng minh thắng)**, điều khiển bởi Fisher H (v2) + refined depth (v3)
> + theory (v3). Luận điểm trung tâm — **MỚI**: *sparse-view depth quá thiếu chính xác để làm LOSS (bias vĩnh
> viễn), nhưng đủ để làm INIT (chỉ cần đúng gần đúng, RGB tinh chỉnh sau).*

---

## 0. Ba đóng góp
- **C-THEORY (v1–v3, đã PASS CPU):** bias-variance của depth supervision; ngưỡng `δ<δ_thresh`; `a*(H,δ)`; hợp nhất + giải thích loss-gating thất bại.
- **C-GAP (MỚI, novelty lõi):** chứng minh **"densification gap"** — 3DGS densify ∝ texture, nên **bỏ sót đúng vùng low-H (bất định cao nhất)**; depth-LOSS không vá được (chỉ reweight Gaussian sẵn có), depth-DENSIFICATION vá được (thêm capacity). + **luận điểm precision: init bao dung lỗi depth hơn loss rất nhiều.**
- **C-METHOD (FRGD):** densification có-kiểm-chứng tại `D_ref`, ở vùng `H` thấp, khi `δ_ref<E_hole` → cải thiện *non-zero-sum*, đáng tin.

---

## 1. THÔNG TIN & GIẢ ĐỊNH (xác định rõ — "đảm bảo mọi thứ")

### 1.1 Biết CHÍNH XÁC
`P_i,K_i` (pose+intrinsic) · `{X_k, track_k}` (điểm SfM = **depth THẬT thưa**, neo + validate) · `I_i` (ảnh → `|∇I|`).

### 1.2 Ước lượng (có lỗi)
`D_mono^i` (DAv2 căn-SfM, lỗi δ_mono) · `D_ref` (refined đa-view, lỗi δ_ref) · `D_render` (model) · `Ĥ` (Fisher) · `c` (reliability) · `E_cur` (lỗi hình học hiện tại của model tại vùng).

### 1.3 Giả định
- **A1–A5** (v3): local-quadratic, SGD noise `V=σ²/2H`, depth term curvature `H_d` kéo về `θ_true+δ`, photometric ~unbiased nơi đủ texture, local constancy. *(A1,A2 đã CPU-PASS v1.)*
- **A6 (kiểm T3):** `δ ↔ H` tương quan **dương** (low-H = ít view/textureless = mono sai nhiều).
- **A7 (kiểm T4):** fusion đa-view **giảm** sai số mono (`δ_ref<δ_mono`) tại điểm SfM.
- **A8 (kiểm T6, MỚI):** **densification chuẩn của 3DGS ∝ texture** → vùng low-H **dưới-tái-tạo** (ít Gaussian/lỗ). Proxy CPU: mật độ điểm SfM ∝ texture → low-texture **khởi tạo thưa sẵn**.
- **A9 (kiểm T6/GPU, MỚI):** vùng dưới-tái-tạo có `E_cur` lớn ≫ `δ_ref` → đặt điểm tại `D_ref` **giảm lỗi**.

---

## 2. PHẦN I — Lý thuyết nền (v1–v3, tóm tắt)
- **Bổ đề 1:** `V=σ²/(2H)`. **Bổ đề 2:** depth `H_d` → `V'=σ²/(2(H+H_d))`, bias `H_dδ/(H+H_d)`.
- **Định lý 1 (ngưỡng):** depth giúp ⟺ `δ² < δ_thresh²=(σ²/2)(H+H_d)/(H H_d)`.
- **Định lý 2 (tối ưu):** `a*(H,δ)=cH/(2δ²H−c)`, giảm theo cả H,δ.
- **Định lý 3 (hợp nhất + thất bại loss-gating):** coverage-gate = `a*(H)` bỏ δ; dưới A6 → bơm depth nơi `δ²>δ_thresh` → net âm (khớp đo `uniform>covonly>fisher`).

---

## 3. PHẦN II — LÝ THUYẾT MỚI: vì sao DENSIFICATION thắng nơi LOSS thua

### Định lý 5 — "Densification gap" (cốt novelty)
*3DGS densify một Gaussian khi trung bình `‖∇_{μ2D} L_photo‖ > τ`.* Mà
$$\big\lVert \nabla_{\mu_{2D}} L_{photo}\big\rVert \approx \underbrace{|r|}_{\text{residual}}\cdot \underbrace{|\nabla I|}_{\text{texture}}$$
(dịch vị trí 2D của Gaussian đổi ảnh **tỉ lệ gradient ảnh cục bộ**). Vì `H ∝ Σ|∇I|²`:
$$\text{densification-signal} \propto |\nabla I| \propto \sqrt{H}.$$
⟹ **vùng `H` thấp (ít texture/ít view) nhận densification ÍT NHẤT.** Nhưng theo Bổ đề 1, đó **chính là vùng phương sai hình học CAO NHẤT** (`V=σ²/2H` lớn). 

**Kết luận:** tồn tại *lệch pha cấu trúc* — nơi cần thêm hình học nhất (low-H) thì 3DGS **không densify**. Depth-LOSS **không sửa được** (nó chỉ reweight Gaussian ĐANG CÓ, không tạo capacity ở chỗ trống); depth-DENSIFICATION **sửa được** (thêm Gaussian, độc lập texture). ∎

> Đây là phát biểu mạnh: **giải thích vì sao depth-loss bão hoà** (không tạo được geometry ở lỗ) **và vì sao densification là lever đúng.**

### Định lý 6 — Init bao dung lỗi depth hơn LOSS (luận điểm trung tâm)
- **LOSS** giúp ⟺ `δ < δ_thresh` (Định lý 1) — **thất bại** ở low-H-high-δ (A6). Bias do loss là **vĩnh viễn** (kéo nghiệm về `θ_true+δ`).
- **DENSIFICATION**: đặt Gaussian mới tại `D_ref` (lỗi `δ_ref`) ở vùng lỗi hiện tại `E_cur`. Gaussian mới là **DOF mới mà RGB sẽ tối ưu tiếp** → lỗi cuối ≈ `min(δ_ref, RGB-achievable)`. Đặt điểm **giúp ⟺ `δ_ref < E_cur`** — điều kiện **YẾU** vì ở lỗ/floater `E_cur` rất lớn. Hơn nữa (basin-of-attraction): depth chỉ cần đưa Gaussian vào **vùng hút** của photometric, **không cần dưới `δ_thresh`**.

**Hệ quả (định lượng so sánh):** trong vùng low-H-high-δ:
- loss: cần `δ < δ_thresh` (thường KHÔNG đạt) → hại.
- densify: cần `δ_ref < E_cur` (thường ĐẠT vì E_cur lớn) → giúp.
⟹ **densification có "biên dung sai depth" lớn hơn loss đúng tại vùng loss thất bại.** ∎

> Luận điểm bán được: *"Sparse-view depth quá nhiễu để làm LOSS (bias vĩnh viễn), nhưng đủ để làm INIT (gần đúng + RGB sửa). Vì vậy hãy DÙNG DEPTH ĐỂ ĐẶT ĐIỂM, ĐỪNG ĐỂ PHẠT."* — đi ngược trend depth-loss/gating, có chứng minh + có số đo hậu thuẫn (loss-gating đã âm).

---

## 4. PHẦN III — METHOD: FRGD

Định kỳ (mỗi `K` iter, như recompute v2), với mỗi train view:
1. **Bản đồ thiếu hụt (WHERE):** tính `Ĥ` (Fisher, v2) + so `D_render` với `D_ref` → đánh dấu vùng **low-H + lệch lớn** = dưới-tái-tạo/floater/lỗ.
2. **Refined depth + reliability (v3):** `D_ref` = fuse robust {mono i, mono lân cận warp về (occlusion-checked), **SfM**}; `c` = đồng thuận; `δ̂_ref` = độ tản.
3. **Quyết WHETHER (theory):** seed Gaussian mới tại back-proj `D_ref` chỉ khi **(low-H) ∧ (c cao) ∧ (`δ̂_ref < E_cur` ⇐ Định lý 6)**. Màu = ảnh tại pixel; scale nhỏ (≈ k-NN); opacity vừa.
4. **RGB tinh chỉnh:** Gaussian mới tham gia optimization bình thường → RGB kéo về đúng (basin). **Không** áp depth-loss vĩnh viễn lên chúng (tránh bias — đúng Định lý 6).
5. (tùy chọn, sau) prune floater multi-view-bất-nhất — rủi ro, để pha 2.

Đầu vào dùng đúng §1: Fisher (∇I) + refined depth (mono+đa-view+SfM) + render depth. **Không hallucinate** (khác diffusion/pseudo-view); **SfM neo** chống đồng-thuận-sai.

---

## 5. Vì sao v4 > v2/v3 (novelty CAO hơn + đáng tin hơn)
| | loss-gating (v2/v3) | **FRGD (v4)** |
|---|---|---|
| Lever | loss zero-sum → **đo CHẾT** | **densification non-zero-sum → lever đã chứng minh thắng** |
| Dung sai depth | cần `δ<δ_thresh` (chặt) | cần `δ_ref<E_cur` (lỏng) + RGB sửa (Định lý 6) |
| Kỳ vọng metric | parity/âm | **dương** (thêm surface đúng chỗ 3DGS bỏ sót) |
| Novelty | "một gate nữa" | **densification-gap (Đl5) + init-vs-loss precision (Đl6) + contrarian "đặt điểm, đừng phạt"** |

---

## 6. LIMITATION đánh trúng (map survey của bạn)
- **FSGS §1.2** (Euclidean midpoint → empty space): FRGD đặt tại **depth kiểm-thị-giác+đa-view** → sửa đúng.
- **CoMapGS** (unproject **raw** mono, 1 lần, covis=0): FRGD = **refined** depth + **ongoing** + **Fisher-guided** + theory-thresholded.
- **§23.x zero-sum / loss-bias:** densification không zero-sum; không áp loss vĩnh viễn (Đl6).
- **§23.5 pruning nguy hiểm:** FRGD chủ yếu THÊM (an toàn), prune để optional/pha sau.
- **Oracle gap (§2-Lim5):** `D_ref` kéo target gần truth; điểm SfM = truth neo.
- **Densification yếu của 3DGS ở textureless (Đl5):** vá trực tiếp.

---

## 7. PHÉP THỬ CPU QUYẾT ĐỊNH (làm TRƯỚC; "chắc chắn rồi mới GPU")
| # | Kiểm | Dữ liệu | PASS = | FAIL ⇒ |
|---|---|---|---|---|
| T1 | Đl1 ngưỡng | toy | khớp | xem lại mô hình |
| T2 | Đl2 + đơn điệu | toy (đã 8/8) | — | — |
| **T3 (A6)** | δ̂ vs H tại điểm SfM, room thật | SfM+mono | tương quan **dương** | đổi narrative thất-bại |
| **T4 (A7) QUYẾT** | `\|D_mono−D_true\|` vs `\|D_ref−D_true\|` tại SfM | SfM=truth | refined **giảm lỗi** | bỏ "refined", dùng mono cho init |
| T5 | Đl6 (toy): densify (δ_ref<E_cur) > loss (δ>δ_thresh) | toy | đúng | xem lại |
| **T6 (A8) QUYẾT** | mật độ SfM/điểm-init vs texture(\|∇I\|), room thật | SfM+ảnh | low-texture ⇒ **init thưa** (under-rep) | densification-gap yếu → cân nhắc |
| T7 | (GPU, sau) FRGD vs uniform, room 5-seed | — | FRGD ≥ uniform | dừng/điều chỉnh |

**T4 + T6 là hai cổng quyết định, CHẠY ĐƯỢC TRÊN CPU bằng SfM (truth) + ảnh, KHÔNG train:**
- T4: refinement có thật sự kéo mono về gần truth không (nền của `D_ref`).
- T6: 3DGS có thật sự under-init vùng low-texture không (nền của densification-gap Đl5).
Nếu **T3+T4+T6 PASS** → C-THEORY + C-GAP + C-METHOD có nền số thật → **mới xin code FRGD**. T7 (GPU nhỏ) là xác nhận cuối.

---

## 8. RỦI RO & RANH GIỚI NOVELTY (trung thực — không hứa 100%)
- **KHÔNG có 100% accept.** v4 tối đa hoá *xác suất*: số dương đáng tin (lever đúng) + framing novel (Đl5/Đl6) + đánh nhiều limitation.
- **Giáp ranh FSGS/CoMapGS** (đều densify/enhance từ depth). Delta = *Fisher-guided WHERE + refined depth + theory WHETHER (Đl6) + ongoing + densification-gap framing*. Modest-nhưng-thật → **phải ablation** (FRGD vs raw-mono-densify vs FSGS-unpool) chứng minh từng phần đáng giá.
- **A9 (densify giảm render-lỗi) chỉ xác nhận đầy đủ trên GPU (T7).** CPU chỉ chứng minh *nền* (T4,T6). Tôi KHÔNG khẳng định metric trước khi T7.
- **Floater risk:** đặt điểm sai → floater. Giảm bởi `c` cao + refined + `δ_ref<E_cur` + RGB sửa + (optional) prune.
- **Compute:** fusion+Fisher ở low-res (rẻ); densify-step thưa.

---

## 9. CHỐT TRƯỚC KHI CHO PHÉP SỬA CODE
1. Bạn duyệt §1 (info/assumptions) + §2–3 (Đl1–6) + §4 (cơ chế FRGD).
2. Tôi viết & chạy **T3, T4, T6 (CPU, room thật, SfM=truth, KHÔNG train)** — chứng minh nền A6/A7/A8 bằng số.
3. **Chỉ khi T4 ∧ T6 PASS** → tôi xin phép thêm `gate_mode/densify=frgd` + chạy T7 (GPU nhỏ: FRGD vs uniform).
4. Nếu T4 FAIL → init bằng raw mono (bỏ refine). Nếu T6 FAIL → densification-gap yếu → lùi về C-THEORY + consistency (do-no-harm).

*Nguồn: v1/v2/v3 proofs; 3DGS densification (∇_{μ2D}); AbsGS/Pixel-GS (densification yếu ở low-gradient); FSGS/CoMapGS (đối tượng phân biệt); MVS triangulation; FisherRF.*
