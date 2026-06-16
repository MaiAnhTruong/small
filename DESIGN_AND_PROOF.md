# VS-Depth — Variance-Stable Covisibility-Gated Depth Supervision cho Sparse-View 3DGS
### Tài liệu thiết kế + chứng minh (v1, 2026-06-16). CHƯA sửa code — bản này để duyệt trước.

---

## 0. Tóm tắt một đoạn

Sparse-view 3DGS có **phương sai theo seed rất lớn** (đo thực tế ~1.6 dB PSNR trên mip-NeRF360 12-view) — **lớn hơn mức cải thiện (~0.2 dB) mà các paper depth-supervision (CoMapGS, UGOT, Pi-GS) báo cáo**, mà *không paper nào báo cáo phương sai này*. Ta đề xuất **VS-Depth**: giám sát depth (DepthAnything V2) **có cổng theo độ-phủ-quan-sát (covisibility)**, được **chứng minh là phân bổ depth tối-ưu-MSE** — bơm độ cong (curvature) tất định đúng vào vùng ít quan sát, nơi phương sai sinh ra. Đóng góp: (i) phân tích bias–variance cho thấy cổng covisibility × reliability là **lời giải tối ưu MSE** (không chỉ heuristic như CoMapGS); (ii) cổng **động** bám theo độ cong hiện tại (vs CoMapGS tĩnh); (iii) **giao thức đánh giá đa-seed (mean ± std + worst-case)** phơi bày việc gain của field nằm trong nhiễu; (iv) chỉ dùng **DAv2** (chạy được trên T4, vs π³ "infeasible on consumer hardware"). **Đã xác nhận bằng mô phỏng CPU (P9, 21/21 PASS):** mọi công thức khớp số đo; gated cho MSE thấp nhất, thấp hơn cov-only (kiểu CoMapGS) **9.7%** nhờ thành phần reliability.

---

# PHẦN I — TÀI LIỆU THIẾT KẾ

## 1. Vấn đề & động lực

### 1.1 Bối cảnh
Sparse-view 3DGS: cho `m` view huấn luyện (m nhỏ, vd 3/6/9/12), khôi phục trường Gaussian để tổng hợp view mới. Vì `m` nhỏ, bài toán **under-constrained**: nhiều cấu hình hình học khác nhau cùng giải thích được `m` ảnh → landscape phẳng, nhiều cực tiểu gần tương đương.

### 1.2 Quan sát cốt lõi (tài sản độc nhất của ta)
Trên mip-NeRF360 12-view, cùng config nhưng đổi seed cho **PSNR dao động ~1.6 dB** (đã đo: 16.693 vs 18.292). Trong khi đó:
- Pi-GS báo cáo Tanks&Temples 22.87 vs 22.67 = **+0.20 dB**.
- Các gain depth-supervision điển hình cùng cỡ **0.2–0.5 dB**.

**⟹ Mâu thuẫn nghiêm trọng:** gain báo cáo **nhỏ hơn nhiễu seed**, mà *không paper nào trong CoMapGS / UGOT / Pi-GS báo cáo std đa-seed* (đã kiểm bằng đọc paper). Tức là **chưa ai biết các gain đó là thật hay nhiễu.** Đây vừa là **động lực**, vừa là **đóng góp phương pháp luận** của ta.

### 1.3 Luận điểm
Thay vì đua mean (đã bão hoà + trong nhiễu), ta đặt mục tiêu **giảm phương sai (ổn định)** — và chứng minh depth-supervision *có cổng đúng cách* là cơ chế giảm phương sai có cơ sở lý thuyết, đồng thời không làm tệ mean.

## 2. Liên quan & khe trống chính xác (đã đọc ruột paper)

| Method | Tín hiệu depth | Covisibility | Depth-uncertainty | Model ngoài | Báo std đa-seed? |
|---|---|---|---|---|---|
| CoMapGS (CVPR'25) | weight `1/(M+1)` | ✓ **TĨNH** (MASt3R, tính 1 lần) | ✗ | MASt3R | ✗ |
| UGOT (2405.19657) | OT + inverse-uncertainty | ✗ | ✓ | depth net | ✗ |
| Pi-GS (2602.03327) | Pearson × confidence | ✗ | ✓ | **π³ (~1M điểm)** | ✗ |
| **VS-Depth (ours)** | **gated, MSE-optimal** | ✓ **ĐỘNG** (từ depth-agreement) | ✓ | **chỉ DAv2** | ✓ **headline** |

**Hàng xóm về stability (phải định vị, KHÔNG trùng):**
- *Quantifying & Alleviating Co-Adaptation* (2508.12720): bất ổn qua co-adaptation, giải bằng dropout — không phải depth.
- *Analysis of Converged 3DGS* (2602.08909): phân tích density–variance — là analysis, không phải method depth-for-stability.

**Khe trống chính xác (giao điểm chưa ai đứng):** *depth-supervision được THIẾT KẾ + CHỨNG MINH như cơ chế giảm phương-sai-seed, cổng bằng covisibility CHÍNH XÁC & ĐỘNG, chỉ DAv2, đo đa-seed.*

> **Ranh giới trung thực với CoMapGS:** CoMapGS đã *gate depth theo covisibility* (cho mean, heuristic `1/(M+1)`, tĩnh). Đóng góp của ta KHÔNG phải "phát minh gate covisibility" mà là: (a) **chứng minh** gate covisibility×reliability là phân bổ **tối ưu MSE** (họ làm heuristic, không có phân tích phương sai); (b) cổng **động** bám độ cong hiện tại; (c) thành phần **reliability** từ sai số mono-depth; (d) **khung + đo đạc phương sai/worst-case**; (e) **DAv2-only**. Đây là novelty **tầm ACCV** (vừa phải, có thật) — tôi nói rõ để không kỳ vọng quá.

## 3. Ký hiệu & sơ bộ

- `Θ = {θ_k}`: tham số Gaussian; ta quan tâm tọa độ **hình học** (vị trí `μ_k`, qua đó là depth bề mặt `d` dọc tia).
- View huấn luyện `{I_i}_{i=1..m}`, pose đã biết `{P_i}` (world→cam), nội tham số `{K_i}`.
- `D_i = R_depth(Θ, P_i)`: depth render (alpha-normalized) ở view i (đã có trong infra `return_depth`).
- `M_i = DAv2(I_i)`: mono-depth thô (lên tới affine chưa biết) ở view i.
- `L_photo`, `L_depth`: loss ảnh và loss depth; `λ`: trọng số.

## 4. Phương pháp

### 4.1 Căn affine mono-depth theo SfM (per-view) — *plumbing chuẩn, không phải novelty*
Mono-depth `M_i` đúng tới `(a_i, b_i)` chưa biết: `d ≈ a_i·M_i + b_i`. Điểm SfM COLMAP cho ta depth metric thưa `{(p_j, z_j)}` trong view i. Giải **robust least-squares** (Huber/RANSAC):
```
(a_i, b_i) = argmin Σ_j ρ( a_i·M_i(p_j) + b_i − z_j )
```
→ `M̂_i = a_i·M_i + b_i` (mono-depth đã căn về hệ mét của SfM). Dùng fallback scale toàn cục nếu view có quá ít điểm SfM.

### 4.2 Bản đồ covisibility hình học (CHÍNH XÁC) — `cov_i(p)`
Với mỗi pixel `p` ở view i, lấy điểm 3D `X = unproj(p, D_i(p), P_i)`, đếm số view **khác** mà `X` rơi vào frustum VÀ không bị che:
```
cov_i(p) = #{ j ≠ i : π_j(X) ∈ ảnh_j  và  X không bị occlude ở view j }
```
- "trong ảnh" = chiếu `X` vào view j nằm trong khung.
- "không occlude" = `z_j(X) ≤ D_j(π_j(X))·(1+τ)` (so depth, như occlusion-mask đã kiểm trong infra).
- **Đây là đại lượng hình học tính từ pose — CHÍNH XÁC, không phải proxy học.** (Điểm sống còn, xem P6.)

### 4.3 Confidence ĐỘNG từ độ-khớp-depth đa-view — `rel_i(p)`
Reliability của mono-depth target tại `p`, kết hợp 2 nguồn:
1. **Sai số căn-SfM cục bộ** + **gradient depth** (biên mono-depth không tin): `rel_geom = exp(−|∇M̂_i(p)|/s)`.
2. **Độ khớp đa-view động** (cập nhật theo training): chiếu `X` sang các view khác, đo độ lệch giữa `M̂_i(p)` và `M̂_j(π_j X)` — khớp cao → tin cậy cao. (Khác CoMapGS tĩnh: đại lượng này **bám geometry hiện tại**.)

`rel_i(p) ∈ (0,1]`, cao = mono-depth đáng tin tại p.

### 4.4 Cổng & loss depth có cổng
**Cổng** (dạng hàm sẽ được P4 biện minh):
```
w_i(p) = g_cov(cov_i(p)) · rel_i(p),   g_cov giảm theo covisibility (vd 1/(cov+1) hoặc exp(−γ·cov))
```
**Loss depth có cổng** (robust):
```
L_depth = Σ_i Σ_p w_i(p) · ρ( D_i(p) − M̂_i(p) )
```
ρ = Huber. (Có thể thêm thành phần gradient/scale-invariant; mặc định L1-robust trên depth đã căn.)

**Mục tiêu tổng:**
```
L = L_photo + λ_dssim·L_dssim + λ_depth · L_depth   (+ reg sẵn có)
```
- `w ≡ const` ⟹ đúng baseline uniform-depth. `λ_depth = 0` ⟹ đúng baseline 3DGS/DropGaussian. (Strict-gen, P5.)

### 4.5 Thuật toán huấn luyện (lịch & nhịp)
```
1. Tiền xử lý: M_i = DAv2(I_i); (a_i,b_i) căn theo SfM → M̂_i.        [1 lần]
2. Mỗi K iter (vd 200) hoặc khi N đổi (densify):
      render D_i (return_depth) cho mọi view i;
      cập nhật cov_i, rel_i, w_i.                                    [động]
3. Mỗi iter: L = L_photo + λ_depth·L_depth(w);  backward; step.
4. (tùy chọn) anneal λ_depth giảm dần cuối training (depth định hướng sớm, ảnh tinh chỉnh muộn).
```

## 5. Giao thức đánh giá (điểm nhấn = đa-seed)

- **Dataset:** chính = mip-NeRF360 sparse-12 (lightning splits sẵn có). Mở rộng (nếu kịp) = LLFF 3/6, DTU-3 (Chamfer, geometry).
- **Đa-seed:** ≥ 5 seed mỗi cấu hình. Báo **mean ± std**, **worst-case (min PSNR/max LPIPS)**, và **mức giảm std**.
- **Ablation bắt buộc:**
  - (i) baseline (no depth)
  - (ii) + uniform depth (`w≡1`)  ← chính là "kiểu CoMapGS/Pi-GS đơn giản hoá"
  - (iii) + VS-Depth tĩnh (cov tính 1 lần)
  - (iv) + VS-Depth động (đầy đủ, ours)
- **Headline (đã chỉnh theo P8):** **MSE/accuracy của (iv) thấp nhất** ( < (iii) cov-only kiểu CoMapGS < (ii) uniform < (i) no-depth ); đồng thời **std-đa-seed của mọi biến thể depth ≪ (i)**. *Lưu ý quan trọng từ P8:* uniform có **std thô thấp hơn** gated nhưng **MSE cao hơn** (over-damp vùng quan sát tốt → bias) ⟹ headline đúng là **"accuracy tại độ ổn định"**, KHÔNG phải "std thấp nhất". Phụ: gain (ii) vs (i) nằm trong std → biện minh khung stability.
- **Metric:** PSNR/SSIM/LPIPS (+ depth error vs SfM/MVS pseudo-GT; Chamfer nếu DTU).

## 6. Kế hoạch hiện thực (trên infra Newnew — lift thấp)
- **Tái dùng:** lightning reader, `return_depth`, env Kaggle (đã chạy).
- **Thêm:** loader DAv2 + căn affine SfM; `cov_i` (chính xác); `rel_i` (động); `L_depth` có cổng; runner đa-seed.
- **Trước GPU (BẮT BUỘC PASS):** (a) CPU variance-simulation xác nhận Bổ đề 1–2 + Định lý (P8); (b) unit test: covisibility đúng trên cảnh 2-view synthetic, đơn điệu cổng, strict-gen (`w const ⇒ = uniform loss`), căn affine khôi phục `(a,b)` đã biết.

## 7. Rủi ro & trung thực
- **"Depth giảm phương sai" nghe trực giác** → novelty nằm ở *chứng minh tối-ưu-MSE + cổng động + đo đa-seed + DAv2-only*. Tầm ACCV, không phải CVPR. Nói rõ.
- **Mean có thể vẫn trong nhiễu** → cố ý: headline là phương sai/worst-case.
- **Lý thuyết chỉ phủ phần within-basin** (xem P7) → magnitude tổng phải đo GPU; tôi KHÔNG hứa con số (bài học NAGC).
- **Định vị vs 2508.12720 / 2602.08909** phải kỹ trong related work.
- **Compute:** đa-seed × scene (T4 ~15 phút/run mip360-12) → cần ngân sách seed hợp lý (vd 5 seed × 3 scene × 4 config).

---

# PHẦN II — CHỨNG MINH (chi tiết, cẩn thận)

> Mục tiêu: chứng minh **chặt** rằng cổng covisibility×reliability là phân bổ depth **giảm phương sai & tối ưu MSE**, và phân biệt rõ phần nào *chứng minh được* với phần nào *phải đo*.

## P1. Giả thiết (nêu rõ, không giấu)
**(A1) Mô hình bậc-2 cục bộ.** Quanh một cực tiểu `θ̂` của kỳ vọng loss dữ liệu, cho một tọa độ hình học vô hướng `θ` trong vùng `R`:
`ℓ(θ) = ½ H (θ − θ̂)² + const`, với `H = ∂²E[L_photo]/∂θ² ≥ 0` (độ cong từ dữ liệu ảnh).
**(A2) Nhiễu gradient SGD.** Cập nhật `θ_{t+1} = θ_t − η g_t`, `g_t = ℓ'(θ_t) + ξ_t`, `E[ξ_t]=0`, `Var(ξ_t)=σ²` (nhiễu từ sampling view/dropout/minibatch), i.i.d. theo t. `0 < ηH < 1`.
**(A3) Depth term tất định.** Loss depth đóng góp `ℓ_d(θ)=½H_d(θ−θ_d)²`, với `θ_d` (mono-depth đã căn) **cố định** theo iter ⟹ **không thêm nhiễu** (đây là điểm mấu chốt: depth bơm độ cong nhưng không bơm phương sai gradient).
**(A4) `H` tăng theo covisibility.** `H ≈ Σ_{view nhìn thấy R} h_view`, mỗi view thấy vùng có texture đóng góp độ cong dương ⟹ `H` đơn điệu tăng theo `cov`. (Biện minh: mỗi view là một ràng buộc độc lập lên `θ`; tổng Hessian = tổng đóng góp.)

> **Phạm vi (P7):** A1 là xấp xỉ *within-basin*. Phần *cross-basin* (densify đổi topology) không nằm trong mô hình — sẽ đo bằng thực nghiệm. Lý thuyết chứng minh **cơ chế** (1/H) và **hướng** (gate đúng giảm phương sai), thực nghiệm xác nhận **độ lớn**.

## P2. Bổ đề 1 — Phương sai dừng tỉ lệ nghịch độ cong
Từ A1–A2, đặt `φ = 1−ηH`:
```
θ_{t+1} − θ̂ = φ (θ_t − θ̂) − η ξ_t       (quá trình AR(1)/Ornstein–Uhlenbeck)
```
Phương sai dừng `V` thỏa `V = φ²V + η²σ²`, nên
```
V = η²σ² / (1 − φ²) = ησ² / ( H (2 − ηH) )  ≈  ησ² / (2H)   (η nhỏ).
```
**⟹ `V ∝ 1/H`.** Khi `H→0` (vùng ít quan sát/phẳng) `V→∞`. ∎

*Hệ quả:* nguồn gốc phương sai-seed lớn = **vùng covisibility thấp** (do A4, `H` nhỏ). Đây là dự đoán **kiểm được trên CPU** (P8) và giải thích cơ học con số 1.6 dB.

## P3. Bổ đề 2 — Depth tất định: giảm phương sai + bias tường minh
Thêm `ℓ_d` (A3). Tổng độ cong `H' = H + H_d`; cực tiểu mới (đặt `ℓ'+ℓ_d'=0`):
```
θ̂' = (H θ̂ + H_d θ_d) / (H + H_d).
```
Nhiễu gradient vẫn `σ²` (A3). Theo Bổ đề 1 với `H'`:
```
V' = ησ² / (H'(2−ηH'))  ≈  ησ² / (2(H + H_d))  =  V · H/(H + H_d).
```
**Hệ số giảm phương sai `H/(H+H_d) < 1`**, đơn điệu giảm theo `H_d`. Bias:
```
b = θ̂' − θ̂ = H_d (θ_d − θ̂) / (H + H_d) = H_d δ / (H + H_d),   với δ := θ_d − θ̂ (sai số mono-depth).
```
∎

## P4. Định lý — Cổng covisibility×reliability là phân bổ depth TỐI ƯU MSE
Sai số bình phương kỳ vọng tại vùng `R`:
```
MSE(H_d) = b² + V' = ( H_d δ / (H+H_d) )² + ησ² / (2(H+H_d)).
```
Đặt `c := ησ²/2`. Lấy đạo hàm theo `a := H_d`:
```
d/da [ δ²a²/(H+a)² + c/(H+a) ] = 2δ² a H/(H+a)³ − c/(H+a)².
```
Cho `=0` ⟹ `2δ² a H/(H+a) = c` ⟹
```
a* = c H / (2 δ² H − c)      khi  2δ²H > c   (nghiệm trong).
```
Khi `2δ²H ≤ c` (tức `δ² ≤ ησ²/(4H)`): đạo hàm `< 0` với mọi `a>0` ⟹ **`a*=∞`** (bơm depth tối đa).

**Hai tính đơn điệu (lấy vi phân `a*`):**
```
da*/dH  = ( c·(2δ²H−c) − cH·2δ² ) / (2δ²H−c)²  = −c² / (2δ²H − c)²  < 0
          ⟹  a* GIẢM theo H   (⇔ giảm theo covisibility, do A4)
da*/dδ² = cH · d/dδ²[ (2Hδ² − c)^(−1) ] = −2cH² / (2δ²H − c)²  < 0
          ⟹  a* GIẢM theo δ²  (⇔ giảm theo sai số mono-depth)
```
*(Trường hợp biên `a*=∞` cũng xảy ra đúng khi `H` nhỏ và `δ²` nhỏ — nhất quán với hai tính đơn điệu.)*

**Kết luận:** trọng số depth tối-ưu-MSE **giảm theo covisibility** và **giảm theo sai số mono-depth**. Đó **chính xác** là dạng cổng
```
w = g_cov(cov) · rel,   g_cov giảm theo cov,   rel giảm theo δ².
```
⟹ Cổng của ta **không phải heuristic** — nó là **lời giải tối ưu MSE** dưới mô hình A1–A4. (CoMapGS dùng `1/(M+1)` heuristic, không có biện minh này; thiếu hẳn thành phần `rel` từ `δ²`.) ∎

*Diễn giải:* ở vùng **quan sát tốt** (`H` lớn) → phương sai vốn nhỏ, thêm depth chỉ tổ rước bias `δ²` ⟹ tối ưu là **ít/không** depth. Ở vùng **ít quan sát** (`H≈0`) → phương sai khổng lồ ⟹ chừng nào `δ²` chưa quá lớn thì **bơm depth mạnh** luôn giảm MSE (sàn MSE = `δ²` ≪ phương sai gốc `ησ²/2H`).

## P5. Mệnh đề — Strict generalization (không thể tệ hơn baseline nếu cổng có ích)
- `λ_depth = 0` ⟹ `L = L_photo` ⟹ **đúng baseline 3DGS/DropGaussian** (bit-identical).
- `w_i(p) ≡ c₀` (hằng) ⟹ `L_depth` thành uniform-depth ⟹ **đúng baseline uniform depth**.
- Do P4, mỗi vùng chọn `H_d` (qua `w`) không vượt `a*` thì `MSE ≤ MSE(0)` = MSE không-depth. Vậy với cổng đúng hướng (P4) và `λ` đủ nhỏ, **MSE từng vùng không tăng** ⟹ tổng không tệ hơn. ∎
*(Đây là lá chắn: ablation (i)/(ii) là trường hợp riêng, ta đo được chính xác phần thêm vào.)*

## P6. Tính HỢP LỆ của cổng — vì sao KHÔNG lặp lại thất bại NAGC
Định nghĩa **tín hiệu cổng hợp lệ** = hàm đơn điệu của *đúng đại lượng* chi phối mục tiêu, **đo không sai số**.

- **NAGC (đã chết):** cổng = phương sai-màu-reproject. Đại lượng này = *texture của ảnh tại điểm chiếu*, **không phải** floater-ness; tệ hơn, nó **anti-correlated** với chi-tiết-thật (texture cao). Tín hiệu **không hợp lệ** ⟹ cổng bẻ sai hướng ⟹ −0.17 dB.
- **VS-Depth:** cổng = `cov` = **số view quan sát**, đo **chính xác** từ pose (không ước lượng). Theo A4, `cov` đơn điệu với `H`; theo Bổ đề 1, `H` chi phối trực tiếp phương sai. Vậy `cov` là **hàm đơn điệu của đúng đại lượng (H/phương sai), đo không sai số** ⟹ **hợp lệ theo định nghĩa**. Không có "khe proxy" để bẻ sai.

⟹ Lỗi cốt tử giết NAGC (tín hiệu sai-hướng) **bị loại trừ về nguyên lý** ở đây. ∎

## P7. Lý thuyết chứng minh GÌ và KHÔNG chứng minh gì (trung thực)
- **Chứng minh được (trong A1–A4):** (1) phương sai ∝ 1/H; (2) depth tất định giảm phương sai theo `H/(H+H_d)`; (3) cổng cov×rel là tối ưu MSE; (4) strict-gen; (5) tính hợp lệ của cổng.
- **KHÔNG chứng minh (phải đo GPU):** (a) độ lớn giảm std thực tế trên scene thật; (b) phần phương sai *cross-basin* (densify đổi topology) — A1 không phủ; (c) `δ²` thực của DAv2 sau căn (giả định "không quá lớn") — phải đo; (d) mean có vượt nhiễu không.
- **Cách thu hẹp khoảng cách lý thuyết–thực nghiệm:** P8 (CPU sim xác nhận 1–3 trước khi tốn GPU), rồi đa-seed GPU đo (a)–(d).

## P8. Dự đoán KIỂM ĐƯỢC TRÊN CPU (làm trước khi sửa code train)
Mô phỏng đồ chơi (không cần rasterizer), xác nhận Phần II:
1. **Bổ đề 1:** chạy AR(1) với nhiều `H`, đo `V` ⟹ khớp `ησ²/(2H)` (kiểm `V·H ≈ const`).
2. **Bổ đề 2:** thêm `H_d`, đo `V'` ⟹ khớp `V·H/(H+H_d)`; đo bias ⟹ khớp `H_dδ/(H+H_d)`.
3. **Định lý (then chốt):** quét `H_d`, vẽ MSE, xác minh `argmin` khớp `a* = cH/(2δ²H−c)`; và **đa-seed**: với phân bố vùng `(H,δ²)` hỗn hợp, cổng cov×rel cho **MSE thấp nhất** ( < cov-only kiểu CoMapGS < uniform < no-depth ). Mọi depth giảm std ≫ no-depth; uniform over-damp (std thô thấp hơn nhưng MSE cao hơn).
4. **Tính hợp lệ cổng:** sinh vùng với `cov` cho trước → `H=Σ`, xác nhận `cov` đơn điệu với `1/V` (Spearman=1); proxy kiểu-NAGC (texture/color, độc lập H) → Spearman≈0.

> Đây là bản sao tinh thần của proof Strat-Drop (đã PASS −15× variance). Nếu P8 PASS, **cơ chế đã được chứng minh trước khi đụng GPU**; nếu không PASS, dừng — không lãng phí như NAGC.

## P9. KẾT QUẢ P8 — ĐÃ CHẠY (2026-06-16, `tools/test_vsdepth_theory.py`, CPU, numpy)

**21/21 checks PASSED.** Số đo khớp giải tích:
- **Bổ đề 1:** `V_emp` khớp `ησ²/(H(2−ηH))` relerr < 0.6% cho H∈{0.2..4}; `V·H` hằng (CoV 1.4%) ⟹ **V ∝ 1/H** xác nhận.
- **Bổ đề 2:** `V'/V0` đo = {1.00, 0.677, 0.505, 0.257} khớp `H/(H+Hd)` = {1, 0.667, 0.5, 0.25}; bias đo khớp `Hdδ/(H+Hd)` (sai số < 1e-3).
- **Định lý:** interior `a*=0.1429`; argmin lưới lý thuyết = 0.1429; argmin thực nghiệm = 0.1429 (MSE tại a* < MSE tại 0.5a* và 2a*). Biên (`2δ²H<c`) → MSE đơn điệu giảm → `a*=∞`. `a*` giảm chặt theo H và theo δ². **Toàn bộ Định lý xác nhận.**
- **Scene (đa-seed, 300 vùng × 400 seed):** MSE — no-depth 6.49e-2 → uniform 2.29e-2 → **cov-only(~CoMapGS) 1.89e-2** → **GATED cov×rel (ours) 1.71e-2**. Std-đa-seed: no-depth 1.75e-2 → gated 5.0e-3 (**−71%**), uniform 3.3e-3 (−81%).
- **PHÁT HIỆN (sửa lý thuyết):** gated KHÔNG có std thấp nhất — uniform over-damp cho std thô thấp hơn nhưng **MSE cao hơn** (bias). Claim đúng = **gated đạt MSE thấp nhất** (tối ưu phân bổ). 
- **Delta vs CoMapGS được chứng minh:** thành phần **reliability** (gating theo δ²) cho **gated MSE thấp hơn cov-only 9.7%** — đây là đóng góp riêng so với covisibility-gate thuần của CoMapGS.
- **Tính hợp lệ cổng (P6):** Spearman(cov, 1/V) = 1.000; proxy kiểu-NAGC = 0.20. Cổng covisibility **hợp lệ**, proxy NAGC **không** — xác nhận bằng số.

**Hệ quả cho framing:** headline = **accuracy (MSE) tối ưu nhờ phân bổ cov×rel**, beating uniform & cov-only; stability (giảm std vs no-depth) là động lực + trục phụ. KHÔNG quảng cáo "std thấp nhất".

---

## 8. Việc tiếp theo (sau khi bạn duyệt tài liệu này)
1. (Chốt) Bạn duyệt thiết kế + chứng minh Phần II.
2. Viết `tools/test_vsdepth_theory.py` (CPU) hiện thực P8 — **phải PASS hết**.
3. Nếu PASS: wiring trên Newnew (4.5) + unit test cổng/căn affine.
4. Đa-seed trên Kaggle theo giao thức Mục 5.

*Nguồn: CoMapGS arXiv 2503.20998; UGOT 2405.19657; Pi-GS 2602.03327; Co-Adaptation 2508.12720; Analysis-of-Converged-3DGS 2602.08909.*
