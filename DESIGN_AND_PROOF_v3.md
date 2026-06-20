# VS-Depth v3 — Bias-Variance Theory of Depth Supervision + Multi-View Refined Depth
### Thiết kế + chứng minh v3 (2026-06-20). CHƯA sửa code. Duyệt + verify CPU xong mới code.

> Bối cảnh đo được (room, 5 seed): `none 18.27 → uniform 19.18 → covonly 19.13 → fisher 19.05`.
> ⟹ depth giúp lớn (+0.9 dB) nhưng **gating theo coverage (CoMapGS/Fisher) THUA uniform**. v3 giải thích VÌ SAO
> bằng lý thuyết, và đề xuất method đánh đúng bottleneck (**độ ĐÚNG của depth = oracle gap**).

---

## 0. Hai đóng góp (chốt)
- **C-THEORY (xương sống novelty):** khung bias-variance cho depth supervision trong 3DGS. Rút ngưỡng *khi nào depth giúp*, công thức trọng số tối ưu `a*(H,δ)`, và chứng minh **mọi gate hiện có (uniform / coverage / uncertainty) là trường hợp riêng KHIẾM KHUYẾT**; giải thích tại sao coverage-gating backfire.
- **C-METHOD (method chính):** **Multi-View Refined Depth (MVRD)** — sửa target mono-depth về phía oracle bằng đồng thuận đa-view + SfM, rồi giám sát bằng `a*`. Đánh thẳng vào δ (bottleneck), không chỉ che.

---

## 1. THÔNG TIN & GIẢ ĐỊNH (xác định rõ trước — không mập mờ)

### 1.1 Thông tin BIẾT CHÍNH XÁC (không lỗi)
| Ký hiệu | Là gì | Vai trò |
|---|---|---|
| `P_i, K_i` | pose + nội tham số view i (COLMAP) | warp đa-view, tính baseline/geom |
| `{X_k, track_k}` | điểm SfM 3D + danh sách view quan sát | **depth THẬT thưa** (neo correctness), validate |
| `I_i` | ảnh train | tính `\|∇I\|` (texture) |

### 1.2 Thông tin ƯỚC LƯỢNG (có lỗi — phải mô hình hoá lỗi)
| Ký hiệu | Là gì | Lỗi |
|---|---|---|
| `D_mono^i` | mono inverse-depth (DAv2) đã căn affine theo SfM | δ_mono (bias, lớn ở biên/textureless/specular) |
| `D_render^i` | depth model render (đang train) | nhiễu, tiến hoá theo iter |
| `Ĥ(p)` | Fisher curvature ước lượng `Σ_j vis_j(b_j f/z²)²\|∇I_j\|²` | proxy của H thật |
| `δ̂(p)` | bất nhất đa-view của D_mono (proxy sai số depth) | proxy của δ thật |

### 1.3 GIẢ ĐỊNH (nêu thẳng, có cái phải KIỂM)
- **A1** (local quadratic): gần hội tụ, `E[L_photo] ≈ ½H(θ−θ̂)²` cho toạ độ hình học θ; `H` = Fisher info quang trắc.
- **A2** (SGD noise): gradient nhiễu phương sai `σ²` → phương sai dừng `V=σ²/(2H)` (v1, đã CPU-PASS).
- **A3** (depth term): thêm curvature `H_d = λ·w·(curv depth)`, kéo về `D_target = θ_true + δ`.
- **A4** (photometric ~unbiased): `θ̂ ≈ θ_true` *nơi đủ texture/view*; vùng under-determined gói vào `H` nhỏ (V lớn).
- **A5** (local constancy): `H, δ, σ²` hằng trong vùng nhỏ.
- **A6 — PHẢI KIỂM (CPU):** `δ` và `H` **tương quan dương** (vùng H thấp = ít view/textureless cũng là nơi mono SAI nhiều = δ cao). Đây là mấu chốt giải thích thất bại coverage-gating. *Nếu A6 sai → lý thuyết-giải-thích yếu, phải xem lại.*
- **A7 — PHẢI KIỂM (CPU):** đồng thuận đa-view **giảm** sai số mono (consensus ≈ gần truth hơn raw mono) tại điểm SfM. *Nếu A7 sai → refinement (C-METHOD) vô ích, lùi về gating/low-freq.*

---

## 2. PHẦN I — LÝ THUYẾT (C-THEORY)

**Bổ đề 1 (v1, đã PASS):** `V(θ)=σ²/(2H)` → variance ∝ 1/H.
**Bổ đề 2 (v1, đã PASS):** thêm `H_d` → `V'=σ²/(2(H+H_d))`, bias `b=H_dδ/(H+H_d)`.

**Định lý 1 — Ngưỡng "khi nào depth giúp" (MỚI, lõi):**
$$\text{depth giúp tại }p \iff MSE'<MSE_0 \iff \boxed{\;\delta^2 < \delta_{\text{thresh}}^2=\frac{\sigma^2}{2}\cdot\frac{H+H_d}{H\,H_d}\;}$$
*Chứng minh:* `(H_dδ/(H+H_d))² + σ²/(2(H+H_d)) < σ²/(2H)` ⟺ `(H_dδ/(H+H_d))² < (σ²/2)·H_d/(H(H+H_d))` ⟺ `δ² < (σ²/2)(H+H_d)/(H H_d)`. ∎
*Hệ quả:* depth là con dao hai lưỡi — giúp khi mono đủ đúng (δ nhỏ), **HẠI khi δ vượt ngưỡng** (bias nuốt variance-reduction).

**Định lý 2 — Trọng số tối ưu MSE (đã PASS 8/8 ở v2):**
$$a^*(H,\delta)=\frac{cH}{2\delta^2H-c},\quad c=\tfrac{\sigma^2}{2}\ \ (a^*=\text{cap khi }2\delta^2H\le c).$$
Đơn điệu: `∂a*/∂δ² < 0` và `∂a*/∂H < 0` (đều giảm). ⟹ trọng số tối ưu phụ thuộc **CẢ** H và δ.

**Định lý 3 — Hợp nhất & giải thích thất bại (đóng góp novelty):**
Các gate hiện có là *trường hợp riêng* của `a*`:
- `uniform`: `w=const` ⇒ bỏ cả H,δ (tối ưu chỉ khi a* phẳng).
- `coverage` (CoMapGS `1/(M+1)`): `w` giảm theo `M∝H` ⇒ **= a*(H), GIẢ ĐỊNH δ hằng**.
- `uncertainty` (UGOT/Pi-GS): `w` từ uncertainty-mạng ⇒ **≈ a*(δ), bỏ H**.
- v2 `fisher`: `a*(H, δ-yếu)` — vẫn lệ thuộc H.

**Mệnh đề thất bại (giải thích đo được):** dưới **A6** (δ↑ khi H↓), tại vùng H thấp ta có δ lớn tới mức `δ²>δ_thresh` (Định lý 1 ⇒ depth HẠI ⇒ tối ưu `w=0`). Nhưng coverage-gate đặt `w=max` đúng tại đó ⟹ **bơm depth cực đại đúng nơi depth có hại** ⟹ net âm so với uniform. **Đây là chứng minh ở mức lý thuyết cho thứ tự đo `uniform>covonly>fisher`** (fisher hung hơn ⇒ âm hơn). ∎(điều kiện A6)

> Đây là phát biểu CHƯA paper nào có: *coverage/uncertainty-gating tối ưu nhầm biến; biến đúng là δ, và vì δ↔H dương nên gating-theo-H phản tác dụng.*

---

## 3. PHẦN II — METHOD: Multi-View Refined Depth (MVRD)

**Nguyên lý (từ Định lý 1):** có 2 cách làm depth giúp nhiều hơn:
1. **Hạ trọng số nơi δ lớn** (gating/consistency) — chỉ *do-no-harm* (≤ uniform-trên-vùng-tốt).
2. **GIẢM chính δ** (refinement) — đẩy thêm pixel xuống dưới `δ_thresh` ⟹ **biến pixel "depth-hại" thành "depth-giúp"**.

**Định lý 4 — Refinement mạnh hơn gating:** với gating tốt nhất (đặt `w=a*` trên `δ_mono`), pixel có `δ_mono²>δ_thresh` cho đóng góp ≤ 0 (tối ưu là tắt). Nếu refinement đạt `δ_ref<δ_mono` sao cho `δ_ref²<δ_thresh`, pixel đó chuyển sang đóng góp DƯƠNG. ⟹ refinement giảm MSE tổng **nghiêm ngặt hơn** gating, *với điều kiện A7* (fusion thật sự giảm δ). Nếu A7 không đạt ở pixel nào → `δ_ref≈δ_mono` → tự thoái về gating (do-no-harm). ∎

**Cơ chế (dùng đúng thông tin §1):** định kỳ (như recompute v2), với mỗi view i, pixel p:
1. Unproject p qua `D_render^i` → điểm 3D `X`.
2. **Gom bằng chứng depth của X:** (a) `D_mono^i(p)`; (b) với mỗi view j đồng-thấy (occlusion-check qua `D_render^j`): chiếu X→j, đọc `D_mono^j` đã-căn, quy đổi về depth của X trong khung i → `d_{j→i}(p)`; (c) `d_SfM` nếu p gần điểm SfM (**trọng số cao — đây là truth**).
3. **Fuse robust** (median/Tukey): `D_ref(p) = robust({d_i, d_{j→i}, d_SfM})`; `δ̂_ref(p) =` độ tản (spread) của tập (nhỏ = đồng thuận = tin).
4. **Giám sát:** `L_depth = mean_p w(p)·ρ(D_render(p) − D_ref(p))`, `ρ=Huber`, `w(p)=clamp(a*(Ĥ(p), δ̂_ref(p)), floor, cap)` (Định lý 2).

⟹ target dịch **raw mono → đồng thuận đa-view (gần oracle)**, và trọng số dùng `a*` ĐÚNG (cả H,δ). Khác hẳn pseudo-view/diffusion: đây là **thông tin THẬT từ các view khác**, không hallucinate; SfM neo chống "đồng thuận-nhưng-sai".

---

## 4. PHẦN III — DỰ ĐOÁN KIỂM ĐƯỢC TRÊN CPU (làm TRƯỚC, 0 giờ GPU thêm)

| # | Kiểm | PASS = | Nếu FAIL |
|---|---|---|---|
| T1 | Định lý 1 (toy): depth giúp ⟺ δ<δ_thresh | ngưỡng khớp mô phỏng | xem lại mô hình |
| T2 | Định lý 2 + đơn điệu (đã 8/8) | a* khớp argmin | — |
| **T3 (A6)** | trên room **thật**: tương quan `δ̂` (cross-view mono disagree) vs `H` (Fisher) tại điểm SfM | **dương rõ** (low-H ⇒ high-δ) | giải thích-thất-bại yếu → đổi narrative |
| **T4 (A7) — QUYẾT ĐỊNH** | tại điểm SfM (**có depth THẬT**): so `\|D_mono−D_true\|` vs `\|D_ref−D_true\|` | **refinement giảm sai số** (δ_ref<δ_mono) | **MVRD vô ích → DỪNG #3, lùi gating/low-freq** |
| T5 | Định lý 4 (toy): refinement (δ↓) cho MSE thấp hơn gating tốt nhất | đúng | — |

**T4 là phép thử rẻ-mà-quyết-định:** ta CÓ depth thật tại điểm SfM (track) → đo trực tiếp multi-view fusion có kéo mono về gần truth không, **không cần train**. Đây chính là "chắc chắn trước khi tốn giờ" bạn yêu cầu. Nếu T3+T4 PASS → C-THEORY đứng + C-METHOD có nền thật → mới code.

---

## 5. RỦI RO (trung thực)
- **A6 yếu ở scene texture-dày** (garden δ↔H lỏng) → giải thích coverage-fail chỉ mạnh ở indoor/structured. Báo per-scene.
- **Consensus ≠ truth** (lỗi mono hệ thống: texture lặp, specular → mọi view sai giống nhau). SfM neo giảm nhưng thưa. → δ_ref vẫn có thể lệch; do-no-harm cứu (fallback gating).
- **Vòng lặp:** warp dùng `D_render` (đang train). Giảm thiểu: neo SfM + recompute định kỳ + dùng `D_mono` cho warp giai đoạn đầu.
- **Compute:** fusion đa-view ở low-res (như fisher), rẻ. 
- **Metric:** vẫn trục depth → kỳ vọng modest; novelty từ **theory + refinement-attacks-δ**, không cần con số to (đúng mục tiêu của bạn).

## 6. CÁI CẦN CHỐT TRƯỚC KHI CHO PHÉP SỬA CODE
1. Bạn duyệt giả định §1 + 3 định lý §2 + cơ chế §3.
2. Tôi viết & chạy **T3 + T4 (CPU, trên room thật, dùng SfM làm truth)** — đây là cổng quyết định.
3. Chỉ khi **T4 PASS** (refinement thật sự giảm sai số tại điểm SfM) → tôi xin phép code `gate_mode=mvrd`.
4. Nếu T4 FAIL → KHÔNG code refinement; chuyển C-METHOD sang **consistency/low-freq** (do-no-harm) + giữ C-THEORY làm xương sống.

*Nguồn lý thuyết: v1/v2 proofs (test_vsdepth_theory 21/21, test_fisher_gate 8/8); MVS triangulation-uncertainty; FisherRF; CoMapGS/UGOT/Pi-GS (đối tượng hợp nhất).*
