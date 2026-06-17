# VS-Depth v2 — Fisher-Information-Gated Depth (FIG-Depth)
### Tài liệu thiết kế + chứng minh v2 (2026-06-17). Mở rộng v1. CHƯA sửa code train — bản này để duyệt.

> v1 (`DESIGN_AND_PROOF.md`) đã chứng minh: depth-supervision giảm phương sai (V∝1/H), trọng số `a*=cH/(2δ²H−c)`
> là tối-ưu-MSE (CPU 21/21), cổng covisibility hợp lệ (anti-NAGC). **v2 trả lời câu hỏi gốc hơn:** *đại lượng `H`
> thật sự là gì?* — và phát hiện rằng "đếm covisibility" (CoMapGS) là một **proxy SAI** của `H`, vá được bằng
> **Fisher information quang trắc**. CPU đã chứng minh (P10, `tools/test_fisher_gate.py`, 8/8).

---

## 0. v2 mới gì so với v1
1. Truy `H` (curvature trong lý thuyết v1) về **công thức gốc**: `H` = **Fisher information quang trắc** = bất định tam giác đạc MVS. Nó **không phải** số-đếm-view; nó là *số view × texture² × hình học²*.
2. Chứng minh **limitation gốc**: cổng đếm-view (CoMapGS) tối ưu **chỉ khi** texture/độ-sâu đồng đều; thực tế nó **phân bổ depth sai**, thậm chí **thua depth-đều**, và **mù** với vùng *phẳng-nhiều-view* (đúng vùng cần depth nhất).
3. Cơ chế **FIG-Depth**: thay đại lượng cổng từ `cov` → `H` (Fisher), trọng số = `a*(H,δ)`. Ghép tối thiểu vào `compute_gates` của Final1606.
4. Vá robustness: loss depth **scale-invariant cục bộ** để không chết khi SfM thưa (bicycle 3/12).

---

# PHẦN A — CÔNG THỨC GỐC (nền của mọi method depth-sparse)

**A1. 3DGS** (Kerbl 2023): `C(p)=Σ_i c_i α_i Π_{j<i}(1−α_j)`, `α_i=o_i exp(−½ Δᵀ Σ⁻¹ Δ)`; densify khi `‖∇_{μ_2D} L‖ > τ`.

**A2. Các dạng depth-loss (gốc):**
- L1 nghịch đảo (DNGaussian / 3DGS-depth — **bản Final1606 đang dùng**): `L_d = Σ_p w(p)·|D_r⁻¹(p) − (s D_m⁻¹(p)+t)|`. Cần `(s,t)` căn theo SfM ⟹ **vỡ khi SfM thưa** (đo thực: bicycle căn 3/12).
- Pearson scale-free (FSGS/Pi-GS): `L_d = 1 − ρ(D_r, D_m)`. Bỏ `(s,t)` nhưng mất scale tuyệt đối.

**A3. Bất định độ sâu MVS (công thức GỐC — chìa khoá):** sai số tam giác đạc của một điểm bề mặt nhìn từ view j:
```
σ_z,j  ≈  z² · ε_j / (b_j f),     ε_j ∝ 1/|∇I_j|      (ε = sai số khớp pixel, nhỏ khi gradient ảnh lớn)
```
⟹ **lượng thông tin độ sâu (nghịch phương sai) view j đóng góp:**
```
H_j(p)  ∝  vis_j(p) · ( b_j f / z(p)² )² · |∇I_j(π_j(p))|²
```
(b_j=baseline tới view j, f=tiêu cự, z=độ sâu, ∇I_j=gradient ảnh tại điểm chiếu). Nhiều-view tam giác đạc tốt hơn hai-view ≥1 bậc → tổng theo view.

**A4. Fisher information (FisherRF, ECCV'24):** `H = E[∇_θ log p · ∇_θ log pᵀ]` = Hessian của NLL ảnh, tính từ **gradient ảnh + depth render**. Với toạ độ độ sâu, Fisher info **đồng nhất** với A3. FisherRF dùng `H` để **chọn view**; ta dùng `H` để **đặt trọng số depth** — đây là điểm mới.
```
H(p) = Σ_j vis_j(p) · ( b_j f / z(p)² )² · |∇I_j(π_j(p))|²          (★ curvature quang trắc = Fisher info)
```

**A5. Trọng số depth tối-ưu-MSE (v1, đã chứng minh 21/21):**
```
V(p) ∝ 1/H(p)  ;   a*(H,δ) = cH / (2δ²H − c),  c = ησ²/2   (a*=cap khi 2δ²H ≤ c, tức depth-rất-cần)
```
`δ(p)` = sai số mono-depth (bias). v1 chứng minh `a*` cực tiểu hoá MSE = bias² + variance.

---

# PHẦN B — KHE & LIMITATION GỐC

**B1. covisibility ≠ độ-ràng-buộc.** CoMapGS đặt `w(p) ∝ 1/(M(p)+1)`, `M=Σ_j vis_j` = **chỉ đếm view**. So với (★): công thức đếm **giữ `vis_j`, vứt `(b_j f/z²)²` và `|∇I_j|²`**. Tức nó giả định mọi view ràng buộc bằng nhau, bất kể texture hay hình học.

**B2. Thất bại "phẳng-nhiều-view" (chứng minh trên giấy + đo P10).** Xét hai vùng cùng `M=12`:
- A = có texture (`|∇I|=0.8`): `H_A` lớn → `V_A` nhỏ → đã ràng buộc tốt → **ít cần depth**.
- B = phẳng (`|∇I|=0.05`): `H_B` nhỏ → `V_B` lớn → **dưới ràng buộc → rất cần depth**.

Đo (P10): `H_A=48.0`, `H_B=0.19` ⟹ **V_B/V_A = 256×**. Cổng đếm: `w_A=w_B=1/13` — **mù tuyệt đối, gán y hệt**. Fisher: `a*_A=0.50`, `a*_B=50.0` — **gửi depth đúng vào B**. ⟹ CoMapGS **dưới-giám-sát đúng vùng cần depth nhất** (tường/sàn/counter) → floater/mờ vùng phẳng. *Đây là limitation gốc, chưa ai vá.*

**B3. Hai nguồn sai của cổng đếm (phân tách đo được, P10).** So Fisher với CoMapGS-heuristic, MSE giảm **31%**, tách thành:
- **Dạng hàm sai:** `1/(M+1)` ≠ `a*` ⟹ ngay cả khi tín hiệu đúng vẫn lệch → **22.2%**.
- **Tín hiệu sai:** `M` thay vì `H` (bỏ texture/hình học) → **11.2%** (và **biến mất = 0%** khi texture đều — P10 mục 3, xác nhận đây *chính xác* là texture/geom-awareness).
- Quan trọng: P10 đo **CoMapGS-heuristic còn TỆ hơn depth-đều** (.00824 vs .00806) — đếm-view là tín hiệu đủ sai để thua cả uniform.

---

# PHẦN C — CƠ CHẾ FIG-Depth (ghép tối thiểu vào Final1606)

**C1. Ước lượng `H` (★) trong `compute_gates` HIỆN CÓ.** Đường ống đã chiếu mỗi pixel→mọi view qua rendered depth. Chỉ thêm:
- precompute `g_j = |∇I_j|` (Sobel ảnh input, 1 lần).
- tại điểm chiếu: `geom_j = b_j f / z²` (b_j từ pose tương đối, z từ rendered depth).
- `H(p) = Σ_j vis_j · (geom_j · g_j)²`.

**C2. Ước lượng `δ(p)`** = độ lệch chuẩn của mono-depth-đã-căn giữa các view đồng-thấy (hoặc residual căn-SfM). Cao = mono-depth kém tin.

**C3. Trọng số = `a*(H,δ)`** (thay `g_cov(cov)` heuristic của v1), clamp `[floor, cap]`, pin mean=1 (giữ ngân sách). **Cấu trúc code không đổi** — chỉ đổi *đại lượng per-pixel* từ "đếm" → "Fisher". `gate_mode`: thêm `fisher` (giữ `none/uniform/covonly` để ablation). Strict-gen: `w≡1`=uniform.

**C4. Vá robustness SfM-thưa:** dùng loss depth **scale-invariant cục bộ** (Pearson theo patch) thay L1-căn-SfM → không chết khi ít điểm SfM (bicycle). `H, δ` vẫn tính như trên.

---

# PHẦN D — CHỨNG MINH (đã đo, không hand-wave)

**D1.** `a*` tối-ưu-MSE + `V∝1/H` + tính hợp lệ cổng: **v1, CPU 21/21** (`test_vsdepth_theory.py`).

**D2. Mệnh đề (cổng đếm tối ưu ⇔ texture đều).** Cổng đếm `1/(M+1)` (và bất kỳ hàm chỉ-của-`M`) cực-tiểu-MSE *chỉ khi* `H ∝ M` (texture+hình học đồng đều). P10 mục 3: lợi-ích-tín-hiệu của Fisher = **0.0%** khi texture đều, **11.2%** khi texture biến thiên. ⟹ với scene thật (texture luôn biến thiên), cổng đếm **không tối ưu**.

**D3. Phân tách lợi ích (P10, đo).** Fisher vs CoMapGS = **31%** MSE = 22.2% (dạng `a*`) + 11.2% (tín hiệu Fisher). CoMapGS-heuristic ≥ uniform (không đáng tin hơn depth-đều).

**D4. Ca phân biệt (P10).** Textureless-covisible: `V_B/V_A=256×`; count gán bằng nhau; Fisher `a*_B/a*_A≈100×`. Spearman(trọng số, `a*` tối ưu): **Fisher 1.000 vs count 0.083**.

**D5. CHỨNG MINH GÌ / PHẢI ĐO GPU (trung thực).**
- *Chứng minh (CPU, mô hình bậc-2):* Fisher `H` là đại lượng đúng; cổng đếm sai-hướng ở vùng phẳng; `a*(H,δ)` tối ưu; lợi ích = dạng + tín hiệu.
- *Phải đo GPU:* (a) `H` ước-lượng-từ-gradient có nhiễu — độ lớn lợi ích thật; (b) `fisher > covonly` trên mip360 đa-seed (đặc biệt scene indoor phẳng room/counter); (c) loss scale-invariant có cứu bicycle không. **Một mô hình đồ chơi không thay được đo thật** (bài học NAGC).

---

# PHẦN E — NOVELTY / ĐỊNH VỊ / RỦI RO (khắt khe)

- **Novelty (modest, có gốc, chưa bị chiếm):** trọng số depth = **closed-form MSE-optimal từ Fisher information quang trắc**. FisherRF→chọn view (không phải trọng số depth); CoMapGS→đếm-view thô; UGOT/Pi-GS→uncertainty-của-mạng (≈δ, bỏ `H` hình học). Tổ hợp "Fisher-H × a* × stability-framing" là khe trống. Tầng **ACCV/BMVC/WACV**, không CVPR.
- **Limitation vá rõ:** (i) "covisibility ≠ ràng buộc" (vùng phẳng-nhiều-view); (ii) scale-fragility SfM-thưa. Cả hai *đo được*.
- **Thí nghiệm CHỐT:** `fisher > covonly(=CoMapGS) > uniform > none`, đa-seed đa-scene; lợi ích lớn nhất ở scene phẳng. Nếu `fisher ≈ covonly` thực tế → tôi nói thẳng là chưa đủ, không cố.
- **Rủi ro:** reviewer có thể coi là "confidence-weighting xịn hơn" → phản biện = closed-form a* + tín hiệu vật lý Fisher (không phải uncertainty-mạng) + khung phương sai. `H` nhiễu khi ảnh ít texture — nhưng đó *chính là* cái nó đo (ít texture → H thấp → đúng ý).

---

# PHẦN F — KẾ HOẠCH (sau khi bạn duyệt v2)
1. (Chốt) Bạn duyệt v2.
2. CPU mở rộng: P10 đã PASS 8/8 (`tools/test_fisher_gate.py`). (xong)
3. Sửa code Final1606 (tối thiểu): thêm ước lượng `H` (Sobel + geom) trong `compute_gates`, `gate_mode=fisher`, (tùy chọn) loss scale-invariant. + unit test mới.
4. Kaggle đa-seed: `none / uniform / covonly / fisher` trên scene depth-sống (garden/room/counter), báo mean±std + worst-case + Δ.

*Nguồn gốc: 3DGS (Kerbl SIGGRAPH'23); DNGaussian (CVPR'24); FSGS (ECCV'24); Pi-GS; CoMapGS (CVPR'25); FisherRF (ECCV'24); MVS triangulation-uncertainty (escholarship qt6nk233jn).*
