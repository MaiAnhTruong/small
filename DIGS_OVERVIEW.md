# DIGS — Depth-Initialized Gaussian Splatting: tài liệu ý tưởng cải tiến (hoàn chỉnh)

*Tổng hợp toàn bộ: trực giác → lý thuyết → cơ chế → bằng chứng → khung paper. Đọc file này là nắm trọn DIGS.*

---

## 1. Ý tưởng một câu

> **Thay vì** khởi tạo sparse-view 3DGS từ đám mây SfM **thưa** rồi thêm điểm depth **giữa** quá trình train
> (FRGD), **DIGS** dựng một đám mây **dày** bằng cách back-project monocular-depth của **tất cả** train view
> **ngay tại iteration 0** — mỗi điểm là một *surfel đúng footprint* (frustum `z/f`, hướng camera) — để các
> điểm này nhận **toàn bộ ngân sách tối ưu**, thứ quyết định chất lượng geometry trong chế độ **sparse + ít
> iteration**.

Một câu ngắn hơn: **"đưa capacity depth vào lúc khởi tạo, không phải giữa chừng."**

---

## 2. Bối cảnh & hành trình (vì sao đến DIGS)

Sparse-view 3DGS (12 view, 10k iter) thiếu ràng buộc hình học → Gaussian trôi/phình → floater/haze. Depth
được đưa vào như tín hiệu dày hơn RGB. Chúng tôi đã đo **8 hướng dùng depth** một cách có kiểm soát:

| # | Hướng | Trục tác động | Kết quả (room, multi-seed) |
|---|---|---|---|
| 1 | depth-loss gating (covonly/gated/fisher) | reweight loss | **chết** (zero-sum) |
| 2 | **plain depth densification (FRGD)** | thêm capacity giữa chừng | **+0.3 PSNR** (đòn bẩy đầu tiên) |
| 3 | refine placement | *vị trí* điểm thêm | hoà PSNR |
| 4 | FRGD-G (frustum shape) | *hình* điểm thêm | trade-off (LPIPS+/SSIM−) |
| 5 | CGD (confidence→opacity) | *opacity* điểm thêm | hại (calibration) |
| 6 | BDVR (suppress floater) | *xoá* điểm | hại (over-broad) |
| 7 | aligned depth (L1) | *chất lượng* depth | hoà (densify chịu được sai-số) |
| 8 | **DIGS (dense init @ iter 0)** | **TIMING đưa capacity** | **THẮNG sạch (xem §6)** |

**Bài học cốt lõi:** 7 hướng đầu đều giữ NGUYÊN hai giả định ẩn — *(a) khởi tạo SfM thưa, (b) thêm capacity
GIỮA train*. Tất cả "dùng depth khôn hơn" đều bão hoà (đúng trần thông tin / oracle-gap). **DIGS là hướng đầu
tiên thay đổi giả định (a)+(b): đổi TIMING.** Và nó thắng.

---

## 3. Vì sao DIGS hoạt động — lý thuyết (init-persistence)

### 3.1 Trực giác
Trong 3DGS, kết quả cuối của một Gaussian = **init + tối ưu**. Mức tối ưu nhận được ∝ số lần nó được cập nhật
gradient = (số view thấy nó) × (số iteration nó còn sống). Trong **dense-view + 30k iter** (3DGS chuẩn), mọi
điểm được tối ưu rất nhiều → init bị "wash" → *không ai quan tâm init*. Trong **sparse-view (12) + few-iter
(10k)** → mỗi điểm nhận **rất ít** cập nhật → **init TỒN TẠI tới render cuối**.

### 3.2 Công thức (đã chứng minh CPU, test_digs.py D1/D2)
Điểm thêm tại iteration `t_add` nhận:
```
K_eff(t_add) ≈ (N − t_add) · (V_see / V_tot)          (N = tổng iter, V_see = #view thấy nó)
```
và sai số init giảm theo `(1 − ηH)^K_eff`. Suy ra:
```
K_eff(0) / K_eff(t_add) = N / (N − t_add)  > 1     →  1.25× (t=2000) … 10× (t=9000)
```
- **FRGD** thêm điểm ở `t_add ∈ [2000, 10000]` → điểm muộn được tối ưu rất ít → init kém **tồn tại** → *đây
  chính là lý do hướng 3/4/5 (refine/shape/confidence trên điểm muộn) đều washed.*
- **DIGS** đặt điểm ở `t_add = 0` → tối ưu **đủ 10k iter** → hội tụ sát optimum → lợi ích densification được
  **hiện thực hoá đầy đủ**, và các cơ chế shape/confidence (vốn neutral trên điểm muộn) **sống lại**.

> **Đây là "hidden lever":** không phải sắp xếp lại cùng thông tin (đã bão hoà), mà đổi **ngân sách tối ưu/điểm**
> — một trục hoàn toàn khác, có headroom thật.

---

## 4. Cơ chế chi tiết (đúng như code)

`densify_mode=none --init_mode depth` → hàm `digs_init()` chạy **một lần trước vòng train**:

1. **Refined depth + reliability:** `refine_depth_maps` fuse mono-depth đa-view → `D_ref` (depth tinh) +
   `rel` (độ tin đa-view), per view. (Không cần render — iter 0 chưa có geometry.)
2. **Back-project dày:** với mỗi train view, lấy lưới pixel **subsample (stride `s`)**, giữ pixel `rel > floor`,
   back-project qua `D_ref` → điểm 3D **nằm trên bề mặt** (visibility-verified). Màu = ảnh tại pixel.
3. **Hình đúng (FRGD-G):** mỗi điểm = **đĩa frustum**: scale ngang `σ_lat = c_f·z/f` (footprint 1 pixel ở
   depth z — đúng ở MỌI mật độ subsample, khác `distCUDA2` over-size), mỏng theo tia nhìn (`σ_n = β·σ_lat`),
   xoay hướng camera → ít haze.
4. **Opacity:** mặc định 0.1 (robust); tùy chọn `0.1·rel` (CGD-style).
5. **Dedup + cap:** voxel-dedup so với điểm SfM (tránh chồng), cap `digs_max_points`.
6. **Thêm vào cloud** qua `add_frgd_points` → vào optimizer → train 3DGS chuẩn (base densify + uniform depth
   loss) như thường.

**Tham số:** `init_mode(sfm|depth)`, `digs_stride` (mật độ init — **quan trọng**, xem §6.2), `digs_rel_floor`,
`digs_max_points`, `digs_conf_opacity`. **Base 3DGS không bị đụng** — DIGS chỉ là một init mode, ghép được với
mọi `densify_mode`/`gate_mode`.

**Tấn công đúng limitation (survey):**
- FSGS §1.1 (sparse init thiếu) → init dày (D3: dense phủ 100% vùng textureless, SfM 0%).
- FSGS §1.2 (Euclidean unpool → empty space) → back-project depth nằm ON surface (D4: 0% empty vs Euclid 100%).
- §23.x → hình đúng + reliability, tái dùng v6/v7.

---

## 5. Khác gì các phương pháp trước

| | Dense geometry | DIGS khác ở đâu |
|---|---|---|
| **FSGS** (Unpool) | midpoint Euclidean từ SfM thưa | **back-project depth** (on-surface) + **frustum disk** + **lý thuyết timing** |
| **NexusGS** | dense init từ optical-flow/epipolar (nặng) | mono-depth (nhẹ) + shape/conf + init-persistence |
| **FRGD (của ta)** | thêm điểm GIỮA train | **đưa về iter 0** (chứng minh tốt hơn: D1/D2 + đo) |
| 3DGS/CoMapGS init | SfM / covisibility | surfel depth dày per-pixel, confidence-aware, đo geometry-faithful |

**Đóng góp (novelty):** (i) **khởi tạo depth dày, geometry-correct, confidence-aware**; (ii) **lý thuyết
init-persistence** giải thích *vì sao* iter-0 > mid-densify ở sparse-few-iter (và vì sao dense-30k bỏ qua);
(iii) **đo geometry-faithful** (geom-consistency, efficiency) phơi bày gain mà interp-PSNR che; (iv) **chuỗi
7-negative** làm động lực (depth chỉ giúp như capacity-at-init, không phải supervision/shape/quality). Trung
thực: kề cận FSGS/NexusGS → đóng góp là *bộ depth-surfel-confidence init + lý thuyết + đo lường*, incremental
nhưng vững, đúng tầm ACCV.

---

## 6. Bằng chứng (room, cùng depth/loss, multi-seed)

### 6.1 Bảng
| model | PSNR | SSIM | LPIPS↓ | geom↓ | #G |
|---|---|---|---|---|---|
| frgd s0 | 19.105 | 0.7018 | 0.3888 | 0.238 | 814,703 |
| frgd s1 | 19.386 | 0.7023 | 0.3869 | 0.227 | 817,239 |
| digs8 s0 | 19.416 | 0.7018 | 0.3661 | 0.209 | 953,750 |
| digs8 s1 | 19.605 | 0.7032 | 0.3653 | 0.209 | 953,268 |
| **digs12 s0** | **19.607** | **0.7103** | **0.3643** | **0.206** | **870,186** |

### 6.2 Ba lớp bằng chứng (đều chặt)
1. **Tách phân phối hoàn toàn:** mọi DIGS > mọi FRGD trên PSNR (19.416 > 19.386), LPIPS (0.366 < 0.387), geom
   (0.209 < 0.227). Không chồng lấn → không phải seed-luck.
2. **Chống capacity (anti-monotonic #G):** ÍT điểm hơn lại TỐT hơn — digs12 (870k) > digs8 (954k) > digs4
   (1.31M) trên PSNR. Nếu gain do số điểm thì digs4 phải thắng; nó thua xa → **gain là chất lượng/timing của
   init, không phải capacity**. (⇒ stride 12 là "sweet spot"; quá dày = nhiễu/over-densify.)
3. **#G-matched (digs12 870k vs frgd 816k, +6.6%):** thắng MỌI metric — PSNR **+0.22…+0.36**, SSIM **+0.008**,
   LPIPS **−0.024**, geom **−0.026**.

→ **Kết luận:** DIGS thắng sạch, ổn định seed, chống-capacity, cùng-#G — **cú phá trần đầu tiên** sau 8 thí
nghiệm, đúng tiên đoán của lý thuyết init-persistence.

---

## 7. Đã chứng minh trên CPU trước khi chạy GPU (tools/test_digs.py, 9/9)
- **D1/D2** budget: `K_eff(0)/K_eff(t)=N/(N−t)` (1.25–10×), residual iter-0 < mid → timing là lever thật.
- **D3** coverage: dense phủ 100% surface textureless, SfM 0% (FSGS §1.1).
- **D4** on-surface: depth-backproj 0% empty, Euclidean-unpool 100% empty (FSGS §1.2).
- **E1** geometry-blind: floater displacement `b·f·|1/zF−1/zS|` = 0.8px (interp) vs 10.7px (extrap) → interp-PSNR
  mù với geometry → cần đo geom-consistency/extrapolation (§23.6).

(Và shape v6: test_frgd_g 13/13; confidence v7: test_cgd 11/11 — tái dùng trong DIGS.)

---

## 8. Khung PAPER (ACCV)

**Tiêu đề ý tưởng:** *"Where, not how: depth helps sparse-view 3DGS as a dense initialization, not as
supervision."*

**Story:** depth-as-supervision/mid-densify/shape/confidence/quality đều metric-neutral dưới interp-PSNR (ta
chứng minh 7 ablation có kiểm soát). Vì (a) gain là hiệu ứng **init-geometry** và (b) interp-PSNR **mù
geometry** (E1). DIGS đưa capacity depth vào **init dày geometry-correct**, và — đo bằng giao thức
geometry-faithful — cho gain mà interp-PSNR che.

**4 đóng góp:** (1) method DIGS; (2) lý thuyết init-persistence; (3) chuỗi 7-negative (study); (4) đo
geometry-faithful (geom-consistency + efficiency, đập §23.6).

**Bảng chính:** FRGD vs DIGS, #G-matched, multi-seed, PSNR/SSIM/LPIPS + geom + #G, trên nhiều scene.

---

## 9. Trung thực: còn thiếu gì để thành paper

- **Tổng quát:** mới **1 scene (room, indoor thuận lợi)**. **Phải có 2–3 scene nữa** (garden outdoor,
  counter/bicycle) lặp lại pattern → đây là lỗ hổng lớn nhất hiện tại.
- **Seed:** digs12 mới 1 seed (digs8 đã 2 seed, đều thắng) → thêm digs12 s1 cho headline mean±std.
- **Độ lớn gain:** PSNR +0.2…+0.3, LPIPS −0.02 — **khiêm tốn nhưng sạch (Pareto)**, đúng tầm ACCV
  incremental, không phải breakthrough về magnitude.
- **Novelty:** kề cận FSGS/NexusGS → phải định vị rõ (depth-surfel on-surface + frustum + timing-theory +
  geometry eval).
- **Tùy chọn nâng cấp:** stride 16 (có thể match #G chính xác + xu hướng "sparser→better" gợi ý còn tốt hơn);
  `digs_conf_opacity` (chưa bật mặc định).

---

## 10. TL;DR
DIGS = **đưa depth-capacity vào iter 0 (dense, surfel đúng footprint) thay vì giữa train.** Lý thuyết
init-persistence giải thích vì sao điều này thắng ở sparse-few-iter. Trên room: **thắng FRGD mọi metric, cùng
#G, chống-capacity, ổn định seed** — cú thắng thật đầu tiên. Việc còn lại: **chứng minh trên ≥3 scene** rồi
viết paper (DIGS + theory + 7-negative study + geometry-faithful eval).
