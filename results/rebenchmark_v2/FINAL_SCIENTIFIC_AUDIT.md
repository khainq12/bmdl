# FINAL SCIENTIFIC AUDIT — R12 (toàn bộ: 504-run, 54-run ablation, 18-run extension)

Audit cuối cùng trước khi cập nhật báo cáo Word. **Không chạy GPU mới, không huấn
luyện thêm** — toàn bộ phân tích dưới đây dùng dữ liệu đã có trong
`results/rebenchmark_v2/{full_benchmark,ablation_adaptive_norm_only,convergence_extension}/`.

## 0. Ba nghiên cứu — phân biệt rõ phạm vi (bắt buộc, yêu cầu #7)

| | Benchmark 504-run | Ablation 54-run | Convergence extension 18-run |
|---|---|---|---|
| Thư mục | `full_benchmark/` | `ablation_adaptive_norm_only/` | `convergence_extension/` |
| Phạm vi | 7 defense × 6 attack × 4 partition × 3 seed | 3 defense (Norm/AdaptiveNormOnly/C4-DA) × 3 attack × 2 partition × 3 seed | 3 defense (FedAvg/Multi-Krum/C4-DA) × **no_attack only** × 2 partition × 3 seed |
| Round budget | 280 | 280 | **500** |
| Đánh giá mỗi vòng | **Test set** (hạn chế đã biết) | Test set (cùng protocol gốc) | **Validation set** (sửa) |
| Code path | numpy per-step (gốc) | numpy per-step (gốc) | **GPU-native** (tối ưu Phase 2, đã kiểm chứng) |
| Checkpoint | Không | Không | Có, mỗi 50 vòng |
| τ/ρ/κ | Hiệu chỉnh riêng | Hiệu chỉnh riêng | **Tái sử dụng** từ 504-run (không hiệu chỉnh lại) |

**Ba nghiên cứu KHÔNG thể ghép thành một đường cong liên tục** (cơ sở đánh giá khác
nhau: test vs validation; code path khác nhau: numpy vs GPU-native) — chỉ so sánh
được ĐỊNH TÍNH (xu hướng, cơ chế) giữa chúng, không được công bố các con số như thể
chúng đến từ cùng một thực nghiệm.

## 1. Xác minh 18 run (extension)

**Đầy đủ 18/18**, đúng combo (3 defense × 2 partition × 3 seed), mỗi run đủ 10
checkpoint (vòng 49,99,...,499), 500/500 vòng ghi nhận, không thiếu/thừa. ✅

### Phát hiện cần sửa: test được đánh giá ở vòng CUỐI (500), không phải tại `selected_round`

Script `convergence_extension.py` tính `selected_round` (vòng được tiêu chí hội tụ
xác định) nhưng đánh giá test trên trạng thái mô hình **ở vòng 500** (trạng thái
cuối cùng sau vòng lặp), KHÔNG rollback về checkpoint tại `selected_round`. Tên
trường `test_accuracy_at_final_checkpoint` phản ánh ĐÚNG những gì được đo (checkpoint
cuối) nhưng không khớp với ý định thiết kế ban đầu ("test sau khi đã chọn mô hình").

**Định lượng mức độ ảnh hưởng** (dùng lại val_accuracy đã log, không cần GPU mới — so
sánh val tại `selected_round−1` [điểm neo thật của tiêu chí] với val tại vòng 500):

- Chênh lệch tối đa: **0,0252** (multi_krum/dirichlet_a0.5/seed42)
- 3/18 run có |chênh lệch| > 0,01; 15/18 run dưới 0,01
- Đa số run có `selected_round` rất gần vòng 500 (vd FedAvg/Multi-Krum nhiều run
  converged_round=None → selected_round=499 → không có sai lệch)

**Đánh giá**: sai lệch tối đa 2,5 điểm % — nhỏ hơn 1 bậc độ lớn so với phát hiện
chính (khoảng cách IID 15 điểm %) — **không đủ để đảo ngược bất kỳ kết luận định
tính nào**, nhưng là một lỗi giao thức thật cần sửa nếu dùng lại script này, và cần
công bố rõ trong báo cáo (không được nói "test đánh giá tại mô hình đã chọn" mà không
có chú thích này).

## 2. C4-DA dưới IID — điều tra chi tiết (dùng dữ liệu 504-run gốc, 280 vòng, 3 seed)

*(Lưu ý: phần mở rộng 500-vòng KHÔNG log accept/reject/FPR/norm — chỉ có
val_accuracy/loss. Toàn bộ phân tích accept/reject dưới đây dùng dữ liệu đã có từ
`full_benchmark/client_decisions.csv`, tại 280 vòng, code gốc KHÔNG tối ưu. Tính nhất
quán định tính với đường cong 500-vòng mới được kiểm tra riêng ở mục 2d.)*

### 2a. Tỷ lệ accept/reject theo thời gian (3 seed)

| Giai đoạn | seed42 | seed43 | seed44 |
|---|---|---|---|
| Vòng 0-30 | 98,0% accept | 96,0% accept | 94,7% accept |
| Vòng 100-150 | 21,6% accept | 9,2% accept | 13,2% accept |
| Vòng 230-280 | 5,6% accept | 8,0% accept | 3,2% accept |

### 2b. FPR tổng thể dưới no_attack (toàn bộ 280 vòng)

seed42: 67,7% · seed43: 72,0% · seed44: 70,9% — **rất nhất quán giữa 3 seed**
(chênh lệch <5 điểm %).

### 2c. Cơ chế gốc rễ — xác định chính xác nhánh nào chịu trách nhiệm

| Nhánh reject (vòng 230-280) | Tỷ lệ kích hoạt |
|---|---|
| `norm_fail` (τ drift-aware) | 5,5% |
| **`cos_fail` (ρ tĩnh)** | **94,1%** |
| `sign_peer_outlier` (điều kiện phụ của sign) | 50,0% |

**`cos_fail` là nguyên nhân chi phối gần như tuyệt đối**, không phải Norm (đã
drift-aware, hoạt động đúng thiết kế) hay Sign. Đối chiếu giá trị thực:

| | ρ đã khóa | cosine_score thực tế (TB vòng 230-280) |
|---|---|---|
| IID | **0,650** | **0,395** (thấp hơn ρ ~0,25) |
| Dirichlet α=0,5 | **−0,442** | 0,078 (cao hơn ρ ~0,52, dư biên lớn) |

**Kết luận cơ chế (đã xác nhận, không còn là giả thuyết)**: ρ được hiệu chỉnh từ mô
phỏng calibration 5-vòng (mô hình gần-ngẫu-nhiên) tìm thấy độ tương đồng cosine giữa
các client IID RẤT CAO ở giai đoạn đầu (vì cùng data phân phối, gradient gần giống hệt
nhau ngay sau khởi tạo ngẫu nhiên) → khóa ρ=0,65 (ngưỡng khắt khe, kỳ vọng gần như
song song tuyệt đối). Khi huấn luyện thật tiến triển hàng trăm vòng, cosine similarity
THẬT của client lương thiện trôi xuống tự nhiên (~0,40) — y hệt cơ chế trôi-ngưỡng đã
phát hiện với τ (Norm), nhưng xảy ra ở CHIỀU COSINE, và nghiêm trọng hơn nhiều ở IID
vì ρ hiệu chỉnh quá khắt khe ngay từ đầu. Ở Dirichlet α=0,5, calibration (cùng quy
trình) tình cờ tìm ra ρ rất khoan dung (−0,44) vì bản thân client non-IID đã kém tương
đồng ngay từ lúc hiệu chỉnh — nên dư biên lớn, không bao giờ bị vi phạm dù cosine thật
cũng trôi (0,08, vẫn cách xa −0,44).

### 2d. Đóng băng thật hay chỉ plateau? Tính nhất quán 3 seed (dùng dữ liệu 500-vòng mới)

Từ `convergence_extension/c4_drift_aware__iid__seed*/round_metrics.csv`:

| Seed | Vòng "hội tụ" (tiêu chí gốc) | Val acc tại hội tụ | Val acc vòng 500 | Biến thiên sau hội tụ |
|---|---|---|---|---|
| 42 | 198 | 0,659 | 0,658 | ~phẳng (đúng là đóng băng) |
| 43 | 310 | 0,660 (đỉnh vòng 309) | 0,650 | dao động quanh đỉnh, không vượt qua |
| 44 | 447 | 0,668 | 0,657 | dao động, không vượt qua |

**Cả 3 seed đều thể hiện cùng một mẫu hình: đạt đỉnh sớm (vòng 200-450) rồi KHÔNG BAO
GIỜ vượt qua đỉnh đó nữa cho tới hết 500 vòng** — đây là **plateau có dao động nhỏ
quanh một trần cố định**, nhất quán với cơ chế cos_fail~94% đã xác định ở 2c (hầu hết
mọi vòng chỉ 0-1/5 client được accept → cập nhật gần như luôn gần 0 hoặc dựa trên rất
ít client, không đủ tín hiệu học thêm). Gọi đây là "đóng băng" theo nghĩa thực dụng
(plateau cứng, không phải update=0 tuyệt đối mọi vòng) là chính xác; gọi là "hội tụ"
theo nghĩa tích cực (đã học xong, không còn gì để học) là SAI — cần phân biệt rõ
trong Chương 4-5.

## 3. Vì sao C4-DA tốt hơn ở Dirichlet α=0,5 so với IID

Đã trả lời đầy đủ ở mục 2c: **ρ hiệu chỉnh khoan dung hơn nhiều ở Dirichlet α=0,5**
(do bản chất calibration 5-vòng từ mô hình gần-ngẫu-nhiên tình cờ phản ánh đúng mức độ
phân tán tự nhiên của non-IID), khiến nhánh Cosine không bao giờ kích hoạt (0,0% vs
94,1% ở IID). Đây là **bằng chứng trực tiếp** (không phải suy luận), dựa trên dữ liệu
client-vòng đã ghi nhận.

## 4. Đối chiếu với ablation AdaptiveNormOnly — phân biệt bằng chứng và giả thuyết

| Phát hiện | Ablation (trước) | Audit này (sau) |
|---|---|---|
| "ρ/κ tĩnh cũng là vấn đề giống τ" | **Giả thuyết** định tính, không phân biệt ρ vs κ | **Xác nhận + tinh chỉnh**: cụ thể là ρ (Cosine), κ (Sign) không phải nguyên nhân chính (sign_peer_outlier ~50% cả 2 partition, không phân biệt) |
| Mức độ ảnh hưởng | Chỉ có FPR tổng hợp (~0,65-0,67 IID) | **Định lượng chính xác**: cos_fail=94,1% vs norm_fail=5,5%; cosine_score thực tế vs ρ đã khóa, cho cả 2 partition |
| AdaptiveNormOnly tại 500 vòng | Chưa test (ablation chỉ 280 vòng, không có trong 18-run extension) | **Chưa có bằng chứng trực tiếp** — phần mở rộng 500-vòng KHÔNG bao gồm AdaptiveNormOnly làm nhánh thí nghiệm. Mọi suy luận về hành vi AdaptiveNormOnly tại 500 vòng là NGOẠI SUY từ xu hướng 280-vòng, không phải bằng chứng trực tiếp. |

**Phân biệt rõ**: kết luận "ρ tĩnh là nguyên nhân chính, không phải κ" là BẰNG CHỨNG
(dữ liệu client-vòng trực tiếp). Kết luận "điều này cũng đúng ở 500 vòng cho
AdaptiveNormOnly" là GIẢ THUYẾT CHƯA KIỂM CHỨNG (không có dữ liệu AdaptiveNormOnly ở
500 vòng).

## 5. Sai số số học GPU-native có ảnh hưởng tới kết luận không?

**Không, dựa trên phân tích độ lớn và một kiểm tra chéo trực tiếp:**

1. **Độ lớn**: sai số đã định lượng ở preflight (lệch quyết định 1/100 client-vòng,
   lệch accuracy tối đa 0,009/20 vòng) nhỏ hơn 3 bậc độ lớn so với phát hiện chính
   (khoảng cách IID 15 điểm %, duy trì suốt ~300 vòng, nhất quán cả 3 seed).
2. **Kiểm tra chéo trực tiếp** (không cần chạy thêm): cơ chế cos_fail~94% ở mục 2
   được đo từ **benchmark 504-run GỐC, dùng code numpy-per-step CHƯA tối ưu** (không
   phải GPU-native). Tức là hiện tượng "C4-DA kém ở IID" **đã tồn tại và được giải
   thích đầy đủ TRƯỚC KHI có bất kỳ thay đổi tối ưu hóa nào** — phần mở rộng 500-vòng
   (dùng GPU-native) chỉ XÁC NHẬN TIẾP TỤC một hiện tượng đã biết, không phải TẠO RA
   nó. Đây là bằng chứng mạnh nhất cho thấy sai số số học không phải nguyên nhân.

**Không khẳng định "tương đương tuyệt đối"** — đã nêu rõ ở preflight rằng 1% quyết
định có thể lệch; chỉ khẳng định sai số này **không đủ lớn để giải thích hay đảo
ngược** các kết luận định tính đã báo cáo.

## 6. Giao thức validation-only — xác minh lại

Đã xác nhận ở preflight (code: `X_test`/`y_test` chỉ xuất hiện 1 lần, sau vòng lặp).
**Bổ sung ở audit này**: `selected_round` được xác định HOÀN TOÀN bằng tiêu chí
PATIENCE/ACC_EPS/LOSS_EPS trên validation — **không phải** argmax(val_accuracy) (xác
minh trực tiếp: selected_round trùng argmax chỉ 1/18 lần, và đó là trùng hợp khi
converged_round=None nên mặc định về vòng cuối). Tức là KHÔNG có hành vi "chọn điểm
validation tốt nhất" kiểu cherry-pick — chỉ có lỗi ở chỗ ĐÁNH GIÁ test tại vòng sai
(mục 1), không phải lỗi ở việc CHỌN vòng nào.

## 7. Danh sách sửa cần thiết cho Chương 3-5 (trước khi cập nhật Word)

1. **Chương 3 (Phương pháp)**: bổ sung mô tả 3 nghiên cứu riêng biệt (504-run/
   ablation/extension) với bảng phân biệt phạm vi như mục 0; nêu rõ phần mở rộng dùng
   validation, benchmark gốc dùng test (hạn chế đã biết, không sửa hồi tố).
2. **Chương 4 (Kết quả)**: KHÔNG được viết "C4-DA hội tụ ở IID" mà không kèm chú
   thích "đây là plateau cứng ở mức thấp, không phải học xong"; PHẢI trình bày khoảng
   cách IID (15 điểm %) và Dirichlet α=0,5 (C4-DA vượt Multi-Krum) như hai kết quả
   TÁCH BIỆT, không gộp thành một câu "C4-DA cạnh tranh với Multi-Krum" chung chung.
3. **Chương 4**: thêm cơ chế cos_fail=94,1% (không phải norm) làm nguyên nhân chính
   của plateau IID — sửa lại mọi chỗ trước đó quy nguyên nhân chung chung cho "ngưỡng
   tĩnh" mà không chỉ rõ là Cosine.
4. **Chương 5 (Thảo luận/Hạn chế)**: bổ sung rõ (a) giới hạn phạm vi no_attack-only
   của phần mở rộng, không suy rộng sang có-tấn-công; (b) sai lệch test-tại-vòng-sai
   tối đa 2,5 điểm %, đã định lượng, không đổi kết luận định tính; (c) AdaptiveNormOnly
   chưa được test ở 500 vòng — khuyến nghị cải tiến ρ/κ thích nghi (C4-DA v2) vẫn là
   đề xuất, chưa kiểm chứng thực nghiệm ở độ dài huấn luyện dài.
5. **Toàn báo cáo**: không dùng cụm "ổn định thống kê" hay "có ý nghĩa thống kê" cho
   bất kỳ so sánh nào chỉ dựa trên n=3 seed — chỉ dùng "nhất quán qua 3 seed" (mô tả),
   không dùng ngôn ngữ suy diễn thống kê hình thức.

## 8. Giới hạn còn tồn đọng (không đổi, nhắc lại ngắn gọn)

- Chỉ no_attack, chỉ 3/7 phương pháp, chỉ 2/4 partition trong phần mở rộng 500-vòng.
- n=3 seed — đủ cho mẫu hình định tính nhất quán, không đủ cho kiểm định hình thức.
- AdaptiveNormOnly chưa kiểm chứng ở 500 vòng.
- Nguyên nhân CHÍNH XÁC vì sao calibration 5-vòng cho ra ρ quá khắt khe ở IID cụ thể
  (so với lý do nó khoan dung ở Dirichlet) được giải thích ĐỊNH TÍNH hợp lý (client IID
  tương đồng cao lúc khởi tạo ngẫu nhiên) nhưng chưa có mô hình ĐỊNH LƯỢNG dự đoán
  trước mức độ cho partition/seed bất kỳ.

## 9. KHUYẾN NGHỊ CUỐI CÙNG: GO

**GO — có thể cập nhật báo cáo Word**, với điều kiện áp dụng đầy đủ 5 sửa đổi ở mục 7
trước khi hoàn tất. Lý do:

- Toàn bộ 18 run đã xác minh đầy đủ, không thiếu/sai dữ liệu.
- Phát hiện chính (C4-DA plateau mạnh ở IID, cạnh tranh/vượt trội ở Dirichlet α=0,5)
  có cơ chế gốc rễ đã XÁC NHẬN bằng dữ liệu trực tiếp (không phải suy đoán), nhất quán
  cả 3 seed, và đã kiểm tra chéo với code gốc (không phải artifact của tối ưu hóa).
- Hai vấn đề phát hiện trong audit này (test-tại-vòng-sai ≤2,5 điểm %; sai số GPU-
  native ~1%) đều đã định lượng và đều NHỎ HƠN NHIỀU so với các hiệu ứng chính đang
  báo cáo (10-15 điểm %) — không đủ để đảo ngược kết luận, nhưng BẮT BUỘC phải công
  bố minh bạch như hạn chế đã biết.
- Không phát hiện lỗi logic nghiêm trọng, không phát hiện số liệu bịa đặt hay thiếu
  sót âm thầm ở bất kỳ trong 3 nghiên cứu.
