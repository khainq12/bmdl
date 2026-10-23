# Phase 3+4 — Minimal Convergence Extension: Design, Cost Estimate, Recommendation

**Trạng thái: CHƯA CHẠY. Đây là kế hoạch + ước tính, chờ xác nhận trước khi thực thi
trên GPU**, đúng theo yêu cầu. Script đã viết và kiểm chứng đúng đắn (mục 3), nhưng
thí nghiệm chính (18 run × 500 vòng) chưa khởi động.

## 1. Thiết kế thí nghiệm (pre-specified, balanced)

| Trục | Giá trị |
|---|---|
| Phương pháp | FedAvg, Multi-Krum, C4-DA v1 |
| Partition | IID, Dirichlet α=0,5 |
| Seed | 42, 43, 44 |
| Số round mục tiêu | 500 (đầu trên của khoảng 430-500 được yêu cầu — chi phí tăng thêm từ 430→500 chỉ ~16%, đổi lấy biên an toàn lớn hơn để quan sát hội tụ thật) |
| **Tổng số run** | **18** (3×2×3) — đầy đủ, CÂN BẰNG, không rút gọn/chọn lọc theo kết quả test-set |

**Vì sao không rút gọn thêm**: chi phí ước tính (mục 4) đã rất thấp (~2-3 giờ GPU) sau
tối ưu hóa Phase 2, nên không cần "chọn lọc" run theo khả năng thành công (vốn sẽ vi
phạm yêu cầu "justify independently of test-set performance") — chạy đủ bộ 18 run cân
bằng là lựa chọn sạch nhất về phương pháp luận.

## 2. Sửa đổi phương pháp so với benchmark 504-run gốc (có chủ đích)

| | Benchmark 504-run gốc | Phần mở rộng này |
|---|---|---|
| Đánh giá mỗi vòng | **Test set** (hạn chế đã biết, xem SCIENTIFIC_AUDIT.md) | **Validation set** |
| Đánh giá test | Mỗi vòng (280 lần/run) | **Đúng 1 lần**, tại round được chọn bởi tiêu chí hội tụ (hoặc round cuối nếu chưa hội tụ) |
| Checkpoint | Không có | **Có** — trọng số mô hình, trạng thái RNG, round index, config, lưu mỗi 50 vòng (`.npz`) |
| Resume | Không hỗ trợ | **Hỗ trợ, đã kiểm chứng đúng tuyệt đối** (mục 3) |
| Tốc độ cục bộ | numpy get/set_weights mỗi bước SGD | GPU-native (tối ưu Phase 2, sai số ~1e-5, nhanh ~2× — xem GPU_OPTIMIZATION_REPORT.md) |
| Tiêu chí hội tụ | PATIENCE=15, ACC_EPS=0,005, LOSS_EPS=0,002 | **Giữ nguyên y hệt** (không đổi, không nới lỏng) |
| τ/ρ/κ (C4-DA) | Hiệu chỉnh riêng cho benchmark 504-run | **Tái sử dụng đúng giá trị đã khóa** từ `full_benchmark/calibration_results.csv` (không hiệu chỉnh lại) |

Vì đánh giá chuyển từ test sang validation, số liệu round-by-round của phần mở rộng
**không thể ghép trực tiếp** vào đường cong test-based của benchmark 504-run gốc —
đây là hai nghiên cứu riêng biệt, bổ sung cho nhau, không thay thế nhau (đúng yêu cầu
Phase 4: "Do not treat the new validation-based study as directly interchangeable").

## 3. Kiểm chứng script (đã thực hiện, chi phí ~10 giây GPU, không phải thí nghiệm chính)

Script: `scripts/rebenchmark_v2_convergence_extension.py`. Kiểm tra:
1. **Checkpoint + resume chính xác**: chạy liên tục 12 vòng (fedavg/iid/seed42) vs
   chạy 6 vòng → lưu checkpoint → resume → chạy tiếp 6 vòng. Kết quả: **khớp tuyệt
   đối từng bit** ở cả 4 chỉ số (val_accuracy, val_loss, test_accuracy, test_loss) —
   xác nhận resume an toàn và chính xác.
2. **Tải đúng τ/ρ/κ đã khóa** cho C4-DA từ file calibration gốc (không hiệu chỉnh lại).

## 4. Ước tính chi phí GPU (18 run × 500 vòng, chạy đơn luồng, pipeline tối ưu)

| Defense | s/vòng tối ưu | Số run | Tổng giây | Giờ |
|---|---|---|---|---|
| FedAvg | 0,519 | 6 | 1.557 | 0,43 |
| Multi-Krum | 0,876 | 6 | 2.628 | 0,73 |
| C4-DA | 0,963 | 6 | 2.889 | 0,80 |
| **Tổng** | | **18** | **7.074** | **≈2,0 giờ** |

Cộng thêm overhead (hiệu chỉnh — tái sử dụng nên ~0; checkpoint I/O — không đáng kể;
đánh giá test 1 lần cuối — không đáng kể) và biên an toàn: **ước tính thực tế 2,5-3,5
giờ GPU**, chạy trên avis-1 (hiện đang rảnh).

## 5. Phase 4 — Trả lời các câu hỏi khoa học (dựa trên Phase 1, CHƯA có dữ liệu mở rộng)

| Câu hỏi | Trả lời (từ dữ liệu đã có) |
|---|---|
| 280 vòng có đủ cho kết luận ổn định? | **Có, cho thứ hạng phương pháp** (Spearman ρ=1,0 giữa round 150 và 280). Chưa đủ để khẳng định đã hội tụ tuyệt đối (34 run no_attack chưa genuinely_converged). |
| Huấn luyện thêm có cải thiện accuracy đáng kể? | Có cải thiện nhưng **tốc độ giảm dần rõ rệt** (mọi defense: Δ(150→200) > Δ(240→280)) — lợi nhuận biên giảm, không phải dấu hiệu chưa học được gì. |
| Thứ hạng phương pháp có đổi theo thời lượng huấn luyện? | **Không**, ổn định từ round 150. |
| C4-DA còn cạnh tranh với Multi-Krum? | Khoảng cách (có lợi cho Multi-Krum) **nới rộng nhẹ** theo thời gian (round150: gần hòa; round280: Multi-Krum hơn ~2 điểm %) — xu hướng chưa đủ mạnh về thống kê (std > mean diff), đây là câu hỏi CHÍNH mà phần mở rộng nhắm trả lời dứt khoát hơn. |
| Phương pháp nào thực sự hội tụ? | Multi-Krum (9/12, phần còn lại đang improving — có khả năng đạt genuinely_converged với thêm vòng) tốt nhất trong 4 phương pháp ưu tiên; C4-DA 9/12 nhưng phần còn lại (plateau/oscillating) khó cải thiện thêm; Norm kém nhất (4/12, 3 đóng băng vĩnh viễn ở IID). |

## 6. Khuyến nghị rõ ràng

**Khuyến nghị: NÊN chạy thí nghiệm mở rộng 18-run (không bắt buộc cho kết luận xếp
hạng, nhưng chi phí đã rất thấp sau tối ưu hóa và trực tiếp trả lời 2 câu hỏi mở còn
tồn đọng).**

Lý do:
- Kết luận xếp hạng tổng thể đã đủ vững chắc từ 280 vòng (không cần mở rộng CHỈ để
  xác nhận lại điều đã rõ).
- NHƯNG câu hỏi "khoảng cách C4-DA vs Multi-Krum nới rộng hay sẽ ổn định" và "Multi-
  Krum có thực sự hội tụ nếu được thêm vòng" vẫn MỞ, có giá trị khoa học thật, và chi
  phí trả lời (~2,5-3,5 giờ GPU, nhờ tối ưu Phase 2) thấp hơn nhiều so với ước tính
  ban đầu (trước tối ưu: ~5-7 giờ cho phạm vi tương đương).
- Không có run nào trong phạm vi 18-run này bị loại vì đóng băng/diverge mà chưa rõ cơ
  chế (norm — nguồn gốc đóng băng đã biết rõ — không nằm trong 3 phương pháp mục tiêu
  của Phase 3 nên không áp dụng câu hỏi này).

**Nếu KHÔNG muốn chạy**: kết luận xếp hạng hiện tại (Multi-Krum/C4-DA xấp xỉ nhau,
Multi-Krum nhỉnh hơn nhẹ, không có ý nghĩa thống kê mạnh) đã đủ để báo cáo trung thực
với caveat rõ ràng — không bắt buộc phải mở rộng để có kết luận "đủ dùng" ở mức xếp
hạng tổng quát.

## 7. Giới hạn còn tồn đọng (dù có chạy mở rộng hay không)

1. Chỉ 3/7 phương pháp, 2/4 partition được test trong mở rộng — không tổng quát hóa
   sang Norm/Cosine/Sign Consensus hay Dirichlet α=1,0/0,1.
2. Chỉ điều kiện no_attack — **không chứng minh hội tụ dưới tấn công** (yêu cầu rõ
   ràng không được suy diễn điều này).
3. n=3 seed — đủ cho xu hướng định tính, không đủ cho kiểm định thống kê hình thức.
4. Tiêu chí hội tụ PATIENCE/ACC_EPS/LOSS_EPS có tính hồi cứu — không thể dừng sớm
   thật sự giữa chừng (chỉ đánh giá tại các mốc 50 vòng), đã nêu rõ trong thiết kế.
5. Hai pha (test-based 504-run gốc, validation-based mở rộng mới) không thể ghép trực
   tiếp thành một đường cong liên tục — chỉ so sánh được tại các điểm tương ứng
   (round 280) với hiểu biết rằng cơ sở đánh giá khác nhau.
