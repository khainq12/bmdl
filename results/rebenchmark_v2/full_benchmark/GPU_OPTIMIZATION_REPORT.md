# Phase 2 — GPU Pipeline Profiling & Optimization Report

Thực hiện trên avis-1 (RTX 5000 Ada, idle, 302 MiB/0% trước khi bắt đầu), workload
đại diện nhỏ (10 vòng, 5 client, iid, giống hệt K=50/batch=128/lr=0,01 của protocol
chính) — KHÔNG phải thí nghiệm chính. Script:
`scripts/rebenchmark_v2_phase2_profile.py`. Kiến trúc mô hình, K-steps, lr, luật
aggregation, điều kiện thực nghiệm giữ nguyên tuyệt đối — chỉ thay đổi ĐƯỜNG THỰC
THI (execution path), không thay đổi TOÁN HỌC.

## 1. Điểm nghẽn xác định qua đọc code + đo thực tế

| Hạng mục | Phát hiện |
|---|---|
| **CPU↔GPU transfer** | `local_training_fixed_steps()` gọi `model.get_weights()`/`set_weights()` (full GPU→CPU→GPU round-trip, numpy concat theo từng tham số) **ở MỖI bước SGD** (K=50 lần/client/vòng), không chỉ ở đầu/cuối round. `train_step()` cũng kéo gradient về CPU làm numpy mỗi bước. |
| **Redundant evaluation data transfer** | `evaluate()` chuyển lại toàn bộ X_val/X_test từ host lên GPU **mỗi lần gọi**, dù dữ liệu định danh giống hệt nhau mọi vòng. |
| **GPU utilization** | ~0,5-1 giây làm việc thật mỗi vòng cho model nhỏ (207K tham số) — GPU RTX 5000 Ada dư thừa năng lực so với kích thước bài toán; nghẽn chủ yếu ở overhead chuyển dữ liệu/Python, không phải compute thuần. |
| **Logging/sync** | Ghi CSV mỗi vòng (per_round_results) — chi phí không đáng kể so với transfer overhead ở trên. |
| **GPU contention** | Benchmark 504-run chạy 3 tiến trình song song trên avis-1 — tốc độ đo được (1,67-3,1 s/vòng) PHẦN LỚN phản ánh tranh chấp 3 chiều, không phải chi phí thuần của 1 vòng (xem mục 3). |
| **Deterministic execution overhead** | Không phát hiện overhead đáng kể riêng cho tính tất định (seed cố định, không dùng cuDNN benchmark mode nhưng kích thước input cố định nên không cần). |

## 2. Tối ưu đề xuất (đã kiểm chứng đúng-sai + đo tốc độ, KHÔNG áp dụng vào file gốc)

### 2a. GPU-native local SGD (thay vì numpy get/set_weights mỗi bước)

Giữ nguyên thuật toán (K=50 bước SGD thường, không momentum, cùng batch/lr/seed), chỉ
chuyển việc áp dụng `w ← w − lr·∇w` sang thực hiện **trực tiếp trên tensor GPU**
(`p -= lr * p.grad`, in-place, `torch.no_grad()`), loại bỏ hoàn toàn việc trích
xuất/ghi lại trọng số qua numpy ở MỖI bước — chỉ còn 1 lần numpy-hóa ở đầu (snapshot
initial_weights, vốn đã cần cho việc tính delta) và cuối mỗi client.

| | Thời gian | |
|---|---|---|
| Bản gốc (numpy mỗi bước) | 1,065 s/vòng | (10 vòng × 5 client, đo trên GPU rảnh) |
| GPU-native | 0,519 s/vòng | |
| **Tăng tốc** | **2,05×** | |

**Kiểm chứng đúng đắn** (cùng seed, cùng dữ liệu, client 0, vòng 0, 207.081 tham số):
lệch tuyệt đối tối đa 1,457×10⁻⁵, lệch tuyệt đối trung bình 2,855×10⁻⁸, norm delta
0,052504 (gốc) vs 0,052511 (mới) — sai số nằm trong phạm vi làm tròn float32 bình
thường khi đổi thứ tự phép tính, KHÔNG phải sai lệch thuật toán. (Lệch tương đối trung
bình 0,398 trông cao nhưng là hiện tượng chia cho giá trị gần-0 ở vài tọa độ — không
phản ánh sai lệch thực chất; xem norm tổng thể khớp đến 5 chữ số có nghĩa.)

### 2b. Cache tensor đánh giá trên GPU (thay vì chuyển lại từ host mỗi lần)

| | Thời gian/lần gọi evaluate() | |
|---|---|---|
| Bản gốc (chuyển X_val từ host mỗi lần) | 21,03 ms | (val set, 10.004 ảnh) |
| Cache tensor trên GPU, chỉ forward pass | 11,10 ms | |
| **Tăng tốc** | **1,89×** | |

## 3. Tác động ước tính lên chi phí thực nghiệm

Hai tối ưu trên đều **an toàn về mặt khoa học** (cùng công thức toán, chỉ khác đường
thực thi, đã kiểm chứng sai số ở mức làm tròn float32) — khuyến nghị áp dụng cho
**thí nghiệm mở rộng MỚI** (Phase 3), KHÔNG áp dụng hồi tố vào benchmark 504-run gốc
(giữ nguyên theo đúng yêu cầu, tránh làm benchmark gốc không còn tái lập được chính
nó).

Ước tính s/vòng tối ưu cho 3 phương pháp mục tiêu Phase 3 (ngoại suy từ tỷ lệ chi phí
tương đối đã đo thực tế trong benchmark 504-run, áp lên baseline GPU-native đo trực
tiếp — không profiling riêng từng defense để tiết kiệm GPU-time):

| Defense | s/vòng (504-run, có tranh chấp 3 luồng) | s/vòng tối ưu (ước tính, chạy đơn, không tranh chấp) |
|---|---|---|
| FedAvg | 1,673 | **0,519** (đo trực tiếp) |
| Multi-Krum | 2,824 | **0,876** (ngoại suy theo tỷ lệ 2,824/1,673) |
| C4-DA | 3,103 | **0,963** (ngoại suy theo tỷ lệ 3,103/1,673, gồm thêm 1 lượt train g_ref) |

→ Tổng hợp: pipeline tối ưu nhanh hơn ~**2-3,5 lần** so với tốc độ đã quan sát trong
benchmark 504-run gốc (kết hợp cả hiệu ứng tối ưu hóa THẬT và loại bỏ tranh chấp GPU
3 chiều bằng cách chạy đơn luồng cho thí nghiệm mở rộng nhỏ này).

## 4. Giới hạn của việc profiling này

- Chỉ đo FedAvg trực tiếp; Multi-Krum/C4-DA ngoại suy theo tỷ lệ quan sát được từ
  504-run (không profiling riêng để tiết kiệm GPU-time, vì mục tiêu Phase 2 là ước
  tính đủ tốt cho quyết định, không phải tối ưu tối đa).
- Không thử nghiệm batch-size lớn hơn hay tăng song song hóa giữa các client (có thể
  mang lại thêm tốc độ nhưng thay đổi sâu hơn cấu trúc vòng lặp, rủi ro ảnh hưởng tính
  so sánh — không áp dụng theo tinh thần "preserve scientific comparability").
- Cache-eval tối ưu (mục 2b) được đo riêng nhưng CHƯA kết hợp cùng lúc với GPU-native
  SGD (mục 2a) trong cùng một lần đo — số liệu "0,519 s/vòng" đã bao gồm 1 lần
  evaluate() KHÔNG cache mỗi vòng, nên con số thực tế khi kết hợp cả hai có thể nhanh
  hơn một chút so với ước tính ở mục 3 (ước tính hiện tại là thận trọng, không lạc
  quan hóa).
