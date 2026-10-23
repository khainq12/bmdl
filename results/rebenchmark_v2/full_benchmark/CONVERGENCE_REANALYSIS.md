# Phase 1 — Convergence Reanalysis (CPU-only, no GPU, no new training)

Nguồn: `scripts/rebenchmark_v2_phase1_reanalysis.py`, đọc `run_manifest.csv` +
`per_round_results.csv` + `freeze_detection.csv` đã có sẵn trong
`results/rebenchmark_v2/full_benchmark/`. Kết quả trung gian:
`analysis_artifacts/phase1_*.csv`. Không chạy GPU, không training mới.

**Lưu ý kế thừa từ SCIENTIFIC_AUDIT.md**: đường cong dùng ở đây là accuracy/loss
đánh giá trên **tập TEST mỗi vòng** (hạn chế đã biết của benchmark 504-run gốc —
xem GPU_OPTIMIZATION_REPORT.md / ANALYSIS.md). Tái phân tích này KHÔNG dùng xu hướng
test-set để chọn thí nghiệm mới hay điểm dừng mới — chỉ dùng để mô tả lại hành vi đã
ghi nhận của dữ liệu đã có (một việc tái phân tích dữ liệu cũ, không phải một quyết
định thí nghiệm mới dựa trên test).

## 1. Sửa lỗi phân loại: frozen ≠ converged

Audit trước (`SCIENTIFIC_AUDIT.md` §7) báo **55/84 "hội tụ"** theo tiêu chí
PATIENCE/ACC_EPS/LOSS_EPS — nhưng tiêu chí đó có tính hồi cứu thuần túy (chỉ cần
đường cong phẳng là tự động thỏa mãn cả 2 điều kiện), nên **3 run đóng băng thật sự**
(`norm/no_attack/iid`, cả 3 seed — xem freeze_detection.csv) bị tính nhầm là "hội tụ".

**Phân loại đã sửa (84 run no_attack):**

| Loại | Số run | Định nghĩa |
|---|---|---|
| **genuinely_converged** | 52 | Thỏa tiêu chí PATIENCE/ACC_EPS/LOSS_EPS VÀ không đóng băng |
| **plateau** | 17 | std 30 vòng cuối thấp, slope ~0, nhưng không thỏa tiêu chí nghiêm ngặt |
| **still_improving** | 8 | slope dương rõ rệt (>ACC_EPS/PATIENCE mỗi vòng) |
| **oscillating** | 4 | slope ~0 nhưng std > 0,02 — nhiễu, không có xu hướng |
| **frozen** | 3 | std ≈ 0 tuyệt đối (update=0 mọi vòng) — **tách riêng khỏi genuinely_converged** |

## 2. Theo 4 phương pháp ưu tiên (FedAvg, Median, Multi-Krum, C4-DA)

| Defense | genuinely_converged | plateau | still_improving | oscillating | frozen |
|---|---|---|---|---|---|
| **FedAvg** | 6/12 | 4 | 1 | 1 | 0 |
| **Median** | 8/12 | 3 | 1 | 0 | 0 |
| **Multi-Krum** | 9/12 | 0 | **3** | 0 | 0 |
| **C4-DA** | 9/12 | 1 | **0** | 2 | 0 |

Multi-Krum có tỷ lệ hội tụ thật cao nhất (9/12) và toàn bộ phần chưa hội tụ đều thuộc
loại "still_improving" (có khả năng hội tụ nếu thêm vòng). **C4-DA không có run nào
"still_improving"** — phần chưa hội tụ của nó là plateau/oscillating, ít khả năng cải
thiện chỉ bằng cách thêm vòng.

## 3. Accuracy theo mốc vòng (trung bình mọi defense, mọi partition/seed)

| Defense | r150 | r200 | r240 | r280 | Δ(150→280) |
|---|---|---|---|---|---|
| FedAvg | 0,635 | 0,672 | 0,688 | 0,700 | +0,066 |
| Sign Consensus | 0,635 | 0,672 | 0,688 | 0,700 | +0,066 |
| Cosine | 0,618 | 0,662 | 0,671 | 0,683 | +0,066 |
| Median | 0,611 | 0,649 | 0,674 | 0,676 | +0,065 |
| Multi-Krum | 0,603 | 0,639 | 0,662 | 0,674 | +0,071 |
| C4-DA | 0,597 | 0,641 | 0,646 | 0,655 | +0,058 |
| Norm | 0,508 | 0,533 | 0,551 | 0,562 | +0,054 |

Mọi phương pháp vẫn còn cải thiện nhẹ ở round 280 (chưa "cạn" hoàn toàn), nhưng tốc độ
cải thiện giảm dần rõ rệt (độ dốc round 240→280 nhỏ hơn round 150→200 ở mọi defense).

## 4. Ổn định thứ hạng (ranking stability) qua các mốc vòng

| Mốc vòng | Thứ hạng (tốt→kém) |
|---|---|
| 150 | FedAvg, Sign, Cosine, Median, Multi-Krum, C4-DA, Norm |
| 200 | FedAvg, Sign, Cosine, Median, C4-DA, Multi-Krum, Norm |
| 240 | FedAvg, Sign, Median, Cosine, Multi-Krum, C4-DA, Norm |
| 280 | FedAvg, Sign, Cosine, Median, Multi-Krum, C4-DA, Norm |

**Hệ số tương quan hạng Spearman giữa round 150 và round 280: ρ=1,000 (p<0,001).**
Chỉ có hoán đổi vị trí nhỏ giữa các cặp liền kề ở round 200/240 (Cosine↔Median,
Multi-Krum↔C4-DA), tự điều chỉnh lại đúng thứ tự ban đầu ở round 280. **Thứ hạng tổng
thể đã ổn định từ rất sớm (ít nhất từ round 150) và không đổi tới round 280** —
bằng chứng mạnh rằng 280 vòng đã đủ để rút ra kết luận xếp hạng đáng tin cậy.

## 5. C4-DA vs Multi-Krum theo thời gian huấn luyện

| Mốc vòng | Chênh lệch TB (C4-DA − Multi-Krum) | C4-DA thắng |
|---|---|---|
| 150 | −0,0063 ± 0,0490 | 6/12 (ngang ngửa) |
| 200 | +0,0021 ± 0,0675 | 6/12 |
| 240 | −0,0157 ± 0,0579 | 4/12 |
| 280 | −0,0195 ± 0,0462 | 4/12 |

Khoảng cách **nới rộng dần theo hướng có lợi cho Multi-Krum** khi huấn luyện dài hơn
(từ gần như hòa ở round 150 đến chênh ~2 điểm phần trăm ở round 280) — xu hướng này
LÀ động lực chính đáng cho câu hỏi "huấn luyện dài hơn nữa có đảo ngược xu hướng này
không, hay sẽ tiếp tục nới rộng" (xem MINIMAL_EXTENSION_PLAN.md). Độ lệch chuẩn vẫn
lớn hơn chênh lệch trung bình ở mọi mốc — chưa đủ bằng chứng thống kê để khẳng định
dứt khoát, chỉ là một xu hướng quan sát được.

## 6. Kết luận Phase 1

1. Thứ hạng phương pháp đã ổn định (ρ=1,0) từ round 150 — 280 vòng là đủ cho kết luận
   xếp hạng tổng thể.
2. Vẫn còn dư địa cải thiện nhỏ ở round 280 (mọi defense vẫn có slope dương nhẹ),
   nhưng tốc độ đang giảm dần — phù hợp với lợi nhuận biên giảm dần theo thời gian
   huấn luyện, không phải dấu hiệu "chưa học được gì".
3. Trong 4 phương pháp ưu tiên: Multi-Krum là ứng viên có khả năng cải thiện thêm rõ
   nhất (3/3 run chưa hội tụ đều đang improving); C4-DA ít có khả năng cải thiện thêm
   bằng cách đơn thuần thêm vòng (0 run improving).
4. Khoảng cách C4-DA vs Multi-Krum có xu hướng nới rộng theo thời gian huấn luyện,
   chưa đủ mạnh về thống kê để kết luận dứt khoát — đây là câu hỏi mở chính đáng cho
   phần mở rộng.
