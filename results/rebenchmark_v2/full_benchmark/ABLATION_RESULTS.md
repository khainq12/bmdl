# R12 Ablation — AdaptiveNormOnly: threshold adaptation vs multi-signal fusion

## 0. Mục tiêu và phạm vi (reuse, không chạy lại)

Mục tiêu: tách riêng đóng góp của (a) **threshold adaptation** (ngưỡng τ tính lại mỗi
vòng từ median+MAD của round hiện tại, công thức giống hệt C4-DA v1) khỏi (b)
**multi-signal fusion** (thêm Cosine + Sign + peer-outlier) trong cải thiện của C4-DA
v1 so với Norm tĩnh.

Dữ liệu dùng lại nguyên vẹn từ `results/rebenchmark_v2/ablation_adaptive_norm_only/`
(54 run, hoàn thành trước khi có yêu cầu audit này — **không chạy lại** theo đúng chỉ
thị "avoid unnecessary reruns"). Đã kiểm tra tính toàn vẹn ở `SCIENTIFIC_AUDIT.md` §
"Ablation pilot integrity": 54/54 combo đúng kỳ vọng, 0 thiếu.

**Phạm vi có chủ đích, không phải sai sót**: 3 defense (StaticNorm, AdaptiveNormOnly,
C4-DA v1 đầy đủ) × 3 attack (no_attack, large_norm, directional_poisoning τ-calibrated)
× 2 partition (iid, dirichlet_a0,5) × 3 seed × 280 vòng = 54 run. KHÔNG bao phủ
dirichlet_a1,0/dirichlet_a0,1 hay low_norm/full_sign_flip/sparse_coordinate_attack —
xem mục 5 (giới hạn) để biết tác động của việc thu hẹp phạm vi này lên tính tổng quát
của kết luận.

**Kiểm chứng chéo**: điều kiện `norm/no_attack/iid` trong ablation (chạy trên GPU local,
độc lập) và trong benchmark chính 504-run (chạy trên avis-1) dùng cùng seed —
chênh lệch tuyệt đối tối đa giữa hai lần chạy độc lập: **0,000139** (xem
`analysis_artifacts/ablation_vs_main_crosscheck.csv`) — xác nhận tính tái lập cao giữa
hai phần cứng khác nhau.

## 1. Cài đặt AdaptiveNormOnly

`benchmark/defenses/combined.py::AdaptiveNormOnly` — tái sử dụng chính xác hàm
`_drift_tau()` mà `C4DriftAware` dùng cho nhánh Norm (median + 3,5×1,4826×MAD của
norm_scores trong vòng hiện tại, fallback về τ tĩnh nếu MAD=0), loại bỏ hoàn toàn
Cosine, Sign, và peer-outlier. 3 unit test trong `tests/test_benchmark_core.py` xác
nhận quyết định accept/reject của AdaptiveNormOnly **khớp tuyệt đối từng bit** với
nhánh Norm nội bộ của C4DriftAware khi Cosine/Sign được đặt thành luôn-pass (tất cả
pass, chi tiết trong `test_adaptive_norm_only_matches_c4_drift_aware_norm_branch`).

## 2. Bảng kết quả đầy đủ (mean trên 3 seed)

| Partition | Attack | Defense | Accuracy | TPR | FPR | Diverge |
|---|---|---|---|---|---|---|
| IID | no_attack | StaticNorm | 0,233 ± 0,133 | 1,000* | 0,964 | 0% |
| IID | no_attack | AdaptiveNormOnly | **0,746 ± 0,004** | 0,059 | 0,058 | 0% |
| IID | no_attack | C4-DA | 0,658 ± 0,060 | 0,915 | 0,674 | 0% |
| IID | large_norm | StaticNorm | 0,233 ± 0,133 | 1,000 | 0,964 | 0% |
| IID | large_norm | AdaptiveNormOnly | **0,752 ± 0,016** | 1,000 | 0,032 | 0% |
| IID | large_norm | C4-DA | 0,673 ± 0,029 | 1,000 | 0,659 | 0% |
| IID | directional_poisoning | StaticNorm | 0,149 ± 0,053 | 0,763 | 0,922 | 0% |
| IID | directional_poisoning | AdaptiveNormOnly | 0,733 ± 0,019 | **0,002** | 0,066 | 0% |
| IID | directional_poisoning | C4-DA | 0,670 ± 0,002 | **1,000** | 0,669 | 0% |
| Dirichlet α=0,5 | no_attack | StaticNorm | 0,704 ± 0,023 | 0,000 | 0,200 | 0% |
| Dirichlet α=0,5 | no_attack | AdaptiveNormOnly | 0,688 ± 0,028 | 0,006 | 0,012 | 0% |
| Dirichlet α=0,5 | no_attack | C4-DA | 0,705 ± 0,034 | 0,015 | 0,012 | 0% |
| Dirichlet α=0,5 | large_norm | StaticNorm | 0,678 ± 0,019 | 1,000 | 0,200 | 0% |
| Dirichlet α=0,5 | large_norm | AdaptiveNormOnly | 0,700 ± 0,016 | 1,000 | 0,006 | 0% |
| Dirichlet α=0,5 | large_norm | C4-DA | 0,707 ± 0,021 | 1,000 | 0,005 | 0% |
| Dirichlet α=0,5 | directional_poisoning | StaticNorm | 0,670 ± 0,019 | 0,669 | 0,200 | 0% |
| Dirichlet α=0,5 | directional_poisoning | AdaptiveNormOnly | 0,690 ± 0,042 | **0,974** | 0,006 | 0% |
| Dirichlet α=0,5 | directional_poisoning | C4-DA | 0,693 ± 0,013 | 0,991 | 0,006 | 0% |

\* TPR=1,000 của StaticNorm khi đã đóng băng là giá trị **vô nghĩa về mặt toán học**
(mọi client đều bị từ chối bất kể nội dung — "phát hiện" 100% chỉ là hệ quả của từ
chối toàn bộ, không phản ánh năng lực phát hiện thật). Xem SCIENTIFIC_AUDIT.md §6.

Nguồn: `results/rebenchmark_v2/ablation_adaptive_norm_only/run_manifest.csv`,
tổng hợp tại `analysis_artifacts/ablation_summary_table.csv`. Hình minh họa:
`figures/fig5_ablation_accuracy_tpr.png`.

## 3. Trả lời câu hỏi: adaptation hay fusion?

**Câu trả lời phụ thuộc mức độ trôi ngưỡng — vốn phụ thuộc partition:**

- **IID (trôi ngưỡng rất mạnh)**: threshold adaptation MỘT MÌNH giải quyết hoàn toàn
  đóng băng (0,233→0,746) và phát hiện hoàn hảo tấn công độ lớn (large_norm, TPR=1,0,
  FPR=0,032) — nhưng **mù hoàn toàn** trước tấn công ghim độ lớn ở ngưỡng tĩnh cũ
  (directional_poisoning, TPR=0,002). Multi-signal fusion (thêm Cosine+Sign trong
  C4-DA) là cơ chế DUY NHẤT cứu được điểm mù này (TPR 0,002→1,000), nhưng trả giá FPR
  cao và gần như hằng định ở MỌI điều kiện kể cả khi không cần (no_attack: FPR 0,67;
  large_norm: FPR 0,66) — vì ρ/κ (ngưỡng Cosine/Sign) vẫn tĩnh, cùng một lớp lỗ hổng
  "trôi ngưỡng" như Norm, chỉ ít nghiêm trọng hơn (không có cơ chế all-reject→0).
- **Dirichlet α=0,5 (trôi ngưỡng nhẹ)**: threshold adaptation MỘT MÌNH đã gần như đủ —
  TPR 0,97-1,0 ở CẢ BA loại tấn công test, FPR thấp (0,006-0,012). Multi-signal fusion
  gần như không thêm giá trị (C4-DA ≈ AdaptiveNormOnly, chênh lệch accuracy/TPR/FPR
  trong phạm vi nhiễu giữa seed).

**Kết luận tổng thể**: cải thiện của C4-DA v1 so với Norm tĩnh **chủ yếu đến từ
threshold adaptation** — một mình nó khôi phục gần hết accuracy và xử lý tốt mọi tấn
công theo độ lớn ở mọi điều kiện test. **Multi-signal fusion chỉ thiết yếu trong
trường hợp cực đoan** (non-IID thấp/IID, nơi trôi ngưỡng rất mạnh VÀ kẻ tấn công cố
tình khai thác đúng điểm trôi đó qua ghim độ lớn ở giá trị τ tĩnh cũ) — ở các điều
kiện khác, fusion chỉ thêm chi phí FPR mà không thêm lợi ích đo được. Gợi ý thiết kế
rõ ràng cho C4-DA v2: làm ρ/κ thích nghi theo cùng nguyên tắc τ đã áp dụng.

## 4. Vì sao AdaptiveNormOnly thất bại ở IID nhưng thành công ở Dirichlet α=0,5

Cơ chế: `directional_poisoning` ghim độ lớn tấn công đúng bằng τ **tĩnh** (hiệu chỉnh
1 lần từ mô hình gần-ngẫu-nhiên) của partition đó. Khi huấn luyện tiến triển, ngưỡng
thích nghi (median+MAD của norm thật) trôi lên theo norm hợp lệ thực tế:
- **IID**: norm hợp lệ trôi rất xa khỏi τ tĩnh ban đầu (từ ~0,04-0,06 lên ~0,75-0,85,
  xem SCIENTIFIC_AUDIT.md §6) → độ lớn tấn công ghim ở τ cũ giờ trông "nhỏ" so với
  ngưỡng mới → lọt qua hoàn toàn.
- **Dirichlet α=0,5**: norm hợp lệ trôi ít hơn nhiều → độ lớn tấn công ghim ở τ cũ vẫn
  đủ lớn so với ngưỡng thích nghi hiện tại → vẫn bị bắt (TPR 0,97-1,0).

**Chưa có mô hình định lượng** dự đoán trước mức độ trôi cho một partition/seed bất kỳ
mà không cần chạy thực nghiệm — đây là giới hạn còn tồn đọng, xem mục 5.

## 5. Giới hạn của ablation này

1. Chỉ test 2/4 partition (iid, dirichlet_a0,5) — **chưa xác nhận** mẫu hình trên
   dirichlet_a1,0 (có thể trôi ít hơn dirichlet_a0,5) hay dirichlet_a0,1 (có thể trôi
   theo cách khác do dữ liệu lệch cực mạnh).
2. Chỉ test 3/6 attack — chưa test low_norm (kỳ vọng không ảnh hưởng do bất biến theo
   tỷ lệ dương đã chứng minh ở R10/R11), full_sign_flip, sparse_coordinate_attack.
3. Kết luận "fusion không thêm giá trị ở Dirichlet α=0,5" dựa trên đúng 1 mức non-IID
   trung bình — chưa biết ngưỡng chuyển tiếp (mức α nào bắt đầu cần fusion) nằm ở đâu
   giữa α=1,0 (nhẹ hơn) và α=0,1 (nặng hơn α=0,5).
4. n=3 seed/điều kiện — đủ cho thấy mẫu hình nhất quán (TPR của AdaptiveNormOnly dưới
   directional_poisoning/IID: 0,000/0,000/0.006 — rất nhất quán) nhưng không đủ cho
   kiểm định thống kê hình thức.
