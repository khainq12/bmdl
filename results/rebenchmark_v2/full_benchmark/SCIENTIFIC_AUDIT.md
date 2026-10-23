# R12 Scientific Audit — results/rebenchmark_v2/full_benchmark/ (504-run sweep)

Mọi số liệu trong tài liệu này được tính bởi `scripts/rebenchmark_v2_scientific_audit.py`,
đọc trực tiếp từ `run_manifest.csv` (504 dòng), `per_round_results.csv` (141.120 dòng),
`client_decisions.csv` (705.600 dòng) trong thư mục này. Kết quả trung gian lưu tại
`analysis_artifacts/` (bao gồm `audit_summary.json`) — mọi con số dưới đây truy được
ngược về một file cụ thể, không phải khẳng định suông.

## 1. Tính đầy đủ (completeness)

| Kiểm tra | Kết quả |
|---|---|
| Số combo kỳ vọng (7 defense × 6 attack × 4 partition × 3 seed) | 504 |
| Số dòng trong run_manifest.csv | 504 |
| run_id trùng lặp | 0 |
| combo thiếu | 0 |
| combo thừa ngoài kỳ vọng | 0 |
| Dòng per_round_results.csv | 141.120 (= 504 × 280, khớp chính xác) |
| Dòng client_decisions.csv | 705.600 (= 504 × 280 × 5, khớp chính xác) |
| Run có số vòng ≠ 280 | 0 |
| Combo (defense,attack,partition) không đủ 3 seed | 0 |

**Kết luận**: không có run nào bị bỏ sót thầm lặng, không có trùng lặp, không có combo
thiếu seed. Toàn bộ 504 run như thiết kế đã thực thi đầy đủ.

## 2. Dữ liệu NaN / thiếu

Cột `tp/fp/tn/fn/tpr/fpr` có đúng 216 dòng NaN = 3 defense không-phải-detector
(fedavg, median, multi_krum) × 6 attack × 4 partition × 3 seed = 216. Đây là NaN
**đúng như thiết kế** (các aggregator này không có cơ chế accept/reject nên không có
TPR/FPR) — không có dòng NaN ngoài dự kiến ở bất kỳ cột nào khác (`final_accuracy`,
`final_loss`, `runtime_s` đều 0 NaN). Không có dòng detector nào bị NaN tpr ngoài ý
muốn, không có dòng non-detector nào có tpr khác NaN ngoài ý muốn.

## 3. Đối chiếu công thức TPR/FPR với dữ liệu thô

Với cả 288 run thuộc 4 defense có detector (norm, cosine, sign_consensus,
c4_drift_aware), tp/fp/tn/fn được **tính lại độc lập** từ `client_decisions.csv`
(đếm trực tiếp theo `attack_active` và `accepted` ở mức client-vòng) và so sánh với
giá trị đã lưu trong `run_manifest.csv`.

**Kết quả: 0/288 run có sai lệch.** Công thức TPR/FPR trong pipeline khớp tuyệt đối
với dữ liệu client-vòng thô — không có lỗi tính toán hay lệch pha giữa ghi nhận và
tổng hợp.

## 4. Audit 56 run diverge

**Định nghĩa divergence** (từ `scripts/rebenchmark_v2_full_benchmark.py`,
`DIVERGENCE_LOSS_THRESHOLD=50.0`): một run được đánh dấu `diverged=True` tại vòng
đầu tiên mà `loss` không hữu hạn (NaN/Inf) HOẶC `loss > 50.0`. Từ vòng đó trở đi,
toàn bộ số liệu (accuracy, loss) được **copy-forward** nguyên trạng vòng diverge cho
các vòng còn lại — không huấn luyện tiếp, tiết kiệm compute mà vẫn giữ đủ 280 dòng
cho mỗi run.

**Kiểm tra tính đúng đắn của cơ chế này**:
- 56/504 run diverge. Toàn bộ 56 run: dữ liệu accuracy/loss sau vòng diverge **phẳng
  tuyệt đối** (0 vi phạm copy-forward).
- Toàn bộ 56 run: vòng được đánh dấu diverge **đều thỏa mãn đúng** điều kiện loss>50
  hoặc non-finite tại đúng vòng đó (0 vi phạm ngưỡng).

→ Cơ chế divergence hoạt động chính xác như thiết kế, không có bug.

### Phân bố divergence

| Theo defense | Tỷ lệ diverge | Theo attack | Tỷ lệ diverge | Theo partition | Tỷ lệ diverge |
|---|---|---|---|---|---|
| fedavg | 33,3% (24/72) | large_norm | 39,3% (33/84) | dirichlet_a0,1 | 15,9% (20/126) |
| cosine | 23,6% (17/72) | sparse_coordinate_attack | 20,2% (17/84) | dirichlet_a0,5 | 11,9% (15/126) |
| sign_consensus | 19,4% (14/72) | full_sign_flip | 6,0% (5/84) | dirichlet_a1,0 | 10,3% (13/126) |
| norm | 1,4% (1/72) | directional_poisoning | 1,2% (1/84) | iid | 6,3% (8/126) |
| median | 0% | low_norm | 0% | | |
| multi_krum | 0% | no_attack | 0% | | |
| c4_drift_aware | 0% | | | | |

**Median, Multi-Krum, C4-DA v1 không bao giờ diverge** trong toàn bộ 504 run — đây là
ưu điểm thực chất của 3 phương pháp này, không phụ thuộc vào accuracy trung bình.
`large_norm` là tấn công gây diverge nhiều nhất (dễ hiểu: biên độ delta cực lớn dễ phá
vỡ số trị nếu không bị lọc). Non-IID càng mạnh thì diverge càng nhiều (dirichlet_a0,1:
15,9% so với iid: 6,3%) — nhất quán với trực giác rằng calibration khó chính xác hơn
khi phân phối dữ liệu lệch mạnh.

### Xử lý run diverge trong tính mean accuracy — điểm cần lưu ý khi đọc báo cáo

| Defense | Acc TB (bao gồm diverge) | Acc TB (loại diverge) | Số run diverge |
|---|---|---|---|
| fedavg | 0,454 | 0,628 | 24 |
| cosine | 0,536 | 0,664 | 17 |
| sign_consensus | 0,555 | 0,670 | 14 |
| norm | 0,555 | 0,562 | 1 |
| median | 0,652 | 0,652 | 0 |
| multi_krum | 0,690 | 0,690 | 0 |
| c4_drift_aware | 0,667 | 0,667 | 0 |

**Quyết định phương pháp luận**: báo cáo dùng accuracy TRUNG BÌNH BAO GỒM các run
diverge làm số liệu chính thức (không loại bỏ) — vì diverge dưới tấn công LÀ một
thất bại thực chất cần được phản ánh, không phải nhiễu cần lọc bỏ. Loại bỏ run diverge
sẽ "làm đẹp" số liệu một cách không trung thực (chỉ giữ lại các trường hợp phương pháp
hoạt động). Tuy nhiên, vì 3 defense (fedavg, cosine, sign_consensus) có accuracy trung
bình bị kéo tụt đáng kể bởi divergence (chênh 0,11-0,17), **mọi so sánh accuracy phải
đi kèm tỷ lệ diverge**, không được trích accuracy đơn lẻ mà không nêu bối cảnh.

## 5. Phát hiện kiểm chứng: Sign Consensus ≡ FedAvg dưới no_attack (không phải lỗi)

Khi đối chiếu đường cong hội tụ no_attack, phát hiện `sign_consensus` và `fedavg` cho
đường accuracy **giống hệt từng bit** (`max_abs_diff = 0.0`) ở MỌI partition. Điều tra
cho thấy đây là hệ quả toán học đúng, không phải lỗi: trong 16.800 bản ghi
client-vòng của sign_consensus/no_attack, **100% được accept** (κ không bao giờ từ
chối một client lương thiện nào khi thực sự không có tấn công) — khi accepted=[5/5]
mọi vòng, công thức `_aggregate_from_accept` của sign_consensus suy biến đúng bằng
`mean(deltas)` không trọng số, tức công thức của FedAvg. Đối chiếu: dưới
`full_sign_flip` (có tấn công thật), hai đường cong sign_consensus và fedavg **khác
nhau rõ rệt** đúng như kỳ vọng — xác nhận đây là kiểm chứng chéo tích cực cho tính
đúng đắn của RNG tất định và công thức aggregate, không phải một dấu hiệu lỗi.

## 6. Freeze (đóng băng) vs Diverge — hai hiện tượng tách biệt

Định nghĩa "freeze": 0/5 client được accept liên tục trong 30 vòng cuối cùng (vòng
250-279) của một run thuộc defense có detector.

| | Không diverge | Diverge |
|---|---|---|
| **Không freeze** | 235 | 32 |
| **Freeze** | 21 | 0 |

Freeze và diverge **tách biệt hoàn toàn** (0 run vừa freeze vừa diverge) — đúng như kỳ
vọng về mặt khái niệm: freeze là trạng thái ổn định-nhưng-bế-tắc (update=0, loss
không đổi nhưng vẫn hữu hạn), còn diverge là nổ số trị. 21 run freeze, phân bố:

| Defense | Partition | Số run freeze |
|---|---|---|
| norm | iid | 13 |
| c4_drift_aware | iid | 5 |
| cosine | iid | 3 |

**Toàn bộ 21 run freeze đều xảy ra ở partition IID**, không có freeze nào ở bất kỳ
partition Dirichlet nào — nhất quán với phát hiện R12 trước đó rằng hiện tượng trôi
ngưỡng tĩnh nghiêm trọng nhất ở IID. 5 run freeze của c4_drift_aware đều thuộc
`directional_poisoning` hoặc `full_sign_flip` (tấn công đảo hướng) — gợi ý rằng dù
τ của C4-DA là drift-aware, ρ/κ (ngưỡng Cosine/Sign) vẫn tĩnh và có thể cùng lúc từ
chối đa số client lương thiện dưới IID, thỉnh thoảng trùng hợp đủ 30 vòng liên tiếp.

## 7. Hội tụ no_attack ở 280 vòng — áp dụng đúng tiêu chí đã khóa ở R12 BASELINE

Dùng lại nguyên vẹn tiêu chí hội tụ đã định trước (pre-registered) từ nghiên cứu
baseline convergence R12 (PATIENCE=15, ACC_EPS=0,005, LOSS_EPS=0,002/vòng — xem
`results/rebenchmark_v2/baseline_convergence/`), áp dụng cho toàn bộ 84 run
`no_attack` ở 280 vòng:

**55/84 (65,5%) hội tụ** theo tiêu chí nghiêm ngặt này — cải thiện rõ rệt so với 0/24
ở 150 vòng (nghiên cứu baseline trước). Theo partition: iid 16/21, dirichlet_a1,0
16/21, dirichlet_a0,5 11/21, dirichlet_a0,1 12/21 — xác nhận lại rằng non-IID càng
mạnh càng khó hội tụ. **Không phải mọi điều kiện đều hội tụ ở 280 vòng** — đây là giới
hạn thực sự cần nêu rõ trong báo cáo (xem UNRESOLVED_LIMITATIONS bên dưới), không
được ngầm hiểu là "đã hội tụ" chỉ vì round budget tăng từ 25 → 280.

Trường hợp đặc biệt: `norm/iid` không "hội tụ" theo nghĩa thực chất — nó đứng yên vì
đóng băng (slope ≈ 0 một cách giả tạo, không phải vì đã học xong). Khi trình bày kết
quả hội tụ, tài liệu ANALYSIS.md phân biệt rõ "hội tụ thật" và "đóng băng trông giống
hội tụ".

## 8. So sánh có ghép cặp (paired), không khẳng định hơn kém khi chưa đủ bằng chứng

So sánh C4-DA v1 với Multi-Krum trên **đúng 72 cặp (attack, partition, seed) giống
nhau**: Multi-Krum thắng 49/72, C4-DA thắng 23/72, chênh lệch trung bình
`-0,0224 ± 0,0496` (C4-DA thấp hơn Multi-Krum trung bình 2,24 điểm phần trăm, độ lệch
chuẩn lớn hơn cả chênh lệch trung bình → **không đủ bằng chứng để khẳng định một
phương pháp vượt trội phương pháp kia về accuracy thuần túy**). Multi-Krum thắng
thường xuyên hơn về accuracy nhưng không cung cấp quyết định accept/reject theo từng
client (không có TPR/FPR) — đây là sự đánh đổi giữa accuracy thô và khả năng diễn giải
(interpretability), không phải một phương pháp "tốt hơn" tuyệt đối.

## 9. Audit tính hợp lệ khoa học của protocol

| Hạng mục | Kết luận | Nguồn |
|---|---|---|
| FedAvg unweighted vs weighted | `benchmark/defenses/fedavg.py` dùng `mean()` không trọng số có chủ đích (nhất quán thiết kế fixed-K-step). Công thức (2.2)/(2.3) trong báo cáo Word mô tả FedAvg có trọng số n_i/n — **cần sửa công thức trong báo cáo cho khớp code**, không sửa code. | `results/rebenchmark_v2/AUDIT_REPORT.md` (R12 AUDIT, mục #1) |
| Hiệu chỉnh ngưỡng τ/ρ/κ | Hiệu chỉnh 1 lần qua mô phỏng calibration-client riêng (5 vòng, CALIB_SEED=1000 cố định), độc lập với seed chính. Quyết định ngưỡng theo tiêu chí tối đa TPR với FPR validation ≤ 5%. | `docs/BENCHMARK_PROTOCOL.md` §8 |
| Định nghĩa tấn công | Đã unit-test đầy đủ (17+3=20 test trong `tests/test_benchmark_core.py`), bao gồm khóa hồi quy cho directional_poisoning τ-calibrated (không còn trùng full_sign_flip như các đợt trước). | `tests/test_benchmark_core.py` |
| Tiêu chí hội tụ | Tái sử dụng nguyên vẹn công thức PATIENCE/ACC_EPS/LOSS_EPS đã khóa ở R12 BASELINE — không tạo tiêu chí mới, không nới lỏng để "đạt hội tụ" dễ hơn. | Mục 7 ở trên |
| Round budget & ATTACK_FROM_ROUND | N_ROUNDS=280 (lựa chọn người dùng, Option B); ATTACK_FROM_ROUND=100 suy ra thực nghiệm từ đường cong baseline (mô hình đạt ~75-90% accuracy cuối cùng tại vòng 100 tùy partition), không phải hằng số sao chép từ protocol cũ (vốn dùng ATTACK_FROM_ROUND=10 cho 25 vòng). | `scripts/rebenchmark_v2_full_benchmark.py` docstring |

## 10. Giới hạn chưa giải quyết (unresolved limitations)

1. **34,5% run no_attack chưa hội tụ ở 280 vòng** theo tiêu chí nghiêm ngặt — đặc
   biệt dirichlet_a0,5/dirichlet_a0,1. Muốn kết luận chắc chắn hơn về hiệu năng trần
   cần round budget lớn hơn, chưa thực hiện (cân nhắc chi phí GPU).
2. **AdaptiveNormOnly thất bại hoàn toàn (TPR≈0) trước directional_poisoning ở IID**
   nhưng lại phát hiện tốt (TPR 0,97-1,0) ở dirichlet_a0,5 — cơ chế gốc rễ (độ lớn
   tấn công ghim ở τ tĩnh cũ, ngưỡng thích nghi trôi xa đến mức nào tùy partition) đã
   được xác định định tính nhưng chưa có mô hình định lượng dự đoán trước mức độ trôi
   cho một partition bất kỳ — chi tiết tại ABLATION_RESULTS.md.
3. **ρ/κ (ngưỡng Cosine/Sign) vẫn tĩnh**, chịu chung một loại lỗ hổng trôi-ngưỡng như
   Norm tĩnh (dù ít nghiêm trọng hơn vì không có cơ chế all-reject→0 giống Norm) —
   minh chứng: freeze xảy ra ở cả c4_drift_aware/iid (5 run) dù τ đã drift-aware. Đây
   là hướng cải tiến (C4-DA v2) chưa được triển khai hay kiểm chứng thực nghiệm.
4. **So sánh C4-DA vs Multi-Krum chưa đủ mạnh về mặt thống kê** (n=72 cặp, độ lệch
   chuẩn chênh lệch lớn hơn giá trị trung bình chênh lệch) — không nên trình bày một
   trong hai là "tốt hơn" mà không kèm theo caveat paired-comparison này.
5. **n=3 seed/điều kiện** — đủ để phát hiện mẫu hình nhất quán rõ ràng (như hiện tượng
   đóng băng, vốn lặp lại cả 3 seed) nhưng KHÔNG đủ cho kiểm định thống kê hình thức
   (t-test, Wilcoxon) có ý nghĩa với các chênh lệch nhỏ. Mọi khẳng định "A tốt hơn B"
   trong báo cáo cần dựa trên **tính nhất quán qua cả 3 seed + chênh lệch đủ lớn so với
   std giữa seed**, không dựa trên p-value hình thức.
6. **Ablation AdaptiveNormOnly chỉ chạy trên 2/4 partition và 3/6 attack** (xem
   ABLATION_RESULTS.md) — quyết định có chủ đích để tiết kiệm compute, không phải sai
   sót, nhưng kết luận về "threshold adaptation vs multi-signal fusion" chưa được xác
   nhận trên dirichlet_a1,0, dirichlet_a0,1, hay 3 attack còn lại (low_norm,
   full_sign_flip, sparse_coordinate_attack).
