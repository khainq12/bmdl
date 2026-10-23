# R12 Analysis — results/rebenchmark_v2/full_benchmark/ (504 run, 280 vòng/run)

Tiền đề kiểm chứng: xem `SCIENTIFIC_AUDIT.md` (504/504 đầy đủ, 0 sai lệch TPR/FPR,
0 vi phạm cơ chế divergence). Số liệu dưới đây lấy từ
`analysis_artifacts/method_x_attack.csv` và `method_x_attack_x_partition.csv`;
mean±std tính trên 3 seed {42,43,44}, **bao gồm cả run diverge** (xem SCIENTIFIC_AUDIT
§4 vì sao không loại bỏ). Hình minh họa: `figures/`.

## 1. Tổng quan theo phương pháp (accuracy, toàn bộ 6 attack × 4 partition)

| Phương pháp | Acc TB | TPR TB | FPR TB | Diverge | Ghi chú |
|---|---|---|---|---|---|
| Multi-Krum | 0,69 | — | — | 0% | Không có cơ chế accept/reject tường minh |
| C4-DA v1 | 0,67 | 0,72 | 0,19 | 0% | Detector duy nhất không bao giờ diverge |
| Median | 0,65 | — | — | 0% | |
| Sign Consensus | 0,56 | 0,22 | 0,00 | 19,4% | FPR=0 tuyệt đối nhưng TPR thấp, diverge cao |
| Norm (tĩnh) | 0,56 | 0,57 | 0,33 | 1,4% | Bị kéo tụt bởi đóng băng ở IID (xem SCIENTIFIC_AUDIT §6) |
| Cosine | 0,54 | 0,43 | 0,16 | 23,6% | |
| FedAvg (không phòng thủ) | 0,45 | — | — | 33,3% | Baseline tham chiếu |

(Xem hình `fig1_accuracy_by_defense.png`, `fig2_divergence_rate_by_defense.png`.)

## 2. Theo tấn công (trung bình mọi phương pháp)

| Tấn công | Acc TB | Diverge TB |
|---|---|---|
| large_norm | 0,44 | 39,3% |
| sparse_coordinate_attack | 0,54 | 20,2% |
| full_sign_flip | 0,59 | 6,0% |
| directional_poisoning | 0,63 | 1,2% |
| no_attack | 0,66 | 0% |
| low_norm | 0,66 | 0% |

`large_norm` là tấn công gây hại nhất tổng thể. `low_norm`/`no_attack` gần như vô hại
theo thiết kế (low_norm là tấn công "free-rider", không có ý định phá hoại chất
lượng mô hình — xem `docs/BENCHMARK_PROTOCOL.md`).

## 3. Phát hiện quan trọng: Sign Consensus thất bại hoàn toàn trước `large_norm`

| | Sign Consensus / large_norm | FedAvg / large_norm (không phòng thủ) |
|---|---|---|
| Accuracy TB | 0,0818 | 0,0818 |
| TPR | **0,000 (chính xác, cả 3 seed)** | — |
| Diverge rate | **100% (12/12 run)** | 100% (12/12 run) |

Hai dòng số liệu **giống hệt nhau** — vì κ (ngưỡng Sign Consensus) không bao giờ từ
chối client độc hại dùng `large_norm` trong giai đoạn kiểm thử thực tế (`accepted=True`
mọi vòng sau ATTACK_FROM_ROUND=100, xác nhận trực tiếp từ `client_decisions.csv`), nên
Sign Consensus sụp đổ y hệt FedAvg không có phòng thủ. Điều này **bất ngờ** vì κ được
hiệu chỉnh chính bằng `large_norm` làm CALIB_ATTACK (`scripts/rebenchmark_v2_full_benchmark.py`,
`CALIB_ATTACK = "large_norm"`) — tức công cụ được tinh chỉnh cho đúng loại tấn công
này nhưng vẫn không phát hiện được khi áp dụng thực tế. **Cơ chế gốc rễ vì sao điểm
sign-agreement của client độc hại không tách biệt khỏi κ chưa được xác định dứt khoát
trong audit này** (cần replay có log raw sign-score, chưa thực hiện để tránh chạy GPU
không cần thiết) — ghi vào UNRESOLVED_LIMITATIONS, không suy đoán cơ chế khi chưa có
bằng chứng trực tiếp.

Ngược lại, Sign Consensus **phát hiện hoàn hảo** `sparse_coordinate_attack` (TPR=1,0,
FPR=0,0 — xem bảng method_x_attack.csv) — cho thấy hiệu năng của defense này phụ thuộc
rất mạnh vào loại tấn công, không đồng đều.

## 4. Static Norm: đóng băng chi phối kết quả IID (đã audit chi tiết ở SCIENTIFIC_AUDIT §6)

Norm tĩnh đạt accuracy TB 0,72 ở dirichlet_a1,0 nhưng chỉ 0,23 ở IID — không phải vì
Norm "yếu hơn" ở IID về bản chất thống kê, mà vì ngưỡng τ tĩnh (hiệu chỉnh từ mô hình
gần-ngẫu-nhiên) bị vượt qua bởi norm thật của delta khi huấn luyện tiến triển, dẫn đến
từ chối toàn bộ client và đóng băng mô hình từ vòng ~20-30 (minh họa
`fig4_norm_freeze_illustration.png`). Đây là hạn chế của **giao thức hiệu chỉnh một
lần** của benchmark này, không phải hạn chế cố hữu của khái niệm "lọc theo độ lớn" —
xem SCIENTIFIC_AUDIT §9 (so sánh với thiết kế trust-score liên tục của FLTrust gốc) và
ABLATION_RESULTS.md (AdaptiveNormOnly giải quyết được vấn đề này).

## 5. Cosine: hiệu năng rất không ổn định (std cao) dưới tấn công độ lớn thuần túy

| Tấn công | Cosine acc TB | Cosine acc std | Diverge |
|---|---|---|---|
| large_norm | 0,276 | 0,247 | 75% |
| sparse_coordinate_attack | 0,282 | 0,270 | 66,7% |
| no_attack | 0,683 | 0,054 | 0% |

Độ lệch chuẩn accuracy của Cosine dưới `large_norm`/`sparse_coordinate_attack` lớn
gấp ~5 lần so với điều kiện no_attack — cho thấy Cosine **không ổn định** (một số seed
bị sụp đổ hoàn toàn, một số seed vẫn ổn) trước tấn công thuần túy theo độ lớn, hợp lý
vì Cosine đo hướng (direction), không đo độ lớn (magnitude) — một vector ngẫu nhiên
biên độ cực lớn vẫn có thể "tình cờ" đủ gần hướng tham chiếu để vượt qua ngưỡng ρ ở
một số lần chạy.

## 6. So sánh có ghép cặp: C4-DA v1 vs Multi-Krum (72 cặp cùng điều kiện, cùng seed)

Multi-Krum thắng 49/72 cặp, C4-DA thắng 23/72, chênh lệch trung bình
`-0,0224 ± 0,0496` (xem SCIENTIFIC_AUDIT §8). **Không đủ bằng chứng để khẳng định một
phương pháp vượt trội phương pháp kia về accuracy** — độ lệch chuẩn của chênh lệch lớn
hơn bản thân chênh lệch trung bình. Khác biệt thực chất: Multi-Krum không cung cấp
quyết định accept/reject theo từng client (TPR/FPR không xác định), trong khi C4-DA
cung cấp (TPR TB 0,72) — sự đánh đổi giữa độ chính xác thô và khả năng diễn giải.

## 7. Hội tụ no_attack ở 280 vòng

55/84 (65,5%) run no_attack hội tụ theo tiêu chí nghiêm ngặt đã khóa từ R12 BASELINE
(xem SCIENTIFIC_AUDIT §7, hình `fig3_no_attack_convergence_curves.png`). Non-IID càng
mạnh càng khó hội tụ ở cùng round budget (iid: 16/21 hội tụ; dirichlet_a0,5: 11/21).
`norm/iid` "phẳng" ngay từ vòng ~20 nhưng đây là đóng băng, không phải hội tụ thật —
phân biệt rõ hai khái niệm khi đọc đường cong.

## 8. Kết luận tổng hợp (cho mục đích viết lại báo cáo, CHƯA áp dụng vào DOCX)

1. Median và Multi-Krum là lựa chọn an toàn nhất về độ ổn định số trị (không bao giờ
   diverge), nhưng không cung cấp cơ chế phát hiện tường minh.
2. C4-DA v1 là defense CÓ detector duy nhất không bao giờ diverge trong 504 run — ưu
   điểm thực chất, dù accuracy trung bình không vượt trội Multi-Krum một cách có ý
   nghĩa thống kê.
3. Norm tĩnh, Cosine, Sign Consensus đều có một "gót chân Achilles" riêng biệt: Norm
   đóng băng ở IID; Cosine cực kỳ không ổn định trước tấn công độ lớn thuần túy; Sign
   Consensus thất bại hoàn toàn (TPR=0) trước chính loại tấn công nó được hiệu chỉnh
   để bắt. Không có phương pháp đơn lẻ nào "thắng" ở mọi điều kiện.
4. `large_norm` là tấn công khó phòng thủ nhất tổng thể (39,3% toàn bộ 84 run diverge
   bất kể phương pháp).
5. Giới hạn quan trọng cần trình bày trung thực trong báo cáo: 34,5% run no_attack
   chưa hội tụ dù đã tăng round budget lên 280; cơ chế thất bại của Sign Consensus
   trước large_norm chưa được giải thích đầy đủ.
