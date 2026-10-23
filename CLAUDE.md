# CLAUDE.md — pq-zkfl-medical

## Scope
Khảo sát các phương pháp xác thực gradient (Norm, Cosine, Sign Consensus, kết hợp
C4-DA v1) để phát hiện Byzantine model poisoning trong Federated Learning, xây trên
baseline ZKFL-PQ (arXiv:2603.03398, [1] trong báo cáo). Báo cáo học thuật đích:
`~/Downloads/Bao_cao_Bao_mat_Gradient_FL_FINAL_REVISED.docx` (5 chương, cấu trúc
TOC hiện dùng Word SEQ/REF fields cho đánh số hình/bảng tự động — không cần renumber
thủ công khi thêm/bớt hình).

## Cấu trúc project
- `benchmark/` — package chính: `attacks/` (suite.py: large_norm, low_norm,
  full_sign_flip, directional_poisoning, sparse_coordinate_attack), `defenses/`
  (norm, cosine, sign_consensus, median, multi_krum, c4_drift_aware — flat OR của
  3 điều kiện Norm/Cosine/Sign, không phải pipeline tuần tự), `calibration/`,
  `models/`, `runner.py`, `metrics.py`.
- `scripts/` — mỗi thí nghiệm có script riêng + script aggregate/analysis tương ứng
  (vd `pathmnist_benchmark_v2.py` → `pathmnist_benchmark_v2_aggregate.py`).
  `fedisic_benchmark.py` resume-safe: skip run_id đã có trong run_manifest.csv,
  mở rộng bằng cách append vào list `combos` trong `main()`.
- `results/pathmnist/` — các thư mục theo thí nghiệm, quan trọng nhất:
  - `benchmark_v2_corrected/` — sweep đơn phương pháp chính (594 lượt ghi nhận,
    6 attack × 6 defense × 4 non-IID × 3 seed = 432 theo thiết kế cơ bản; 594 là số
    ghi nhận trong manifest, KHÔNG tự suy ra bằng tích — xem Bảng 4.4 báo cáo).
  - `analysis_v2/attack_defense_matrix.csv` — bảng tổng hợp TPR/FPR/accuracy từ sweep
    trên, cột `calibration_regime` (dùng `A_condition_specific_oracle`/`not_applicable`
    cho số liệu chính thức trong báo cáo).
  - `combined_design_v1/` — sàng lọc offline 7 phương án kết hợp (C0–C6) trên
    9.000 dòng trace; `CANDIDATE_DEFINITIONS.md` là nguồn gốc định nghĩa C0–C6.
    C2 ≡ C0 theo thiết kế (sanity check, không phải bằng chứng độc lập).
  - `c4_drift_aware_confirmation/run_manifest.csv` — cột `malicious_tpr`,
    `honest_fpr`, filter theo `candidate == "c4_drift_aware"`.
  - `c4_da_ablation/run_manifest.csv` — 216 lượt, 4 biến thể × 3 kịch bản × 3 seed
    (42,43,44).
  - `lownorm_sensitivity/run_manifest.csv` — sweep η bổ sung (0.1/0.3/0.5/0.8),
    FedAvg, Dirichlet α=0.5, 2 seed.
- `results/fed_isic2019/main_benchmark/` — `run_manifest.csv` (level run) +
  `client_decisions.csv` (level client-round, cột `delta_modified`, `accepted`,
  dùng để tính TPR/FPR thủ công qua `(mal.accepted==False).mean()`).

## Quy tắc kiểm chứng số liệu
1. Số liệu trong Word PHẢI truy được về một file CSV/manifest cụ thể trong `results/`
   — không coi số liệu đã có trong báo cáo là bằng chứng cho chính nó.
2. TPR/FPR tính theo đơn vị **client-vòng** (client-round), không phải theo run —
   chỉ áp dụng cho phương pháp có quyết định accept/reject tường minh (Norm, Cosine,
   Sign Consensus, C4-DA v1). FedAvg/Median/Multi-Krum không có TPR/FPR theo định
   nghĩa này.
3. Khi một ô kết quả hiển thị rỗng/"chưa thử" — kiểm tra trước xem có phải lỗi script
   vẽ biểu đồ (tra nhầm dataframe nguồn) trước khi kết luận là thí nghiệm chưa chạy.
   Đã từng xảy ra: C4-DA v1/PathMNIST bị vẽ nhầm thành "chưa thử" dù dữ liệu có sẵn
   trong `c4_drift_aware_confirmation/run_manifest.csv`.
4. low_norm TPR=0 ở Norm/Cosine/Sign là hệ quả toán học, không phải lỗi: với
   g'=ηg, η∈(0,1): cos(g',g₀)=cos(g,g₀) và sgn(g')=sgn(g) (bất biến theo tỉ lệ
   dương), Norm chỉ kiểm tra cận trên nên không bao giờ vượt ngưỡng khi bị co nhỏ.
5. Không tự động coi kết quả bất thường (0%, 100%, dao động mạnh giữa seed) là lỗi —
   trước hết đối chiếu với định nghĩa tấn công/phương pháp, chỉ sửa khi có mâu thuẫn
   thực sự với implementation hoặc log.

## Quy tắc trích dẫn
- Hệ trích dẫn dùng số thứ tự [1]…[21], giữ nguyên thứ tự xuất hiện lần đầu trong văn bản.
- Khi không thể truy cập nguồn gốc để xác minh DOI/arXiv/venue, đánh dấu UNVERIFIED —
  không tự khẳng định đã kiểm tra.
- Phân biệt rõ kết quả của tác giả gốc (vd accuracy 93,3% công bố trong [1]) với kết
  quả nhóm tự tái hiện (74,2% đo lại) — không gộp chung.

## Quy tắc chỉnh sửa báo cáo
- Giữ cấu trúc 5 chương, không viết lại Chương 1–3 trừ khi phát hiện sai sót thực sự
  hoặc mâu thuẫn với thực nghiệm.
- Không đổi kết quả/kết luận chỉ để "đẹp" hơn; nếu C4-DA v1 không vượt trội, phải
  trình bày trung thực.
- Không đưa tên file `.py`, đường dẫn log, hoặc câu kiểu "chúng tôi đã sửa lỗi" vào
  nội dung học thuật.
- Hình/bảng hiện dùng Word SEQ field tự động đánh số — khi thêm/bớt, không cần
  renumber thủ công như các bản trước, chỉ cần update field (F9) trước khi export PDF.

## Phát hiện quan trọng từ đợt kiểm định R10 (2026-10-21)
- `directional_poisoning` trong `benchmark/attacks/suite.py` nhận tham số tùy chọn
  `target_norm` để ghim độ lớn tại một ngưỡng chỉ định (ví dụ τ) — nhưng **không có
  script nào trong `scripts/` từng truyền tham số này** (luôn gọi
  `apply_attack(name, d, rng, {})`). Hệ quả: `directional_poisoning` đã chạy luôn
  tương đương `full_sign_flip` (`gi'=-gi`) trong TOÀN BỘ thực nghiệm đã có. Nếu cần
  đánh giá đúng ý tưởng thiết kế ban đầu (độ lớn ghim tại τ), phải gọi
  `apply_attack("directional_poisoning", d, rng, {"target_norm": tau})` tường minh.
- Tham số `k` trong ngưỡng Norm thích nghi của C4-DA v1 (`_drift_tau()` trong
  `benchmark/defenses/combined.py`) là hằng số module-level cố định
  (`MODIFIED_Z_CUTOFF=3.5`, `MAD_TO_STD=1.4826`), KHÔNG qua bước hiệu chỉnh calibration
  nào — chỉ τ (static fallback), ρ, κ mới được hiệu chỉnh (`benchmark/calibration/`).
- File Word báo cáo hiện dùng Word field `SEQ ... \s 1` (đánh số lại theo từng
  Heading1/chương) cho mọi Hình/Bảng/Công thức — đánh số đúng và nhất quán khi đọc
  giá trị cache trực tiếp trong XML, và hiển thị đúng khi mở bằng Microsoft Word.
  **LibreOffice headless (dùng để xuất PDF QA) tính lại các field này sai** (ví dụ
  hiển thị "Bảng 4.7" thay vì "Bảng 4.3" đúng) — đây là giới hạn hỗ trợ `\s` switch
  của LibreOffice, không phải lỗi trong file. Khi QA bằng PDF xuất từ LibreOffice,
  không dùng số thứ tự hiển thị trên hình/bảng làm bằng chứng — đọc giá trị cache
  trực tiếp từ `<w:fldChar w:fldCharType="separate"/><w:t>N</w:t>` trong XML thay vào đó.

## Phát hiện từ R11 (2026-10-21, thí nghiệm τ-calibrated directional_poisoning)
- Script `scripts/directional_tau_calibrated_benchmark.py` chạy
  `apply_attack("directional_poisoning", d, rng, {"target_norm": tau})` đúng thiết
  kế ban đầu (khác với mọi script trước đó, vốn luôn gọi với `params={}`). Kết quả
  tại `results/pathmnist/directional_tau_calibrated/`.
- **Cosine, Sign Consensus**: TPR khớp gần như tuyệt đối với trường hợp own-norm đã
  chạy trước đó (sai số trong phạm vi nhiễu seed) — xác nhận thực nghiệm tính bất
  biến theo tỷ lệ dương (đã chứng minh toán học).
- **Norm**: ghim độ lớn đúng tại τ đặt cập nhật vào ranh giới quyết định
  (`scores <= tau`) — kết quả chấp nhận/từ chối dao động mạnh theo seed (dominated
  by floating-point boundary noise), KHÔNG phải một đặc tính bảo mật ổn định. Không
  nên dùng "pin tại τ" làm benchmark "hardest case" cho Norm trong các thí nghiệm
  tương lai — cần một margin rõ ràng dưới hoặc trên τ (vd τ×0.95 hoặc τ×1.05) để có
  kết quả xác định/tái lập được.
- **C4-DA v1**: TPR vẫn cao (0,889–1,000) bất kể độ lớn tấn công đặt ở đâu — phát
  hiện chủ yếu qua nhánh dấu (Si∧Pi), không phụ thuộc nhánh Norm.

## Giới hạn tài nguyên tính toán (đã xác nhận)
- Fed-ISIC2019 ~18 lần chi phí/vòng so với PathMNIST (38,5 vs 2,1 giây/vòng,
  `results/fed_isic2019/round_budget_study/`).
- GPU đơn (local, GTX 1660 Super), không có cluster — mọi thí nghiệm bổ sung cần
  cân nhắc thời gian chạy trước khi thực hiện (tham khảo runtime cột trong các
  run_manifest.csv). Từ R12: có thêm máy remote `avis-1` (SSH alias, xem
  `~/.ssh/config`) — NVIDIA RTX 5000 Ada, 32.760 MiB VRAM, nhanh hơn ~2,4 lần so
  với máy local (đo thực tế: ~0,82s/vòng vs ~2,0s/vòng, fedavg/dirichlet_a0.5).
  Repo được đồng bộ sang đó qua tar+scp (không phải git remote).

## Phát hiện từ R12 (2026-10-21, rebuild benchmark cho valid/reproducible)
- **AUDIT** (`results/rebenchmark_v2/AUDIT_REPORT.md`, 17 unit test mới trong
  `tests/test_benchmark_core.py`): model sync, định nghĩa delta, data leakage,
  công thức TPR/FPR, fixed-K-step training — tất cả VERIFIED, không có bug.
  Phát hiện duy nhất: `benchmark/defenses/fedavg.py` dùng `mean(deltas)` KHÔNG
  trọng số, trong khi công thức (2.2)/(2.3) trong báo cáo mô tả FedAvg có trọng
  số n_i/n (McMahan et al. 2017). Đây là lựa chọn CÓ CHỦ ĐÍCH, nhất quán với
  thiết kế fixed-K-step (mọi client train đúng K bước bất kể cỡ dữ liệu, để
  delta không bị lệch theo cỡ client — cộng trọng số theo n_i sẽ làm mất một
  phần tác dụng này). Đã thêm `FedAvgWeighted` (`ctx['client_weights']`) để so
  sánh, KHÔNG đổi hành vi `FedAvg` gốc. Cần sửa công thức (2.2)/(2.3) trong báo
  cáo cho khớp code (unweighted), không phải sửa code theo báo cáo.
- **BASELINE CONVERGENCE** (`results/rebenchmark_v2/baseline_convergence/`,
  24 run = {fedavg, fedavg_weighted} × {iid, dirichlet a=1.0/0.5/0.1} × 3 seed,
  tới 150 vòng): round budget CŨ (25 vòng) chỉ đạt ~22% độ chính xác validation
  cuối cùng (0,409 ở dirichlet a=0,5 round 24 so với 0,604 ở round 149) —
  **under-trained nghiêm trọng**, không phải tham số mô hình/tấn công sai.
  Tiêu chí hội tụ (pre-registered, PATIENCE=15, ACC_EPS=0,005, LOSS_EPS=0,002,
  xem `scripts/rebenchmark_v2_analyze_convergence.py`): **0/24 run hội tụ
  trong 150 vòng** — dirichlet a=0,1 vẫn cải thiện nhanh nhất ở cuối. FedAvg có
  trọng số cho kết quả nhỉnh hơn một chút, nhiễu nhất ở a=0,1 (0,500±0,025
  không trọng số vs 0,532±0,047 có trọng số) — không đủ khác biệt để đổi
  phương pháp chính.
- **FULL BENCHMARK** (`results/rebenchmark_v2/full_benchmark/`,
  `scripts/rebenchmark_v2_full_benchmark.py`): 504 run = 7 defense (fedavg,
  norm, cosine, sign_consensus, median, multi_krum, c4_drift_aware) × 6 attack
  (no_attack, large_norm, low_norm, full_sign_flip, directional_poisoning
  [τ-calibrated — fix R10/R11], sparse_coordinate_attack) × 4 non-IID × 3 seed.
  **N_ROUNDS=280** (user chọn "Option B" 250-300 vòng, sau khi so sánh GPU-hours
  trên avis-1 ~32h vs local ~77h). **ATTACK_FROM_ROUND=100** (KHÔNG phải 10 như
  sweep cũ) — do baseline convergence cho thấy mô hình chỉ đạt ~22% accuracy
  cuối cùng ở vòng 10-15 (không có điểm plateau sớm như Fed-ISIC2019), nên tấn
  công chỉ bắt đầu sau khi mô hình đã học được phần lớn (round 100 ≈ 75-90% độ
  chính xác cuối cùng tùy partition). Chỉ dùng calibration Regime A
  (condition-specific oracle) — Regime B không lặp lại (đã có sẵn kết quả độc
  lập trong `benchmark_v2_corrected`, trực giao với vấn đề round-budget).
  Chạy trên avis-1, nohup detached (sống sót qua mất kết nối SSH), kết quả
  sync định kỳ về local qua rsync.
- Dataset cache (`~/.medmnist/pathmnist.npz`) tải lần đầu trên avis-1 bị HỎNG
  do mạng chập chờn (kích thước đúng 206MB nhưng MD5 sai) — `medmnist` không tự
  phát hiện lỗi này cho tới khi train script gọi `download=True` và kiểm MD5
  thất bại. Luôn kiểm `md5sum` sau khi tải dataset lần đầu trên máy mới, đừng
  chỉ tin kích thước file.
