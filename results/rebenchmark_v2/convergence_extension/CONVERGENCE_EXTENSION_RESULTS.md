# Convergence Extension Results — 18 run × 500 vòng

**Hoàn tất 18/18 run.** FedAvg, Multi-Krum, C4-DA v1 × {IID, Dirichlet α=0,5} × seed
{42,43,44}, restart từ vòng 0 (không có checkpoint cũ), đánh giá dừng bằng
**validation**, test chỉ 1 lần cuối. Dữ liệu thô: `summary.csv`,
`checkpoint_comparison.csv`, `<run_id>/round_metrics.csv`,
`<run_id>/final_result.json`. Preflight (4 bước, báo cáo ở lượt trước) đã xác nhận:
tách val/test sạch, resume chính xác tuyệt đối (kể cả C4-DA ở mốc 50 vòng), tốc độ
GPU-native đúng đắn trong phạm vi sai số float32 đã định lượng (1/100 quyết định
accept/reject của C4-DA lệch do boundary noise — xem lại báo cáo preflight).

## 1. Tỷ lệ hội tụ theo phương pháp × partition

| Defense | Partition | Hội tụ thật (tiêu chí PATIENCE/ACC_EPS/LOSS_EPS gốc, không đổi) |
|---|---|---|
| C4-DA | IID | **3/3** (round 198, 310, 447) |
| C4-DA | Dirichlet α=0,5 | **3/3** (round 462, 465, 483) |
| FedAvg | IID | 1/3 (round 465; 2 run vẫn đang cải thiện ở vòng 500) |
| FedAvg | Dirichlet α=0,5 | 3/3 (round 474-477) |
| Multi-Krum | IID | 1/3 (round 475; 2 run vẫn đang cải thiện ở vòng 500) |
| Multi-Krum | Dirichlet α=0,5 | 2/3 (round 476-477; 1 run vẫn cải thiện) |

**C4-DA là phương pháp hội tụ nhất quán nhất (6/6)** — nhưng vì lý do KHÁC với kỳ
vọng ban đầu: nó không "học nhanh hơn", mà **bị mắc kẹt ở một mức plateau thấp hơn**
(rõ nhất ở IID, xem hình) nên tiêu chí hội tụ (vốn chỉ đòi hỏi đường cong phẳng) được
thỏa mãn sớm. FedAvg/Multi-Krum tiếp tục học chậm nhưng đều đặn tới tận vòng 500,
nên thường CHƯA thỏa tiêu chí hội tụ nghiêm ngặt dù accuracy cao hơn nhiều.

## 2. Accuracy validation tại các mốc vòng (trung bình 3 seed)

| Defense | Partition | r280 | r350 | r400 | r450 | r500 |
|---|---|---|---|---|---|---|
| FedAvg | IID | 0,742 | 0,766 | 0,783 | 0,797 | **0,805** |
| Multi-Krum | IID | 0,735 | 0,764 | 0,782 | 0,795 | **0,805** |
| C4-DA | IID | 0,655 | 0,656 | 0,657 | 0,645 | **0,655** |
| FedAvg | Dirichlet α=0,5 | 0,697 | 0,721 | 0,734 | 0,754 | **0,764** |
| Multi-Krum | Dirichlet α=0,5 | 0,644 | 0,680 | 0,705 | 0,708 | **0,729** |
| C4-DA | Dirichlet α=0,5 | 0,689 | 0,725 | 0,720 | 0,738 | **0,760** |

Mean±std đầy đủ theo seed: xem `checkpoint_comparison.csv`.

**Quan sát chính**: ở IID, FedAvg và Multi-Krum **gần như trùng khít nhau hoàn toàn**
từ vòng 280 đến 500 (chênh lệch <0,01 ở mọi mốc) — cả hai tiếp tục học đều đặn. C4-DA
**đóng băng ở mức plateau ~0,65 ngay sau vòng ~250** (xem hình `fig_convergence_500.png`,
đường hồng dao động quanh 0,65 không tăng thêm). Ở Dirichlet α=0,5, thứ tự khác hẳn:
C4-DA vượt Multi-Krum rõ rệt (0,760 vs 0,729 ở vòng 500), xấp xỉ FedAvg.

## 3. So sánh C4-DA vs Multi-Krum có đổi theo thời lượng huấn luyện không?

**Có, và đổi theo HƯỚNG KHÁC NHAU tùy partition:**

- **IID**: khoảng cách (có lợi cho Multi-Krum) **nới rộng mạnh** khi huấn luyện dài
  hơn — từ gần hòa lúc đầu đến **Multi-Krum vượt C4-DA tới 15 điểm phần trăm** ở vòng
  500 (0,805 vs 0,655). Đây là xác nhận THỰC NGHIỆM (không còn ngoại suy từ dữ liệu
  280-vòng/test-based cũ) cho xu hướng đã nghi ngờ ở Phase 1 — và mạnh hơn nhiều so
  với ước tính ngoại suy trước đó (~2 điểm %).
- **Dirichlet α=0,5**: ngược lại — **C4-DA dần vượt qua Multi-Krum** (0,689 vs 0,644
  ở vòng 280 → 0,760 vs 0,729 ở vòng 500), khoảng cách nới rộng theo hướng có lợi cho
  C4-DA.

**Kết luận**: "C4-DA có còn cạnh tranh với Multi-Krum không" **không có câu trả lời
đơn nhất** — phụ thuộc hoàn toàn vào mức độ non-IID. Ở IID, câu trả lời rõ ràng là
KHÔNG (cách biệt lớn và nới rộng theo thời gian). Ở Dirichlet α=0,5, C4-DA cạnh tranh
tốt, thậm chí nhỉnh hơn khi huấn luyện đủ dài.

## 4. Tốc độ GPU thực tế (đo được, không ngoại suy)

| Defense | s/vòng thực đo (3 luồng song song) |
|---|---|
| FedAvg | 1,661 |
| Multi-Krum | 1,733 |
| C4-DA | 1,842 |

Tổng compute: **4,36 giờ GPU** (tổng cộng dồn 18 run), chạy song song 3 luồng nên
**wall-clock thực tế chỉ ~1h40** — nhanh hơn ước tính ban đầu (2,5-3,5 giờ). So với
chi phí NẾU KHÔNG tối ưu (ngoại suy theo tỷ lệ chi phí quan sát ở benchmark 504-run
gốc cho cùng phạm vi 18×500 vòng): ước tính ~6,33 giờ compute → **tốc độ tăng thực tế
≈1,45×** (thấp hơn con số 2,05× đo ở Phase 2 vì lần đo này gồm cả hiệu chỉnh,
checkpoint I/O, đánh giá, và tranh chấp GPU 3 luồng — không phải chỉ phần vòng lặp SGD
thuần túy).

## 5. Phân biệt hội tụ thật vs đóng băng (bắt buộc theo yêu cầu)

C4-DA/IID: "hội tụ" theo tiêu chí hình thức nhưng **là đóng băng giả danh hội tụ**
(accuracy dao động quanh một mức thấp cố định ~0,65, không tăng thêm dù còn 250 vòng
nữa) — PHẢI diễn giải khác với "hội tụ thật" của Dirichlet α=0,5 (accuracy vẫn tăng
chậm, mượt, và việc dừng ở vòng ~470 phản ánh thực sự gần hết dư địa cải thiện trong
khung thời gian quan sát). Không được báo cáo "C4-DA hội tụ 6/6, tốt nhất" mà không
kèm theo phân biệt này — con số hội tụ cao của C4-DA ở IID là dấu hiệu TIÊU CỰC
(mắc kẹt), không phải tích cực.

## 6. Giới hạn còn tồn đọng

1. Chỉ 3/7 phương pháp, 2/4 partition, điều kiện no_attack — không tổng quát hóa sang
   Norm/Cosine/Sign Consensus, Dirichlet α=1,0/0,1, hay dưới tấn công thật.
2. n=3 seed — xu hướng IID (khoảng cách nới rộng) nhất quán cả 3 seed (độ lệch chuẩn
   nhỏ: C4-DA 0,0043, FedAvg/Multi-Krum ~0,003-0,004 ở vòng 500) nên đáng tin; xu
   hướng Dirichlet α=0,5 có độ lệch chuẩn lớn hơn (0,01-0,017) — kết luận "C4-DA vượt
   Multi-Krum" ở đây ít chắc chắn hơn về mặt thống kê hình thức dù hướng nhất quán.
3. Không thể dừng sớm thật sự giữa chừng (tiêu chí hồi cứu) — mọi run đều chạy đủ 500
   vòng dù một số đã "hội tụ" từ sớm (C4-DA/IID/seed42 ở vòng 198) — tốn một phần GPU-
   time cho các vòng không mang thêm thông tin, nhưng cần thiết để đo đường cong đầy
   đủ phục vụ phân biệt hội tụ-thật/đóng-băng ở mục 5.
4. Nguyên nhân CƠ CHẾ vì sao C4-DA đóng băng ở IID (dù τ đã drift-aware) nhưng không
   ở Dirichlet α=0,5 — đã có giả thuyết định tính từ ABLATION_RESULTS.md (ρ/κ tĩnh
   gây từ chối hàng loạt ở IID) nhưng CHƯA kiểm chứng trực tiếp trong các run 500-vòng
   này (không log lại accepted-count per-round cho phần mở rộng, chỉ có val_accuracy/
   loss) — cần một đợt log bổ sung nếu muốn xác nhận cơ chế chính xác ở độ dài 500
   vòng.

## 7. Không cập nhật báo cáo Word

Đúng theo yêu cầu — toàn bộ phần này nằm trong `results/rebenchmark_v2/
convergence_extension/`, tách biệt hoàn toàn khỏi `full_benchmark/` gốc. File Word
chưa bị đụng tới.
