# SimpleHTR PyTorch: Nhận dạng chữ viết tay với CRNN + CTC

## Giới thiệu
Dự án triển khai bài toán Handwritten Text Recognition (HTR) offline bằng PyTorch.
Pipeline sử dụng kiến trúc CRNN kết hợp CTC Loss cho nhận dạng chuỗi ký tự từ ảnh từ đơn.
Toàn bộ vòng đời train/infer được thiết kế theo hướng ổn định, tái lập và tối ưu tốc độ.
Cấu hình mặc định đã tinh chỉnh cho GPU lớp NVIDIA RTX 5060 Ti.

Mục tiêu hệ thống:
- Input: ảnh xám chứa một từ (single-word image)
- Output: chuỗi ký tự dự đoán
- Loss: CTC với blank token tường minh

## Kiến trúc mô hình
### 1) CNN Backbone
Backbone CNN trích xuất đặc trưng nét chữ từ ảnh đã chuẩn hóa kích thước 32x256.
Chiều rộng được giảm có kiểm soát để giữ đủ độ phân giải theo trục thời gian.

Điểm kỹ thuật quan trọng:
- Dùng Flatten Height thay cho phép lấy trung bình theo chiều cao.
- Tensor đặc trưng được đổi về [B, W, C, H], sau đó flatten thành [B, W, CxH].
- Cách làm này giữ thông tin không gian theo chiều dọc (ascender/descender, giao nét).

### 2) RNN Sequence Model
Khối Bi-LSTM 2 lớp học ngữ cảnh chuỗi theo cả hai chiều trái-phải và phải-trái.
Projection layer trước RNN giúp nén đặc trưng và ổn định gradient.

### 3) CTC Head
Classifier xuất log-probabilities dạng [T, B, C] cho CTCLoss.
Pha suy luận sử dụng Greedy CTC Decoder để:
- gộp ký tự lặp liên tiếp
- loại bỏ blank token

## Tối ưu huấn luyện
### Mixed Precision (AMP)
Huấn luyện dùng torch.amp.autocast và GradScaler để tăng throughput.
Trình tự cập nhật chuẩn:
1. scale(loss).backward()
2. unscale gradient
3. clip gradient
4. optimizer.step()
5. scaler.update()

Unscale trước clip là bắt buộc để clipping đúng về mặt số học khi dùng AMP.

### Data Augmentation chống overfitting
Augmentation được áp dụng trong bước preprocess của dataset ở chế độ train:
- Random Rotation
- Random Horizontal Shear
- Morphological transforms (erosion/dilation)

### Regularization
- Dropout trong RNN: 0.5
- Dropout2d ở các block CNN sâu
- Weight Decay: 1e-4
- Early stopping theo xu hướng CER validation

## Tăng tốc phần cứng
Training loop bật các tối ưu GPU chuyên dụng:
- TF32 matmul cho CUDA
- cuDNN benchmark + TF32
- DataLoader pin_memory
- persistent_workers
- prefetch_factor

Các thiết lập này giúp chồng lấp I/O và compute tốt hơn trên RTX 5060 Ti.

## Cách sử dụng
Giả sử đang đứng tại thư mục gốc dự án.

Huấn luyện:
python src/train.py --mode train --data_root dataset --label_file dataset/label.txt --batch_size 64 --epochs 80

Suy luận 1 ảnh:
python src/inference.py --checkpoint src/checkpoints/best_model.pth --input dataset/words/c06/c06-138/c06-138-10-01.png --device cuda

Suy luận cả thư mục:
python src/inference.py --checkpoint src/checkpoints/best_model.pth --input dataset/words/c06 --device cuda

Định dạng output ảnh đơn:
Prediction: [TEXT]

## Ghi chú kỹ thuật
- Inference khôi phục idx_to_char và blank_idx từ checkpoint để decode nhất quán.
- Preprocess trong inference đồng bộ với dataset (resize giữ tỉ lệ, pad nền trắng, normalize).
- Có thể ép chạy CPU bằng --device cpu khi không có CUDA.
