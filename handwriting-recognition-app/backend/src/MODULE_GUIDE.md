# Backend Module Guide

## Mô tả từng file trong thư mục `backend/src/`

### 📊 dataset.py
**Mục đích:** Load và tiền xử lý dữ liệu từ IAM Handwriting Dataset  
**Chức năng chính:**
- Lớp `IAMDataset(Dataset)` - Đọc ảnh + labels từ IAM format
- Parse file `label.txt` và map tới đúng cấu trúc thư mục
- Tiền xử lý ảnh: resize, padding, normalize (-0.5 to 0.5)
- Tự động chia train/validation set
- Loại bỏ ảnh lỗi được IAM thông báo

**Cách sử dụng:**
```python
from dataset import IAMDataset
dataset = IAMDataset(data_dir="/path/to/iam", img_size=(128, 32), is_train=True)
```

---

### 🧠 model.py
**Mục đích:** Định nghĩa kiến trúc mô hình CNN + LSTM + CTC  
**Thành phần:**
- `CNNFeatureExtractor` - 4 block CNN để extract features từ ảnh
- `RNNDecoder` - 2-layer Bidirectional LSTM để mô hình hóa sequences
- `HandwritingRecognitionModel` - Kết hợp CNN + LSTM + CTC Loss
- `create_model()` - Khởi tạo model và chuyển lên device

**Output shape:** (T=32, B, num_classes=80) cho inference  
**Tổng parameters:** ~3.06M  

**Cách sử dụng:**
```python
from model import create_model
model = create_model(num_classes=80, device="cuda")
log_probs = model(images)  # Inference
```

---

### 🚀 train.py
**Mục đích:** Script huấn luyện mô hình với progress bar  
**Chức năng:**
- Lớp `Trainer` - Quản lý training loop, validation, checkpoint saving
- Tqdm progress bar cho epochs, batches
- Tự động lưu best model và training log (JSON)
- Early stopping support qua best_val_loss tracking
- Encoder text -> CTC targets

**Command line args:**
```bash
python train.py --data_dir /path/to/iam \
                 --num_epochs 50 \
                 --batch_size 32 \
                 --lr 0.001 \
                 --device cuda \
                 --save_dir models
```

---

### 🛠️ utils.py
**Mục đích:** Utility functions cho inference, preprocessing, metrics  
**Chức năng chính:**
- `load_checkpoint()` - Tải model từ .pth file
- `preprocess_image()`, `batch_preprocess_images()` - Processing ảnh
- `decode_predictions()` - Chuyển indices thành text
- `visualize_result()` - Vẽ kết quả lên ảnh (BGR format)
- `calculate_cer()`, `calculate_wer()` - Tính Character/Word Error Rate
- Character ↔ Index mapping

**Cách sử dụng:**
```python
from utils import load_checkpoint, preprocess_image

model, char_list, epoch, val_loss = load_checkpoint(
    "models/best_model.pth", model, device="cpu"
)
tensor, original_img = preprocess_image("test.png", img_size=(128, 32))
```

---

### 🔍 inference_model.py
**Mục đích:** Interface để test mô hình trên ảnh đơn/batch/thư mục  
**Lớp chính:**
- `HandwritingRecognizer` - Wrapper cho inference dễ sử dụng
  - `recognize_single()` - Nhận dạng 1 ảnh
  - `recognize_batch()` - Nhận dạng batch
  - `recognize_directory()` - Nhận dạng thư mục

**Command line usage:**
```bash
# Ảnh đơn lẻ
python inference_model.py --checkpoint models/best_model.pth --image test.png

# Với ground truth (tính accuracy)
python inference_model.py --checkpoint models/best_model.pth \
                          --image test.png \
                          --ground_truth "hello"

# Batch từ thư mục
python inference_model.py --checkpoint models/best_model.pth \
                          --directory /path/to/images \
                          --pattern "*.png" \
                          --save_all
```

**Output:** Predicted text, CER/WER metrics, visualization

---

## 📋 Workflow truy vấn

### 1️⃣ Chuẩn bị
```bash
cd handwriting-recognition-app
pip install -r ../../requirements.txt
python ../../scripts/create_dirs.py
```

### 2️⃣ Huấn luyện
```bash
python backend/src/train.py --data_dir backend/data/iam --num_epochs 50
# Output: models/best_model.pth, models/training_log.json
```

### 3️⃣ Testing
```bash
python backend/src/inference_model.py \
    --checkpoint backend/models/best_model.pth \
    --directory /path/to/test/images \
    --save_all
```

---

## 🔧 Customization

### Thay đổi kích thước ảnh
- Sửa `img_size` trong các hàm (default: 128x32)
- Update CNN architecture nếu cần

### Thay đổi alphabet
- IAM alphabet tự động extract từ `label.txt`
- Model output: num_classes = len(char_list) + 1 (for blank token)

### Tuning model
- CNN hidden_dim: `model.py` line ~133
- LSTM layers/hidden_dim: `model.py` line ~135-137
- Learning rate, batch size: `train.py` args

---

## 📝 Notes

- CTC Loss tự động handle variable-length outputs
- Blank token (0) được thêm tự động trong decoding
- Character Error Rate (CER) cho character-level accuracy
- Word Error Rate (WER) cho word-level accuracy
