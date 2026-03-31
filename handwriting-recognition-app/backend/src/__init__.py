"""
Backend module cho hệ thống nhận dạng chữ viết tay từ IAM dataset.

Các module chính:
- dataset.py: Load và tiền xử lý ảnh từ IAM Handwriting Database
- model.py: Kiến trúc CNN + LSTM + CTC cho nhận dạng chữ viết tay
- train.py: Script huấn luyện mô hình với progress bar (tqdm)
- utils.py: Utility functions cho inference, tiền xử lý, metrics (CER/WER)
- inference_model.py: Interface để test model trên ảnh đơn/batch/thư mục

Workflow:
1. Chuẩn bị: pip install -r ../../requirements.txt
2. Huấn luyện: python train.py --data_dir /path/to/iam/data --num_epochs 50
3. Inference: python inference_model.py --checkpoint models/best_model.pth --image test.png
"""
