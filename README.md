# Handwriting Recognition System

Dự án xây dựng hệ thống nhận dạng chữ viết tay cơ bản (chữ cái và chữ số) sử dụng mạng nơ-ron tích chập (CNN) và thư viện OpenCV.

## 1. Đội ngũ thực hiện
- Đinh Thái Sơn - 202416746
- Đỗ Hải Đăng - 202400035
- Nguyễn Trí Hiếu - 202416203

## 2. Mô tả dự án
Dự án tập trung vào việc chuyển đổi các ký tự viết tay từ hình ảnh thành văn bản số. Hệ thống áp dụng các kỹ thuật Học máy và Xử lý ảnh để tự động hóa quy trình nhập liệu từ các ghi chú đơn giản.

Phạm vi kỹ thuật:
- Ký tự hỗ trợ: Chữ cái tiếng Anh (a-z, A-Z) và chữ số (0-9).
- Đầu vào: Ảnh tĩnh chụp rõ nét các ký tự đơn lẻ hoặc từ đơn.
- Công nghệ: Mạng nơ-ron tích chập (CNN), OpenCV, PyQt6.
- Mục tiêu: Độ chính xác trên tập kiểm thử đạt trên 85%.

## 3. Cấu trúc thư mục chi tiết
handwriting-recognition-app/
├── .gitignore                  # Cấu hình bỏ qua các file nặng (data, models)
├── README.md                   # Tài liệu hướng dẫn dự án
├── requirements.txt            # Danh sách thư viện và phiên bản cụ thể
├── backend/                    # Tầng xử lý logic và trí tuệ nhân tạo
│   ├── data/                   # Thư mục chứa dataset EMNIST/MNIST (tải cục bộ)
│   ├── models/                 # Lưu trữ file trọng số (.pth) sau khi huấn luyện
│   └── src/                    # Mã nguồn xử lý chính
│       ├── __init__.py
│       ├── model.py            # Kiến trúc mạng CNN (Conv2D, MaxPooling, Dense)
│       ├── image_processing.py # Tiền xử lý OpenCV (Gray, Binary, Adaptive Threshold)
│       ├── dataset_loader.py   # Tải và chuẩn hóa dữ liệu đầu vào
│       ├── train.py            # Kịch bản huấn luyện và tinh chỉnh siêu tham số
│       ├── predict.py          # Luồng suy luận dự đoán từ ảnh thực tế
│       └── utils.py            # Các hàm bổ trợ đo lường (Accuracy, Loss)
├── frontend/                   # Tầng giao diện người dùng
│   ├── app.py                  # Điểm khởi chạy chính của ứng dụng Desktop
│   ├── ui/                     # Thành phần giao diện PyQt6
│   │   ├── __init__.py
│   │   ├── main_window.py      # Thiết kế cửa sổ chính và các sự kiện nút bấm
│   │   └── widgets.py          # Các thành phần giao diện tùy chỉnh
│   └── assets/                 # Tài nguyên tĩnh
│       └── style.qss           # Định dạng giao diện ứng dụng
└── docs/                       # Tài liệu đi kèm
    ├── diagrams/               # Sơ đồ kiến trúc và luồng dữ liệu
    ├── reports/                # Kết quả đánh giá mô hình (Confusion Matrix)
    └── Proposal_introAI.pdf    # Bản đề xuất đồ án gốc

## 4. Hướng dẫn cài đặt
1. Cài đặt các thư viện cần thiết:
   pip install -r requirements.txt

2. Huấn luyện mô hình (thực hiện trên máy có GPU để tối ưu tốc độ):
   python backend/src/train.py

3. Khởi chạy ứng dụng Desktop:
   python frontend/app.py

## 5. Quy trình thực hiện
- Giai đoạn 1: Thu thập và tiền xử lý dữ liệu EMNIST.
- Giai đoạn 2: Xây dựng và huấn luyện mô hình CNN.
- Giai đoạn 3: Phát triển thuật toán cắt tách ký tự bằng OpenCV.
- Giai đoạn 4: Tích hợp mô hình vào giao diện PyQt6 và đóng gói.