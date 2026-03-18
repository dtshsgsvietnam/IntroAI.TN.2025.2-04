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
  
```text
handwriting-recognition-app/
├── .gitignore                  # Cau hinh bo qua file nang
├── README.md                   # Tai lieu huong dan
├── requirements.txt            # Danh sach thu vien
├── backend/                    # Tang xu ly AI
│   ├── data/                   # Dataset EMNIST/MNIST
│   ├── models/                 # File trong so .pth
│   └── src/                    # Ma nguon chinh
│       ├── __init__.py
│       ├── model.py            # Kien truc CNN
│       ├── image_processing.py # Tien xu ly OpenCV
│       ├── dataset_loader.py   # Tai du lieu
│       ├── train.py            # Huan luyen
│       ├── predict.py          # Du doan
│       └── utils.py            # Ham bo tro
├── frontend/                   # Tang giao dien
│   ├── app.py                  # File chay chính (Tkinter)
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py      # Cua so chính
│   │   └── widgets.py          # Thanh phan tuy chinh
│   └── assets/
│       └── style.qss           # Dinh dang giao dien
└── docs/                       # Tai lieu
    ├── diagrams/               # So do
    ├── reports/                # Ket qua
    └── Proposal_introAI.pdf    # Ban de xuat
```
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