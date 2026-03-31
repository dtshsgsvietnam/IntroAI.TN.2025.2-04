"""
Dataset loader cho IAM Handwriting Database.
Tự động chia 90% train, 5% val, 5% test từ label.txt
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, NamedTuple
import random

class Sample(NamedTuple):
    gt_text: str
    file_path: Path

class IAMDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 img_size: Tuple[int, int] = (128, 32), 
                 split_type: str = "train",
                 seed: int = 42):
        """
        Dataset loader IAM - tự động chia train/val/test (90/5/5)
        
        :param data_dir: Đường dẫn gốc (D:/Downloads/dataset)
        :param img_size: Kích thước ảnh (Width, Height)
        :param split_type: "train", "val", hoặc "test"
        :param seed: Random seed để reproducible
        """
        random.seed(seed)
        
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.split_type = split_type
        self.samples = []
        
        # Load tất cả samples từ label.txt
        all_samples = []
        chars = set()
        bad_samples = {'a01-117-05-02', 'r06-022-03-05'}
        
        label_file = self.data_dir / 'label.txt'
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        print(f"Loading dataset from {label_file}...")
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line[0] == '#':
                    continue
                
                # Format: words/a01/a01-000u/a01-000u-00-00.png\tA
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                img_rel_path = parts[0]  # words/a01/a01-000u/...
                gt_text = parts[1]        # Text label
                img_path = self.data_dir / img_rel_path
                
                if img_path.exists():
                    chars.update(list(gt_text))
                    all_samples.append(Sample(gt_text, img_path))
        
        print(f"Total samples loaded: {len(all_samples)}")
        
        # Chia 90-5-5
        random.shuffle(all_samples)
        train_idx = int(0.90 * len(all_samples))
        val_idx = train_idx + int(0.05 * len(all_samples))
        
        if split_type == "train":
            self.samples = all_samples[:train_idx]
        elif split_type == "val":
            self.samples = all_samples[train_idx:val_idx]
        else:  # test
            self.samples = all_samples[val_idx:]
        
        self.char_list = sorted(list(chars))
        print(f"Loaded {split_type.upper():5} set: {len(self.samples):6} samples")

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh: Giữ nguyên tỷ lệ khung hình (Aspect Ratio), 
        chèn viền trắng (Padding) để đạt đúng kích thước (128x32).
        """
        wt, ht = self.img_size
        
        # Nếu lỗi đọc ảnh, trả về ma trận toàn số 0
        if img is None:
            return np.zeros((1, ht, wt), dtype=np.float32)

        h, w = img.shape
        
        # Tính tỷ lệ thu phóng sao cho chữ không bị kéo giãn méo mó
        scale = min(wt / w, ht / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Tạo một khung phông bạt màu trắng (255) với kích thước chuẩn 128x32
        target = np.ones((ht, wt), dtype=np.uint8) * 255
        
        # Đặt bức ảnh chữ đã thu phóng vào góc trên bên trái của khung trắng
        target[0:new_h, 0:new_w] = img_resized

        # Chuẩn hóa giá trị pixel về dải [-0.5, 0.5] để mạng Nơ-ron học dễ hơn
        target = target.astype(np.float32) / 255.0 - 0.5
        
        # PyTorch yêu cầu tensor ảnh có dạng (Channels, Height, Width). Ở đây là ảnh xám -> 1 Channel
        return np.expand_dims(target, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        
        # Dùng OpenCV đọc ảnh dưới dạng thang độ xám (Grayscale)
        img = cv2.imread(str(sample.file_path), cv2.IMREAD_GRAYSCALE)
        
        # Gọi hàm tiền xử lý
        img_tensor = self.preprocess(img)
        
        # Trả về dữ liệu kiểu Tensor của PyTorch và Nhãn chữ
        return torch.FloatTensor(img_tensor), sample.gt_text