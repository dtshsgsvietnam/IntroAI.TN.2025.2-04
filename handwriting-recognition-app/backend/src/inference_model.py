"""
Script inference siêu gọn: Test mô hình trên ảnh đơn hoặc thư mục.
Đã tích hợp Otsu Binarization và chuẩn hóa [-0.5, 0.5] chống lỗi nhòe chữ.
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from .model import create_model
from .utils import create_idx_to_char_map, decode_predictions

class HandwritingRecognizer:
    def __init__(self, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.char_list = ckpt["char_list"]
        
        self.model = create_model(num_classes=len(self.char_list) + 1, device=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
        self.idx_to_char = create_idx_to_char_map(self.char_list)
        print(f"✓ Model loaded | Alphabet: {len(self.char_list)} | Device: {device}\n")

    def preprocess_image(self, img_path):
        """Khử nhiễu, dồn viền và chuẩn hóa ảnh y hệt lúc Train"""
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: 
            raise ValueError("Không thể đọc ảnh")
            
        # 1. Binarize: Ép nét xám thành đen/trắng tuyệt đối
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 2. Resize giữ tỷ lệ & Dồn góc trái
        h, w = img.shape
        scale = min(128 / w, 32 / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        
        target = np.ones((32, 128), dtype=np.float32) * 255
        target[0:nh, 0:nw] = cv2.resize(img, (nw, nh))
        
        # 3. Chuẩn hóa pixel về [-0.5, 0.5] và nén thành Tensor 4D
        tensor = torch.FloatTensor(target / 255.0 - 0.5).unsqueeze(0).unsqueeze(0)
        return tensor

    def recognize(self, img_path):
        """Hàm dự đoán nhả thẳng ra text"""
        tensor = self.preprocess_image(img_path).to(self.device)
        with torch.no_grad():
            preds = self.model.decode_greedy(self.model(tensor))
        return decode_predictions(preds, self.idx_to_char)[0]


def main():
    parser = argparse.ArgumentParser(description="Inference HTR")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--directory", type=str, help="Path to directory of images")
    args = parser.parse_args()

    recognizer = HandwritingRecognizer(args.checkpoint)

    # Gộp chung logic xử lý 1 ảnh và 1 thư mục cho gọn
    files = [Path(args.image)] if args.image else list(Path(args.directory).glob("*.png"))
    files += list(Path(args.directory).glob("*.jpg")) if args.directory else []
    
    if not files:
        print("⚠️ Không tìm thấy ảnh nào để test!")
        return

    for f in files:
        try:
            pred = recognizer.recognize(f)
            print(f"📷 {f.name:15} -> ✓ Predicted: {pred}")
        except Exception as e:
            print(f"❌ Lỗi {f.name}: {e}")

if __name__ == "__main__":
    main()