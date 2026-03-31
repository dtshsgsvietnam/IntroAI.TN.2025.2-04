"""
Utility functions cho inference và xử lý dữ liệu.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def load_checkpoint(checkpoint_path: str, model, device: str = "cpu"):
    """
    Tải model từ checkpoint.
    
    Args:
        checkpoint_path: Đường dẫn tới checkpoint file
        model: Model architecture
        device: Device (cpu/cuda)
    
    Returns:
        Model đã tải, char_list, epoch, val_loss
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    char_list = checkpoint.get("char_list", [])
    epoch = checkpoint.get("epoch", 0)
    val_loss = checkpoint.get("val_loss", float("inf"))
    
    return model, char_list, epoch, val_loss


def load_training_log(log_path: str) -> Dict:
    """Tải training log từ JSON file"""
    with open(log_path, "r") as f:
        log = json.load(f)
    return log


def create_char_to_idx_map(char_list: List[str]) -> Dict[str, int]:
    """Tạo mapping từ character sang index"""
    return {c: i + 1 for i, c in enumerate(char_list)}  # 0 = blank


def create_idx_to_char_map(char_list: List[str]) -> Dict[int, str]:
    """Tạo mapping từ index sang character"""
    return {i + 1: c for i, c in enumerate(char_list)}  # 0 = blank


def preprocess_image(
    img_path: str, 
    img_size: Tuple[int, int] = (128, 32)
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Tiền xử lý ảnh để input vào model.
    
    Args:
        img_path: Đường dẫn ảnh
        img_size: Kích thước (width, height)
    
    Returns:
        (tensor đã xử lý, ảnh gốc numpy)
    """
    # Đọc ảnh
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {img_path}")
    
    img_original = img.copy()
    
    # Tiền xử lý giống như trong dataset
    wt, ht = img_size
    h, w = img.shape
    
    # Tính tỷ lệ thu phóng
    scale = min(wt / w, ht / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Padding với màu trắng
    target = np.ones((ht, wt), dtype=np.uint8) * 255
    target[0:new_h, 0:new_w] = img_resized
    
    # Normalize
    target = target.astype(np.float32) / 255.0 - 0.5
    
    # (H, W) -> (1, H, W) -> (1, 1, H, W) for batch
    tensor = torch.FloatTensor(target).unsqueeze(0).unsqueeze(0)
    
    return tensor, img_original


def decode_predictions(
    predictions: List[List[int]],
    idx_to_char: Dict[int, str]
) -> List[str]:
    """
    Giải mã predictions (list of indices) thành text.
    
    Args:
        predictions: List[List[int]] từ model.decode_greedy()
        idx_to_char: Mapping từ index sang character
    
    Returns:
        List[str] - các text dự đoán
    """
    texts = []
    for pred_indices in predictions:
        text = "".join(idx_to_char.get(idx, "") for idx in pred_indices)
        texts.append(text)
    return texts


def visualize_result(
    img: np.ndarray,
    predicted_text: str,
    ground_truth: Optional[str] = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Vẽ kết quả dự đoán lên ảnh.
    
    Args:
        img: Ảnh grayscale (numpy array)
        predicted_text: Text dự đoán
        ground_truth: Ground truth text (nếu có)
        save_path: Lưu ảnh nếu được cung cấp
    
    Returns:
        Ảnh với text được vẽ lên
    """
    # Convert grayscale to BGR để vẽ text màu
    if len(img.shape) == 2:
        img_viz = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_viz = img.copy()
    
    # Vẽ predicted text
    cv2.putText(
        img_viz,
        f"Predicted: {predicted_text}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    # Vẽ ground truth nếu có
    if ground_truth:
        cv2.putText(
            img_viz,
            f"Ground Truth: {ground_truth}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
    
    # Lưu ảnh nếu được yêu cầu
    if save_path:
        cv2.imwrite(str(save_path), img_viz)
        print(f"✓ Ảnh được lưu: {save_path}")
    
    return img_viz


def batch_preprocess_images(
    img_paths: List[str],
    img_size: Tuple[int, int] = (128, 32)
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Tiền xử lý batch ảnh.
    
    Args:
        img_paths: List các đường dẫn ảnh
        img_size: Kích thước output
    
    Returns:
        (batch tensor, list ảnh gốc)
    """
    tensors = []
    originals = []
    
    for img_path in img_paths:
        try:
            tensor, original = preprocess_image(img_path, img_size)
            tensors.append(tensor)
            originals.append(original)
        except Exception as e:
            print(f"⚠️  Lỗi xử lý {img_path}: {e}")
            continue
    
    if not tensors:
        raise ValueError("Không có ảnh nào được xử lý thành công")
    
    batch_tensor = torch.cat(tensors, dim=0)  # (B, 1, H, W)
    return batch_tensor, originals


def calculate_cer(predicted: str, ground_truth: str) -> float:
    """
    Tính Character Error Rate (CER) giữa dự đoán và ground truth.
    
    Args:
        predicted: Chuỗi dự đoán
        ground_truth: Chuỗi gốc
    
    Returns:
        CER (float, 0-1)
    """
    from difflib import SequenceMatcher
    
    # Normalize
    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()
    
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    
    # Đếm edit distance đơn giản
    matcher = SequenceMatcher(None, pred, gt)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    
    cer = 1 - (matches / max(len(pred), len(gt)))
    return cer


def calculate_wer(predicted: str, ground_truth: str) -> float:
    """
    Tính Word Error Rate (WER) giữa dự đoán và ground truth.
    
    Args:
        predicted: Chuỗi dự đoán
        ground_truth: Chuỗi gốc
    
    Returns:
        WER (float, 0-1)
    """
    pred_words = predicted.strip().lower().split()
    gt_words = ground_truth.strip().lower().split()
    
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    # Đếm word mismatches
    mismatches = sum(1 for p, g in zip(pred_words, gt_words) if p != g)
    mismatches += abs(len(pred_words) - len(gt_words))
    
    wer = mismatches / len(gt_words)
    return min(wer, 1.0)
