"""
Mô hình CNN + LSTM + CTC cho nhận dạng chữ viết tay từ IAM dataset.
Kiến trúc:
  - CNN: Extract visual features từ ảnh (128x32)
  - LSTM: Mô hình hóa dependencies giữa các ký tự trong sequence
  - CTC Loss: Xử lý variable-length output sequences
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class CNNFeatureExtractor(nn.Module):
    """
    Trích xuất đặc trưng từ ảnh bằng CNN.
    Input: (B, 1, H=32, W=128)
    Output: (B, C=256, H'=4, W'=32) -> reshape thành (B, T=32, D=1024)
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 256):
        super(CNNFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim

        # Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # (B, 32, 16, 64)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # (B, 64, 8, 32)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # (B, 128, 4, 32)
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4, 1)),  # (B, 256, 1, 32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 32, 128)
        output: (B, T=32, D=256)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # (B, 256, 1, 32) -> (B, 256, 32) -> (B, 32, 256)
        x = x.squeeze(2)  # (B, 256, 32)
        x = x.permute(0, 2, 1)  # (B, 32, 256)
        return x


class RNNDecoder(nn.Module):
    """
    Mô hình hóa dependencies giữa các ký tự bằng LSTM.
    Input: (B, T=32, D=256)
    Output: (B, T=32, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 80,
    ):
        super(RNNDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Projection layer: (B, T, 2*hidden_dim) -> (B, T, num_classes)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        output: (B, T, num_classes)
        """
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits


class HandwritingRecognitionModel(nn.Module):
    """
    Mô hình CNN + LSTM + CTC cho nhận dạng chữ viết tay.
    """

    def __init__(
        self,
        num_classes: int = 80,
        cnn_hidden_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super(HandwritingRecognitionModel, self).__init__()
        self.num_classes = num_classes

        self.cnn = CNNFeatureExtractor(in_channels=1, hidden_dim=cnn_hidden_dim)
        self.rnn = RNNDecoder(
            input_dim=cnn_hidden_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )

        # CTC Loss tự động tính toán alignment giữa input và target
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass của mô hình.

        Args:
            images: (B, 1, 32, 128) - ảnh grayscale đã resize
            targets: (N,) - concatenated ground truth labels cho CTC loss
                    (N = sum của độ dài tất cả text trong batch)
            target_lengths: (B,) - độ dài của từng target text trong batch

        Returns:
            Nếu targets là None: log_probs (T=32, B, num_classes)
            Nếu targets được cung cấp: loss value
        """
        # CNN Feature extraction
        features = self.cnn(images)  # (B, 32, 256)

        # RNN prediction
        logits = self.rnn(features)  # (B, 32, num_classes)

        # Log softmax cho CTC (CTC loss mong muốn log-softmax input)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        # Transpose cho CTC: (T, B, C) thay vì (B, T, C)
        log_probs = log_probs.permute(1, 0, 2)  # (T=32, B, num_classes)

        if targets is None:
            return log_probs

        # Tính CTC loss
        batch_size = images.size(0)
        input_lengths = torch.full(
            (batch_size,), log_probs.size(0), dtype=torch.long, device=images.device
        )

        if target_lengths is None:
            raise ValueError("target_lengths phải được cung cấp khi training")

        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss

    def decode_greedy(self, log_probs: torch.Tensor) -> List[int]:
        """
        Greedy decoding: lấy ký tự có xác suất cao nhất tại mỗi time step.
        Loại bỏ blank (0) và collapse consecutive duplicates theo CTC convention.

        Args:
            log_probs: (T, B, num_classes) - log softmax output từ model

        Returns:
            List[List[int]] - danh sách predicted indices cho mỗi sample trong batch
        """
        # (T, B, num_classes) -> (B, T, num_classes)
        log_probs = log_probs.permute(1, 0, 2)

        # Lấy ký tự có xác suất cao nhất
        predicted_indices = torch.argmax(log_probs, dim=2)  # (B, T)

        predictions = []
        for pred_seq in predicted_indices:
            # Bước 1: Loại bỏ blank (token 0)
            non_blank = [idx.item() for idx in pred_seq if idx.item() != 0]
            
            # Bước 2: Collapse consecutive duplicates
            if non_blank:
                pred_text = [non_blank[0]]
                for idx in non_blank[1:]:
                    if idx != pred_text[-1]:
                        pred_text.append(idx)
            else:
                pred_text = []
            
            predictions.append(pred_text)

        return predictions


def create_model(
    num_classes: int = 80,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> HandwritingRecognitionModel:
    """
    Khởi tạo mô hình và chuyển lên device (GPU/CPU).

    Args:
        num_classes: Số lượng ký tự trong alphabet
        device: Device để chạy model (cuda/cpu)

    Returns:
        HandwritingRecognitionModel đã được khởi tạo
    """
    model = HandwritingRecognitionModel(num_classes=num_classes)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Kiểm tra dimensionality
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(num_classes=80, device=device)

    # Test input
    batch_size = 4
    images = torch.randn(batch_size, 1, 32, 128).to(device)

    # Forward pass (inference)
    with torch.no_grad():
        log_probs = model(images)
        print(f"Log probs shape: {log_probs.shape}")  # (T=32, B=4, num_classes=80)

        predictions = model.decode_greedy(log_probs)
        print(f"Predictions: {predictions}")

    # Test với targets (training)
    targets = torch.randint(1, 80, (10,)).to(device)  # 10 ký tự tổng cộng
    target_lengths = torch.tensor([3, 2, 3, 2], dtype=torch.long).to(device)  # tổng = 10
    loss = model(images, targets, target_lengths)
    print(f"Loss: {loss.item():.4f}")

    print(f"Model running on: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
