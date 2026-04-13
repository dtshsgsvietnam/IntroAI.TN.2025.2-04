from typing import Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CRNN(nn.Module):
    """CRNN model for handwritten text recognition with CTC head.

    Input shape:  [B, 1, H, W], where H is fixed (default 32)
    Output shape: [T, B, C] log-probabilities for CTC
    """

    def __init__(
        self,
        num_classes: int,
        rnn_hidden_size: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            ConvBlock(1, 32, 5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, 5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, 3),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBlock(128, 256, 3),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBlock(256, 512, 3),
            nn.Dropout2d(p=0.2),
        )

        # With input height=32 and pooling scheme, final feature map height is 2.
        cnn_feature_dim = 512 * 2
        self.sequence_projector = nn.Sequential(
            nn.Linear(cnn_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_classes)

    def _compute_output_lengths(self, input_widths: torch.Tensor) -> torch.Tensor:
        # Only first two pools downsample width by factor 2 each; total factor = 4.
        out = torch.div(input_widths, 4, rounding_mode="floor")
        out = torch.clamp(out, min=1)
        return out

    def forward(self, images: torch.Tensor, input_widths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.cnn(images)  # [B, C, H', W']

        # Preserve vertical information by flattening C and H for each timestep W.
        feats = feats.permute(0, 3, 1, 2).contiguous()  # [B, W', C, H']
        feats = feats.flatten(start_dim=2)  # [B, W', C*H']

        feats = self.sequence_projector(feats)
        rnn_out, _ = self.rnn(feats)  # [B, W', 2H]
        logits = self.classifier(rnn_out)  # [B, W', C]

        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2).contiguous()  # [T, B, C]
        output_lengths = self._compute_output_lengths(input_widths).to(dtype=torch.long)
        return log_probs, output_lengths

