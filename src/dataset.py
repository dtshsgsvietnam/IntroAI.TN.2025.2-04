import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    img_path: Path
    text: str


class CTCLabelConverter:
    """Character-level tokenizer for CTC with an explicit blank token."""

    def __init__(self, texts: List[str]) -> None:
        charset = sorted({ch for t in texts for ch in t})
        self.blank_token = "<CTC_BLANK>"
        self.idx_to_char: List[str] = [self.blank_token] + charset
        self.char_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.idx_to_char)}
        self.blank_idx = 0

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_char)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        chars = []
        for idx in ids:
            if idx == self.blank_idx:
                continue
            chars.append(self.idx_to_char[idx])
        return "".join(chars)


class IAMWordDataset(Dataset):
    """Dataset parser for label file with format: relative_path <ws/tab> label."""

    def __init__(
        self,
        data_root: str,
        label_file: str,
        is_train: bool = False,
        img_height: int = 32,
        img_width: int = 256,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5,
        converter: Optional[CTCLabelConverter] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.label_file = Path(label_file)
        self.is_train = is_train
        self.img_height = img_height
        self.img_width = img_width
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self.samples = self._parse_label_file()
        all_texts = [s.text for s in self.samples]
        self.converter = converter if converter is not None else CTCLabelConverter(all_texts)

    def _parse_label_file(self) -> List[Sample]:
        if not self.label_file.exists():
            raise FileNotFoundError(f"Label file not found: {self.label_file}")

        samples: List[Sample] = []
        with self.label_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = re.split(r"\s+", line, maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid label format at line {line_no}: {line}")

                rel_path, text = parts[0], parts[1]
                rel_path = rel_path.replace("\\", "/")

                if rel_path.startswith("words/"):
                    img_path = self.data_root / rel_path
                else:
                    img_path = self.data_root / "words" / rel_path

                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found at line {line_no}: {img_path}")

                samples.append(Sample(img_path=img_path, text=text))

        if not samples:
            raise RuntimeError("No valid samples found in label file.")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply geometric and morphology augmentation for training only."""
        h, w = img.shape

        # Random rotation in [-5, 5] degrees.
        angle = random.uniform(-5.0, 5.0)
        rot_m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
        img = cv2.warpAffine(
            img,
            rot_m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        # Random horizontal shear.
        shear = random.uniform(-0.15, 0.15)
        shear_m = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        sheared_w = max(1, int(round(w + abs(shear) * h)))
        img = cv2.warpAffine(
            img,
            shear_m,
            (sheared_w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        # Randomly thicken or thin strokes using 2x2 morphology.
        kernel = np.ones((2, 2), dtype=np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

        return img

    def _preprocess_image(self, img_path: Path) -> Tuple[torch.Tensor, int]:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        if self.is_train:
            img = self._augment_image(img)

        h, w = img.shape
        scale = self.img_height / float(h)
        new_w = max(1, int(round(w * scale)))
        new_w = min(new_w, self.img_width)

        resized = cv2.resize(img, (new_w, self.img_height), interpolation=cv2.INTER_CUBIC)

        canvas = np.full((self.img_height, self.img_width), 255, dtype=np.uint8)
        canvas[:, :new_w] = resized

        img_f32 = canvas.astype(np.float32) / 255.0
        img_f32 = (img_f32 - self.normalize_mean) / self.normalize_std
        img_tensor = torch.from_numpy(img_f32).unsqueeze(0)

        return img_tensor, new_w

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image, effective_width = self._preprocess_image(sample.img_path)
        target_ids = self.converter.encode(sample.text)

        return {
            "image": image,
            "target": torch.tensor(target_ids, dtype=torch.long),
            "target_length": len(target_ids),
            "text": sample.text,
            "img_path": str(sample.img_path),
            "orig_width": effective_width,
        }


def collate_fn(batch: List[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = torch.cat([item["target"] for item in batch], dim=0)
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    widths = torch.tensor([item["orig_width"] for item in batch], dtype=torch.long)
    texts = [item["text"] for item in batch]
    paths = [item["img_path"] for item in batch]

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "widths": widths,
        "texts": texts,
        "paths": paths,
    }
