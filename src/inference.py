import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from metrics import ctc_greedy_decode
from model import CRNN


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_checkpoint_path(checkpoint_arg: str) -> Path:
    """Resolve checkpoint path from cwd first, then relative to this file's parent."""
    candidate = Path(checkpoint_arg)
    if candidate.exists():
        return candidate

    alt = (Path(__file__).resolve().parent / checkpoint_arg).resolve()
    if alt.exists():
        return alt

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")


def preprocess_like_dataset(
    image_path: Path,
    img_height: int = 32,
    img_width: int = 256,
    normalize_mean: float = 0.5,
    normalize_std: float = 0.5,
) -> Tuple[torch.Tensor, int]:
    """Mirror IAMWordDataset._preprocess_image for inference consistency."""
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img.shape
    scale = img_height / float(h)
    new_w = max(1, int(round(w * scale)))
    new_w = min(new_w, img_width)

    resized = cv2.resize(img, (new_w, img_height), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((img_height, img_width), 255, dtype=np.uint8)
    canvas[:, :new_w] = resized

    img_f32 = canvas.astype(np.float32) / 255.0
    img_f32 = (img_f32 - normalize_mean) / normalize_std

    img_tensor = torch.from_numpy(img_f32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return img_tensor, new_w


def decode_to_text(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    idx_to_char: Sequence[str],
    blank_idx: int,
) -> str:
    decoded_ids = ctc_greedy_decode(
        log_probs,
        blank_idx=blank_idx,
        input_lengths=output_lengths.detach().cpu(),
    )
    if not decoded_ids:
        return ""

    text_chars: List[str] = []
    for idx in decoded_ids[0]:
        if idx == blank_idx:
            continue
        ch = idx_to_char[idx]
        if ch == "<CTC_BLANK>":
            continue
        text_chars.append(ch)
    return "".join(text_chars)


class CRNNInference:
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device)
        if "converter" not in ckpt:
            raise KeyError("Checkpoint missing 'converter' dictionary.")

        converter = ckpt["converter"]
        if "idx_to_char" not in converter or "blank_idx" not in converter:
            raise KeyError("Checkpoint converter must contain 'idx_to_char' and 'blank_idx'.")

        self.idx_to_char: List[str] = list(converter["idx_to_char"])
        self.blank_idx: int = int(converter["blank_idx"])

        if not self.idx_to_char:
            raise ValueError("Checkpoint converter 'idx_to_char' is empty.")
        if not (0 <= self.blank_idx < len(self.idx_to_char)):
            raise ValueError("Checkpoint converter 'blank_idx' is out of range.")

        self.model: nn.Module = CRNN(num_classes=len(self.idx_to_char)).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: Path) -> str:
        image_tensor, effective_width = preprocess_like_dataset(image_path)
        image_tensor = image_tensor.to(self.device, non_blocking=True)
        input_widths = torch.tensor([effective_width], dtype=torch.long, device=self.device)

        log_probs, output_lengths = self.model(image_tensor, input_widths)
        return decode_to_text(log_probs, output_lengths, self.idx_to_char, self.blank_idx)


def list_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise RuntimeError(f"No image files found in directory: {input_path}")
    return sorted(files)


def maybe_render_overlay(image_path: Path, prediction: str, show: bool, save_vis_dir: str) -> None:
    if not show and not save_vis_dir:
        return

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Warning: cannot visualize unreadable image: {image_path}")
        return

    canvas = bgr.copy()
    cv2.putText(
        canvas,
        f"Pred: {prediction}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    if save_vis_dir:
        out_dir = Path(save_vis_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / image_path.name
        cv2.imwrite(str(out_path), canvas)

    if show:
        cv2.imshow("Inference", canvas)
        cv2.waitKey(0)


def run_single_image(engine: CRNNInference, image_path: Path, show: bool, save_vis_dir: str) -> None:
    prediction = engine.predict(image_path)
    print(f"Prediction: {prediction}")
    maybe_render_overlay(image_path, prediction, show=show, save_vis_dir=save_vis_dir)


def run_directory(engine: CRNNInference, image_paths: Iterable[Path], show: bool, save_vis_dir: str) -> None:
    for path in tqdm(list(image_paths), desc="Infer", leave=False):
        try:
            prediction = engine.predict(path)
            print(f"{path.as_posix()}\t{prediction}")
            maybe_render_overlay(path, prediction, show=show, save_vis_dir=save_vis_dir)
        except Exception as exc:
            print(f"{path.as_posix()}\t[ERROR] {exc}")

    if show:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for CRNN + CTC model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--input", type=str, required=True, help="Path to one image file or an image directory")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")

    # Optional visualization flags.
    parser.add_argument("--show", action="store_true", help="Show image with prediction overlay")
    parser.add_argument("--save_vis_dir", type=str, default="", help="Directory to save overlaid images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
        input_path = Path(args.input)
        device = resolve_device(args.device)

        engine = CRNNInference(checkpoint_path=checkpoint_path, device=device)
        image_paths = list_images(input_path)

        if len(image_paths) == 1 and input_path.is_file():
            run_single_image(engine, image_paths[0], show=args.show, save_vis_dir=args.save_vis_dir)
        else:
            run_directory(engine, image_paths, show=args.show, save_vis_dir=args.save_vis_dir)
    except Exception as exc:
        print(f"Inference failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()