import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import IAMWordDataset, collate_fn
from metrics import cer, ctc_greedy_decode, wer
from model import CRNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_validation(
    model: nn.Module,
    criterion: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    blank_idx: int,
    idx_to_char: List[str],
    use_amp: bool,
) -> Tuple[float, float, float]:
    model.eval()

    total_val_loss = 0.0
    pred_texts: List[str] = []
    gt_texts: List[str] = []

    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        widths = batch["widths"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            log_probs, output_lengths = model(images, widths)
            val_loss = criterion(log_probs, targets, output_lengths, target_lengths)

        total_val_loss += val_loss.item()

        decoded_ids = ctc_greedy_decode(
            log_probs.detach(),
            blank_idx=blank_idx,
            input_lengths=output_lengths.detach().cpu(),
        )
        batch_pred = ["".join(idx_to_char[i] for i in ids if i != blank_idx) for ids in decoded_ids]

        pred_texts.extend(batch_pred)
        gt_texts.extend(batch["texts"])

        pbar.set_postfix(val_loss=f"{val_loss.item():.4f}")

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    val_cer = cer(pred_texts, gt_texts)
    val_wer = wer(pred_texts, gt_texts)

    if val_cer >= 0.99 and pred_texts:
        print("Debug (first 5) pred -> gt:")
        for i, (pred, gt) in enumerate(zip(pred_texts[:5], gt_texts[:5]), start=1):
            print(f"{i}. \"{pred}\" -> \"{gt}\"")

    return avg_val_loss, val_cer, val_wer


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    total_train_loss = 0.0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        widths = batch["widths"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            log_probs, output_lengths = model(images, widths)
            loss = criterion(log_probs, targets, output_lengths, target_lengths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        pbar.set_postfix(train_loss=f"{loss.item():.4f}")

    avg_train_loss = total_train_loss / max(1, len(train_loader))
    return avg_train_loss


@torch.no_grad()
def infer_single_image(
    model: nn.Module,
    img_path: Path,
    dataset: IAMWordDataset,
    device: torch.device,
) -> str:
    model.eval()
    image, width = dataset._preprocess_image(img_path)

    images = image.unsqueeze(0).to(device)
    widths = torch.tensor([width], dtype=torch.long, device=device)

    log_probs, _ = model(images, widths)
    decoded_ids = ctc_greedy_decode(
        log_probs,
        blank_idx=dataset.converter.blank_idx,
        input_lengths=torch.tensor([widths.item()], dtype=torch.long),
    )[0]
    text = "".join(
        dataset.converter.idx_to_char[i]
        for i in decoded_ids
        if i != dataset.converter.blank_idx
    )
    return text


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_cer: float,
    best_val_loss: float,
    converter_data: Dict,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_cer": best_cer,
        "best_val_loss": best_val_loss,
        "converter": converter_data,
    }
    torch.save(payload, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRNN + CTC training and inference (PyTorch)")
    parser.add_argument("--data_root", type=str, default=r"D:\Downloads\SimpleHTR\dataset")
    parser.add_argument("--label_file", type=str, default=r"D:\Downloads\SimpleHTR\dataset\label.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_h", type=int, default=32)
    parser.add_argument("--img_w", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train")
    parser.add_argument("--img_path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")

    full_dataset = IAMWordDataset(
        data_root=args.data_root,
        label_file=args.label_file,
        is_train=False,
        img_height=args.img_h,
        img_width=args.img_w,
    )

    converter = full_dataset.converter
    num_classes = converter.num_classes

    train_base_dataset = IAMWordDataset(
        data_root=args.data_root,
        label_file=args.label_file,
        is_train=True,
        img_height=args.img_h,
        img_width=args.img_w,
        converter=converter,
    )
    val_base_dataset = IAMWordDataset(
        data_root=args.data_root,
        label_file=args.label_file,
        is_train=False,
        img_height=args.img_h,
        img_width=args.img_w,
        converter=converter,
    )

    model = CRNN(num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=converter.blank_idx, zero_infinity=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    if args.mode == "infer":
        if not args.resume:
            raise ValueError("--resume must be provided in infer mode")
        if not args.img_path:
            raise ValueError("--img_path must be provided in infer mode")

        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        pred_text = infer_single_image(model, Path(args.img_path), full_dataset, device)
        print(f"Prediction: {pred_text}")
        return

    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    if val_size == 0:
        raise ValueError("Validation split is empty. Increase dataset size or val_ratio.")

    split_generator = torch.Generator().manual_seed(args.seed)
    permuted_indices = torch.randperm(len(full_dataset), generator=split_generator).tolist()
    train_indices = permuted_indices[:train_size]
    val_indices = permuted_indices[train_size:]
    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collate_fn,
    }
    if args.workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 4,
        })

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    best_cer = float("inf")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_cer = ckpt.get("best_cer", best_cer)
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from epoch {ckpt['epoch']}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        avg_train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        avg_val_loss, val_cer, val_wer = run_validation(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            blank_idx=converter.blank_idx,
            idx_to_char=converter.idx_to_char,
            use_amp=use_amp,
        )

        scheduler.step(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss:   {avg_val_loss:.6f}")
        print(f"CER:        {val_cer:.6f}")
        print(f"WER:        {val_wer:.6f}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "cer": val_cer,
                "wer": val_wer,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        improved = (val_cer < best_cer) or (val_cer == best_cer and avg_val_loss < best_val_loss)
        if improved:
            best_cer = val_cer
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                save_path=ckpt_dir / "best_model.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_cer=best_cer,
                best_val_loss=best_val_loss,
                converter_data={
                    "idx_to_char": converter.idx_to_char,
                    "blank_idx": converter.blank_idx,
                },
            )
            print(f"Saved best checkpoint to {(ckpt_dir / 'best_model.pth').as_posix()}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping triggered: no val_cer improvement for "
                    f"{args.patience} consecutive epochs."
                )
                break

    with (ckpt_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
