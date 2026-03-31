"""
Script huấn luyện mô hình nhận dạng chữ viết tay từ IAM dataset.
Sử dụng CNN + LSTM + CTC Loss với chia tập 90-5-5 (train-val-test)
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from .model import create_model
from .dataset import IAMDataset
from .config import (
    DATA_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    SCHEDULER_TYPE, SCHEDULER_FACTOR, SCHEDULER_PATIENCE, SCHEDULER_MIN_LR,
    DEVICE, NUM_WORKERS, MODELS_DIR, WEIGHT_DECAY,
    USE_EARLY_STOPPING, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA
)


class CTCCollate:
    """Top-level class for collate_fn (picklable on Windows multiprocessing)"""
    def __init__(self, char_list):
        self.char_list = char_list
        # Cache char_to_idx dictionary to avoid recreating in every batch
        self.char_to_idx = {c: i + 1 for i, c in enumerate(char_list)}
    
    def __call__(self, batch):
        """
        Collate function for DataLoader.
        Returns tensors on CPU only - device transfer happens in main loop.
        This avoids multiprocessing issues on Windows.
        """
        images = torch.stack([item[0] for item in batch])
        texts = [item[1] for item in batch]
        
        # Encode targets in CPU worker (parallel processing)
        targets = []
        target_lengths = []
        
        for text in texts:
            encoded = [self.char_to_idx.get(c, 0) for c in text]
            targets.extend(encoded)
            target_lengths.append(len(encoded))
        
        # Create tensors on CPU (no .to(device) here!)
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
        
        return images, targets_tensor, target_lengths_tensor, texts


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        device,
        num_epochs=50,
        save_dir="models",
        char_list=None,
        scheduler=None,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.char_list = char_list or []
        
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        self.start_time = datetime.now()
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_counter = 0

    def encode_targets(self, texts):
        """Chuyển đổi list text thành targets + target_lengths"""
        char_to_idx = {c: i + 1 for i, c in enumerate(self.char_list)}
        
        targets = []
        target_lengths = []
        
        for text in texts:
            encoded = [char_to_idx.get(c, 0) for c in text]
            targets.extend(encoded)
            target_lengths.append(len(encoded))
        
        targets_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long, device=self.device)
        
        return targets_tensor, target_lengths_tensor

    def decode_indices_to_text(self, indices_list):
        """Convert list of indices to text using char_list"""
        texts = []
        idx_to_char = {i + 1: c for i, c in enumerate(self.char_list)}
        
        for indices in indices_list:
            text = "".join([idx_to_char.get(idx, "?") for idx in indices])
            texts.append(text)
        
        return texts

    def train_epoch(self):
        """Huấn luyện 1 epoch - Chỉ tính loss (tránh double forward pass)"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            images, targets, target_lengths, texts = batch
            images = images.to(self.device)
            # Move GPU tensors to device in main loop (after workers finish)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            self.optimizer.zero_grad()
            loss = self.model(images, targets, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """Validation - Tính loss + accuracy"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating", leave=False)
            for batch in pbar:
                images, targets, target_lengths, texts = batch
                images = images.to(self.device)
                # Move GPU tensors to device in main loop (after workers finish)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                loss = self.model(images, targets, target_lengths)
                total_loss += loss.item()
                
                # Calculate accuracy
                logits = self.model(images)
                pred_indices = self.model.decode_greedy(logits)
                pred_texts = self.decode_indices_to_text(pred_indices)
                
                for pred, gt in zip(pred_texts, texts):
                    total += 1
                    if pred.strip() == gt.strip():
                        correct += 1
                
                acc = correct / total if total > 0 else 0
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2%}"})
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        avg_acc = correct / total if total > 0 else 0
        return avg_loss, avg_acc

    def test(self):
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing", leave=False)
            for batch in pbar:
                images, targets, target_lengths, texts = batch
                images = images.to(self.device)
                # Move GPU tensors to device in main loop (after workers finish)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                loss = self.model(images, targets, target_lengths)
                total_loss += loss.item()
                
                # Calculate accuracy
                logits = self.model(images)
                pred_indices = self.model.decode_greedy(logits)
                pred_texts = self.decode_indices_to_text(pred_indices)
                
                for pred, gt in zip(pred_texts, texts):
                    total += 1
                    if pred.strip() == gt.strip():
                        correct += 1
                
                acc = correct / total if total > 0 else 0
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2%}"})
        
        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0
        avg_acc = correct / total if total > 0 else 0
        return avg_loss, avg_acc

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Luu checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "char_list": self.char_list,
        }
        
        latest_path = self.save_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path} (val_loss: {val_loss:.4f})")

    def train(self):
        """Training loop với Early Stopping"""
        print(f"\n== TRAINING START ==")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Alphabet size: {len(self.char_list)}")
        print(f"Early Stopping: patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta}")
        print()
        
        with tqdm(range(self.num_epochs), desc="Epochs") as pbar_epoch:
            for epoch in pbar_epoch:
                train_loss = self.train_epoch()
                val_loss, val_acc = self.validate()
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                # Check if val_loss improved
                improvement = self.best_val_loss - val_loss
                is_best = improvement > self.early_stopping_min_delta
                
                if is_best:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0  # Reset counter khi improve
                else:
                    self.early_stopping_counter += 1
                
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                self.save_checkpoint(epoch, val_loss, is_best=is_best)
                
                elapsed = (datetime.now() - self.start_time).seconds
                pbar_epoch.set_postfix({
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc:.2%}",
                    "es": f"{self.early_stopping_counter}/{self.early_stopping_patience}",
                    "time": f"{elapsed}s"
                })
                
                # Early stopping
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                    break
        
        print(f"\nTraining finished! Best val_loss: {self.best_val_loss:.4f}")
        
        # Test on test set
        print("\n== TESTING ==")
        test_loss, test_acc = self.test()
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2%}")
        
        self.save_training_log(test_loss, test_acc)

    def save_training_log(self, test_loss, test_acc):
        """Luu log training"""
        log = {
            "num_epochs": self.num_epochs,
            "best_val_loss": self.best_val_loss,
            "test_loss": test_loss,
            "test_acc": float(test_acc),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "char_list": self.char_list,
            "duration_seconds": (datetime.now() - self.start_time).seconds,
        }
        
        log_path = self.save_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training log saved: {log_path}")





def main():
    parser = argparse.ArgumentParser(description="Train handwriting recognition model")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Dataset path")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default=str(MODELS_DIR), help="Model save directory")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers")
    
    args = parser.parse_args()
    
    # Check data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    if not (data_path / "label.txt").exists():
        print(f"Error: label.txt not found in {args.data_dir}")
        return
    
    print("Loading dataset...")
    
    # Load train/val/test datasets (automatically splits 90-5-5)
    try:
        train_dataset = IAMDataset(data_dir=str(data_path), img_size=(128, 32), split_type="train")
        val_dataset = IAMDataset(data_dir=str(data_path), img_size=(128, 32), split_type="val")
        test_dataset = IAMDataset(data_dir=str(data_path), img_size=(128, 32), split_type="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create single CTCCollate instance for all DataLoaders (consistent char alphabet)
    collate_fn_obj = CTCCollate(train_dataset.char_list)
    
    # DataLoaders (all use same collate_fn for synchronization)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_obj,
        num_workers=args.num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_obj,
        num_workers=args.num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_obj,
        num_workers=args.num_workers,
    )
    
    # Create model
    print("Creating model...")
    num_classes = len(train_dataset.char_list) + 1
    model = create_model(num_classes=num_classes, device=args.device)
    
    # Optimizer với Weight Decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    print(f"Optimizer: Adam (lr={args.lr}, weight_decay={WEIGHT_DECAY})")
    
    # Scheduler
    scheduler = None
    if SCHEDULER_TYPE == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR
        )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=args.device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        char_list=train_dataset.char_list,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
