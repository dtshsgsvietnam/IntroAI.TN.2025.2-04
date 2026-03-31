"""
Configuration - Ngắn gọn, dễ customize.
"""

from pathlib import Path
import torch

# PATHS
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data" / "iam"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# DATASET
IMG_SIZE = (128, 32)
TRAIN_VAL_SPLIT = 0.95
BAD_SAMPLES = {"a01-117-05-02", "r06-022-03-05"}

# MODEL
NUM_CLASSES = 80
CNN_HIDDEN_DIM = 256
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
DROPOUT = 0.5
CTC_BLANK = 0

# TRAINING - Optimized for RTX 5060 Ti (16GB)
NUM_EPOCHS = 60  # Quick training to test
BATCH_SIZE = 256  # RTX 5060 Ti 16GB -> batch 256 OK (~1GB)
LEARNING_RATE = 0.0005  # Giảm vì batch size lớn, gradient ít noisy
OPTIMIZER = "adam"
GRAD_CLIP = 1.0
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
SAVE_BEST_ONLY = True

# SCHEDULER - Learning Rate Schedule
USE_SCHEDULER = True
SCHEDULER_TYPE = "reduce_on_plateau"  # reduce_on_plateau: adaptive, tốt hơn exponential
SCHEDULER_FACTOR = 0.5  # Giảm LR * 0.5 khi val_loss stuck
SCHEDULER_PATIENCE = 5  # Chờ 5 epochs không improve rồi giảm
SCHEDULER_MIN_LR = 1e-6  # LR tối thiểu

# DEVICE & DATALOADER
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 10  # Parallel data loading (Windows multiprocessing - increased from 6)

# INFERENCE
DECODING_METHOD = "greedy"
VISUALIZE = True

# METRICS
COMPUTE_CER = True
COMPUTE_WER = True


def create_dirs():
    """Tạo thư mục cần thiết"""
    for d in [MODELS_DIR, DATA_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"IMG_SIZE: {IMG_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
