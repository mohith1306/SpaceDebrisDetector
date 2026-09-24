import os
from pathlib import Path

try:
    import torch
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    DEVICE = get_device()
    print(f"Using device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
except ImportError:
    DEVICE = "cpu"
    print("PyTorch not installed, defaulting to CPU")

# Base directory: use CAPSTONE_DIR env var if set, otherwise use the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path(os.environ.get("CAPSTONE_DIR", PROJECT_ROOT))

# Google Drive sync settings
GDRIVE_CACHE_DIR = BASE_DIR / "gdrive_cache"
GDRIVE_FOLDER_NAME = "Capstone"  # Folder name in "Shared with me"

DATA_DIR = BASE_DIR
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
YOLO_LABEL_DIR = DATA_DIR / "labels_yolo"
YOLO_BASE = DATA_DIR / "yolo_dataset"
CHECKPOINT_DIR = BASE_DIR / "model_checkpoints"
RUNS_DIR = BASE_DIR / "runs"

IMAGE_DIR = DATA_DIR / "train"
LABEL_CSV = LABELS_DIR / "train.csv"

NUM_CLASSES = 1
CLASS_NAMES = ["space_object"]

def get_yaml_config():
    return f"""path: {YOLO_BASE}
train: images/train
val: images/val

nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""

def verify_paths():
    paths = {
        "IMAGE_DIR": IMAGE_DIR,
        "LABEL_CSV": LABEL_CSV,
    }
    for name, path in paths.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name}: {path}")
    return all(p.exists() for p in paths.values())

def ensure_data_synced():
    """Sync data from Google Drive if not present locally."""
    if not verify_paths():
        print("📥 Local data not found, syncing from Google Drive...")
        try:
            from .data.gdrive_sync import sync_capstone_data
            local_path = sync_capstone_data(
                cache_dir=str(GDRIVE_CACHE_DIR),
                folder_name=GDRIVE_FOLDER_NAME
            )
            # Update paths to use synced data
            global DATA_DIR, IMAGE_DIR, LABEL_CSV, PROCESSED_DIR, LABELS_DIR
            DATA_DIR = Path(local_path)
            IMAGE_DIR = DATA_DIR / "train"
            LABEL_CSV = DATA_DIR / "labels" / "train.csv"
            PROCESSED_DIR = DATA_DIR / "processed"
            LABELS_DIR = DATA_DIR / "labels"
            return verify_paths()
        except Exception as e:
            print(f"❌ Failed to sync from Google Drive: {e}")
            return False
    return True

for d in [PROCESSED_DIR, YOLO_LABEL_DIR, CHECKPOINT_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)