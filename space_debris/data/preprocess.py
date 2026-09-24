import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from ..config import IMAGE_DIR, PROCESSED_DIR

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    processed = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

    return processed

def preprocess_dataset(num_images=2000, force=False):
    image_files = sorted([
        f for f in IMAGE_DIR.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ][:num_images])

    pending_images = [
        img_path for img_path in image_files
        if force or not (PROCESSED_DIR / img_path.name).exists()
    ]

    print(f"Preprocessing {len(pending_images)} images ({len(image_files) - len(pending_images)} cached)...")

    failed = []
    for img_path in tqdm(pending_images, desc="Preprocessing"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                failed.append(img_path.name)
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed = preprocess_image(img)

            save_path = PROCESSED_DIR / img_path.name
            cv2.imwrite(str(save_path), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

        except Exception as e:
            failed.append(img_path.name)

    print(f"Completed. Failed: {len(failed)}")
    return failed

def preview_preprocessing(num_samples=3):
    image_files = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))

    plt.figure(figsize=(12, 8))
    for i, img_path in enumerate(sample_images):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed = preprocess_image(img)

        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(img)
        plt.title(f"Original: {img_path.name}")
        plt.axis("off")

        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(processed)
        plt.title("Processed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preview_preprocessing()
    preprocess_dataset()