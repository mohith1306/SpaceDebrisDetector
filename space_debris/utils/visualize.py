import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path

from ..config import IMAGE_DIR, PROCESSED_DIR

def show_preprocessing_comparison(num_samples=3):
    image_files = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))

    from ..data.preprocess import preprocess_image

    plt.figure(figsize=(12, 4 * num_samples))
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

def show_dataset_samples(image_dir, num_samples=5):
    image_files = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))

    plt.figure(figsize=(15, 3 * num_samples))
    for i, img_path in enumerate(sample_images):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Sample: {img_path.name}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_preprocessing_comparison()