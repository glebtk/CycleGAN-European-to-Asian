import os
import config
import numpy as np

from PIL import Image
from utils import *


def test(img_dir="test_images", save_dir="saved_images", n=1):
    for i in range(n):
        images = [img for img in os.listdir(img_dir) if img.endswith(".png") or img.endswith(".jpg")]  # Names
        images = [Image.open(os.path.join(img_dir, img)).convert("RGB") for img in images]  # Names -> PIL Images
        images = [np.array(img) for img in images]  # PIL Images -> np.arrays
        images = [config.train_transforms(image=img)["image"] for img in images]  # Transforms
        images = [postprocessing(img) for img in images]
        images = np.concatenate(images, axis=2)
        images = np.moveaxis(images, 0, -1)

        path = os.path.join(save_dir, f"dataset_test_{get_current_time()}.png")

        images = Image.fromarray(images)
        images.save(path)

    print(f"{n} images successfully saved!")


if __name__ == "__main__":
    test(img_dir="test_images", save_dir="saved_images", n=5)
