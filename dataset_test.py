import os
import cv2
import config
import numpy as np

from utils import postprocessing
from utils import get_current_time


def test(img_dir="test_images", save_dir="saved_images", n=1):
    for i in range(n):
        images = [img for img in os.listdir(img_dir) if img.endswith(".png") or img.endswith(".jpg")]  # Names
        images = [cv2.imread(os.path.join(img_dir, img), 1) for img in images]
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        images = [config.train_transforms(image=img)["image"] for img in images]  # Transforms
        images = [postprocessing(img) for img in images]
        images = np.concatenate(images, axis=2)
        images = np.moveaxis(images, 0, -1)

        path = os.path.join(save_dir, f"dataset_test_{get_current_time()}.png")

        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, images)

    print(f"{n} images successfully saved!")


if __name__ == "__main__":
    test(img_dir="test_images", save_dir="saved_images", n=5)
