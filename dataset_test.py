import os

import numpy as np
import torch
import torch.optim as optim
import torchvision
from PIL import Image

import config
import utils
from generator import Generator


def test(img_dir="dataset/train/class_A", save_dir="saved_images/", name="test.png"):
    for i in range(5):
        # Загружаем и подготавливаем картинки:
        images = [img for img in os.listdir(img_dir)]  # Names
        images = images[:10]
        images = [Image.open(img_dir + img).convert("RGB") for img in images]  # Names -> PIL Images
        images = [np.array(img) for img in images]  # PIL Images -> np.arrays
        images = [config.train_transforms(image=img)["image"] for img in images]  # Transforms
        images = torch.stack(images)  # List of tensors -> Tensor

        # Собираем результат в одну картинку
        result = torchvision.utils.make_grid(images, nrow=len(images))

        # Сохраняем
        torchvision.utils.save_image(result*config.DATASET_STD+config.DATASET_MEAN, save_dir + f"{i}_" + name)
        print("Successfully saved!")


if __name__ == "__main__":
    test(img_dir="test_images/", save_dir="saved_images/")
