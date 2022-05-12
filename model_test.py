import os

import numpy as np
import torch.optim as optim

from PIL import Image
from generator import Generator
from utils import *


def load_generators():
    gen_E = Generator(in_channels=config.IN_CHANNELS, num_residuals=9).to(config.DEVICE)
    gen_A = Generator(in_channels=config.IN_CHANNELS, num_residuals=9).to(config.DEVICE)

    optimizer = optim.Adam(
        params=list(gen_E.parameters()) + list(gen_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    load_checkpoint(gen_E, optimizer, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_GEN_EUROPEAN))
    load_checkpoint(gen_A, optimizer, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_GEN_ASIAN))

    return gen_E, gen_A


def test(img_dir="test_images", save_dir="saved_images", name="test.png"):

    # Загружаем и подготавливаем картинки:
    images = [img for img in os.listdir(img_dir) if img.endswith(".png") or img.endswith(".jpg")]  # Names
    images = [Image.open(os.path.join(img_dir, img)).convert("RGB") for img in images]  # Names -> PIL Images
    images = [np.array(img) for img in images]  # PIL Images -> np.arrays
    images = [config.test_transforms(image=img)["image"] for img in images]  # Transforms
    images = [img.to(config.DEVICE) for img in images]

    # Загружаем модели генераторов:
    gen_E, gen_A = load_generators()

    # Генерируем изображения:
    pred_E = [tensor_to_array(gen_E(img)) for img in images]
    pred_A = [tensor_to_array(gen_A(img)) for img in images]

    images = [tensor_to_array(img) for img in images]

    # Собираем всё вместе и сохраняем:
    images = np.concatenate(images)
    pred_E = np.concatenate(pred_E)
    pred_A = np.concatenate(pred_A)

    result = np.concatenate((images, pred_E, pred_A), axis=1)

    result = Image.fromarray(result)
    path = os.path.join(save_dir, name)
    result.save(path)

    print(f"Изображение {path} успешно сохранено!")


if __name__ == "__main__":
    test(img_dir="test_images", save_dir="saved_images")
