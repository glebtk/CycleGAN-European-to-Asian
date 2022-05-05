import os

import numpy as np
import torch
import torch.optim as optim
import torchvision
from PIL import Image

import config
import utils
from generator import Generator


def test(img_dir="test_images/", save_dir="saved_images/", name="test.png"):

    # Загружаем и подготавливаем картинки:
    images = [img for img in os.listdir(img_dir)]  # Names
    images = [Image.open(img_dir + img).convert("RGB") for img in images]  # Names -> PIL Images
    images = [np.array(img) for img in images]  # PIL Images -> np.arrays
    images = [config.test_transforms(image=img)["image"] for img in images]  # Transforms
    images = torch.stack(images)  # List of tensors -> Tensor

    # Загружаем модели генераторов:
    gen_B = Generator(in_channels=config.IN_CHANNELS, num_residuals=9).to(config.DEVICE)
    gen_A = Generator(in_channels=config.IN_CHANNELS, num_residuals=9).to(config.DEVICE)

    optimizer = optim.Adam(
        params=list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    utils.load_checkpoint(gen_B, optimizer, config.LEARNING_RATE, "checkpoints/" + config.CHECKPOINT_GEN_B)
    utils.load_checkpoint(gen_A, optimizer, config.LEARNING_RATE, "checkpoints/" + config.CHECKPOINT_GEN_A)

    # Генерируем картинки
    pred_B = gen_B(images)
    pred_A = gen_A(images)

    # Собираем результат в одну картинку
    result = torch.cat((images, pred_B, pred_A), dim=0)
    result = torchvision.utils.make_grid(result, nrow=len(images))

    # Сохраняем
    result = result*0.5+0.5
    torchvision.utils.save_image(result, save_dir + name)
    print("Successfully saved!")


if __name__ == "__main__":
    test(img_dir="test_images/", save_dir="saved_images/")
