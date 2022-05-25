import time
import numpy as np
import torch.optim as optim

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from generator import Generator
from utils import *


def load_generators():
    gen_E = Generator(in_channels=config.IN_CHANNELS, num_residuals=config.NUM_RESIDUALS).to(config.DEVICE)
    gen_A = Generator(in_channels=config.IN_CHANNELS, num_residuals=config.NUM_RESIDUALS).to(config.DEVICE)

    optimizer = optim.Adam(
        params=list(gen_E.parameters()) + list(gen_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    load_checkpoint(gen_E, optimizer, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_GEN_EUROPEAN))
    load_checkpoint(gen_A, optimizer, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_GEN_ASIAN))

    return gen_E, gen_A


def test(img_dir="test_images"):
    # Загружаем и подготавливаем картинки:
    images = [img for img in os.listdir(img_dir) if img.endswith(".png") or img.endswith(".jpg")]  # Names
    images = [Image.open(os.path.join(img_dir, img)).convert("RGB") for img in images]  # Names -> PIL Images
    images = [np.array(img) for img in images]  # PIL Images -> np.arrays
    images = [config.test_transforms(image=img)["image"] for img in images]  # Transforms
    images = [img.to(config.DEVICE) for img in images]

    # Загружаем модели генераторов:
    gen_E, gen_A = load_generators()

    # Генерируем изображения:
    pred_E = [postprocessing(gen_E(img.detach())) for img in images]
    pred_A = [postprocessing(gen_A(img.detach())) for img in images]

    images = [postprocessing(img.detach()) for img in images]

    # Собираем всё вместе:
    images = np.concatenate(images, axis=2)
    pred_E = np.concatenate(pred_E, axis=2)
    pred_A = np.concatenate(pred_A, axis=2)

    return np.concatenate((images, pred_E, pred_A), axis=1)


if __name__ == "__main__":
    writer = SummaryWriter()
    current = test()
    writer.add_image("Generated images", current, global_step=0)
    time.sleep(10)
