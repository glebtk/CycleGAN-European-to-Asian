import os
import sys

import numpy as np
import torch
from PIL import Image

import config

from datetime import datetime


def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, lr, checkpoint_file):
    try:
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    except FileNotFoundError:
        print(f"Ошибка: не удалось найти {checkpoint_file}")
        return


def get_last_checkpoint(model_name):
    try:
        checkpoints = os.listdir(config.CHECKPOINT_DIR)
        checkpoints = [d for d in checkpoints if os.path.isdir(os.path.join(config.CHECKPOINT_DIR, d))]
        checkpoints.sort()

        last_checkpoint = os.path.join(config.CHECKPOINT_DIR, checkpoints[-1])

        return os.path.join(last_checkpoint, model_name)
    except IndexError:
        print(f"Ошибка: в директории {config.CHECKPOINT_DIR} нет сохраненных чекпоинтов")
        sys.exit(1)
    except FileNotFoundError:
        print(f'Ошибка: не удалось загрузить {model_name}')
        sys.exit(1)


def make_directory(folder_path):
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print("Ошибка: директория ", folder_path, " уже существует")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def tensor_to_array(tensor):
    array = tensor.cpu().detach().numpy()

    # Костыль собственной персоной :)
    if len(array.shape) == 4:
        array = array[0, :, :, :]

    array = np.moveaxis(array, 0, -1)  # Меняем порядок осей на такой, как в PIL Image.   (C, H, W) -> (H, W, C)
    array = (array * config.DATASET_STD + config.DATASET_MEAN) * 255  # Производим денормализацию изображения
    array = np.array(array, dtype=np.uint8)  # Меняем тип данных
    return array


def postprocessing(tensor, return_format="array"):
    tensor = tensor.cpu().detach()

    if return_format == "array":
        image = tensor.numpy()

        if len(image.shape) == 4:
            image = image[-1, ...]

        for channel in range(3):
            image[channel] = image[channel] * config.DATASET_STD[channel] + config.DATASET_MEAN[channel]

        image *= 255

        # image = np.moveaxis(image, 0, -1)
        # image = (image * config.DATASET_STD + config.DATASET_MEAN) * 255

        return image.astype('uint8')

    elif return_format == "tensor":
        image = tensor

        if len(image.size) == 4:
            image = image[-1, ...]

        for channel in range(3):
            image[channel] = image[channel] * config.DATASET_STD[channel] + config.DATASET_MEAN[channel]

        image *= 255

        return image

    else:
        print("Ошибка: неверный формат")
        return tensor
