import os
import sys

import torch
import config

from datetime import datetime


def save_checkpoint(model, optimizer, filename):
    print("=> Сохраняется " + filename)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, lr, checkpoint_file):
    try:
        print("=> Загружается " + checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    except FileNotFoundError:
        print(f"Ошибка: не удалось найти {checkpoint_file}")
        return

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_last_checkpoint(model_name):
    try:
        checkpoints = os.listdir(config.CHECKPOINT_DIR)
        checkpoints = [d for d in checkpoints if os.path.isdir(os.path.join(config.CHECKPOINT_DIR, d))]

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
