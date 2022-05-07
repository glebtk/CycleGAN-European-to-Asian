import os
import torch
import config

from datetime import datetime


def save_checkpoint(model, optimizer, filename):
    print("=> Saving " + filename)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, lr, checkpoint_file):
    print("=> Loading " + checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def make_directory(folder_path):
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print("Ошибка: директория ", folder_path, " уже существует")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
