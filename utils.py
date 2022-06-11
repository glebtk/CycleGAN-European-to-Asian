import os
import sys
import torch
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

        last_checkpoint_directory = os.path.join(config.CHECKPOINT_DIR, checkpoints[-1])

        return os.path.join(last_checkpoint_directory, model_name)
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


def postprocessing(tensor):
    # Конвертируем в np.array
    image = tensor.cpu().detach().numpy()

    # Если на вход пришел батч, берем из него только последнее изображение
    if len(image.shape) == 4:
        image = image[-1, :, :, :]

    # Производим денормализацию
    for channel in range(config.IN_CHANNELS):
        image[channel] = image[channel] * config.DATASET_STD[channel] + config.DATASET_MEAN[channel]
        image[channel] *= 255

    return image.astype('uint8')
