import os
import cv2
import sys
import torch
import config
import numpy as np

from PIL import Image
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
        print(f"Error: couldn't find {checkpoint_file}")
        return


def get_last_checkpoint(model_name):
    try:
        checkpoints = os.listdir(config.CHECKPOINT_DIR)
        checkpoints = [d for d in checkpoints if os.path.isdir(os.path.join(config.CHECKPOINT_DIR, d))]
        checkpoints.sort()

        last_checkpoint_directory = os.path.join(config.CHECKPOINT_DIR, checkpoints[-1])

        return os.path.join(last_checkpoint_directory, model_name)
    except IndexError:
        print(f"Error: there are no saved checkpoints in the {config.CHECKPOINT_DIR} directory")
        sys.exit(1)
    except FileNotFoundError:
        print(f'Error: failed to load {model_name}')
        sys.exit(1)


def make_directory(folder_path):
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print(f"Error: directory \"{folder_path}\" already exists")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def model_test(gen_E, gen_A, img_dir="test_images"):
    # We upload and prepare images:
    images = [img for img in os.listdir(img_dir) if img.endswith(".png") or img.endswith(".jpg")]  # Names
    images = [cv2.imread(os.path.join(img_dir, img), 1) for img in images]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    images = [config.test_transforms(image=img)["image"] for img in images]  # Transforms
    images = [img.to(config.DEVICE) for img in images]

    # Generating images:
    pred_E = [postprocessing(gen_E(img.detach())) for img in images]
    pred_A = [postprocessing(gen_A(img.detach())) for img in images]

    images = [postprocessing(img.detach()) for img in images]

    # Putting everything together:
    images = np.concatenate(images, axis=2)
    pred_E = np.concatenate(pred_E, axis=2)
    pred_A = np.concatenate(pred_A, axis=2)

    return np.concatenate((images, pred_E, pred_A), axis=1)


def postprocessing(tensor):
    image = tensor.cpu().detach().numpy()

    if len(image.shape) == 4:
        image = image[0, :, :, :]

    for channel in range(config.IN_CHANNELS):
        image[channel] = image[channel] * config.DATASET_STD[channel] + config.DATASET_MEAN[channel]
        image[channel] *= 255

    return image.astype('uint8')
