import os
import cv2
import sys
import torch
import random
import numpy as np

from transforms import Transforms
from datetime import datetime


def save_checkpoint(checkpoint, filepath):
    if not filepath.endswith(".pth.tar"):
        raise ValueError("Filepath should end with .pth.tar extension")

    try:
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        raise


def load_checkpoint(filepath, device):
    if not filepath.endswith(".pth.tar"):
        raise ValueError("Filepath should end with .pth.tar extension")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    try:
        checkpoint = torch.load(filepath, map_location=device)
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def get_last_checkpoint(checkpoint_dir):
    try:
        checkpoints = os.listdir(checkpoint_dir)
        checkpoints = [c for c in checkpoints if c.endswith(".pth.tar")]
        checkpoints.sort()

        return os.path.join(checkpoint_dir, checkpoints[-1])
    except IndexError:
        print(f"Error: there are no saved checkpoints in the {checkpoint_dir} directory")
        raise


def make_directory(folder_path):
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print(f"Error: directory \"{folder_path}\" already exists")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def model_test(checkpoint, config, device, img_dir="test_images"):
    gen_E = checkpoint["models"]["gen_European"]
    gen_A = checkpoint["models"]["gen_Asian"]
    transforms = Transforms(config.image_size, config.dataset_mean, config.dataset_std)

    # We upload and prepare images:
    images = [img for img in os.listdir(img_dir) if img.endswith(".png") or img.endswith(".jpg")]  # Names
    images = [cv2.imread(os.path.join(img_dir, img), 1) for img in images]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    images = [transforms.test_transforms(image=img)["image"] for img in images]  # Transforms
    images = [img.to(device) for img in images]

    # Generating images:
    pred_E = [postprocessing(gen_E(img.detach()), config) for img in images]
    pred_A = [postprocessing(gen_A(img.detach()), config) for img in images]

    images = [postprocessing(img.detach(), config) for img in images]

    # Putting everything together:
    images = np.concatenate(images, axis=2)
    pred_E = np.concatenate(pred_E, axis=2)
    pred_A = np.concatenate(pred_A, axis=2)

    return np.concatenate((images, pred_E, pred_A), axis=1)


def postprocessing(tensor, config):
    image = tensor.cpu().detach().numpy()

    if len(image.shape) == 4:
        image = image[0, :, :, :]

    for channel in range(config.in_channels):
        image[channel] = image[channel] * config.dataset_std[channel] + config.dataset_mean[channel]
        image[channel] *= 255

    return image.astype('uint8')


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set as {seed}")
