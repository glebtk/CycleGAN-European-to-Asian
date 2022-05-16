import torch
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2


# Предустановки
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

IMAGE_SIZE = 256
IN_CHANNELS = 3
NUM_RESIDUALS = 8

# Обучение
NUM_EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 1e-5

LAMBDA_CYCLE = 10

LOAD_MODEL = True
SAVE_MODEL = True
USE_TENSORBOARD = True

# Датасет
TRAIN_DIR = "dataset/train"
CHECKPOINT_DIR = "checkpoints"

DATASET_MEAN = np.array([0.5298, 0.4365, 0.3811])
DATASET_STD = np.array([0.2654, 0.2402, 0.2382])

# Другое
CHECKPOINT_GEN_EUROPEAN = "gen_european.pth.tar"
CHECKPOINT_GEN_ASIAN = "gen_asian.pth.tar"
CHECKPOINT_DISC_EUROPEAN = "disc_european.pth.tar"
CHECKPOINT_DISC_ASIAN = "disc_asian.pth.tar"

train_transforms = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, always_apply=True),
        A.ISONoise(p=0.15, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.025, hue=0.02, always_apply=True),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=255.0),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)

test_transforms = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=255),
        ToTensorV2(),
     ],
)
