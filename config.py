import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2


# Предустановки
IMAGE_SIZE = 256
IN_CHANNELS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

# Обучение
NUM_EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 3e-4

LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10

LOAD_MODEL = False
SAVE_MODEL = True
TEST_EVERY_EPOCH = False

# Датасет
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
CHECKPOINT_DIR = "checkpoints"

CHECKPOINT_GEN_EUROPEAN = "gen_european.pth.tar"
CHECKPOINT_GEN_ASIAN = "gen_asian.pth.tar"
CHECKPOINT_DISC_EUROPEAN = "disc_european.pth.tar"
CHECKPOINT_DISC_ASIAN = "disc_asian.pth.tar"

DATASET_MEAN = [0.5298, 0.4365, 0.3811]
DATASET_STD = [0.2654, 0.2402, 0.2382]


# For training:
train_transforms = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=1.0, limit=10),
        A.ISONoise(p=0.15, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
        A.RandomBrightnessContrast(p=1.0, brightness_limit=0.2, contrast_limit=0.2),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=255.0),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)

# For test:
test_transforms = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=255),
        ToTensorV2(),
     ],
)