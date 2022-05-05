import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
BATCH_SIZE = 1
IN_CHANNELS = 3
LEARNING_RATE = 3e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 2
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
TEST_EVERY_EPOCH = False
CHECKPOINT_GEN_A = "gen_a.pth.tar"
CHECKPOINT_GEN_B = "gen_b.pth.tar"
CHECKPOINT_DISC_A = "disc_a.pth.tar"
CHECKPOINT_DISC_B = "disc_b.pth.tar"

DATASET_MEAN = 0.4491
DATASET_STD = 0.1909

# DATASET_MEAN = torch.tensor([0.5298, 0.4365, 0.3811])
# DATASET_STD = torch.tensor([0.2104, 0.1828, 0.1795])

# For training:
train_transforms = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=1.0, limit=(-5, 5)),
        A.ElasticTransform(p=1.0, alpha=1.0, sigma=50.0, alpha_affine=5.0),
        A.ISONoise(p=1.0, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
        A.RandomContrast(p=1.0, limit=(-0.1, 0.1)),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)

# For test:
test_transforms = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=255),
        # A.Normalize(mean=[0.5298, 0.4365, 0.3811], std=[0.2104, 0.1828, 0.1795], max_pixel_value=255),
        ToTensorV2(),
     ],
)


