import albumentations as A
from albumentations.pytorch import ToTensorV2


class Transforms:
    def __init__(self, image_size, mean, std):
        self.train_transforms = A.Compose(
            [
                A.Resize(width=image_size, height=image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, always_apply=True),
                A.ISONoise(p=0.15, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.025, hue=0.02, always_apply=True),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )

        self.test_transforms = A.Compose([
            A.Resize(width=image_size, height=image_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ])
