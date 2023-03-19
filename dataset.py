import os
import cv2
import torch

from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


class EuropeanAsianDataset(Dataset):
    def __init__(self, root_european, root_asian, transform=None):
        self.root_european = root_european
        self.root_asian = root_asian
        self.transform = transform

        # Getting lists of image names of both classes
        self.european_names = [name for name in os.listdir(root_european) if name.endswith(".jpg")]
        self.asian_names = [name for name in os.listdir(root_asian) if name.endswith(".jpg")]

        # Find the number of images of each class
        self.european_len = len(self.european_names)
        self.asian_len = len(self.asian_names)

        # We determine the length of the dataset
        self.dataset_len = max(self.european_len, self.asian_len)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        # Getting the names of the images
        european_name = self.european_names[index % self.european_len]
        asian_name = self.asian_names[index % self.asian_len]

        # Getting full paths to images
        european_path = os.path.join(self.root_european, european_name)
        asian_path = os.path.join(self.root_asian, asian_name)

        # Opening images
        european_image = cv2.imread(european_path, 1)
        european_image = cv2.cvtColor(european_image, cv2.COLOR_BGR2RGB)
        asian_image = cv2.imread(asian_path, 1)
        asian_image = cv2.cvtColor(asian_image, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        augmentations = self.transform(image=european_image, image0=asian_image)

        european_image = augmentations["image"]
        asian_image = augmentations["image0"]

        return european_image, asian_image
