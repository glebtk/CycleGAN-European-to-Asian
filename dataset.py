import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import config


class ABDataset(Dataset):
    def __init__(self, root_a, root_b, transform=None):
        self.root_a = root_a
        self.root_b = root_b
        self.transform = transform

        # Получаем списки имен изображений обоих классов
        self.a_names_list = os.listdir(root_a)
        self.b_names_list = os.listdir(root_b)

        # Находим количество изображений каждого класса
        self.a_list_len = len(self.a_names_list)
        self.b_list_len = len(self.b_names_list)

        # Определяем условную длину датасета
        self.dataset_length = max(self.a_list_len, self.b_list_len)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        a_index = index
        b_index = index

        # Если количество изображений A и B разное,
        if self.a_list_len != self.b_list_len:
            a_index = self.a_names_list[index % self.dataset_length]
            b_index = self.b_names_list[index % self.dataset_length]

        # Получаем имена изображений
        a_name = self.a_names_list[a_index]
        b_name = self.b_names_list[b_index]

        # Получаем полные пути к изображениям
        a_path = os.path.join(self.root_a, a_name)
        b_path = os.path.join(self.root_b, b_name)

        # Получаем изображения и конвертируем в np.array
        a_image = np.array(Image.open(a_path).convert("RGB"))
        b_image = np.array(Image.open(b_path).convert("RGB"))

        # Если нужно, применяем аугментации
        if self.transform:
            augmentations = self.transform(image=a_image, image0=b_image)

            a_image = augmentations["image"]
            b_image = augmentations["image0"]

        return a_image, b_image


def test():
    dataset = ABDataset(
        root_a=config.TRAIN_DIR + "/class_A",
        root_b=config.TRAIN_DIR + "/class_B",
        transform=config.train_transforms
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )


if __name__ == "__main__":
    test()
