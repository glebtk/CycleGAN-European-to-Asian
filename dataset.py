import config
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class EuropeanAsianDataset(Dataset):
    def __init__(self, root_european, root_asian, transform=None):
        self.root_european = root_european
        self.root_asian = root_asian
        self.transform = transform

        # Получаем списки имен изображений обоих классов
        self.european_names_list = os.listdir(root_european)
        self.asian_names_list = os.listdir(root_asian)

        # Находим количество изображений каждого класса
        self.european_list_len = len(self.european_names_list)
        self.asian_list_len = len(self.asian_names_list)

        # Определяем условную длину датасета
        self.dataset_length = max(self.european_list_len, self.asian_list_len)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        european_index = index
        asian_index = index

        # Если количество изображений разное,
        if self.european_list_len != self.asian_list_len:
            european_index = self.european_names_list[index % self.dataset_length]
            asian_index = self.asian_names_list[index % self.dataset_length]

        # Получаем имена изображений
        european_name = self.european_names_list[european_index]
        asian_name = self.asian_names_list[asian_index]

        # Получаем полные пути к изображениям
        european_path = os.path.join(self.root_european, european_name)
        asian_path = os.path.join(self.root_asian, asian_name)

        # Получаем изображения и конвертируем в np.array
        european_image = np.array(Image.open(european_path).convert("RGB"))
        asian_image = np.array(Image.open(asian_path).convert("RGB"))

        # Применяем аугментации
        augmentations = self.transform(image=european_image, image0=asian_image)

        european_image = augmentations["image"]
        asian_image = augmentations["image0"]

        return european_image, asian_image


def test():
    dataset = EuropeanAsianDataset(
        root_european=config.TRAIN_DIR + "/European",
        root_asian=config.TRAIN_DIR + "/Asian",
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
