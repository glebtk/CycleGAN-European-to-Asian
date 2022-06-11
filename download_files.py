import os
import zipfile
import urllib.request


# Предустановки
LOAD_DATASET = True         # Загрузить датасет для обучения
LOAD_CHECKPOINT = True      # Загрузить чекпоинт с предобученными весами
LOAD_TEST_IMAGES = True     # Загрузить тестовые картинки


def download_and_unzip(url, path, name):
    full_path = os.path.join(path, name)

    urllib.request.urlretrieve(url, full_path)

    dataset_zip = zipfile.ZipFile(full_path, 'r')
    dataset_zip.extractall(path)


# Выполнение
if __name__ == "__main__":
    if LOAD_DATASET:
        url = "https://gitlab.com/glebtutik/european_to_asian_files/-/raw/main/dataset/dataset.zip"
        path = "dataset"
        file_name = "dataset.zip"

        download_and_unzip(url, path, file_name)
        os.remove(os.path.join(path, file_name))

        print("=> Датасет загружен!")

    if LOAD_CHECKPOINT:
        url = "https://gitlab.com/glebtutik/european_to_asian_files/-/raw/main/checkpoints/checkpoints.zip"
        path = "checkpoints"
        file_name = "checkpoints.zip"

        download_and_unzip(url, path, file_name)
        os.remove(os.path.join(path, file_name))

        print("=> Чекпоинт загружен!")

    if LOAD_TEST_IMAGES:
        url = "https://gitlab.com/glebtutik/european_to_asian_files/-/raw/main/test_images/test_images.zip"
        path = "test_images"
        file_name = "test_images.zip"

        download_and_unzip(url, path, file_name)
        os.remove(os.path.join(path, file_name))

        print("=> Тестовые изображения загружены!")
