import config
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ABDataset


def compute_stats():
    pass


def main():
    dataset = ABDataset(
        root_a=config.TRAIN_DIR + "/class_A",
        root_b=config.TRAIN_DIR + "/class_B",
        transform=config.transforms,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(data_loader):
        x = (inputs[0] + inputs[1]) / 2
        psum += x.sum(axis=[0, 2, 3])
        psum_sq += (x ** 2).sum(axis=[0, 2, 3])

    print(psum)
    print(psum_sq)

    print(len(data_loader))
    count = len(data_loader) * config.IMAGE_SIZE * config.IMAGE_SIZE

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))



if __name__ == "__main__":
    main()
