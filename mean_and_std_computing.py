import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ABDataset
from tqdm import tqdm
import config


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data in tqdm(loader):
        data = torch.mean(data[0], dim=[0, 2, 3])
        data += torch.mean(data[1], dim=[0, 2, 3])
        data /= 2

        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean_ds = channels_sum / num_batches
    std_ds = (channels_sqrd_sum / num_batches - mean_ds ** 2) ** 0.5

    return mean_ds, std_ds


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = ABDataset(
        root_a=config.TRAIN_DIR + "/B1",
        root_b=config.TRAIN_DIR + "/B2",
        transform=config.mean_transforms,
    )
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False)

    mean, std = get_mean_std(train_loader)
    print(mean)
    print(std)


if __name__ == "__main__":
    test()


# import config
# import torch
#
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from dataset import ABDataset
#
#
# def main():
#     dataset = ABDataset(
#         root_a=config.TRAIN_DIR + "/class_A",
#         root_b=config.TRAIN_DIR + "/class_B",
#         transform=config.mean_transforms,
#     )
#
#     data_loader = DataLoader(
#         dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=True
#     )
#
#     # placeholders
#     psum = torch.tensor([0.0, 0.0, 0.0])
#     psum_sq = torch.tensor([0.0, 0.0, 0.0])
#
#     # loop through images
#     for inputs in tqdm(data_loader):
#         x = (inputs[0] + inputs[1]) / 2
#         psum += x.sum(axis=[0, 2, 3])
#         psum_sq += (x ** 2).sum(axis=[0, 2, 3])
#
#     print(psum)
#     print(psum_sq)
#
#     print(len(data_loader))
#     count = len(data_loader) * config.IMAGE_SIZE * config.IMAGE_SIZE
#
#     # mean and std
#     total_mean = psum / count
#     total_var = (psum_sq / count) - (total_mean ** 2)
#     total_std = torch.sqrt(total_var)
#
#     # output
#     print('mean: ' + str(total_mean))
#     print('std:  ' + str(total_std))
#
#
# if __name__ == "__main__":
#     main()
