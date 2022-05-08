import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from dataset import ABDataset
from utils import *
from model_test import test


import config

import time


def train(gen_A, gen_B, disc_A, disc_B, data_loader, opt_gen, opt_disc, L1, MSE, d_scaler, g_scaler):
    loop = tqdm(data_loader, leave=True)

    for idx, (a_image, b_image) in enumerate(loop):
        a_image = a_image.to(config.DEVICE)
        b_image = b_image.to(config.DEVICE)

        # ---------- Обучаем дискриминаторы: ---------- #
        # Из картинки класса B генерируем фейковую картинку класса A
        fake_a_image = gen_A(b_image)

        # Получаем предсказания дискриминатора на реальной и на фейковой картинке
        real_disc_prediction = disc_A(a_image)
        fake_disc_prediction = disc_A(fake_a_image.detach())

        # Вычисляем и суммируем loss
        real_loss = MSE(real_disc_prediction, torch.ones_like(real_disc_prediction))
        fake_loss = MSE(fake_disc_prediction, torch.zeros_like(fake_disc_prediction))
        disc_A_loss = real_loss + fake_loss

        # Из картинки класса A генерируем фейковую картинку класса B
        fake_b_image = gen_B(a_image)

        # Получаем предсказания дискриминатора на реальной и на фейковой картинке
        real_disc_prediction = disc_B(b_image)
        fake_disc_prediction = disc_B(fake_b_image.detach())

        # Вычисляем и суммируем loss
        real_loss = MSE(real_disc_prediction, torch.ones_like(real_disc_prediction))
        fake_loss = MSE(fake_disc_prediction, torch.zeros_like(fake_disc_prediction))
        disc_B_loss = real_loss + fake_loss

        # Объединяем loss дискриминаторов
        D_loss = (disc_A_loss + disc_B_loss) / 2

        # Обновляем веса дискриминаторов
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # ---------- Обучаем генераторы: ---------- #
        # Вычисляем adversarial loss для обоих генераторов
        disc_A_pred = disc_A(fake_a_image)
        disc_B_pred = disc_B(fake_b_image)
        gen_A_adversarial_loss = MSE(disc_A_pred, torch.ones_like(disc_A_pred))
        gen_B_adversarial_loss = MSE(disc_B_pred, torch.ones_like(disc_B_pred))

        # Вычисляем cycle loss для обоих генераторов
        cycle_fake_A = gen_A(fake_b_image)
        cycle_fake_B = gen_B(fake_a_image)
        gen_A_cycle_loss = L1(a_image, cycle_fake_A)
        gen_B_cycle_loss = L1(b_image, cycle_fake_B)

        # Объединяем loss
        G_loss = (
                gen_A_adversarial_loss
                + gen_B_adversarial_loss
                + gen_A_cycle_loss * config.LAMBDA_CYCLE
                + gen_B_cycle_loss * config.LAMBDA_CYCLE
        )

        # Обновляем веса генераторов
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 50 == 0:
            save_image(fake_a_image * config.DATASET_STD + config.DATASET_MEAN, f"saved_images/fake_a/pic_{get_current_time()}.png")
            save_image(fake_b_image * config.DATASET_STD + config.DATASET_MEAN, f"saved_images/fake_b/pic_{get_current_time()}.png")


def main():
    gen_A = Generator(in_channels=config.IN_CHANNELS, num_residuals=9).to(config.DEVICE)
    gen_B = Generator(in_channels=config.IN_CHANNELS, num_residuals=9).to(config.DEVICE)

    disc_A = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    disc_B = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)

    opt_gen = optim.Adam(
        params=list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
    )

    opt_disc = optim.Adam(
        params=list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
    )

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    if config.LOAD_MODEL:
        # Загружаем последний чекпоинт
        load_checkpoint(gen_A, opt_gen, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_GEN_A))
        load_checkpoint(gen_B, opt_gen, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_GEN_B))
        load_checkpoint(disc_A, opt_disc, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_DISC_A))
        load_checkpoint(disc_B, opt_disc, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_DISC_B))

    dataset = ABDataset(
        root_a=config.TRAIN_DIR + "/class_A",
        root_b=config.TRAIN_DIR + "/class_B",
        transform=config.train_transforms,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train(gen_A, gen_B, disc_A, disc_B, data_loader, opt_gen, opt_disc, L1, MSE, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            # Создаем директорию для сохранения
            directory = os.path.join(config.CHECKPOINT_DIR, get_current_time())
            make_directory(directory)

            # Сохраняем модели
            save_checkpoint(gen_A, opt_gen, os.path.join(directory, config.CHECKPOINT_GEN_A))
            save_checkpoint(gen_B, opt_gen, os.path.join(directory, config.CHECKPOINT_GEN_B))
            save_checkpoint(disc_A, opt_disc, os.path.join(directory, config.CHECKPOINT_DISC_A))
            save_checkpoint(disc_B, opt_disc, os.path.join(directory, config.CHECKPOINT_DISC_B))

        # if config.TEST_EVERY_EPOCH:
        #     test(img_dir="test_images/", save_dir="saved_images/", name=f"test_{epoch}_epoch.png")


if __name__ == "__main__":
    main()
