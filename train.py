import config
import model_test
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from dataset import EuropeanAsianDataset
from discriminator import Discriminator
from generator import Generator
from utils import *


def train(gen_European, gen_Asian, disc_European, disc_Asian, data_loader, opt_gen, opt_disc, L1, MSE, d_scaler, g_scaler):
    loop = tqdm(data_loader, leave=True)

    for idx, (european_image, asian_image) in enumerate(loop):
        european_image = european_image.to(config.DEVICE)
        asian_image = asian_image.to(config.DEVICE)

        # ---------- Обучаем дискриминаторы: ---------- #
        # Из картинки азиатского лица генерируем фейковую картинку европейского лица
        fake_european_image = gen_European(asian_image)

        # Получаем предсказания дискриминатора на реальной и на фейковой картинке
        real_disc_European_prediction = disc_European(european_image)
        fake_disc_European_prediction = disc_European(fake_european_image.detach())

        # Вычисляем и суммируем loss
        real_disc_European_loss = MSE(real_disc_European_prediction, torch.ones_like(real_disc_European_prediction))
        fake_disc_European_loss = MSE(fake_disc_European_prediction, torch.zeros_like(fake_disc_European_prediction))
        disc_European_loss = real_disc_European_loss + fake_disc_European_loss

        # Из картинки европейского лица генерируем фейковую картинку азиатского лица
        fake_asian_image = gen_Asian(european_image)

        # Получаем предсказания дискриминатора на реальной и на фейковой картинке
        real_disc_Asian_prediction = disc_Asian(asian_image)
        fake_disc_Asian_prediction = disc_Asian(fake_asian_image.detach())

        # Вычисляем и суммируем loss
        real_disc_Asian_loss = MSE(real_disc_Asian_prediction, torch.ones_like(real_disc_Asian_prediction))
        fake_disc_Asian_loss = MSE(fake_disc_Asian_prediction, torch.zeros_like(fake_disc_Asian_prediction))
        disc_Asian_loss = real_disc_Asian_loss + fake_disc_Asian_loss

        # Объединяем loss дискриминаторов
        D_loss = (disc_European_loss + disc_Asian_loss) / 2

        # Обновляем веса дискриминаторов
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # ---------- Обучаем генераторы: ---------- #
        # Вычисляем adversarial loss для обоих генераторов
        disc_European_pred = disc_European(fake_european_image)
        disc_Asian_pred = disc_Asian(fake_asian_image)
        gen_European_adversarial_loss = MSE(disc_European_pred, torch.ones_like(disc_European_pred))
        gen_Asian_adversarial_loss = MSE(disc_Asian_pred, torch.ones_like(disc_Asian_pred))

        # Вычисляем cycle loss для обоих генераторов
        cycle_fake_European = gen_European(fake_asian_image)
        cycle_fake_Asian = gen_Asian(fake_european_image)
        gen_European_cycle_loss = L1(european_image, cycle_fake_European)
        gen_Asian_cycle_loss = L1(asian_image, cycle_fake_Asian)

        # Объединяем loss
        G_loss = (
                gen_European_adversarial_loss
                + gen_Asian_adversarial_loss
                + gen_European_cycle_loss * config.LAMBDA_CYCLE
                + gen_Asian_cycle_loss * config.LAMBDA_CYCLE
        )

        # Обновляем веса генераторов
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 50 == 0:
            fake_E = Image.fromarray(tensor_to_array(fake_european_image))
            fake_A = Image.fromarray(tensor_to_array(fake_asian_image))
            fake_E.save(f"saved_images/fake_european/pic_{get_current_time()}.png")
            fake_A.save(f"saved_images/fake_asian/pic_{get_current_time()}.png")


def main():
    gen_European = Generator(in_channels=config.IN_CHANNELS, num_residuals=9).to(config.DEVICE)
    gen_Asian = Generator(in_channels=config.IN_CHANNELS, num_residuals=9).to(config.DEVICE)

    disc_European = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    disc_Asian = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)

    opt_gen = optim.Adam(
        params=list(gen_European.parameters()) + list(gen_Asian.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_disc = optim.Adam(
        params=list(disc_European.parameters()) + list(disc_Asian.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    if config.LOAD_MODEL:
        # Загружаем последний чекпоинт
        load_checkpoint(gen_European, opt_gen, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_GEN_EUROPEAN))
        load_checkpoint(gen_Asian, opt_gen, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_GEN_ASIAN))
        load_checkpoint(disc_European, opt_disc, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_DISC_EUROPEAN))
        load_checkpoint(disc_Asian, opt_disc, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_DISC_ASIAN))

    dataset = EuropeanAsianDataset(
        root_european=config.TRAIN_DIR + "/European",
        root_asian=config.TRAIN_DIR + "/Asian",
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
        train(gen_European, gen_Asian, disc_European, disc_Asian, data_loader, opt_gen, opt_disc, L1, MSE, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            # Создаем директорию для сохранения
            directory = os.path.join(config.CHECKPOINT_DIR, get_current_time())
            make_directory(directory)

            # Сохраняем модели
            save_checkpoint(gen_European, opt_gen, os.path.join(directory, config.CHECKPOINT_GEN_EUROPEAN))
            save_checkpoint(gen_Asian, opt_gen, os.path.join(directory, config.CHECKPOINT_GEN_ASIAN))
            save_checkpoint(disc_European, opt_disc, os.path.join(directory, config.CHECKPOINT_DISC_EUROPEAN))
            save_checkpoint(disc_Asian, opt_disc, os.path.join(directory, config.CHECKPOINT_DISC_ASIAN))

            # Если нужно, тестируем
            if config.TEST_EVERY_SAVE:
                model_test.test(save_dir=directory, name=f"{epoch+1}_epoch_test.png")


if __name__ == "__main__":
    main()
