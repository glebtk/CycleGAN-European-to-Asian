import time
import config
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import EuropeanAsianDataset
from discriminator import Discriminator
from generator import Generator
from tqdm import tqdm
from utils import *


def train():
    # Initializing models
    gen_European = Generator(in_channels=config.IN_CHANNELS, num_residuals=config.NUM_RESIDUALS).to(config.DEVICE)
    gen_Asian = Generator(in_channels=config.IN_CHANNELS, num_residuals=config.NUM_RESIDUALS).to(config.DEVICE)
    disc_European = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    disc_Asian = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)

    # Initialize optimizers
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

    # Loading the dataset
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

    # Loading the latest checkpoint models
    if config.LOAD_MODEL:
        print("\033[32m{}".format("=> Loading the last checkpoint"))
        load_checkpoint(gen_European, opt_gen, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_GEN_EUROPEAN))
        load_checkpoint(gen_Asian, opt_gen, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_GEN_ASIAN))
        load_checkpoint(disc_European, opt_disc, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_DISC_EUROPEAN))
        load_checkpoint(disc_Asian, opt_disc, config.LEARNING_RATE, get_last_checkpoint(config.CHECKPOINT_DISC_ASIAN))

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter()

    # ----- Training cycle ----- #
    for epoch in range(config.NUM_EPOCHS):

        loop = tqdm(data_loader)
        for idx, (real_european_image, real_asian_image) in enumerate(loop):
            real_european_image = real_european_image.to(config.DEVICE)
            real_asian_image = real_asian_image.to(config.DEVICE)

            # ---------- Training of discriminators: ---------- #
            # We generate a fake picture of a European face from an Asian face picture
            fake_european_image = gen_European(real_asian_image)

            # We get the discriminator's predictions on the real and fake picture
            real_disc_European_prediction = disc_European(real_european_image)
            fake_disc_European_prediction = disc_European(fake_european_image.detach())

            # Calculate the loss
            real_disc_European_loss = MSE(real_disc_European_prediction, torch.ones_like(real_disc_European_prediction))
            fake_disc_European_loss = MSE(fake_disc_European_prediction, torch.zeros_like(fake_disc_European_prediction))
            disc_European_loss = real_disc_European_loss + fake_disc_European_loss

            # From a picture of a European face, we generate a fake picture of an Asian face
            fake_asian_image = gen_Asian(real_european_image)

            # We get the discriminator's predictions on the real and fake picture
            real_disc_Asian_prediction = disc_Asian(real_asian_image)
            fake_disc_Asian_prediction = disc_Asian(fake_asian_image.detach())

            # Calculate the loss
            real_disc_Asian_loss = MSE(real_disc_Asian_prediction, torch.ones_like(real_disc_Asian_prediction))
            fake_disc_Asian_loss = MSE(fake_disc_Asian_prediction, torch.zeros_like(fake_disc_Asian_prediction))
            disc_Asian_loss = real_disc_Asian_loss + fake_disc_Asian_loss

            # Combining discriminator's loss
            D_loss = (disc_European_loss + disc_Asian_loss) / 2

            # Updating discriminator weights
            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # ---------- Generator training: ---------- #
            # Calculate the adversarial loss for both generators
            disc_European_pred = disc_European(fake_european_image)
            disc_Asian_pred = disc_Asian(fake_asian_image)
            gen_European_adversarial_loss = MSE(disc_European_pred, torch.ones_like(disc_European_pred))
            gen_Asian_adversarial_loss = MSE(disc_Asian_pred, torch.ones_like(disc_Asian_pred))

            # Calculate the cycle loss for both generators
            cycle_fake_European_image = gen_European(fake_asian_image)
            cycle_fake_Asian_image = gen_Asian(fake_european_image)
            gen_European_cycle_loss = L1(real_european_image, cycle_fake_European_image)
            gen_Asian_cycle_loss = L1(real_asian_image, cycle_fake_Asian_image)

            # Combining the loss. Cycle loss is multiplied by the increasing coefficient
            G_loss = (
                gen_European_adversarial_loss
                + gen_Asian_adversarial_loss
                + gen_European_cycle_loss * config.LAMBDA_CYCLE
                + gen_Asian_cycle_loss * config.LAMBDA_CYCLE
            )

            # Updating the weights of generators
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            # Updating tensorboard (current fake images)
            if config.USE_TENSORBOARD and idx % 128 == 0 and idx != 0:
                fake_asian_image = postprocessing(fake_asian_image)
                fake_european_image = postprocessing(fake_european_image)
                current_images = np.concatenate((fake_asian_image, fake_european_image), axis=2)
                writer.add_image(f"Current images", current_images, global_step=int(time.time()))

                writer.add_scalar("Gen European Adversarial Loss", gen_European_adversarial_loss.item(), global_step=int(time.time()))
                writer.add_scalar("Gen Asian Adversarial Loss", gen_Asian_adversarial_loss.item(), global_step=int(time.time()))
                writer.add_scalar("Gen European Cycle Loss", gen_European_cycle_loss.item(), global_step=int(time.time()))
                writer.add_scalar("Gen Asian Cycle Loss", gen_Asian_cycle_loss.item(), global_step=int(time.time()))
                writer.add_scalar("Disc European Loss", disc_European_loss.item(), global_step=int(time.time()))
                writer.add_scalar("Disc Asian Loss", disc_Asian_loss.item(), global_step=int(time.time()))

        # Saving models
        if config.SAVE_MODEL:
            print("\033[32m{}".format("=> Saving a checkpoint"))

            # Creating a directory to save
            save_dir = os.path.join(config.CHECKPOINT_DIR, get_current_time())
            make_directory(save_dir)

            # Save
            save_checkpoint(gen_European, opt_gen, os.path.join(save_dir, config.CHECKPOINT_GEN_EUROPEAN))
            save_checkpoint(gen_Asian, opt_gen, os.path.join(save_dir, config.CHECKPOINT_GEN_ASIAN))
            save_checkpoint(disc_European, opt_disc, os.path.join(save_dir, config.CHECKPOINT_DISC_EUROPEAN))
            save_checkpoint(disc_Asian, opt_disc, os.path.join(save_dir, config.CHECKPOINT_DISC_ASIAN))

        # Updating tensorboard (test images)
        if config.USE_TENSORBOARD:
            writer.add_image("Generated images", model_test(gen_European, gen_Asian), global_step=epoch)


if __name__ == "__main__":
    train()
