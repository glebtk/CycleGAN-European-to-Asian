import yaml
import argparse
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import EuropeanAsianDataset
from discriminator import Discriminator
from generator import Generator
from transforms import Transforms
from tqdm import tqdm
from utils import *


def train_one_epoch(checkpoint, data_loader, device, writer, config):
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    loop = tqdm(data_loader)
    for idx, (real_european_image, real_asian_image) in enumerate(loop):
        global_step = (checkpoint["epoch"] * len(data_loader) + idx) * len(real_european_image)

        real_european_image = real_european_image.to(device)
        real_asian_image = real_asian_image.to(device)

        # ---------- Training of discriminators: ---------- #
        # We generate a fake picture of a European face from an Asian face picture
        fake_european_image = checkpoint["models"]["gen_European"](real_asian_image)

        # We get the discriminator's predictions on the real and fake picture
        real_disc_European_prediction = checkpoint["models"]["disc_European"](real_european_image)
        fake_disc_European_prediction = checkpoint["models"]["disc_European"](fake_european_image.detach())

        # Calculate the loss
        real_disc_European_loss = MSE(real_disc_European_prediction, torch.ones_like(real_disc_European_prediction))
        fake_disc_European_loss = MSE(fake_disc_European_prediction, torch.zeros_like(fake_disc_European_prediction))
        disc_European_loss = real_disc_European_loss + fake_disc_European_loss

        # From a picture of a European face, we generate a fake picture of an Asian face
        fake_asian_image = checkpoint["models"]["gen_Asian"](real_european_image)

        # We get the discriminator's predictions on the real and fake picture
        real_disc_Asian_prediction = checkpoint["models"]["disc_Asian"](real_asian_image)
        fake_disc_Asian_prediction = checkpoint["models"]["disc_Asian"](fake_asian_image.detach())

        # Calculate the loss
        real_disc_Asian_loss = MSE(real_disc_Asian_prediction, torch.ones_like(real_disc_Asian_prediction))
        fake_disc_Asian_loss = MSE(fake_disc_Asian_prediction, torch.zeros_like(fake_disc_Asian_prediction))
        disc_Asian_loss = real_disc_Asian_loss + fake_disc_Asian_loss

        # Combining discriminator's loss
        D_loss = (disc_European_loss + disc_Asian_loss) / 2

        # Updating discriminator weights
        checkpoint["optimizers"]["opt_disc"].zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(checkpoint["optimizers"]["opt_disc"])
        d_scaler.update()

        # ---------- Generator training: ---------- #
        # Calculate the adversarial loss for both generators
        disc_European_pred = checkpoint["models"]["disc_European"](fake_european_image)
        disc_Asian_pred = checkpoint["models"]["disc_Asian"](fake_asian_image)
        gen_European_adversarial_loss = MSE(disc_European_pred, torch.ones_like(disc_European_pred))
        gen_Asian_adversarial_loss = MSE(disc_Asian_pred, torch.ones_like(disc_Asian_pred))

        # Calculate the cycle loss for both generators
        cycle_fake_European_image = checkpoint["models"]["gen_European"](fake_asian_image)
        cycle_fake_Asian_image = checkpoint["models"]["gen_Asian"](fake_european_image)
        gen_European_cycle_loss = L1(real_european_image, cycle_fake_European_image)
        gen_Asian_cycle_loss = L1(real_asian_image, cycle_fake_Asian_image)

        # Combining the loss. Cycle loss is multiplied by the increasing coefficient
        G_loss = (
            gen_European_adversarial_loss
            + gen_Asian_adversarial_loss
            + gen_European_cycle_loss * config.lambda_cycle
            + gen_Asian_cycle_loss * config.lambda_cycle
        )

        # Updating the weights of generators
        checkpoint["optimizers"]["opt_gen"].zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(checkpoint["optimizers"]["opt_gen"])
        g_scaler.update()

        # Updating tensorboard (current fake images)
        if idx % 64 == 0 and idx != 0:
            fake_asian_image = postprocessing(fake_asian_image, config)
            fake_european_image = postprocessing(fake_european_image, config)
            current_images = np.concatenate((fake_asian_image, fake_european_image), axis=2)
            writer.add_image(f"Current images", current_images, global_step=global_step)

            writer.add_scalar("Disc European Loss", disc_European_loss.item(), global_step=global_step)
            writer.add_scalar("Disc Asian Loss", disc_Asian_loss.item(), global_step=global_step)
            writer.add_scalar("Gen European Adversarial Loss", gen_European_adversarial_loss.item(), global_step=global_step)
            writer.add_scalar("Gen Asian Adversarial Loss", gen_Asian_adversarial_loss.item(), global_step=global_step)
            writer.add_scalar("Gen European Cycle Loss", gen_European_cycle_loss.item(), global_step=global_step)
            writer.add_scalar("Gen Asian Cycle Loss", gen_Asian_cycle_loss.item(), global_step=global_step)

    checkpoint["epoch"] += 1


def train(checkpoint, data_loader, device, config):
    train_name = get_current_time()
    writer = SummaryWriter(f"tb/train_{train_name}")

    for epoch in range(checkpoint["epoch"], config.num_epochs):

        train_one_epoch(checkpoint, data_loader, device, writer, config)

        # Save checkpoint
        if config.save_checkpoint:
            print("\033[32m{}".format("=> Saving a checkpoint"))
            save_checkpoint(checkpoint, os.path.join(config.checkpoint_dir, f"checkpoint_{train_name}.pth.tar"))

        # Updating tensorboard (test images)
        writer.add_image("Generated images", model_test(checkpoint, config, device), global_step=epoch)


def get_config():
    # Load default configuration from YAML file
    with open("config.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    # Create argparse parser and add arguments
    parser = argparse.ArgumentParser(description="Train a CycleGAN model")

    for key, value in default_config.items():
        if isinstance(value, (int, float, str, bool)):
            parser.add_argument(f"--{key}", type=type(value), default=value)

    parser.set_defaults(**default_config)

    return parser.parse_args()


def main():
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading the dataset
    transforms = Transforms(config.image_size, config.dataset_mean, config.dataset_std)
    dataset = EuropeanAsianDataset(
        root_european=config.train_dir + "/European",
        root_asian=config.train_dir + "/Asian",
        transform=transforms.train_transforms,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Loading the latest checkpoint models
    if config.load_checkpoint:
        print("\033[32m{}".format("=> Loading the last checkpoint"))

        checkpoint_path = get_last_checkpoint(config.checkpoint_dir)
        checkpoint = load_checkpoint(checkpoint_path, device)
    else:
        models = {
            "gen_European": Generator(in_channels=config.in_channels, num_residuals=config.num_residuals).to(device),
            "gen_Asian": Generator(in_channels=config.in_channels, num_residuals=config.num_residuals).to(device),
            "disc_European": Discriminator(in_channels=config.in_channels).to(device),
            "disc_Asian": Discriminator(in_channels=config.in_channels).to(device)
        }

        optimizers = {
            "opt_gen": optim.Adam(
                params=list(models["gen_European"].parameters()) + list(models["gen_Asian"].parameters()),
                lr=config.generator_learning_rate,
                betas=(0.5, 0.999),
            ),
            "opt_disc": optim.Adam(
                params=list(models["disc_European"].parameters()) + list(models["disc_Asian"].parameters()),
                lr=config.discriminator_learning_rate,
                betas=(0.5, 0.999),
            )
        }

        checkpoint = {
            "models": models,
            "optimizers": optimizers,
            "epoch": 0
        }

    train(checkpoint, data_loader, device, config)


if __name__ == "__main__":
    main()
