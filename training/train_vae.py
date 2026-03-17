"""
Train a Variational Autoencoder (VAE) on CIFAR-10.

Usage:
    python training/train_vae.py
    python training/train_vae.py --config configs/vae_config.yaml
"""
import os
import sys
import time
import argparse

import yaml
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae.vae import VAE
from models.vae.vae_loss import vae_loss
from utils.dataset import get_dataloader
from utils.visualization import (
    save_image_grid,
    save_reconstruction_comparison,
    plot_training_curves,
)
from utils.checkpoint import save_checkpoint


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataloader = get_dataloader(
        dataset_name=config["dataset"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
    )

    # Model
    model = VAE(
        in_channels=config["in_channels"],
        encoder_channels=config["encoder_channels"],
        latent_dim=config["latent_dim"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VAE parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training
    epoch_losses = []
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['epochs']}")
        for images, _ in pbar:
            images = images.to(device)
            recon, mu, logvar = model(images)

            loss, recon_l, kl_l = vae_loss(
                recon, images, mu, logvar,
                kl_weight=config["kl_weight"],
                epoch=epoch - 1,
                kl_warmup_epochs=config["kl_warmup_epochs"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_l.item()
            running_kl += kl_l.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}")

        n_batches = len(dataloader)
        avg_loss = running_loss / n_batches
        avg_recon = running_recon / n_batches
        avg_kl = running_kl / n_batches
        epoch_losses.append(avg_loss)

        effective_kl_w = config["kl_weight"]
        if config["kl_warmup_epochs"] > 0 and (epoch - 1) < config["kl_warmup_epochs"]:
            effective_kl_w = config["kl_weight"] * ((epoch - 1) / config["kl_warmup_epochs"])

        print(
            f"  Loss: {avg_loss:.4f}  Recon: {avg_recon:.4f}  "
            f"KL: {avg_kl:.4f}  KL-weight: {effective_kl_w:.3f}"
        )

        # Save sample images
        if epoch % config["sample_every"] == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                samples = model.sample(config["num_samples"], device)
                save_image_grid(
                    samples,
                    f"results/vae_samples/samples_epoch_{epoch:04d}.png",
                    title=f"VAE Samples - Epoch {epoch}",
                )
                # Reconstruction comparison
                test_images = next(iter(dataloader))[0][:8].to(device)
                recon_imgs, _, _ = model(test_images)
                save_reconstruction_comparison(
                    test_images, recon_imgs,
                    f"results/vae_samples/recon_epoch_{epoch:04d}.png",
                )

        # Save checkpoint
        if epoch % config["save_every"] == 0:
            save_checkpoint(
                f"checkpoints/vae/vae_epoch_{epoch:04d}.pt",
                model, optimizer, epoch=epoch,
            )

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes.")

    # Save final model and training curves
    save_checkpoint("checkpoints/vae/vae_final.pt", model, optimizer, epoch=config["epochs"])
    plot_training_curves(epoch_losses, "results/vae_samples/loss_curve.png", title="VAE Training Loss")
    print("Final model and loss curve saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--config", type=str, default="configs/vae_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)
