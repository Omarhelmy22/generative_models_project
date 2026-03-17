import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def denormalize(images):
    """Convert images from [-1, 1] to [0, 1] for display."""
    return (images.clamp(-1, 1) + 1) / 2


def save_image_grid(images, path, nrow=8, title=None):
    """Save a batch of images as a grid."""
    images = denormalize(images)
    grid = vutils.make_grid(images.cpu(), nrow=nrow, padding=2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_reconstruction_comparison(originals, reconstructions, path):
    """Save original vs reconstructed images side by side."""
    n = min(originals.size(0), 8)
    originals = denormalize(originals[:n]).cpu()
    reconstructions = denormalize(reconstructions[:n]).cpu()
    interleaved = torch.stack([originals, reconstructions], dim=1).reshape(-1, *originals.shape[1:])

    grid = vutils.make_grid(interleaved, nrow=n, padding=2)
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title("Top: Original  |  Bottom: Reconstructed", fontsize=14)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_comparison_grid(real, vae_samples, ddpm_samples, path):
    """Save a three-row comparison: Real, VAE, DDPM."""
    n = min(8, real.size(0), vae_samples.size(0), ddpm_samples.size(0))
    rows = [denormalize(img[:n]).cpu() for img in [real, vae_samples, ddpm_samples]]
    all_images = torch.cat(rows, dim=0)

    grid = vutils.make_grid(all_images, nrow=n, padding=2)
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title("Row 1: Real  |  Row 2: VAE  |  Row 3: DDPM", fontsize=14)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_training_curves(losses, path, title="Training Loss"):
    """Plot and save training loss curve."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(losses, linewidth=1.5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
