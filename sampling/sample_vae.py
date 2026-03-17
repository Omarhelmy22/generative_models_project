"""
Generate images using a trained VAE model.

Usage:
    python sampling/sample_vae.py
    python sampling/sample_vae.py --checkpoint checkpoints/vae/vae_final.pt --num_samples 10000
"""
import os
import sys
import argparse

import yaml
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae.vae import VAE
from utils.visualization import denormalize, save_image_grid


def main():
    parser = argparse.ArgumentParser(description="Sample from trained VAE")
    parser.add_argument("--config", type=str, default="configs/vae_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae/vae_final.pt")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results/vae_samples/generated")
    parser.add_argument("--save_grid", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = VAE(
        in_channels=config["in_channels"],
        encoder_channels=config["encoder_channels"],
        latent_dim=config["latent_dim"],
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)
    batch_size = min(args.num_samples, 128)
    generated = 0

    print(f"Generating {args.num_samples} images...")
    with torch.no_grad():
        while generated < args.num_samples:
            n = min(batch_size, args.num_samples - generated)
            samples = model.sample(n, device)
            samples = denormalize(samples)

            for i in range(n):
                save_image(
                    samples[i],
                    os.path.join(args.output_dir, f"{generated + i:05d}.png"),
                )
            generated += n

    print(f"Saved {generated} images to {args.output_dir}")

    if args.save_grid:
        all_samples = model.sample(min(64, args.num_samples), device)
        save_image_grid(
            all_samples,
            "results/vae_samples/final_grid.png",
            title="VAE Generated Samples",
        )
        print("Saved sample grid to results/vae_samples/final_grid.png")


if __name__ == "__main__":
    main()
