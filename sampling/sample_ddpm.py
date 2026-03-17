"""
Generate images using a trained DDPM model (with EMA weights).

Usage:
    python sampling/sample_ddpm.py
    python sampling/sample_ddpm.py --checkpoint checkpoints/ddpm/ddpm_final.pt --num_samples 10000
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

from models.ddpm.unet import UNet
from models.ddpm.diffusion import GaussianDiffusion
from utils.ema import EMA
from utils.visualization import denormalize, save_image_grid


def main():
    parser = argparse.ArgumentParser(description="Sample from trained DDPM")
    parser.add_argument("--config", type=str, default="configs/ddpm_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ddpm/ddpm_final.pt")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results/ddpm_samples/generated")
    parser.add_argument("--save_grid", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNet(
        in_channels=config["in_channels"],
        base_channels=config["base_channels"],
        channel_mult=tuple(config["channel_mult"]),
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=tuple(config["attention_resolutions"]),
        time_emb_dim=config["time_emb_dim"],
        dropout=0.0,
    ).to(device)

    # Load checkpoint and apply EMA weights
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    if "ema_state_dict" in state:
        ema = EMA(model, decay=config["ema_decay"])
        ema.load_state_dict(state["ema_state_dict"])
        ema.apply_shadow(model)
        print("Using EMA weights for sampling.")
    else:
        print("No EMA weights found; using raw model weights.")

    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    diffusion = GaussianDiffusion(
        timesteps=config["timesteps"],
        beta_schedule=config["beta_schedule"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    batch_size = min(args.num_samples, 64)
    generated = 0
    shape_tail = (config["in_channels"], config["image_size"], config["image_size"])

    print(f"Generating {args.num_samples} images...")
    while generated < args.num_samples:
        n = min(batch_size, args.num_samples - generated)
        samples = diffusion.sample(model, (n, *shape_tail), progress=True)
        samples = denormalize(samples)

        for i in range(n):
            save_image(
                samples[i],
                os.path.join(args.output_dir, f"{generated + i:05d}.png"),
            )
        generated += n
        print(f"  {generated}/{args.num_samples} images generated.")

    print(f"Saved {generated} images to {args.output_dir}")

    if args.save_grid:
        grid_samples = diffusion.sample(
            model, (min(64, args.num_samples), *shape_tail), progress=True,
        )
        save_image_grid(
            grid_samples,
            "results/ddpm_samples/final_grid.png",
            title="DDPM Generated Samples",
        )
        print("Saved sample grid to results/ddpm_samples/final_grid.png")


if __name__ == "__main__":
    main()
