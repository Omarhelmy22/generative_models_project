"""
Train a Denoising Diffusion Probabilistic Model (DDPM) on CIFAR-10.

Uses step-based training, EMA model weights, and optional mixed-precision.
Supports resuming from a checkpoint.

Usage:
    python training/train_ddpm.py
    python training/train_ddpm.py --config configs/ddpm_config.yaml
    python training/train_ddpm.py --resume checkpoints/ddpm/ddpm_step_0010000.pt
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

from models.ddpm.unet import UNet
from models.ddpm.diffusion import GaussianDiffusion
from utils.dataset import get_dataloader, infinite_dataloader
from utils.visualization import save_image_grid, plot_training_curves
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.ema import EMA


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config, resume_path=None):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataloader = get_dataloader(
        dataset_name=config["dataset"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
    )
    data_iter = infinite_dataloader(dataloader)

    # Model
    model = UNet(
        in_channels=config["in_channels"],
        base_channels=config["base_channels"],
        channel_mult=tuple(config["channel_mult"]),
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=tuple(config["attention_resolutions"]),
        time_emb_dim=config["time_emb_dim"],
        dropout=config.get("dropout", 0.0),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet parameters: {param_count:,}")

    # Diffusion
    diffusion = GaussianDiffusion(
        timesteps=config["timesteps"],
        beta_schedule=config["beta_schedule"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    ema = EMA(model, decay=config["ema_decay"])

    # Resume from checkpoint
    start_step = 0
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        state = load_checkpoint(resume_path, model, optimizer, ema, device=device)
        start_step = state.get("step", 0)
        print(f"Resumed at step {start_step}")

    # Mixed precision
    use_amp = config.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"Mixed precision: {use_amp}")

    # Training loop
    losses = []
    log_every = 100
    start_time = time.time()
    total_steps = config["training_steps"]

    pbar = tqdm(range(start_step + 1, total_steps + 1), desc="Training DDPM",
                initial=start_step, total=total_steps)
    for step in pbar:
        model.train()
        images, _ = next(data_iter)
        images = images.to(device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss = diffusion.training_loss(model, images)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = diffusion.training_loss(model, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ema.update(model)
        losses.append(loss.item())

        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            pbar.set_postfix(loss=f"{avg:.4f}")

        # Save sample images using EMA weights
        if step % config["sample_every"] == 0:
            model.eval()
            ema.apply_shadow(model)
            samples = diffusion.sample(
                model,
                (config["num_samples"], config["in_channels"],
                 config["image_size"], config["image_size"]),
                progress=False,
            )
            ema.restore(model)
            save_image_grid(
                samples,
                f"results/ddpm_samples/samples_step_{step:07d}.png",
                title=f"DDPM Samples - Step {step}",
            )

        # Save checkpoint
        if step % config["save_every"] == 0:
            save_checkpoint(
                f"checkpoints/ddpm/ddpm_step_{step:07d}.pt",
                model, optimizer, step=step, ema=ema,
            )

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes.")

    # Save final model and loss curve
    save_checkpoint(
        "checkpoints/ddpm/ddpm_final.pt",
        model, optimizer, step=total_steps, ema=ema,
    )
    plot_training_curves(losses, "results/ddpm_samples/loss_curve.png", title="DDPM Training Loss")
    print("Final model and loss curve saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument("--config", type=str, default="configs/ddpm_config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config, resume_path=args.resume)
