"""
Compute Fréchet Inception Distance (FID) between generated and real images.

Requires pre-generated images saved to disk.  Uses the pytorch-fid package.

Usage:
    python evaluation/compute_fid.py --real_dir evaluation/real_images --gen_dir results/vae_samples/generated
    python evaluation/compute_fid.py --real_dir evaluation/real_images --gen_dir results/ddpm_samples/generated
"""
import os
import sys
import argparse

import torch
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_real_images(output_dir, num_images=10000):
    """Save real CIFAR-10 images to disk for FID computation."""
    import torchvision

    os.makedirs(output_dir, exist_ok=True)
    existing = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    if len(existing) >= num_images:
        print(f"Real images already prepared ({len(existing)} found).")
        return

    print(f"Preparing {num_images} real CIFAR-10 images...")
    transform = T.Compose([T.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform,
    )

    for i in range(min(num_images, len(dataset))):
        img, _ = dataset[i]
        save_image(img, os.path.join(output_dir, f"{i:05d}.png"))

    print(f"Saved {min(num_images, len(dataset))} real images to {output_dir}")


def compute_fid(real_dir, gen_dir, batch_size=50, device=None):
    """Compute FID score using the pytorch-fid library."""
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("ERROR: pytorch-fid not installed. Run: pip install pytorch-fid")
        sys.exit(1)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dims = 2048  # InceptionV3 feature dimension
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir],
        batch_size=batch_size,
        device=device,
        dims=dims,
    )
    return fid_value


def main():
    parser = argparse.ArgumentParser(description="Compute FID score")
    parser.add_argument("--real_dir", type=str, default="evaluation/real_images")
    parser.add_argument("--gen_dir", type=str, required=False,
                        default="results/vae_samples/generated")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--prepare_real", action="store_true", default=True)
    parser.add_argument("--num_real", type=int, default=10000)
    args = parser.parse_args()

    if args.prepare_real:
        prepare_real_images(args.real_dir, args.num_real)

    if not os.path.isdir(args.gen_dir):
        print(f"ERROR: Generated images directory not found: {args.gen_dir}")
        print("Run sampling scripts first to generate images.")
        sys.exit(1)

    num_gen = len([f for f in os.listdir(args.gen_dir) if f.endswith(".png")])
    print(f"Found {num_gen} generated images in {args.gen_dir}")

    print("Computing FID score...")
    fid = compute_fid(args.real_dir, args.gen_dir, args.batch_size)
    print(f"\n{'='*40}")
    print(f"  FID Score: {fid:.2f}")
    print(f"{'='*40}")

    return fid


if __name__ == "__main__":
    main()
