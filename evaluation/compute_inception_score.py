"""
Compute Inception Score (IS) for generated images.

Uses torchvision's Inception v3 model. Images are resized to 299x299,
run through the classifier, and the IS is computed from the conditional
and marginal softmax distributions.

Usage:
    python evaluation/compute_inception_score.py --image_dir results/vae_samples/generated
    python evaluation/compute_inception_score.py --image_dir results/ddpm_samples/generated
"""
import os
import sys
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ImageFolderFlat(Dataset):
    """Load all .png images from a flat directory."""

    def __init__(self, image_dir, transform=None):
        self.paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir) if f.endswith(".png")
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def compute_inception_score(image_dir, batch_size=50, splits=10, device=None):
    """
    Compute Inception Score.

    IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )

    where p(y|x) is the conditional class distribution from InceptionV3
    and p(y) is the marginal distribution averaged over all generated images.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolderFlat(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Load pre-trained Inception v3
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    inception.eval()
    inception.to(device)

    all_preds = []

    print(f"Running {len(dataset)} images through Inception v3...")
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Inception Score"):
            images = images.to(device)
            logits = inception(images)
            # inception_v3 returns InceptionOutputs in eval mode; take the main output
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Compute IS with splits for mean/std
    scores = []
    N = all_preds.shape[0]
    split_size = N // splits

    for k in range(splits):
        part = all_preds[k * split_size: (k + 1) * split_size]
        p_y = np.mean(part, axis=0, keepdims=True)  # marginal
        kl_div = part * (np.log(part + 1e-16) - np.log(p_y + 1e-16))
        kl_div = np.sum(kl_div, axis=1)
        scores.append(np.exp(np.mean(kl_div)))

    return float(np.mean(scores)), float(np.std(scores))


def main():
    parser = argparse.ArgumentParser(description="Compute Inception Score")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--splits", type=int, default=10)
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"ERROR: Directory not found: {args.image_dir}")
        sys.exit(1)

    num_images = len([f for f in os.listdir(args.image_dir) if f.endswith(".png")])
    print(f"Found {num_images} images in {args.image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_is, std_is = compute_inception_score(
        args.image_dir, args.batch_size, args.splits, device,
    )

    print(f"\n{'='*40}")
    print(f"  Inception Score: {mean_is:.2f} ± {std_is:.2f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
