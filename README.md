# VAE vs DDPM: A Comparative Study of Deep Generative Models

From-scratch PyTorch implementations of a **Variational Autoencoder (VAE)** and a
**Denoising Diffusion Probabilistic Model (DDPM)**, trained and evaluated on CIFAR-10.

## Project Structure

```
generative_models_project/
├── configs/                    # YAML configuration files
│   ├── vae_config.yaml
│   └── ddpm_config.yaml
├── models/
│   ├── vae/                    # VAE architecture
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── vae.py
│   │   └── vae_loss.py
│   └── ddpm/                   # DDPM architecture
│       ├── unet.py
│       ├── diffusion.py
│       └── noise_schedule.py
├── training/                   # Training scripts
│   ├── train_vae.py
│   └── train_ddpm.py
├── sampling/                   # Image generation scripts
│   ├── sample_vae.py
│   └── sample_ddpm.py
├── evaluation/                 # Quantitative evaluation
│   ├── compute_fid.py
│   └── compute_inception_score.py
├── utils/                      # Shared utilities
│   ├── dataset.py
│   ├── visualization.py
│   ├── checkpoint.py
│   └── ema.py
├── results/                    # Generated outputs
│   ├── vae_samples/
│   ├── ddpm_samples/
│   └── comparisons/
├── checkpoints/                # Saved model weights
├── report/
│   └── report.md               # Full project report
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd generative_models_project

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The project uses **CIFAR-10** (32×32 colour images, 10 classes).  The dataset is
downloaded automatically on first training run.

## Training

### Train VAE

```bash
python training/train_vae.py
```

With custom config:

```bash
python training/train_vae.py --config configs/vae_config.yaml
```

Default: 100 epochs, batch size 128, Adam (lr=1e-3), KL warm-up over 10 epochs.

### Train DDPM

```bash
python training/train_ddpm.py
```

With custom config:

```bash
python training/train_ddpm.py --config configs/ddpm_config.yaml
```

Default: 150k steps, batch size 128, Adam (lr=1e-4), T=1000, linear beta schedule,
EMA decay 0.999, mixed precision (AMP) on CUDA.

## Sampling

### Generate VAE images

```bash
# Quick grid (64 images)
python sampling/sample_vae.py

# Full evaluation set (10k images)
python sampling/sample_vae.py --num_samples 10000 --output_dir results/vae_samples/generated
```

### Generate DDPM images

```bash
# Quick grid (64 images)
python sampling/sample_ddpm.py

# Full evaluation set (10k images)
python sampling/sample_ddpm.py --num_samples 10000 --output_dir results/ddpm_samples/generated
```

## Evaluation

### FID (Fréchet Inception Distance)

```bash
# VAE
python evaluation/compute_fid.py --gen_dir results/vae_samples/generated

# DDPM
python evaluation/compute_fid.py --gen_dir results/ddpm_samples/generated
```

The script automatically prepares 10k real CIFAR-10 images on first run.

### Inception Score

```bash
# VAE
python evaluation/compute_inception_score.py --image_dir results/vae_samples/generated

# DDPM
python evaluation/compute_inception_score.py --image_dir results/ddpm_samples/generated
```

## Configuration

All hyperparameters are in YAML config files under `configs/`.  Key parameters:

| Parameter | VAE | DDPM |
|-----------|-----|------|
| Learning rate | 1e-3 | 1e-4 |
| Batch size | 128 | 128 |
| Training duration | 100 epochs | 150k steps |
| Latent dim / Timesteps | 128 | 1000 |
| KL warm-up | 10 epochs | — |
| EMA decay | — | 0.999 |
| Mixed precision | — | Enabled |

## Reproducing Results

```bash
# 1. Train both models
python training/train_vae.py
python training/train_ddpm.py

# 2. Generate evaluation samples (10k each)
python sampling/sample_vae.py --num_samples 10000
python sampling/sample_ddpm.py --num_samples 10000

# 3. Compute metrics
python evaluation/compute_fid.py --gen_dir results/vae_samples/generated
python evaluation/compute_fid.py --gen_dir results/ddpm_samples/generated
python evaluation/compute_inception_score.py --image_dir results/vae_samples/generated
python evaluation/compute_inception_score.py --image_dir results/ddpm_samples/generated
```

## Report

The full project report with theoretical background, experimental results, and
analysis is available at [`report/report.md`](report/report.md).

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA GPU recommended (especially for DDPM training)
