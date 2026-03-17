import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, kl_weight=1.0, epoch=0,
             kl_warmup_epochs=0):
    """
    VAE loss = Reconstruction Loss + KL Divergence.

    Reconstruction loss: MSE between input and reconstruction.
    KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I).

    KL warmup linearly increases the KL weight from 0 to kl_weight over
    the first kl_warmup_epochs epochs to mitigate posterior collapse.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)

    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    effective_kl_weight = kl_weight
    if kl_warmup_epochs > 0 and epoch < kl_warmup_epochs:
        effective_kl_weight = kl_weight * (epoch / kl_warmup_epochs)

    total_loss = recon_loss + effective_kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss
