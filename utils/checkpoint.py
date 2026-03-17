import os
import torch


def save_checkpoint(path, model, optimizer, step=None, epoch=None, ema=None,
                    extra=None):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if step is not None:
        state["step"] = step
    if epoch is not None:
        state["epoch"] = epoch
    if ema is not None:
        state["ema_state_dict"] = ema.state_dict()
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, ema=None, device="cpu"):
    """Load a training checkpoint. Returns the state dict for metadata access."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if ema is not None and "ema_state_dict" in state:
        ema.load_state_dict(state["ema_state_dict"])
    return state
