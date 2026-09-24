"""Optimizer and checkpoint helpers shared by the training entry points."""

import torch


def checkpoint_weights(model, state):
    """Translate legacy PEFT keys only when the target model requires it.

    Unknown keys remain visible to strict loading; an incompatible resume must
    fail rather than silently leaving part of the model randomly initialized.
    """
    target = model.state_dict()
    remapped = {}
    for key, value in state.items():
        mapped = key
        if key not in target:
            parts = key.split(".")
            if (len(parts) >= 2 and parts[-1] in {"weight", "bias"}
                    and parts[-2] in {"query", "key", "value"}):
                candidate = ".".join(parts[:-1] + ["base_layer", parts[-1]])
                if candidate in target:
                    mapped = candidate
            elif ".base_layer." in key:
                candidate = key.replace(".base_layer.", ".")
                if candidate in target:
                    mapped = candidate
        if mapped in remapped:
            raise ValueError(f"Checkpoint contains duplicate weights for {mapped}")
        remapped[mapped] = value
    return remapped


def optimizer_step(optimizers, schedulers, scaler, parameters, max_grad_norm,
                   normalization=1.0, sync_invalid=None):
    """Apply one complete update, including a partially filled accumulation group.

    Dense and sparse optimizers either both step or both skip. Sparse gradients
    remain sparse and are checked separately because clip_grad_norm_ cannot
    handle them. Schedulers advance only when an update is applied.
    """
    for optimizer in optimizers:
        scaler.unscale_(optimizer)
    parameters = [p for p in parameters if p.grad is not None]
    dense, sparse = [], []
    for parameter in parameters:
        gradient = parameter.grad
        values = gradient._values() if gradient.is_sparse else gradient
        if normalization != 1.0:
            values.mul_(normalization)
        (sparse if gradient.is_sparse else dense).append(parameter)
    norm = torch.nn.utils.clip_grad_norm_(dense, max_grad_norm)
    invalid = not bool(torch.isfinite(norm))
    invalid |= any(not bool(torch.isfinite(p.grad._values()).all()) for p in sparse)
    if sync_invalid is not None:
        invalid = sync_invalid(invalid)
    if invalid:
        # Norm overflow can occur after unscale_; explicitly lower the scale
        # so all optimizers and distributed ranks recover together.
        if scaler.is_enabled():
            scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        else:
            scaler.update()
    else:
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()
        for scheduler in schedulers:
            scheduler.step()
    for optimizer in optimizers:
        optimizer.zero_grad()
    return not invalid
