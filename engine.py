"""Optimiser construction, schedules, EMA, and the train/eval loops."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gpu_upsample(images, size):
    """Resize 32x32 -> size on the GPU instead of in the dataloader.

    On Colab you get 2 vCPUs. Doing a 32->224 bicubic resize in the transform
    pipeline means the CPU produces ~600KB per image and the GPU sits idle
    waiting for it. Sending 32x32 tensors (12KB) and upsampling on-device cuts
    both the CPU work and the PCIe traffic by ~50x.

    This is mathematically equivalent to resizing before normalising:
    interpolation kernels sum to 1, so interp(a*x + b) == a*interp(x) + b for
    the per-channel affine normalisation. Order does not matter.
    """
    if size is None or images.shape[-1] == size:
        return images
    return F.interpolate(images, size=(size, size), mode='bicubic',
                         align_corners=False)


# --------------------------------------------------------------------------
# Layer-wise learning rate decay
# --------------------------------------------------------------------------
# Intuition: early ViT blocks encode generic edges and textures that already
# transfer; late blocks and the head encode task-specific structure that must
# change. So LR should grow with depth. lr_i = base_lr * decay ** (L - i).

def _vit_layer_id(name: str, num_blocks: int) -> int:
    if name.startswith(('cls_token', 'pos_embed', 'patch_embed')):
        return 0
    if name.startswith('blocks.'):
        return int(name.split('.')[1]) + 1
    return num_blocks + 1  # final norm + head


def build_optimizer(model, cfg):
    hp = cfg['hyperparameters']
    base_lr, wd = hp['lr'], hp['weight_decay']
    decay = hp.get('layer_decay', 1.0)

    blocks = getattr(model, 'blocks', None)
    num_blocks = len(blocks) if blocks is not None else 0
    num_layers = num_blocks + 1

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No weight decay on 1-D params: biases, norms, and positional tokens.
        # Decaying these shrinks representations rather than regularising them.
        no_decay = p.ndim <= 1 or name.endswith('.bias') or 'token' in name or 'pos_embed' in name
        layer_id = _vit_layer_id(name, num_blocks) if num_blocks else num_layers

        key = (layer_id, no_decay)
        if key not in groups:
            groups[key] = {
                'params': [],
                'lr': base_lr * (decay ** (num_layers - layer_id)),
                'weight_decay': 0.0 if no_decay else wd,
            }
        groups[key]['params'].append(p)

    param_groups = list(groups.values())
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.999))
    # Store each group's peak LR so the scheduler can scale from it.
    for grp in optimizer.param_groups:
        grp['base_lr'] = grp['lr']
    return optimizer


def set_lr(optimizer, step, total_steps, warmup_steps):
    """Linear warmup then cosine decay, applied per group from its base_lr."""
    if warmup_steps > 0 and step < warmup_steps:
        scale = step / max(1, warmup_steps)
    else:
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    for grp in optimizer.param_groups:
        grp['lr'] = grp['base_lr'] * scale
    return scale


# --------------------------------------------------------------------------
# Exponential moving average of weights
# --------------------------------------------------------------------------

class ModelEMA:
    """Keeps a slowly-moving copy of the weights.

    The EMA weights sit closer to the centre of the loss basin than the final
    SGD iterate, which is usually worth ~0.1-0.3% on a short finetune. Free,
    apart from one extra copy of the model in memory.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.module.state_dict().values(),
                            model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:
                ema_p.copy_(p)


# --------------------------------------------------------------------------
# Loops
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    epoch, cfg, ema=None, global_step=0, total_steps=1,
                    warmup_steps=0):
    model.train()
    hp = cfg['hyperparameters']
    resize_to = cfg['data']['image_size'] if cfg['data'].get('gpu_resize') else None
    accum = max(1, hp.get('grad_accum_steps', 1))
    use_amp = hp.get('amp', True) and device.type == 'cuda'

    running, seen, correct = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for it, (images, targets, _) in enumerate(loader):
        images = gpu_upsample(images.to(device, non_blocking=True), resize_to)
        targets = targets.to(device, non_blocking=True)

        if it % accum == 0:
            set_lr(optimizer, global_step, total_steps, warmup_steps)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets) / accum

        scaler.scale(loss).backward()

        if (it + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            global_step += 1

        running += loss.item() * accum * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        seen += targets.size(0)

        if it % 100 == 0:
            lr_now = optimizer.param_groups[-1]['lr']
            print(f"  epoch {epoch} [{it}/{len(loader)}] "
                  f"loss={running / seen:.4f} acc={correct / seen:.4f} lr={lr_now:.2e}")

    return running / seen, correct / seen, global_step


@torch.no_grad()
def predict_all(model, loader, device, tta_hflip=False, use_amp=True,
                resize_to=None):
    """Per-example predictions, keyed by dataset index.

    This is what makes run-to-run comparison possible: aggregate accuracy tells
    you almost nothing, but knowing exactly which images flipped tells you a lot.
    """
    model.eval()
    amp_on = use_amp and device.type == 'cuda'
    idxs, ys, preds, probs = [], [], [], []

    for images, targets, index in loader:
        images = gpu_upsample(images.to(device, non_blocking=True), resize_to)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_on):
            logits = model(images)
            if tta_hflip:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2
        p = logits.float().softmax(1).cpu()
        idxs.append(index)
        ys.append(targets)
        preds.append(p.argmax(1))
        probs.append(p)

    return {
        'index': torch.cat(idxs).numpy(),
        'label': torch.cat(ys).numpy(),
        'pred': torch.cat(preds).numpy(),
        'probs': torch.cat(probs).numpy(),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta_hflip=False, use_amp=True,
             resize_to=None):
    model.eval()
    amp_on = use_amp and device.type == 'cuda'
    total_loss, correct, seen = 0.0, 0, 0

    for images, targets, _ in loader:
        images = gpu_upsample(images.to(device, non_blocking=True), resize_to)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_on):
            logits = model(images)
            if tta_hflip:
                # Average logits over the image and its mirror. CIFAR classes are
                # left-right symmetric, so this is a safe, near-free ~0.1% gain.
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2
            loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        seen += targets.size(0)

    return total_loss / seen, correct / seen
