"""Fine-tune a pretrained backbone (or the LeNet baseline) on CIFAR-10.

    python train.py                      # uses config.yaml
    python train.py --config other.yaml
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from data import build_dataloaders
from engine import ModelEMA, build_optimizer, evaluate, predict_all, train_one_epoch
from model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hp = cfg['hyperparameters']
    set_seed(hp['seed'])
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[setup] device={device}  model={cfg['model']['name']}")
    if device.type == 'cpu':
        print("[warn] no CUDA device found; ViT fine-tuning on CPU is impractical.")

    model, data_config = build_model(cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] trainable params: {n_params / 1e6:.1f}M  "
          f"norm mean={data_config['mean']} std={data_config['std']}")

    train_dl, val_dl, test_dl = build_dataloaders(cfg, data_config)

    optimizer = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=hp.get('label_smoothing', 0.0))
    eval_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=hp.get('amp', True) and device.type == 'cuda')

    ema = ModelEMA(model, hp['ema_decay']) if hp.get('ema_decay', 0) else None

    accum = max(1, hp.get('grad_accum_steps', 1))
    steps_per_epoch = max(1, len(train_dl) // accum)
    total_steps = steps_per_epoch * hp['epochs']
    warmup_steps = steps_per_epoch * hp.get('warmup_epochs', 0)

    os.makedirs(os.path.dirname(cfg['paths']['model_path']) or '.', exist_ok=True)
    os.makedirs(cfg['paths']['artifacts_dir'], exist_ok=True)

    resize_to = cfg['data']['image_size'] if cfg['data'].get('gpu_resize') else None
    best_acc, best_state, history = 0.0, None, []
    global_step = 0

    for epoch in range(1, hp['epochs'] + 1):
        t0 = time.time()
        tr_loss, tr_acc, global_step = train_one_epoch(
            model, train_dl, optimizer, scaler, criterion, device, epoch, cfg,
            ema=ema, global_step=global_step, total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

        eval_model = ema.module if ema is not None else model
        va_loss, va_acc = evaluate(eval_model, val_dl, eval_criterion, device,
                                   tta_hflip=False, use_amp=hp.get('amp', True),
                                   resize_to=resize_to)

        dt = time.time() - t0
        print(f"[epoch {epoch}/{hp['epochs']}] train_loss={tr_loss:.4f} "
              f"train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
              f"({dt:.0f}s)")
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc,
                        'val_loss': va_loss, 'val_acc': va_acc, 'seconds': dt})

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in eval_model.state_dict().items()}
            torch.save({'model': best_state, 'config': cfg, 'val_acc': va_acc},
                       cfg['paths']['model_path'])
            print(f"  -> new best val_acc={va_acc:.4f}, checkpoint saved")

    if best_state is not None:
        (ema.module if ema is not None else model).load_state_dict(best_state)

    final_model = ema.module if ema is not None else model
    te_loss, te_acc = evaluate(final_model, test_dl, eval_criterion, device,
                               tta_hflip=hp.get('tta_hflip', False),
                               use_amp=hp.get('amp', True), resize_to=resize_to)
    print(f"\n[test] loss={te_loss:.4f} accuracy={te_acc:.4f}")

    # Dump per-example test predictions. compare_runs.py needs these to do the
    # paired analysis; aggregate accuracy alone cannot show which images flipped.
    tag = cfg['paths'].get('run_tag') or (
        f"{cfg['model']['name']}_{cfg['data'].get('label_cleaning', 'none')}"
        f"_seed{hp['seed']}")
    preds = predict_all(final_model, test_dl, device,
                        tta_hflip=hp.get('tta_hflip', False),
                        use_amp=hp.get('amp', True), resize_to=resize_to)
    pred_path = os.path.join(cfg['paths']['artifacts_dir'], f'preds_{tag}.npz')
    np.savez_compressed(pred_path, run_tag=tag, test_acc=te_acc, **preds)
    print(f"[out] per-example predictions -> {pred_path}")

    summary = {
        'run_tag': tag,
        'model': cfg['model'].get('timm_name') if cfg['model']['name'] == 'vit' else 'lenet',
        'label_cleaning': cfg['data'].get('label_cleaning', 'none'),
        'best_val_acc': best_acc,
        'test_acc': te_acc,
        'history': history,
    }
    out = os.path.join(cfg['paths']['artifacts_dir'], 'run_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary written to {out}")


if __name__ == '__main__':
    main()
