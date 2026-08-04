"""Data pipeline.

Two things worth explaining in the write-up:

1. Augmentation happens at 32x32, *before* the upsample to 224. A 4px random
   crop on a 32px image is a meaningful shift; the same crop applied after
   upsampling would be a sub-pixel nudge and do almost nothing.

2. Label cleaning is applied to the TRAIN split only. The val split is left
   noisy on purpose so it stays an honest proxy for the (also noisy) test set,
   and the test split is never touched at all.
"""

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def build_transforms(data_config, cfg, train: bool):
    size = cfg['data']['image_size']
    mean, std = data_config['mean'], data_config['std']
    interp = transforms.InterpolationMode.BICUBIC

    ops = []
    if train:
        pad = cfg['data'].get('random_crop_padding', 4)
        if pad:
            ops.append(transforms.RandomCrop(32, padding=pad, padding_mode='reflect'))
        if cfg['data'].get('hflip', True):
            ops.append(transforms.RandomHorizontalFlip())

    # When gpu_resize is on, keep tensors at 32x32 here; engine.gpu_upsample
    # does the resize on-device. Much faster on CPU-starved machines.
    if size != 32 and not cfg['data'].get('gpu_resize', False):
        ops.append(transforms.Resize(size, interpolation=interp))

    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


class IndexedCIFAR10(Dataset):
    """CIFAR-10 wrapper that exposes the original index and allows relabelling.

    Returning the index is what lets the cleaning script map predictions back
    to specific training images.
    """

    def __init__(self, root, train, transform, indices=None, labels=None, download=True):
        self.base = CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.indices = np.arange(len(self.base)) if indices is None else np.asarray(indices)
        self.labels = (np.asarray(self.base.targets) if labels is None
                       else np.asarray(labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        img, _ = self.base[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx]), idx


def _load_label_issues(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"label_cleaning is enabled but {path} does not exist. "
            "Run `python clean_labels.py` first."
        )
    with open(path) as f:
        payload = json.load(f)
    flagged = {int(r['index']): int(r['suggested_label']) for r in payload['issues']}
    return flagged


def apply_label_cleaning(train_idx, labels, cfg, verbose=True):
    """Return (train_idx, labels) after applying the configured cleaning mode."""
    mode = cfg['data'].get('label_cleaning', 'none')
    if mode == 'none':
        return train_idx, labels

    flagged = _load_label_issues(cfg['data']['label_issues_file'])
    train_set = set(int(i) for i in train_idx)
    # Only act on indices that actually live in the train split.
    relevant = {i: v for i, v in flagged.items() if i in train_set}

    if mode == 'drop':
        keep = np.array([i for i in train_idx if int(i) not in relevant])
        if verbose:
            print(f"[clean] dropped {len(train_idx) - len(keep)} flagged training images")
        return keep, labels

    if mode == 'random_drop':
        # CONTROL CONDITION. Drops the same NUMBER of training images, chosen
        # uniformly at random instead of by the detector. If 'drop' and
        # 'random_drop' give the same gain, the improvement came from shrinking
        # the training set, not from removing bad labels. Without this control
        # you cannot attribute any gain to the cleaning itself.
        rng = np.random.default_rng(cfg['hyperparameters']['seed'])
        n_drop = len(relevant)
        victims = set(rng.choice(train_idx, size=min(n_drop, len(train_idx)),
                                 replace=False).tolist())
        keep = np.array([i for i in train_idx if int(i) not in victims])
        if verbose:
            print(f"[control] randomly dropped {len(train_idx) - len(keep)} training images")
        return keep, labels

    if mode == 'relabel':
        labels = labels.copy()
        changed = 0
        for i, new in relevant.items():
            if labels[i] != new:
                labels[i] = new
                changed += 1
        if verbose:
            print(f"[clean] relabelled {changed} training images")
        return train_idx, labels

    raise ValueError(f"Unknown label_cleaning mode: {mode!r}")


def build_dataloaders(cfg, data_config):
    root = cfg['data']['root']
    g = torch.Generator().manual_seed(cfg['hyperparameters']['seed'])

    full = CIFAR10(root=root, train=True, download=True)
    labels = np.asarray(full.targets)
    n = len(full)

    perm = torch.randperm(n, generator=g).numpy()
    n_val = int(round(cfg['data']['val_split'] * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Cleaning touches the training indices/labels only.
    train_idx, labels = apply_label_cleaning(train_idx, labels, cfg)

    train_tf = build_transforms(data_config, cfg, train=True)
    eval_tf = build_transforms(data_config, cfg, train=False)

    train_ds = IndexedCIFAR10(root, True, train_tf, train_idx, labels)
    val_ds = IndexedCIFAR10(root, True, eval_tf, val_idx, np.asarray(full.targets))
    test_ds = IndexedCIFAR10(root, False, eval_tf)

    hp = cfg['hyperparameters']
    common = dict(num_workers=hp['num_workers'], pin_memory=True,
                  persistent_workers=hp['num_workers'] > 0)

    train_dl = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True,
                          drop_last=True, **common)
    val_dl = DataLoader(val_ds, batch_size=hp['batch_size'] * 2, shuffle=False, **common)
    test_dl = DataLoader(test_ds, batch_size=hp['batch_size'] * 2, shuffle=False, **common)

    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_dl, val_dl, test_dl
