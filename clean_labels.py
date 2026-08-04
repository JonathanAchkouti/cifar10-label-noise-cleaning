"""Find mislabelled images in the CIFAR-10 *training* set.

Method (confident learning, Northcutt et al. 2021):

  1. Every training image needs a predicted probability vector from a model
     that did NOT see that image's label during training. Otherwise the model
     has memorised the wrong label and will happily confirm it.
  2. Rather than k-fold fine-tuning the whole ViT (5x the training cost), we
     freeze the backbone, extract 768-d features once (~1 forward pass over
     50k images), and cross-validate a cheap linear probe on top. The probe is
     what gets held out, so the probabilities are genuinely out-of-sample.
  3. Confident learning estimates the joint distribution of (noisy label,
     true label) from those probabilities and flags examples whose given label
     is inconsistent with the model's confident prediction. This is better
     than a plain "loss is high" threshold because it calibrates per-class:
     a class the model is generally bad at does not get over-flagged.

Output: artifacts/label_issues.json and a contact sheet of the top flagged
images so the errors can be eyeballed rather than trusted blindly.

    python clean_labels.py --config config.yaml --topk 300
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from data import CLASSES
from engine import gpu_upsample
from model import build_feature_extractor


@torch.no_grad()
def extract_features(cfg, device, train_split=True):
    """One clean forward pass over the split. No augmentation."""
    backbone, data_config = build_feature_extractor(cfg['model']['timm_name'])
    backbone.eval().to(device)

    gpu_resize = cfg['data'].get('gpu_resize', False)
    ops = []
    if not gpu_resize:
        ops.append(transforms.Resize(cfg['data']['image_size'],
                                     interpolation=transforms.InterpolationMode.BICUBIC))
    ops += [transforms.ToTensor(),
            transforms.Normalize(data_config['mean'], data_config['std'])]
    tf = transforms.Compose(ops)
    size = cfg['data']['image_size'] if gpu_resize else None

    ds = CIFAR10(root=cfg['data']['root'], train=train_split, download=True, transform=tf)
    dl = DataLoader(ds, batch_size=256, shuffle=False,
                    num_workers=cfg['hyperparameters']['num_workers'], pin_memory=True)

    feats, labels = [], []
    use_amp = device.type == 'cuda'
    for i, (x, y) in enumerate(dl):
        x = gpu_upsample(x.to(device, non_blocking=True), size)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            f = backbone(x)
        feats.append(f.float().cpu())
        labels.append(y)
        if i % 20 == 0:
            print(f"  features {i * 256}/{len(ds)}")

    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def cross_val_probs(features, labels, folds=5, seed=42):
    """Out-of-sample probabilities from a linear probe on frozen features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
    )
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    print(f"[probe] {folds}-fold cross-validated logistic regression...")
    probs = cross_val_predict(clf, features, labels, cv=skf,
                              method='predict_proba', n_jobs=1)
    acc = (probs.argmax(1) == labels).mean()
    print(f"[probe] out-of-sample linear-probe accuracy: {acc:.4f}")
    return probs


def find_issues(labels, probs, topk=None):
    """Prefer cleanlab; fall back to a self-confidence ranking if unavailable."""
    try:
        from cleanlab.filter import find_label_issues
        idx = find_label_issues(
            labels=labels,
            pred_probs=probs,
            return_indices_ranked_by='self_confidence',
            filter_by='prune_by_noise_rate',
        )
        print(f"[cleanlab] flagged {len(idx)} candidate label errors "
              f"({100 * len(idx) / len(labels):.2f}% of the train set)")
    except ImportError:
        print("[warn] cleanlab not installed; falling back to self-confidence ranking.")
        print("       pip install cleanlab  (the calibrated version is better)")
        self_conf = probs[np.arange(len(labels)), labels]
        disagree = probs.argmax(1) != labels
        idx = np.where(disagree)[0]
        idx = idx[np.argsort(self_conf[idx])]

    if topk is not None:
        idx = idx[:topk]
    return np.asarray(idx)


def save_contact_sheet(indices, labels, probs, cfg, path, n=48, split='train'):
    """Grid of the most-suspect images so you can verify by eye."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping contact sheet")
        return

    raw = CIFAR10(root=cfg['data']['root'], train=split == 'train', download=False)
    sel = indices[:n]
    cols = 8
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.0))
    for ax, idx in zip(np.ravel(axes), sel):
        img, _ = raw[int(idx)]
        ax.imshow(img)
        given = CLASSES[labels[idx]]
        sugg = CLASSES[probs[idx].argmax()]
        ax.set_title(f"{given} -> {sugg}", fontsize=7)
        ax.axis('off')
    for ax in np.ravel(axes)[len(sel):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[out] contact sheet: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--topk', type=int, default=None,
                        help='keep only the N most-suspect examples')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--split', choices=['train', 'test'], default='train',
                        help="'test' AUDITS the test labels for the analysis. "
                             "It never modifies them and never feeds training.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art = cfg['paths']['artifacts_dir']
    os.makedirs(art, exist_ok=True)

    cache = os.path.join(art, f'{args.split}_features.npz')
    if os.path.exists(cache):
        print(f"[cache] loading {cache}")
        d = np.load(cache)
        features, labels = d['features'], d['labels']
    else:
        features, labels = extract_features(cfg, device, train_split=args.split == 'train')
        np.savez_compressed(cache, features=features, labels=labels)
        print(f"[cache] saved {cache}")

    probs = cross_val_probs(features, labels, folds=args.folds,
                            seed=cfg['hyperparameters']['seed'])
    idx = find_issues(labels, probs, topk=args.topk)

    issues = [{
        'index': int(i),
        'given_label': int(labels[i]),
        'given_name': CLASSES[labels[i]],
        'suggested_label': int(probs[i].argmax()),
        'suggested_name': CLASSES[probs[i].argmax()],
        'given_prob': float(probs[i][labels[i]]),
        'suggested_prob': float(probs[i].max()),
    } for i in idx]

    out = (cfg['data']['label_issues_file'] if args.split == 'train'
           else os.path.join(art, 'test_label_audit.json'))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'n_train': int(len(labels)), 'n_issues': len(issues),
                   'issues': issues}, f, indent=2)
    print(f"[out] {len(issues)} issues -> {out}")

    save_contact_sheet(idx, labels, probs, cfg,
                       os.path.join(art, f'{args.split}_label_issues_grid.png'),
                       split=args.split)
    if args.split == 'test':
        print("\n[note] These are SUSPECT TEST LABELS, for analysis only.")
        print("       Verify them by eye in the contact sheet, then pass the")
        print("       confirmed list to compare_runs.py --exclude-test-issues.")
        print("       Never edit the test labels themselves.")


if __name__ == '__main__':
    main()
