"""Measure how good the label-noise detector actually is.

The problem with evaluating a label-error detector on real data: you have no
ground truth. You cannot compute recall, because you do not know how many
errors were there to begin with.

Solution: inject a known number of label errors yourself, then see how many the
detector recovers. Because you created them, you know exactly which indices are
corrupted, so precision and recall are directly measurable.

This gives you a hard number to quote -- "the detector recovers 92% of injected
errors at 78% precision" -- which is independent of any downstream accuracy
change and therefore not vulnerable to the "that's within seed noise" objection.

    python validate_detector.py --noise-rate 0.02 --folds 5
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml

from clean_labels import cross_val_probs, extract_features, find_issues


def inject_noise(labels, rate, seed, num_classes=10):
    """Flip a fraction of labels to a different, uniformly-chosen class."""
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    n_flip = int(round(rate * len(labels)))
    victims = rng.choice(len(labels), size=n_flip, replace=False)
    for i in victims:
        choices = [c for c in range(num_classes) if c != labels[i]]
        noisy[i] = rng.choice(choices)
    return noisy, set(int(v) for v in victims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--noise-rate', type=float, default=0.02)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', default='artifacts/detector_validation.json')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art = cfg['paths']['artifacts_dir']
    os.makedirs(art, exist_ok=True)

    cache = os.path.join(art, 'train_features.npz')
    if os.path.exists(cache):
        d = np.load(cache)
        features, labels = d['features'], d['labels']
        print(f"[cache] reusing {cache}")
    else:
        features, labels = extract_features(cfg, device, train_split=True)
        np.savez_compressed(cache, features=features, labels=labels)

    noisy, injected = inject_noise(labels, args.noise_rate, args.seed)
    print(f"[inject] flipped {len(injected)} of {len(labels)} labels "
          f"({args.noise_rate:.1%})")

    probs = cross_val_probs(features, noisy, folds=args.folds, seed=args.seed)
    flagged = set(int(i) for i in find_issues(noisy, probs))

    tp = len(flagged & injected)
    fp = len(flagged - injected)
    fn = len(injected - flagged)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # How often does the detector propose the ORIGINAL label for an image it
    # correctly flagged? This is what decides whether 'relabel' is safe or
    # whether you should stick to 'drop'.
    recovered = sum(1 for i in (flagged & injected)
                    if int(probs[i].argmax()) == int(labels[i]))
    relabel_acc = recovered / max(1, tp)

    print("\n" + "=" * 56)
    print(f"  injected errors        : {len(injected)}")
    print(f"  flagged by detector    : {len(flagged)}")
    print(f"  true positives         : {tp}")
    print(f"  false positives        : {fp}")
    print(f"  missed (false neg)     : {fn}")
    print(f"  precision              : {precision:.3f}")
    print(f"  recall                 : {recall:.3f}")
    print(f"  F1                     : {f1:.3f}")
    print(f"  relabel correctness    : {relabel_acc:.3f}")
    print("=" * 56)
    print("\n  Note: 'false positives' here include genuinely mislabelled images")
    print("  that were already in CIFAR-10 before injection. So the true")
    print("  precision is a little higher than the number above -- this is a")
    print("  conservative estimate, which is the right direction to err in.")
    if relabel_acc < 0.9:
        print(f"\n  relabel correctness is {relabel_acc:.2f}: prefer label_cleaning='drop',")
        print("  since relabelling would inject fresh noise at a meaningful rate.")

    with open(args.out, 'w') as f:
        json.dump({'noise_rate': args.noise_rate, 'seed': args.seed,
                   'n_injected': len(injected), 'n_flagged': len(flagged),
                   'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision,
                   'recall': recall, 'f1': f1,
                   'relabel_correctness': relabel_acc}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == '__main__':
    main()
