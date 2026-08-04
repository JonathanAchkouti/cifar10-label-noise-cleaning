"""Paired comparison of two training runs on the same test set.

Why not just compare test accuracies? Because 98.90% vs 99.02% is 12 images out
of 10,000, and seed-to-seed variance on this setup is comparable. A raw delta
proves nothing on its own.

What this script does instead:

  1. Builds the 2x2 contingency table of paired outcomes on identical test
     images:  (baseline right/wrong) x (cleaned right/wrong).
  2. Runs McNemar's exact test on the two discordant cells. This is the correct
     test for paired binary outcomes measured on the same samples, and it is
     far more sensitive than comparing two independent accuracy numbers,
     because the ~9,880 images both models get right carry no information and
     are correctly ignored.
  3. Isolates the "repaired" set: images the baseline got WRONG and the cleaned
     model got RIGHT. Optionally restricted to test images whose own label has
     been verified correct -- which is precisely the claim you want to make.
  4. Breaks the change down per class and checks whether it lands on the
     classes the training-set noise actually came from. That is the mechanism
     evidence: a real effect should be targeted, not diffuse.

Usage:

    python compare_runs.py \
        --baseline artifacts/preds_vit_none_seed42.npz \
        --treatment artifacts/preds_vit_drop_seed42.npz \
        --label-issues artifacts/label_issues.json \
        --exclude-test-issues artifacts/test_label_audit_verified.json
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

# Defined locally rather than imported from data.py so this script runs in a
# plain numpy/scipy environment -- you can analyse results on your laptop
# without a GPU or a torch install.
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def load_run(path):
    d = np.load(path, allow_pickle=True)
    order = np.argsort(d['index'])
    return {
        'tag': str(d['run_tag']),
        'index': d['index'][order],
        'label': d['label'][order],
        'pred': d['pred'][order],
        'probs': d['probs'][order],
        'acc': float(d['test_acc']),
    }


def mcnemar_exact(b, c):
    """Exact two-sided McNemar. b, c are the discordant counts.

    Null hypothesis: an image is equally likely to break as to be repaired.
    Under H0 the repaired count follows Binomial(b + c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5, alternative='two-sided').pvalue)
    except ImportError:
        from math import comb
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        return float(min(1.0, 2 * tail))


def wilson_interval(k, n, z=1.96):
    """Confidence interval for an accuracy, so the report shows uncertainty."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_excluded_test_indices(path):
    if not path:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"--exclude-test-issues file not found: {path}")
    with open(path) as f:
        payload = json.load(f)
    items = payload['issues'] if isinstance(payload, dict) else payload
    return {int(r['index'] if isinstance(r, dict) else r) for r in items}


def noisy_class_profile(label_issues_path):
    """Which classes did the training-set noise actually involve?"""
    if not label_issues_path or not os.path.exists(label_issues_path):
        return None
    with open(label_issues_path) as f:
        issues = json.load(f)['issues']
    counter = Counter()
    for r in issues:
        counter[int(r['given_label'])] += 1
        counter[int(r['suggested_label'])] += 1
    return counter


def save_flip_sheet(indices, base, treat, cfg_root, path, n=32):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchvision.datasets import CIFAR10
    except ImportError:
        print("[warn] matplotlib/torchvision unavailable; skipping flip sheet")
        return

    raw = CIFAR10(root=cfg_root, train=False, download=False)
    pos = {int(i): k for k, i in enumerate(base['index'])}
    sel = list(indices)[:n]
    if not sel:
        return

    cols = 8
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
    for ax, idx in zip(np.ravel(axes), sel):
        k = pos[int(idx)]
        img, _ = raw[int(idx)]
        ax.imshow(img)
        ax.set_title(
            f"true {CLASSES[base['label'][k]]}\n"
            f"was {CLASSES[base['pred'][k]]} -> now {CLASSES[treat['pred'][k]]}",
            fontsize=6)
        ax.axis('off')
    for ax in np.ravel(axes)[len(sel):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[out] repaired-image sheet: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--treatment', required=True)
    ap.add_argument('--label-issues', default='artifacts/label_issues.json')
    ap.add_argument('--exclude-test-issues', default=None,
                    help='JSON list of VERIFIED-bad test indices to exclude')
    ap.add_argument('--data-root', default='data/cifar10')
    ap.add_argument('--out', default='artifacts/comparison.json')
    args = ap.parse_args()

    base, treat = load_run(args.baseline), load_run(args.treatment)
    assert np.array_equal(base['index'], treat['index']), \
        "Runs were evaluated on different test indices."
    assert np.array_equal(base['label'], treat['label']), \
        "Test labels differ between runs -- the test set must never be modified."

    excluded = load_excluded_test_indices(args.exclude_test_issues)
    keep = np.array([int(i) not in excluded for i in base['index']])

    print("=" * 68)
    print(f"baseline : {base['tag']}   acc={base['acc']:.4f}")
    print(f"treatment: {treat['tag']}   acc={treat['acc']:.4f}")
    if excluded:
        print(f"excluding {len(excluded)} test images with verified-bad labels "
              f"-> analysing {keep.sum()} of {len(keep)}")
    print("=" * 68)

    y = base['label'][keep]
    ok_b = base['pred'][keep] == y
    ok_t = treat['pred'][keep] == y

    both = int((ok_b & ok_t).sum())
    repaired = int((~ok_b & ok_t).sum())    # wrong -> right
    broken = int((ok_b & ~ok_t).sum())      # right -> wrong
    neither = int((~ok_b & ~ok_t).sum())
    n = len(y)

    print("\nPaired contingency (verified-clean test images)")
    print(f"  both correct         : {both}")
    print(f"  repaired (wrong->ok) : {repaired}")
    print(f"  broken   (ok->wrong) : {broken}")
    print(f"  both wrong           : {neither}")

    acc_b, acc_t = ok_b.mean(), ok_t.mean()
    lo_b, hi_b = wilson_interval(int(ok_b.sum()), n)
    lo_t, hi_t = wilson_interval(int(ok_t.sum()), n)
    p = mcnemar_exact(broken, repaired)

    print(f"\n  baseline  accuracy   : {acc_b:.4f}  95% CI [{lo_b:.4f}, {hi_b:.4f}]")
    print(f"  treatment accuracy   : {acc_t:.4f}  95% CI [{lo_t:.4f}, {hi_t:.4f}]")
    print(f"  net change           : {acc_t - acc_b:+.4f} ({repaired - broken:+d} images)")
    print(f"  McNemar exact p      : {p:.4g}")
    print("  " + ("SIGNIFICANT at alpha=0.05" if p < 0.05
                  else "NOT significant -- the two models are statistically indistinguishable"))

    if repaired + broken < 25:
        print("\n  [caution] Only "
              f"{repaired + broken} discordant images. McNemar has very little power")
        print("            here; treat the result as inconclusive regardless of p.")

    # ---- mechanism: where did the change land? ----
    print("\nPer-class breakdown of repaired / broken")
    profile = noisy_class_profile(args.label_issues)
    rows = []
    for c, name in enumerate(CLASSES):
        m = y == c
        r = int((~ok_b & ok_t & m).sum())
        b = int((ok_b & ~ok_t & m).sum())
        involved = profile.get(c, 0) if profile else 0
        rows.append({'class': name, 'repaired': r, 'broken': b,
                     'train_noise_involvement': involved})
        flag = ' <- noisy class' if profile and involved >= (
            np.percentile(list(profile.values()), 60) if profile else 0) else ''
        print(f"  {name:11s} repaired={r:3d}  broken={b:3d}  "
              f"train-noise mentions={involved:4d}{flag}")

    if profile:
        inv = np.array([profile.get(c, 0) for c in range(10)], dtype=float)
        net = np.array([r['repaired'] - r['broken'] for r in rows], dtype=float)
        if inv.std() > 0 and net.std() > 0:
            corr = float(np.corrcoef(inv, net)[0, 1])
            print(f"\n  correlation(train-noise involvement, net per-class gain) = {corr:+.3f}")
            print("  A clearly positive value is the mechanism evidence: the gain")
            print("  concentrates on the classes the training noise came from.")

    repaired_idx = base['index'][keep][(~ok_b & ok_t)]
    broken_idx = base['index'][keep][(ok_b & ~ok_t)]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'baseline': base['tag'], 'treatment': treat['tag'],
            'n_analysed': n, 'n_excluded': len(excluded),
            'both_correct': both, 'repaired': repaired,
            'broken': broken, 'both_wrong': neither,
            'baseline_acc': float(acc_b), 'treatment_acc': float(acc_t),
            'baseline_ci': [lo_b, hi_b], 'treatment_ci': [lo_t, hi_t],
            'mcnemar_p': p,
            'per_class': rows,
            'repaired_indices': repaired_idx.tolist(),
            'broken_indices': broken_idx.tolist(),
        }, f, indent=2)
    print(f"\n[out] {args.out}")

    save_flip_sheet(repaired_idx, base, treat, args.data_root,
                    os.path.join(os.path.dirname(args.out) or '.', 'repaired_images.png'))


if __name__ == '__main__':
    main()
