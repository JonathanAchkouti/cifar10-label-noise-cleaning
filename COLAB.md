# Colab runbook

Paste each block into its own cell. Set **Runtime → Change runtime type → GPU**
before you start.

---

## Cell 1 — check what GPU you got

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Your GPU decides the whole plan:

| GPU | VRAM | batch_size | grad_accum | ~min/epoch @224 |
|---|---|---|---|---|
| T4 | 16 GB | 64 | 2 | 9–12 |
| L4 | 22 GB | 128 | 1 | 5–7 |
| A100 | 40 GB | 256 | 1 | 2–3 |

If you got a T4 and it says you have Colab Pro, use Runtime → Change runtime
type and pick a better accelerator before spending hours.

---

## Cell 2 — mount Drive so a disconnect doesn't destroy your work

```python
from google.colab import drive
drive.mount('/content/drive')

import os
PERSIST = '/content/drive/MyDrive/inmind_cnn'
os.makedirs(f'{PERSIST}/artifacts', exist_ok=True)
os.makedirs(f'{PERSIST}/weights', exist_ok=True)
print('persisting to', PERSIST)
```

Colab reclaims your VM without warning. Checkpoints and prediction dumps written
to `/content` are gone when that happens; ones on Drive survive.

---

## Cell 3 — clone your fork and install

```python
!git clone https://github.com/<your-username>/<your-repo>.git /content/repo
%cd /content/repo

!pip install -q timm cleanlab scikit-learn pyyaml einops matplotlib
import timm, torch
print('torch', torch.__version__, '| timm', timm.__version__, '| cuda', torch.cuda.is_available())
```

Use a **fine-grained personal access token** if the repo is private:
`https://<token>@github.com/<user>/<repo>.git`. Don't paste the token into a
cell you'll commit — put it in Colab's Secrets panel and read it with
`google.colab.userdata.get()`.

---

## Cell 4 — symlink artifacts to Drive

```python
!rm -rf artifacts weights
!ln -s {PERSIST}/artifacts /content/repo/artifacts
!ln -s {PERSIST}/weights   /content/repo/weights
!ls -la artifacts weights
```

Now every checkpoint and `.npz` the scripts write lands on Drive automatically.

---

## Cell 5 — set batch size for your GPU

```python
import yaml
cfg = yaml.safe_load(open('config.yaml'))

gpu = torch.cuda.get_device_name(0)
if 'A100' in gpu:   bs, accum = 256, 1
elif 'L4' in gpu:   bs, accum = 128, 1
else:               bs, accum = 64,  2   # T4

cfg['hyperparameters']['batch_size'] = bs
cfg['hyperparameters']['grad_accum_steps'] = accum
cfg['hyperparameters']['num_workers'] = 2
yaml.safe_dump(cfg, open('config.yaml', 'w'), sort_keys=False)
print(f'{gpu}: batch_size={bs} grad_accum={accum} (effective {bs*accum})')
```

---

## Cell 6 — smoke test (do this before any long run)

```python
import yaml
cfg = yaml.safe_load(open('config.yaml'))
cfg['hyperparameters']['epochs'] = 1
cfg['data']['val_split'] = 0.5     # tiny train split -> finishes in ~2 min
yaml.safe_dump(cfg, open('smoke.yaml', 'w'), sort_keys=False)

!python train.py --config smoke.yaml
```

You are checking three things: it downloads CIFAR-10, one epoch completes, and
val accuracy is already ~90%+. If accuracy is near 10%, the normalisation or the
pretrained weights are wrong — stop and fix it before burning GPU hours.

---

## Cell 7 — extract features and find label errors

```python
!python clean_labels.py --topk 500
```

~5 min (one forward pass over 50k images), then seconds for the probe.
Cached to `artifacts/train_features.npz` — later scripts reuse it for free.

```python
from IPython.display import Image, display
display(Image('artifacts/train_issues_grid.png'))
```

**Actually look at this grid.** Screenshot it for your slides. Some flagged
images are real errors, some are genuinely ambiguous. Being able to say which is
which is worth more marks than any accuracy number.

---

## Cell 8 — validate the detector

```python
!python validate_detector.py --noise-rate 0.02
```

~1 min using the cached features. Gives you precision/recall on errors you
injected yourself, so the numbers don't depend on any downstream accuracy claim.

---

## Cell 9 — audit the test labels (identify only, never modify)

```python
!python clean_labels.py --split test
from IPython.display import Image, display
display(Image('artifacts/test_label_issues_grid.png'))
```

Go through the grid by hand. Keep only the indices you personally confirm are
mislabelled, then write them out:

```python
import json
CONFIRMED = [ ... ]   # paste the indices you verified by eye
json.dump({'issues': [{'index': i} for i in CONFIRMED]},
          open('artifacts/test_label_audit_verified.json', 'w'), indent=2)
print(len(CONFIRMED), 'verified bad test labels')
```

---

## Cell 10 — the three training runs

```python
import yaml, subprocess

for mode in ['none', 'drop', 'random_drop']:
    cfg = yaml.safe_load(open('config.yaml'))
    cfg['data']['label_cleaning'] = mode
    name = f'run_{mode}.yaml'
    yaml.safe_dump(cfg, open(name, 'w'), sort_keys=False)
    print(f'\n===== {mode} =====')
    subprocess.run(['python', 'train.py', '--config', name], check=True)
```

Budget ~1.5 h each on a T4, ~50 min on an L4. Keep the browser tab alive — Colab
disconnects idle sessions. If it dies mid-run, your Drive checkpoints survive and
you restart from Cell 3.

---

## Cell 11 — the paired analysis

```python
!python compare_runs.py \
    --baseline  artifacts/preds_vit_none_seed42.npz \
    --treatment artifacts/preds_vit_drop_seed42.npz \
    --exclude-test-issues artifacts/test_label_audit_verified.json \
    --out artifacts/comparison_cleaned.json

print('\n\n########## CONTROL ##########\n')

!python compare_runs.py \
    --baseline  artifacts/preds_vit_none_seed42.npz \
    --treatment artifacts/preds_vit_random_drop_seed42.npz \
    --out artifacts/comparison_control.json
```

```python
from IPython.display import Image, display
display(Image('artifacts/repaired_images.png'))
```

---

## Cell 12 — seed variance (do not skip)

```python
import yaml, subprocess
for seed in [1, 2, 3]:
    for mode in ['none', 'drop']:
        cfg = yaml.safe_load(open('config.yaml'))
        cfg['hyperparameters']['seed'] = seed
        cfg['data']['label_cleaning'] = mode
        name = f'run_{mode}_s{seed}.yaml'
        yaml.safe_dump(cfg, open(name, 'w'), sort_keys=False)
        subprocess.run(['python', 'train.py', '--config', name], check=True)
```

Without this you cannot interpret any of your numbers. Report mean ± std.

---

## Cell 13 — commit results back

```python
!git config user.email "you@example.com"
!git config user.name "Your Name"
!git add artifacts/*.json artifacts/*.png
!git commit -m "results: ViT-B/16 finetune + label-cleaning ablation"
!git push
```

Never commit `data/`, `weights/*.pth`, or `artifacts/*_features.npz` — the
feature cache alone is ~150 MB.
