# CIFAR-10 Label-Noise Detection and Cleaning

Fine-tuning a pretrained Vision Transformer on CIFAR-10, then finding and
handling mislabelled training images with confident learning, and measuring
whether cleaning the labels actually helps, with statistics that hold up.

CIFAR-10 is known to contain a small number of wrong labels. At the accuracy a
pretrained ViT reaches, those errors are a meaningful part of what is left to
fix. This project asks three questions:

1. **Can the label errors be found reliably?** Validated by injecting known
   errors and measuring precision and recall.
2. **Does removing or correcting them improve the model?** Tested with a paired
   statistical test rather than a raw accuracy difference.
3. **Is the improvement where it should be?** Checked per class against where
   the training-set noise actually came from.

## Pipeline

```
train.py              Fine-tune ViT-B/16 (or the LeNet baseline) on CIFAR-10
clean_labels.py       Find likely mislabelled images with confident learning
validate_detector.py  Measure detector precision/recall on injected noise
compare_runs.py       Paired comparison of two runs (McNemar's exact test)
```

### 1. Fine-tuning the backbone

`vit_base_patch16_224.augreg_in21k` from `timm` (ImageNet-21k, AugReg recipe),
fine-tuned with:

- **AdamW with layer-wise learning-rate decay (0.65):** lower layers hold
  general features and get smaller updates than the top blocks and the head.
- **Linear warmup and cosine decay, gradient accumulation** (effective batch 128), **mixed
  precision**, and **stochastic depth** (0.1).
- **EMA of the weights** (decay 0.999) and **horizontal-flip test-time
  augmentation**.
- **Augmentation at 32×32, before upsampling to 224.** A 4-pixel random crop
  on a 32-pixel image is a real shift. The same crop after upsampling would be
  a sub-pixel nudge. Upsampling runs on the GPU rather than in the data loader.

The original LeNet-style baseline is kept in `model.py` for a like-for-like
comparison.

### 2. Finding label errors (`clean_labels.py`)

Based on confident learning (Northcutt et al., 2021):

- Every training image needs a predicted probability from a model that **did
  not see its label**. Otherwise the model has memorised the wrong label and
  confirms it.
- Instead of k-fold fine-tuning the whole ViT (5× the cost), the backbone is
  frozen, 768-d features are extracted once, and a **cross-validated linear
  probe** produces genuinely out-of-sample probabilities.
- Confident learning calibrates per class, so classes the model finds hard are
  not over-flagged the way a plain high-loss threshold would over-flag them.
- Output: `artifacts/label_issues.json` and a contact sheet of the top flagged
  images for manual inspection.

Cleaning (`label_cleaning: drop | relabel`) is applied to the **training split
only**. The validation split stays noisy on purpose so it remains an honest
proxy for the test set, and the test labels are never modified. They are only
audited (`--split test`) so that affected test images can be excluded from the
analysis.

A `random_drop` control removes the same number of training images chosen at
random. If `drop` and `random_drop` gave the same gain, the improvement would
come from a smaller dataset, not from removing the right images.

### 3. Validating the detector (`validate_detector.py`)

On real data there is no ground truth, so recall cannot be computed. The script
flips a known fraction of labels (2% by default), runs the same detector, and
measures how many of the injected errors it recovers. This gives a precision
and recall that do not depend on any downstream accuracy change.

### 4. Comparing runs (`compare_runs.py`)

Two test accuracies a few tenths of a percent apart can be within
seed-to-seed noise. Instead of comparing them directly, the script:

- builds the 2×2 table of paired outcomes on identical test images and runs
  **McNemar's exact test** on the discordant pairs;
- isolates the **"repaired" set**, meaning images the baseline got wrong and
  the cleaned model got right, optionally restricted to test images with
  verified-correct labels;
- breaks the change down **per class**, to check that it lands on the classes
  the training noise came from. A real effect should be targeted, not diffuse.

Seed-variance runs are part of the protocol, so any effect is judged against
run-to-run noise.

## Running

```bash
pip install timm cleanlab scikit-learn pyyaml einops matplotlib torch torchvision

python train.py                                   # baseline fine-tune (config.yaml)
python clean_labels.py --topk 500                 # find label issues in train
python validate_detector.py --noise-rate 0.02     # detector precision / recall
python clean_labels.py --split test               # audit test labels (identify only)

# set data.label_cleaning to 'drop', 'relabel' or 'random_drop' in config.yaml, then:
python train.py

python compare_runs.py \
    --baseline artifacts/preds_vit_none_seed42.npz \
    --treatment artifacts/preds_vit_drop_seed42.npz \
    --label-issues artifacts/label_issues.json
```

All settings live in `config.yaml`. `COLAB.md` is a step-by-step runbook for
Google Colab, including batch sizes per GPU type and saving artifacts to Drive
so a disconnect doesn't lose a run.

## Repository structure

```
config.yaml           All experiment settings
data.py               CIFAR-10 loading, augmentation, train-only label cleaning
model.py              LeNet baseline, timm ViT builder, feature extractor
engine.py             Optimizer with layer decay, LR schedule, EMA, train/eval loops
train.py              Training entry point
clean_labels.py       Confident-learning label-issue detection
validate_detector.py  Injected-noise validation of the detector
compare_runs.py       Paired statistical comparison of two runs
COLAB.md              Colab runbook
```
