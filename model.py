"""Model definitions.

Keeps the original SimpleNet so the report can show a like-for-like baseline
against the fine-tuned backbone.
"""

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

NUM_CLASSES = 10


class SimpleNet(nn.Module):
    """Original repo baseline (LeNet-style). Kept for ablation."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def build_model(cfg):
    """Return (model, data_config).

    data_config carries the normalisation statistics the checkpoint was trained
    with. Using ImageNet stats on a model pretrained with (0.5, 0.5, 0.5) is a
    silent accuracy leak, so we always read it from the checkpoint metadata
    rather than hardcoding it.
    """
    name = cfg['model']['name'].lower()

    if name == 'lenet':
        model = SimpleNet()
        data_config = {
            'input_size': (3, 32, 32),
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2470, 0.2435, 0.2616),
            'interpolation': 'bilinear',
        }
        return model, data_config

    if name == 'vit':
        import timm
        model = timm.create_model(
            cfg['model']['timm_name'],
            pretrained=cfg['model'].get('pretrained', True),
            num_classes=NUM_CLASSES,
            drop_path_rate=cfg['model'].get('drop_path_rate', 0.0),
        )
        data_config = timm.data.resolve_model_data_config(model)
        return model, data_config

    raise ValueError(f"Unknown model name: {name!r} (expected 'lenet' or 'vit')")


def build_feature_extractor(timm_name: str):
    """Backbone with the classifier removed, for label-noise detection.

    num_classes=0 makes timm return pooled features instead of logits.
    """
    import timm
    model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(model)
    return model, data_config
