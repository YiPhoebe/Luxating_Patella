from typing import Literal

import torch.nn as nn


def build_model(
    name: Literal["resnet18", "resnet50", "resnet50_custom", "simple_cnn"] = "resnet18",
    pretrained: bool = True,
    num_classes: int = 2,
) -> nn.Module:
    # torchvision ResNet variants (no custom file dependency)
    if name in ("resnet18", "resnet50"):
        import torchvision.models as models

        if name == "resnet18":
            model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
        else:
            model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
        # replace classifier
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        return model

    if name == "resnet50_custom":
        from .resnet50 import build_resnet50 as _build_resnet50_custom

        return _build_resnet50_custom(num_classes=num_classes, in_channels=3)

    if name == "simple_cnn":
        # pretrained flag ignored for custom model
        from .simple_cnn import SimpleCNN

        return SimpleCNN(num_classes=num_classes, in_channels=3)

    raise ValueError(f"Unknown model name: {name}")
