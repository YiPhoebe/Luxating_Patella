import json
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(init_temperature)).log())

    @property
    def T(self) -> torch.Tensor:
        return self.log_t.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T


@torch.no_grad()
def _gather_logits_labels(
    model: nn.Module, dataloader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all = []
    labels_all = []
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def fit_temperature_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200
) -> float:
    device = logits.device
    scaler = TemperatureScaler().to(device)

    def nll_criterion(lg: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(lg, lb)

    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.T.detach().cpu().item())


def fit_temperature(
    model: nn.Module, val_loader, device: torch.device, max_iter: int = 200
) -> float:
    logits, labels = _gather_logits_labels(model, val_loader, device)
    return fit_temperature_from_logits(logits, labels, max_iter=max_iter)


def save_calibration(temperature: float, path: str):
    data = {
        "temperature": float(temperature),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_calibration(path: str) -> float:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return float(data.get("temperature", 1.0))
    except Exception:
        return 1.0
