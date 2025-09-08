from typing import Dict

import torch
import torch.nn.functional as F


def _apply_tta(x: torch.Tensor, tta_id: int) -> torch.Tensor:
    # x: (B, C, H, W)
    # 간단한 TTA 변환(수평플립/밝기 조절 등)
    if tta_id == 0:
        return x
    if tta_id == 1:
        return x.flip(-1)
    if tta_id == 2:
        return x * 0.95
    if tta_id == 3:
        return x * 1.05
    return x


@torch.no_grad()
def predict_tta(
    model, x: torch.Tensor, device: torch.device, tta: int = 4, temperature: float = 1.0
) -> Dict:
    model.eval()
    probs_all = []
    logits_all = []
    for i in range(tta):
        xi = _apply_tta(x, i).to(device)
        logits = model(xi)
        logits = logits / float(max(temperature, 1e-6))
        prob = F.softmax(logits, dim=1)
        logits_all.append(logits.detach().cpu())
        probs_all.append(prob.detach().cpu())
    probs = torch.stack(probs_all, dim=0).mean(dim=0)  # (B, C)
    # 불확실성: TTA 간 확률 표준편차(클래스별)
    stds = torch.stack(probs_all, dim=0).std(dim=0)  # (B, C)
    return {
        "probs": probs,
        "uncertainty_std": stds,
        "logits": torch.stack(logits_all, dim=0).mean(dim=0),
    }
