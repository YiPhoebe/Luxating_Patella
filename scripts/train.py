import contextlib
import os
import sys
import time
from os.path import abspath, dirname, join

import torch
import torch.nn as nn
import yaml

try:
    import cv2

    cv2.setNumThreads(1)
except Exception:
    pass

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, abspath(join(dirname(__file__), "..")))
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.knee_xray import KneeXrayDataset
from src.models.factory import build_model
from src.train.calibrate import fit_temperature_from_logits, save_calibration
from src.utils.device import get_device
from src.utils.metrics import accuracy, precision_recall_f1
from src.utils.seed import set_seed


def build_transforms(input_size, train=True):
    if train:
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=0.05,
                    rotate_limit=7,
                    border_mode=0,
                    p=0.7,
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.5
                ),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)), ToTensorV2()]
        )


def main(cfg_path="./config/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"[Device] {device}")

    train_ds = KneeXrayDataset(
        cfg["data"]["train_jsons"],
        cfg["data"]["image_root"],
        input_size=cfg["data"]["input_size"],
        use_bbox=cfg["data"]["use_bbox"],
        train=True,
        transforms=build_transforms(cfg["data"]["input_size"], True),
    )
    val_ds = KneeXrayDataset(
        cfg["data"]["val_jsons"],
        cfg["data"]["image_root"],
        input_size=cfg["data"]["input_size"],
        use_bbox=cfg["data"]["use_bbox"],
        train=False,
        transforms=build_transforms(cfg["data"]["input_size"], False),
    )
    # DataLoader performance tuning for MPS/CPU
    num_workers = 4
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=False,
    )

    model = build_model(
        cfg["model"]["name"], cfg["model"]["pretrained"], cfg["model"]["num_classes"]
    ).to(device)
    # (Optional) channels_last can help, but may break models using .view(); keep disabled for stability
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_f1 = -1
    patience = 0
    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    use_amp = device.type == "mps"

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        t0 = time.time()
        n_samples = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x, y = x.to(device), y.to(device)
            ctx = (
                torch.autocast("mps", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with ctx:
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_samples += x.size(0)
        dur = time.time() - t0
        if dur > 0:
            print(f"[Epoch {epoch}] train throughput: {n_samples/dur:.2f} img/s")

        # 검증 단계
        model.eval()
        accs = []
        precs = []
        recs = []
        f1s = []
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [valid]"):
                x, y = x.to(device), y.to(device)
                ctx = (
                    torch.autocast("mps", dtype=torch.float16)
                    if use_amp
                    else contextlib.nullcontext()
                )
                with ctx:
                    logits = model(x)
                all_logits.append(logits.detach().cpu())
                all_labels.append(y.detach().cpu())
                accs.append(accuracy(logits, y))
                p, r, f1 = precision_recall_f1(logits, y, positive_class=1)
                precs.append(p)
                recs.append(r)
                f1s.append(f1)
        f1 = sum(f1s) / len(f1s)
        scheduler.step(f1)
        print(f"[Epoch {epoch}] F1:{f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            torch.save(
                {"model": model.state_dict(), "config": cfg},
                os.path.join(cfg["train"]["save_dir"], "resnet18_best.pt"),
            )
            # 검증 세트 기준 Temperature Scaling 적용 및 저장
            try:
                logits_cat = torch.cat(all_logits, dim=0)
                labels_cat = torch.cat(all_labels, dim=0)
                T = fit_temperature_from_logits(logits_cat, labels_cat)
                save_calibration(
                    T, os.path.join(cfg["train"]["save_dir"], "calib.json")
                )
                print(f"[Calibration] Temperature updated: T={T:.3f}")
            except Exception as e:
                print(f"[Calibration] 오류로 건너뜀: {e}")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()
