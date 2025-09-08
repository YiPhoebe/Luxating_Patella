import json
import os
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# JSON 항목 예시:
# {
#   "image_path": "images/xxx.jpg",
#   "dog_id": "A001",
#   "label": 0,
#   "bbox": [120,80,360,320]  # [x1,y1,x2,y2] 형식 (선택)
# }


def _square_pad(img: np.ndarray, value=0):
    h, w = img.shape[:2]
    side = max(h, w)
    pad = np.full((side, side), value, dtype=img.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    pad[y0 : y0 + h, x0 : x0 + w] = img
    return pad


def _crop_with_margin(img: np.ndarray, bbox, margin=0.15):
    if bbox is None:
        return img
    h, w = img.shape[:2]
    if len(bbox) == 4:
        x, y, a, b = bbox
        # bbox 포맷 추정: [x,y,w,h] vs [x1,y1,x2,y2]
        if a > 1 and b > 1 and (x + a) <= w + 5 and (y + b) <= h + 5:
            x1, y1, x2, y2 = x, y, x + a, y + b
        else:
            x1, y1, x2, y2 = x, y, a, b
    else:
        x1, y1, x2, y2 = bbox
    bw, bh = float(x2 - x1), float(y2 - y1)
    mx = int(round(margin * bw))
    my = int(round(margin * bh))
    X1 = int(max(0, round(x1 - mx)))
    Y1 = int(max(0, round(y1 - my)))
    X2 = int(min(w, round(x2 + mx)))
    Y2 = int(min(h, round(y2 + my)))
    # Ensure valid box
    if X2 <= X1 or Y2 <= Y1:
        return img
    return img[Y1:Y2, X1:X2]


def _clahe_norm(gray: np.ndarray):
    if gray.dtype != np.uint8:
        g = gray.astype(np.float32)
        g = 255 * (g - g.min()) / (g.ptp() + 1e-6)
        gray = g.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    eq = eq.astype(np.float32)
    eq = (eq - eq.mean()) / (eq.std() + 1e-6)
    return eq


class KneeXrayDataset(Dataset):
    def __init__(
        self,
        json_files: List[str],
        image_root: str,
        input_size=224,
        use_bbox=True,
        train=True,
        transforms=None,
    ):
        self.items = []
        self.image_root = image_root
        self.use_bbox = use_bbox
        self.input_size = input_size
        self.train = train
        self.transforms = transforms
        for jf in json_files:
            if not os.path.exists(jf):
                continue
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else data.get("images", [])
            for e in entries:
                self.items.append(e)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        e = self.items[idx]
        path = os.path.join(self.image_root, e["image_path"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"Image not found: {path}"
        if self.use_bbox and "bbox" in e and e["bbox"] is not None:
            img = _crop_with_margin(img, e["bbox"], margin=0.15)
        img = _square_pad(img, value=0)
        img = cv2.resize(
            img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA
        )
        img = _clahe_norm(img)
        if self.transforms:

            aug = self.transforms(image=np.stack([img, img, img], axis=-1))
            img3 = aug["image"]  # usually torch.Tensor(C,H,W)
        else:
            img3 = np.stack([img, img, img], axis=0)  # (C,H,W) float32 later
        # Convert to torch tensor without unnecessary copy
        if isinstance(img3, torch.Tensor):
            x = img3.to(torch.float32)
        else:
            x = torch.from_numpy(img3).to(torch.float32)
        y = torch.tensor(int(e["label"]), dtype=torch.long)
        return x, y
