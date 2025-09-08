import base64
import json
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from src.datacheck.quality_gate import quality_gate
from src.datasets.knee_xray import _clahe_norm, _crop_with_margin, _square_pad
from src.models.factory import build_model
from src.serving.inference import predict_tta
from src.serving.policy_engine import decide_message
from src.train.calibrate import load_calibration
from src.utils.device import get_device
from src.utils.gradcam import get_cam_on_input

app = FastAPI(title="PatellaLuxation AI MVP")
DEVICE = get_device()
CKPT = "./checkpoints/resnet18_best.pt"
ckpt = torch.load(CKPT, map_location=DEVICE)
cfg = ckpt["config"]
model = build_model(cfg["model"]["name"], False, cfg["model"]["num_classes"]).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()


def preprocess_bytes(bytes_, input_size=224, bbox=None):
    arr = np.frombuffer(bytes_, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if bbox is not None:
        img = _crop_with_margin(img, bbox, margin=0.15)
    img = _square_pad(img, 0)
    img_resized = cv2.resize(img, (input_size, input_size))
    img_u8 = img_resized.copy()
    img = _clahe_norm(img_resized)
    img3 = np.stack([img, img, img], axis=0)
    return torch.tensor(img3, dtype=torch.float32).unsqueeze(0), img_u8


@app.post("/predict")
async def predict(file: UploadFile = File(...), bbox: str = Form(default=None)):
    try:
        bytes_ = await file.read()
        bbox_val = json.loads(bbox) if bbox else None
        x, gray_u8 = preprocess_bytes(
            bytes_, input_size=cfg["data"]["input_size"], bbox=bbox_val
        )
        x = x.to(DEVICE)
        # 품질 게이트 평가
        q = quality_gate(gray_u8, bbox=bbox_val)
        # 캘리브레이션 온도(존재하면 적용)
        T = load_calibration(os.path.join("./checkpoints", "calib.json"))
        # TTA 추론
        out = predict_tta(model, x, DEVICE, tta=4, temperature=T)
        probs_arr = out["probs"][0].cpu().numpy()
        sigma_arr = out["uncertainty_std"][0].cpu().numpy()
        pred = int(np.argmax(probs_arr))
        # 정책 결정 (클래스 1이 양성이라고 가정)
        prob_pos = (
            float(probs_arr[1]) if probs_arr.shape[0] > 1 else float(probs_arr[0])
        )
        sigma_pos = (
            float(sigma_arr[1]) if sigma_arr.shape[0] > 1 else float(sigma_arr[0])
        )
        policy = decide_message(
            prob_pos, sigma_pos, q["quality"], low_th=0.2, high_th=0.8
        )

        # Grad-CAM
        try:
            # 마지막 Conv2d 레이어를 CAM 타깃으로 선택
            import torch.nn as nn

            target_layer = None
            for m in reversed(list(model.modules())):
                if isinstance(m, nn.Conv2d):
                    target_layer = m
                    break
            if target_layer is None:
                raise RuntimeError("Grad-CAM을 위한 Conv2d 레이어를 찾지 못했습니다")
            overlay, _ = get_cam_on_input(model, x, target_layer, class_idx=pred)
            ok, buf = cv2.imencode(".png", overlay[:, :, ::-1])
            gradcam_b64 = base64.b64encode(buf).decode("utf-8") if ok else None
        except Exception:
            gradcam_b64 = None
        return {
            "pred": pred,
            "probs": probs_arr.tolist(),
            "sigma": float(sigma_pos),
            "ood_flag": bool(policy.get("ood_flag", False)),
            "message": policy.get("message", ""),
            "quality": q,
            "gradcam": gradcam_b64,
        }
    except Exception as e:
        return JSONResponse({"error": f"예측 처리 중 오류: {str(e)}"}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}
