import os, sys, json, torch, cv2, numpy as np
from os.path import abspath, dirname, join

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, abspath(join(dirname(__file__), "..")))
from src.utils.device import get_device
from src.models.factory import build_model
from src.datasets.knee_xray import _square_pad, _crop_with_margin, _clahe_norm
from src.train.calibrate import load_calibration
from src.serving.inference import predict_tta


def preprocess(img_path, input_size=224, bbox=None):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if bbox is not None:
        img = _crop_with_margin(img, bbox, margin=0.15)
    img = _square_pad(img, 0)
    img = cv2.resize(img, (input_size, input_size))
    img = _clahe_norm(img)
    img3 = np.stack([img, img, img], axis=0)
    return torch.tensor(img3, dtype=torch.float32).unsqueeze(0)


if __name__ == "__main__":
    device = get_device()
    ckpt = torch.load("./checkpoints/resnet18_best.pt", map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg["model"]["name"], False, cfg["model"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    x = preprocess("./data/example.jpg", input_size=cfg["data"]["input_size"])
    T = load_calibration(os.path.join("./checkpoints", "calib.json"))
    out = predict_tta(model, x, device, tta=4, temperature=T)
    probs = out["probs"][0].numpy()
    sigma = out["uncertainty_std"][0].numpy()
    print("Prediction:", int(probs.argmax()), "Probs:", probs, "Sigma:", sigma)
