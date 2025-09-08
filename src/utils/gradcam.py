import cv2
import numpy as np
import torch.nn.functional as F

# Try external pytorch-grad-cam first; fall back to local impl
try:
    from pytorch_grad_cam import GradCAM as _ExtGradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image as _ext_show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import (
        ClassifierOutputTarget as _ExtTarget,
    )

    _HAS_EXT = True
except Exception:
    _HAS_EXT = False


def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _get_cam_on_input_fallback(model, input_tensor, target_layer, class_idx=None):
    """Local Grad-CAM implementation used when pytorch-grad-cam is unavailable."""
    model.eval()

    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    def bwd_hook(_, grad_in, grad_out):
        # grad_out[0] corresponds to dL/dA
        gradients.append(grad_out[0].detach())

    # Register hooks on target layer
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        input_tensor = input_tensor.requires_grad_(True)
        logits = model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.detach().argmax(dim=1)[0].item())
        loss = logits[:, class_idx].sum()
        model.zero_grad(set_to_none=True)
        loss.backward()

        A = activations[-1]  # (B, C, h, w)
        dA = gradients[-1]  # (B, C, h, w)
        weights = dA.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (A * weights).sum(dim=1)  # (B, h, w)
        cam = F.relu(cam)
        cam_np = cam.detach().cpu().numpy()[0]
        cam_np = _normalize_minmax(cam_np)

        # Resize CAM to input size
        _, _, H, W = input_tensor.shape
        cam_resized = cv2.resize(
            cam_np.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
        )

        # Prepare input image (0..1, HxWxC)
        x = input_tensor.detach().cpu()[0]
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        x = x.permute(1, 2, 0).numpy().astype(np.float32)

        # Colorize CAM and overlay
        heatmap = (cam_resized * 255.0).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = (
            cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )
        overlay = 0.5 * heatmap_color + 0.5 * x
        overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

        return overlay, cam_resized.astype(np.float32)
    finally:
        # Clean up hooks
        try:
            h1.remove()
            h2.remove()
        except Exception:
            pass


def _get_cam_on_input_external(model, input_tensor, target_layer, class_idx=None):
    model.eval()
    cam = _ExtGradCAM(model=model, target_layers=[target_layer], use_cuda=False)
    targets = None if class_idx is None else [_ExtTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    x = input_tensor.detach().cpu()[0]
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    x = x.permute(1, 2, 0).numpy()
    overlay = _ext_show_cam_on_image(x, grayscale_cam[0], use_rgb=True)
    return overlay, grayscale_cam[0]


def get_cam_on_input(model, input_tensor, target_layer, class_idx=None):
    """
    Returns (overlay_rgb_uint8, grayscale_cam_float). Uses pytorch-grad-cam if
    available; otherwise falls back to a lightweight local implementation.
    """
    if _HAS_EXT:
        return _get_cam_on_input_external(model, input_tensor, target_layer, class_idx)
    return _get_cam_on_input_fallback(model, input_tensor, target_layer, class_idx)
