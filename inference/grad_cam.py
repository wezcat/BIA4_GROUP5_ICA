from __future__ import annotations
from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from models.base_interfaces import SupervisedModelBase

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    CAM_AVAILABLE = True
except Exception:
    GradCAM = GradCAMPlusPlus = ClassifierOutputTarget = None
    CAM_AVAILABLE = False


def grad_cam_with_layer(
    model: SupervisedModelBase,
    x: torch.Tensor,
    class_idx: Optional[int],
    layer_name: str
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap using pytorch-grad-cam.
    Supports both CNN-style models and ViT-style models.
    """
    if not CAM_AVAILABLE:
        raise RuntimeError(
            "pytorch-grad-cam is not installed. Run: pip install pytorch-grad-cam"
        )

    x = x.clone().detach().to(model.device)
    model.eval()

    # Resolve target layer
    target_layer = None

    if "vit" in model.__class__.__name__.lower():
        # ViT: use the norm layer inside a selected transformer block (more stable)
        vit_model = getattr(model, "model", model)
        blocks = getattr(vit_model, "blocks", None)
        if blocks is not None:
            try:
                block_idx = int(layer_name.split("_")[1])
                target_layer = blocks[block_idx].norm1
            except Exception:
                target_layer = blocks[-1].norm1
    else:
        # CNN: lookup by module name
        named_modules = dict(model.named_modules())
        target_layer = named_modules.get(layer_name)

    if target_layer is None:
        # If not found, fall back to a heuristic default layer
        target_layer = _find_default_target_layer(model)
        if target_layer is None:
            raise RuntimeError(
                f"Could not find a suitable CAM target layer. Model type: {model.__class__.__name__}"
            )

    # Select CAM method
    cam_class = GradCAMPlusPlus if "vit" in model.__class__.__name__.lower() else GradCAM

    # Create CAM object
    cam = cam_class(
        model=model,
        target_layers=[target_layer],
        # use_cuda=(cfg.device == "cuda")  # keep disabled if you manage device elsewhere
    )

    # Ensure batch dimension
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    # Determine target class
    if class_idx is None:
        with torch.no_grad():
            logits = model(x)
            class_idx = logits.argmax(dim=1).item()

    # Generate CAM
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=x, targets=targets)

    # Post-process
    cam_map = grayscale_cam[0]
    cam_map = cv2.resize(cam_map, (x.shape[2], x.shape[3]))
    cam_map = np.clip(cam_map, 0, 1)

    return cam_map


def _find_default_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """Heuristically find a reasonable default CAM target layer."""
    # Prefer common deep layers (e.g., layer4/features/down4)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            if ("layer4" in name) or ("features" in name) or ("down4" in name):
                return module

    # Otherwise, use the last Conv2d
    conv_modules = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if conv_modules:
        return conv_modules[-1]

    return None
