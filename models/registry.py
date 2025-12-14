import os
import glob
from typing import Optional, List, Tuple
import torch
from config import cfg
from models.base_interfaces import MachineLearningModelBase

# supervised models
from models.supervised.efficientnet_b0 import EfficientNetB0Classifier
from models.supervised.resnet50 import ResNet50
from models.supervised.unet import UNet
from models.supervised.vgg16 import VGG16Classifier
from models.supervised.small_cnn import SmallCNNClassifier
from models.supervised.inception_v3 import InceptionV3Classifier
from models.supervised.resnet34 import ResNet34Classifier
from models.supervised.vit_base16 import ViTBase16Classifier

# self-supervised
from models.selfsupervised.triplet_models import TripletResNet18

# ML models
from models.ml.effnet_xgboost import EffNetB0XGBoost
from models.ml.densenet201_catboost import DenseNet201CatBoost
from models.ml.voting_ensemble import VotingEnsembleML
from models.ml.rf_wrapper import RandomForestClassifierWrapper
from models.ml.svm_wrapper import SVMClassifierWrapper

MODEL_REGISTRY = {
    "supervised": {
        "effnet_b0": lambda dropout=None: EfficientNetB0Classifier(dropout=dropout),
        "resnet50": lambda dropout=None: ResNet50(dropout=dropout),
        "unet": lambda dropout=None: UNet(dropout=dropout),
        "vgg16": lambda dropout=None: VGG16Classifier(dropout=dropout),
        "inceptionv3": lambda dropout=None: InceptionV3Classifier(dropout=dropout),
        "resnet34": lambda dropout=None: ResNet34Classifier(dropout=dropout),
        "vit_base16": lambda dropout=None: ViTBase16Classifier(dropout=dropout),
        "small_cnn": lambda dropout=None: SmallCNNClassifier(dropout=dropout),
    },
    "selfsupervised": {
        "triplet_resnet18": lambda: TripletResNet18(),
    },
    "ml": {
        "effnetb0_xgb": lambda: EffNetB0XGBoost(),
        "densenet201_cat": lambda: DenseNet201CatBoost(),
        "voting_ensemble": lambda: VotingEnsembleML(),
        "handcrafted_RF": lambda: RandomForestClassifierWrapper(),
        "handcrafted_SVM": lambda: SVMClassifierWrapper(),
    },
}


def get_model_types() -> list:
    return list(MODEL_REGISTRY.keys())


def get_model_names(model_type: str) -> list:
    return list(MODEL_REGISTRY.get(model_type, {}).keys())


def create_model(
    model_type: str,
    model_name: str,
    dropout_rate: Optional[float] = None,
):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    factory = MODEL_REGISTRY[model_type].get(model_name)
    if factory is None:
        raise ValueError(f"Unknown model name: {model_name}")

    if model_type == "supervised" and dropout_rate is not None:
        try:
            model = factory(dropout=dropout_rate)
        except ValueError as e:
            # Some models explicitly do not support custom dropout
            if "does not support custom dropout" in str(e):
                raise e
            else:
                model = factory()
    else:
        model = factory()

    # Move torch models to device; ML models stay on CPU
    if not isinstance(model, MachineLearningModelBase):
        model = model.to(cfg.device)

    return model


def scan_weights() -> List[Tuple[str, str]]:
    items = []

    weight_pattern = os.path.join(cfg.weight_dir, "*.pth")
    pkl_pattern = os.path.join(cfg.weight_dir, "*.pkl")

    # Torch model checkpoints
    for p in glob.glob(weight_pattern):
        try:
            mtime = os.path.getmtime(p)
            ckpt = torch.load(p, map_location="cpu")

            model_type = ckpt.get("model_type", "unknown")
            model_name = ckpt.get("model_name", "unknown")
            acc = ckpt.get("acc", 0.0)
            best_tag = "[BEST]" if os.path.basename(p).startswith("[BEST]") else ""

            items.append(
                (
                    f"{best_tag}{model_type}/{model_name} | acc {acc:.5f} | {os.path.basename(p)}",
                    p,
                )
            )
        except Exception as e:
            items.append(
                (f"[CORRUPTED] {os.path.basename(p)} | {str(e)}", p)
            )

    # ML model files
    for p in glob.glob(pkl_pattern):
        items.append((f"ml/{os.path.basename(p)}", p))

    # Sort: BEST first, then lexicographically
    items.sort(key=lambda x: ("[BEST]" in x[0], x[0]), reverse=True)
    return items


def save_supervised_ckpt(model, optimizer, epoch, acc, params, is_best=False):
    ckpt = {
        "model_type": params["model_type"], "model_name": params["model_name"],
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "acc": acc
    }
    if "dropout_rate" in params and params["dropout_rate"] is not None:
        ckpt["dropout_rate"] = params["dropout_rate"]

    prefix = "[BEST]" if is_best else ""

    # 自动设置最佳CAM层
    if hasattr(model, "down4"):
        ckpt["best_cam_layer_name"] = "down4"
    elif hasattr(model, "backbone"):
        if hasattr(model.backbone, "layer4"):
            ckpt["best_cam_layer_name"] = "backbone.layer4"
        elif hasattr(model.backbone, "features"):
            ckpt["best_cam_layer_name"] = "backbone.features"
        elif hasattr(model.backbone, "Mixed_7c"):
            ckpt["best_cam_layer_name"] = "backbone.Mixed_7c"
    elif hasattr(model, "features"):
        ckpt["best_cam_layer_name"] = "features"
    elif "vit" in model.__class__.__name__.lower():
        ckpt["best_cam_layer_name"] = "block_11"  # ViT默认使用最后一个block

    if not is_best:
        return None  # 不保存普通 epoch

    # filename = f"{params['model_name']}_best.pth"
    filename = f"[BEST]{params['model_name']}.pth"
    path = os.path.join(cfg.weight_dir, filename)
    torch.save(ckpt, path)
    return path
