import os
import glob
import traceback
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from PyQt5 import QtCore   # or PySide6: from PySide6 import QtCore
from config import cfg
from gui.state import global_state
from models.registry import create_model, MODEL_REGISTRY
from dataset import get_supervised_transforms, get_ml_transform
from dataset.handcrafted import extract_handcrafted_features_vector
from inference.grad_cam import grad_cam_with_layer
from inference.pseudo_nuclear import pseudo_nuclear_analysis
from inference.report_html import generate_professional_html_report

class InferenceWorker(QtCore.QThread):
    def __init__(self, selected_labels, weight_map, input_path, upload_pil, percentile, signals, data_root):
        super().__init__()
        self.selected_labels = selected_labels
        self.weight_map = weight_map
        self.input_path = input_path
        self.upload_pil = upload_pil
        self.percentile = percentile
        self.signals = signals
        self.data_root = data_root

    def run(self):
        try:
            global_state.reset()
            img_paths = self._get_image_paths()
            if not img_paths:
                self.signals.log.emit("No image files were found.")
                self.signals.finished.emit()
                return

            for img_idx, (img_path, original_name) in enumerate(img_paths):
                if global_state.infer_stop:
                    self.signals.log.emit("Inference was stopped by the user.")
                    break
                self.signals.log.emit(
                    f"Start processing image ({img_idx+1}/{len(img_paths)}): {original_name}"
                )
                self._process_single_image(img_path, original_name)

            if not global_state.infer_stop:
                self.signals.log.emit("All images have been processed.")
            self.signals.finished.emit()

        except Exception:
            error_msg = f"A fatal error occurred during inference:\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                self.signals.log.emit(f"Error while clearing GPU memory: {e}")

    def _get_image_paths(self):
        """Get image paths along with original file names."""
        try:
            if self.upload_pil:
                tmp_path = os.path.join(cfg.temp_dir, "upload_temp.png")
                self.upload_pil.save(tmp_path)
                # Use the input path to infer original file name
                original_name = os.path.basename(self.input_path) if self.input_path else "uploaded_image.png"
                return [(tmp_path, original_name)]

            elif os.path.isdir(self.input_path):
                paths = []
                for ext in ["*.png", "*.jpg", "*.jpeg"]:
                    for path in glob.glob(os.path.join(self.input_path, ext)):
                        original_name = os.path.basename(path)
                        paths.append((path, original_name))
                return sorted(paths)

            elif os.path.isfile(self.input_path) and self.input_path.lower().endswith((".png", ".jpg", ".jpeg")):
                original_name = os.path.basename(self.input_path)
                return [(self.input_path, original_name)]

            return []

        except Exception:
            self.signals.error.emit(f"Error while collecting image paths:\n{traceback.format_exc()}")
            return []

    def _process_single_image(self, img_path, original_filename):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            individual_results = []
            cams = []

            for label_idx, label in enumerate(self.selected_labels):
                if global_state.infer_stop:
                    self.signals.log.emit("Stopped. Cleaning up...")
                    break

                weight_path = self.weight_map[label]
                self.signals.log.emit(
                    f"  Using model {label_idx+1}/{len(self.selected_labels)}: {label}"
                )

                try:
                    if weight_path.endswith(".pth"):
                        self._process_pth_model(label, weight_path, pil_img, individual_results, cams)
                    elif weight_path.endswith(".pkl"):
                        self._process_pkl_model(label, weight_path, pil_img, individual_results)
                except Exception as e:
                    self.signals.log.emit(f"  Model {label} failed: {str(e)}")
                    continue

                if global_state.infer_stop:
                    break

            if individual_results:
                mean_prob = np.mean([r["prob"] for r in individual_results])
                ensemble_pred = "Malignant" if mean_prob > 50 else "Benign"
                mean_cam = np.mean(cams, axis=0) if cams else np.zeros((cfg.input_size, cfg.input_size))

                report_path = generate_professional_html_report(
                    cfg.html_report_dir,
                    img_path,
                    original_filename,
                    individual_results,
                    mean_prob,
                    ensemble_pred,
                    pseudo_nuclear_analysis(mean_cam, self.percentile),
                    mean_cam
                )
                self.signals.log.emit(f"  Report generated: {report_path}")
            else:
                self.signals.log.emit("  Warning: No models were successfully processed. No report was generated.")

        except Exception:
            self.signals.error.emit(
                f"Error while processing image {original_filename}:\n{traceback.format_exc()}"
            )

    def _process_pth_model(self, label, weight_path, pil_img, results, cams):
        try:
            transform = get_supervised_transforms(False, root_dir=self.data_root)
            tensor = transform(pil_img).unsqueeze(0).to(cfg.device)

            ckpt = torch.load(weight_path, map_location=cfg.device)
            mt = ckpt["model_type"]
            mn = ckpt["model_name"]

            if mt == "selfsupervised":
                backbone = create_model("selfsupervised", mn).get_backbone()
                backbone.load_state_dict(ckpt["backbone_state_dict"])
                linear = nn.Linear(backbone.fc.in_features if hasattr(backbone, "fc") else 512, 2).to(cfg.device)
                linear.load_state_dict(ckpt["linear_state_dict"])
                model = nn.Sequential(backbone, linear).to(cfg.device)
            else:
                model = create_model(mt, mn)
                model.load_state_dict(ckpt["model_state_dict"])

            model.eval()
            with torch.no_grad():
                logits = model(tensor)
                prob = torch.softmax(logits, dim=1)[0]
                prob_mal = prob[1].item() * 100

            results.append({
                "model": f"{mt}/{mn}",
                "acc": ckpt.get("acc", 0.0),
                "pred": "Malignant" if prob_mal > 50 else "Benign",
                "prob": prob_mal
            })

            layer_name = ckpt.get("best_cam_layer_name")
            if layer_name:
                try:
                    cam = grad_cam_with_layer(model, tensor, None, layer_name)
                    cams.append(cam)
                except Exception as e:
                    self.signals.log.emit(f"    Grad-CAM failed ({mn}): {e}")

            del model, logits, prob, tensor

        except Exception as e:
            raise RuntimeError(f"PTH model processing failed {os.path.basename(weight_path)}: {str(e)}")

    def _process_pkl_model(self, label, weight_path, pil_img, results):
        try:
            # Extract model name from the file path
            filename = os.path.basename(weight_path)

            # Validate file name format
            if (not filename.startswith("ml_")) or (not filename.endswith(".pkl")):
                raise ValueError(
                    f"Invalid ML weight filename: {filename}. Expected: ml_{{model_name}}_YYYYMMDD_HHMMSS.pkl"
                )

            # Remove prefix "ml_" and suffix ".pkl"
            # Example: ml_handcrafted_RF_20251212_114918.pkl -> handcrafted_RF_20251212_114918
            filename = filename[3:-4]

            # Split by "_" and remove timestamp parts (last two items: YYYYMMDD and HHMMSS)
            parts = filename.split("_")
            if len(parts) < 3:
                raise ValueError(f"Invalid ML weight filename. Cannot parse model name from: {filename}")

            model_name_parts = parts[:-2]
            model_name = "_".join(model_name_parts)

            # Validate model name in registry
            if model_name not in MODEL_REGISTRY["ml"]:
                raise ValueError(
                    f"Unknown ML model name: '{model_name}'. Available: {list(MODEL_REGISTRY['ml'].keys())}\n"
                    f"Please ensure filename follows: ml_{{model_name}}_YYYYMMDD_HHMMSS.pkl"
                )

            # Choose correct feature extraction path
            if "handcrafted" in model_name:
                # Handcrafted features: extract directly from the original PIL image
                img_np = np.array(pil_img)
                feat = extract_handcrafted_features_vector(img_np).reshape(1, -1)
            else:
                # Deep-feature ML models: apply normalization transform then flatten
                ml_transform = get_ml_transform(model_name)
                feat = ml_transform(pil_img).numpy().reshape(1, -1)

            # Load model and predict
            model = create_model("ml", model_name)
            model.load(weight_path)
            proba = model.predict_proba(feat)[0]
            prob_mal = proba[1] * 100

            results.append({
                "model": f"ml/{label}",
                "acc": 0.0,
                "pred": "Malignant" if prob_mal > 50 else "Benign",
                "prob": prob_mal
            })

            del model, feat, proba

        except Exception as e:
            raise RuntimeError(f"PKL model processing failed {os.path.basename(weight_path)}: {str(e)}")
