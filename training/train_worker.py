import os
import time
import datetime
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader

from PyQt5 import QtCore  # or PySide6: from PySide6 import QtCore
from config import cfg
from gui.signals import WorkerSignals          # 你定义的 WorkerSignals 在哪就从哪 import
from models.registry import create_model          # create_model 在哪就从哪 import
from models.registry import save_supervised_ckpt
from gui.state import global_state             # GlobalState 实例
from dataset import BreastHistDataset, MLFeatureDataset, TripletHistDataset, get_supervised_transforms, get_selfsup_transform

class TrainWorker(QtCore.QThread):
    def __init__(self, params: dict, signals: WorkerSignals):
        super().__init__()
        self.params = params
        self.signals = signals

    def run(self):
        try:
            if self.params["model_type"] == "supervised":
                self._run_supervised_with_error_handling()
            elif self.params["model_type"] == "selfsupervised":
                self._run_selfsupervised_with_error_handling()
            elif self.params["model_type"] == "ml":
                self._run_ml_with_error_handling()
        except Exception:
            error_msg = f"A fatal error occurred during training:\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                self.signals.log.emit(f"Error while clearing GPU memory: {e}")
            self.signals.finished.emit()

    def _run_supervised_with_error_handling(self):
        try:
            dropout_rate = self.params.get("dropout_rate")
            model = create_model(
                self.params["model_type"],
                self.params["model_name"],
                dropout_rate=dropout_rate
            )
        except ValueError as e:
            if "does not support custom dropout value" in str(e):
                self.signals.log.emit(f"⚠️ Warning: {e}. Falling back to the model default.")
                model = create_model(self.params["model_type"], self.params["model_name"])
            else:
                raise

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.params["lr"])
        scaler = amp.GradScaler() if cfg.device == "cuda" else None
        start_epoch = 1
        best_acc = 0.0

        if self.params.get("resume_path"):
            try:
                ckpt = torch.load(self.params["resume_path"], map_location=cfg.device)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_acc = ckpt.get("acc", 0.0)
                self.signals.log.emit(f"✅ Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
            except Exception as e:
                self.signals.error.emit(f"Failed to load checkpoint: {e}\n{traceback.format_exc()}")
                return

        try:
            train_ds = BreastHistDataset(
                self.params["data_root"],
                "train",
                get_supervised_transforms(True, self.params["data_root"])
            )
            val_ds = BreastHistDataset(
                self.params["data_root"],
                "test",
                get_supervised_transforms(False, self.params["data_root"])
            )
        except Exception:
            self.signals.error.emit(
                f"Failed to create dataset. Please check the data path and files:\n{traceback.format_exc()}"
            )
            return

        train_loader = DataLoader(
            train_ds,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Record training start time for total-time estimation
        train_start_time = time.time()
        epoch_times = []

        for epoch in range(start_epoch, self.params["max_epoch"] + 1):
            if global_state.train_stop:
                status = "⏹️ Stopped"
                self.signals.train_progress.emit(epoch, self.params["max_epoch"], status, "")
                self.signals.log.emit("⏹️ Training was stopped by the user.")
                break

            epoch_start_time = time.time()

            try:
                model.train()
                running_loss = 0.0

                for i, (x, y, _) in enumerate(train_loader, 1):
                    if global_state.train_stop:
                        break

                    x, y = x.to(cfg.device), y.to(cfg.device)
                    optimizer.zero_grad()

                    if scaler:
                        with amp.autocast():
                            out = model(x)
                            loss = criterion(out, y)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        out = model(x)
                        loss = criterion(out, y)
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()

                if global_state.train_stop:
                    status = "⏹️ Stopped"
                    time_info = ""
                    self.signals.train_progress.emit(epoch, self.params["max_epoch"], status, time_info)
                    break

                model.eval()
                correct = 0
                total = 0
                val_loss = 0.0

                with torch.no_grad():
                    for x, y, _ in val_loader:
                        if global_state.train_stop:
                            break
                        x, y = x.to(cfg.device), y.to(cfg.device)
                        out = model(x)
                        loss = criterion(out, y)
                        val_loss += loss.item()
                        pred = out.argmax(dim=1)
                        correct += (pred == y).sum().item()
                        total += y.size(0)

                if global_state.train_stop:
                    status = "⏹️ Stopped"
                    time_info = ""
                    self.signals.train_progress.emit(epoch, self.params["max_epoch"], status, time_info)
                    break

                # Compute epoch time and estimated remaining time
                epoch_time = time.time() - epoch_start_time
                epoch_times.append(epoch_time)
                avg_epoch_time = np.mean(epoch_times[-5:])  # average over the last 5 epochs

                remaining_epochs = self.params["max_epoch"] - epoch
                estimated_remaining_time = remaining_epochs * avg_epoch_time

                # Format time info
                time_info = f"This epoch: {epoch_time:.1f}s | Remaining: {self._format_time(estimated_remaining_time)}"

                # Normal training status
                status = f"Training ({epoch}/{self.params['max_epoch']})"
                self.signals.train_progress.emit(epoch, self.params["max_epoch"], status, time_info)

                acc = correct / total if total > 0 else 0.0
                self.signals.log.emit(
                    f"📊 Epoch {epoch:3d} | TrainLoss {running_loss/len(train_loader):.4f} | ValAcc {acc:.5f}"
                )

                if epoch % 10 == 0 or epoch == self.params["max_epoch"] or acc > best_acc:
                    try:
                        is_best = acc > best_acc
                        save_path = save_supervised_ckpt(model, optimizer, epoch, acc, self.params, is_best)
                        if is_best:
                            best_acc = acc
                            self.signals.log.emit(
                                f"🏆 New best accuracy! {best_acc:.5f}. Saved [BEST] checkpoint to: {save_path}"
                            )
                        else:
                            self.signals.log.emit(f"💾 Checkpoint saved to: {save_path}")
                    except Exception as e:
                        self.signals.log.emit(f"⚠️ Failed to save checkpoint: {e}")

            except Exception:
                status = "❌ Error"
                time_info = ""
                self.signals.train_progress.emit(epoch, self.params["max_epoch"], status, time_info)
                self.signals.error.emit(f"Error at epoch {epoch}:\n{traceback.format_exc()}")
                self.signals.log.emit("⚠️ This epoch failed. Continuing to the next epoch...")
                continue

        if not global_state.train_stop:
            status = "✅ Done"
            total_time = time.time() - train_start_time
            time_info = f"Total time: {self._format_time(total_time)}"
            self.signals.train_progress.emit(self.params["max_epoch"], self.params["max_epoch"], status, time_info)
            self.signals.log.emit(f"✅ Supervised training finished! Best validation accuracy: {best_acc:.5f}")

    def _format_time(self, seconds):
        """Format a duration in seconds into a readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _run_selfsupervised_with_error_handling(self):
        # Self-supervised training code stays the same...
        try:
            model = create_model("selfsupervised", self.params["model_name"])
            optimizer = optim.Adam(model.parameters(), lr=self.params["lr"])
            scaler = amp.GradScaler() if cfg.device == "cuda" else None

            try:
                train_ds = TripletHistDataset(self.params["data_root"], "train", get_selfsup_transform())
                train_loader = DataLoader(train_ds, batch_size=self.params["batch_size"], shuffle=True, num_workers=4)
            except Exception:
                self.signals.error.emit(
                    f"Failed to create self-supervised dataset:\n{traceback.format_exc()}"
                )
                return

            best_acc = 0.0
            for epoch in range(1, self.params["max_epoch"] + 1):
                if global_state.train_stop:
                    self.signals.log.emit("⏹️ Self-supervised training was stopped.")
                    break

                try:
                    model.train()
                    total_loss = 0.0
                    total_batches = len(train_loader)

                    for batch_idx, (a, p, n, _, _) in enumerate(train_loader):
                        if global_state.train_stop:
                            break

                        a, p, n = a.to(cfg.device), p.to(cfg.device), n.to(cfg.device)
                        optimizer.zero_grad()

                        if scaler:
                            with amp.autocast():
                                loss, _, _, _ = model.forward_triplet(a, p, n)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss, _, _, _ = model.forward_triplet(a, p, n)
                            loss.backward()
                            optimizer.step()

                        total_loss += loss.item()
                        progress = int((batch_idx + 1) / total_batches * 100)
                        # Self-supervised: update batch-level progress only
                        self.signals.train_progress.emit(
                            epoch, self.params["max_epoch"], f"SelfSup Epoch {epoch}", f"Progress: {progress}%"
                        )

                    if global_state.train_stop:
                        break

                    self.signals.log.emit(f"[SelfSup] Epoch {epoch} TripletLoss: {total_loss/len(train_loader):.4f}")

                    if epoch % 10 == 0 or epoch == self.params["max_epoch"]:
                        backbone = model.get_backbone()
                        linear = nn.Linear(
                            backbone.fc.in_features if hasattr(backbone, "fc") else 512, 2
                        ).to(cfg.device)

                        linear.train()
                        opt = optim.Adam(linear.parameters(), lr=1e-3)

                        sup_train = BreastHistDataset(
                            self.params["data_root"], "train",
                            get_supervised_transforms(False, self.params["data_root"])
                        )
                        sup_val = BreastHistDataset(
                            self.params["data_root"], "test",
                            get_supervised_transforms(False, self.params["data_root"])
                        )

                        loader = DataLoader(sup_train, batch_size=32, shuffle=True)
                        for _ in range(5):
                            for x, y, _ in loader:
                                if global_state.train_stop:
                                    break
                                x, y = x.to(cfg.device), y.to(cfg.device)
                                with torch.no_grad():
                                    feat = backbone(x)
                                out = linear(feat.view(feat.size(0), -1))
                                loss = F.cross_entropy(out, y)
                                opt.zero_grad()
                                loss.backward()
                                opt.step()

                        if global_state.train_stop:
                            break

                        linear.eval()
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for x, y, _ in DataLoader(sup_val, batch_size=32):
                                if global_state.train_stop:
                                    break
                                x = x.to(cfg.device)
                                feat = backbone(x)
                                pred = linear(feat.view(feat.size(0), -1)).argmax(1)
                                correct += (pred.cpu() == y).sum().item()
                                total += y.size(0)

                        acc = correct / total
                        self.signals.log.emit(f"📊 Linear probe accuracy: {acc:.5f}")

                        if acc > best_acc:
                            best_acc = acc
                            save_path = os.path.join(
                                cfg.weight_dir, f"[BEST]{self.params['model_name']}_selfsup.pth"
                            )
                            torch.save(
                                {
                                    "model_type": "selfsupervised",
                                    "model_name": self.params["model_name"],
                                    "backbone_state_dict": backbone.state_dict(),
                                    "linear_state_dict": linear.state_dict(),
                                    "epoch": epoch,
                                    "acc": acc,
                                },
                                save_path,
                            )
                            self.signals.log.emit(
                                f"🏆 New best linear probe accuracy! {best_acc:.5f}. Saved to: {save_path}"
                            )

                except Exception:
                    self.signals.error.emit(f"Error at self-supervised epoch {epoch}:\n{traceback.format_exc()}")
                    self.signals.log.emit("⚠️ This epoch failed. Continuing to the next epoch...")
                    continue

            if not global_state.train_stop:
                self.signals.log.emit(
                    f"✅ Self-supervised training finished! Best linear eval accuracy: {best_acc:.5f}"
                )
            self.signals.train_progress.emit(self.params["max_epoch"], self.params["max_epoch"], "✅ Done", "")

        except Exception as e:
            raise RuntimeError(f"Self-supervised training init failed: {str(e)}\n{traceback.format_exc()}")

    def _run_ml_with_error_handling(self):
        try:
            self.signals.log.emit(
                "Starting ML model training (deep feature extraction + ImageNet normalization)..."
            )
            feature_extractor_type = self.params["model_name"].split("_")[0]
            self.signals.log.emit(
                f"Feature extractor: {feature_extractor_type} | Using ImageNet normalization"
            )

            try:
                train_ds = MLFeatureDataset(self.params["data_root"], "train", feature_extractor_type)
                val_ds = MLFeatureDataset(self.params["data_root"], "test", feature_extractor_type)
            except Exception:
                self.signals.error.emit(f"Failed to create ML dataset:\n{traceback.format_exc()}")
                return

            try:
                X_train = np.stack([x for x, _, _ in train_ds])
                y_train = np.array([y for _, y, _ in train_ds])
                X_val = np.stack([x for x, _, _ in val_ds]) if len(val_ds) > 0 else None
                y_val = np.array([y for _, y, _ in val_ds]) if len(val_ds) > 0 else None
            except Exception:
                self.signals.error.emit(f"Failed to prepare ML data:\n{traceback.format_exc()}")
                return

            try:
                model = create_model("ml", self.params["model_name"])
                self.signals.log.emit(f"Training {self.params['model_name']} ...")
                model.fit(X_train, y_train)
            except Exception:
                self.signals.error.emit(f"ML model training failed:\n{traceback.format_exc()}")
                return

            try:
                acc = 0.0
                if X_val is not None:
                    pred = model.predict(X_val)
                    acc = (pred == y_val).mean()

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(cfg.weight_dir, f"ml_{self.params['model_name']}_{timestamp}.pkl")
                model.save(save_path)

                self.signals.log.emit(f"✅ ML training finished! Validation accuracy: {acc:.5f}")
                self.signals.log.emit(f"💾 Model saved to: {save_path}")
                self.signals.train_progress.emit(100, 100, "✅ ML Done", "")

            except Exception:
                self.signals.error.emit(f"ML evaluation or saving failed:\n{traceback.format_exc()}")

        except Exception as e:
            raise RuntimeError(f"ML training init failed: {str(e)}\n{traceback.format_exc()}")
