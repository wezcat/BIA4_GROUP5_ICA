import os
import traceback
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
import torch
from config import cfg
from gui.signals import WorkerSignals
from gui.state import global_state
from gui.logging import append_log_with_limit
from models.registry import (
    create_model,
    MODEL_REGISTRY,
    save_supervised_ckpt,
    get_model_types,
    get_model_names,
    scan_weights,
)
from dataset import (
    BreastHistDataset,
    MLFeatureDataset,
    TripletHistDataset,
    get_supervised_transforms,
    get_ml_transform,
    get_selfsup_transform,
)
from training.train_worker import TrainWorker
from training.inference_worker import InferenceWorker


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BreastCAD-Pro")
        self.resize(1800, 1000)
        self.weight_map = {};
        self.uploaded_pil = None

        self.build_ui()

        self.train_signals = WorkerSignals()
        self.infer_signals = WorkerSignals()

        self.train_signals.log.connect(lambda s: append_log_with_limit(self.train_log, s))
        self.train_signals.error.connect(self.on_train_error)
        self.train_signals.train_progress.connect(self.update_train_progress)  # Connecting to new signal
        self.train_signals.finished.connect(self.on_train_finished)

        self.infer_signals.log.connect(lambda s: append_log_with_limit(self.infer_log, s))
        self.infer_signals.error.connect(self.on_infer_error)
        self.infer_signals.finished.connect(self.on_infer_finished)

        self.refresh_weights()

    def build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        title = QtWidgets.QLabel("Training Console")
        title.setStyleSheet("font-size:18pt;font-weight:bold;color:#1976d2;")
        left_layout.addWidget(title)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Dataset root:"))
        self.data_root_edit = QtWidgets.QLineEdit(cfg.data_root)
        self.data_root_edit.setPlaceholderText("./data/BreaKHis_400X")
        h.addWidget(self.data_root_edit, 3)
        btn = QtWidgets.QPushButton("Browse")
        btn.clicked.connect(self.browse_data_root)
        h.addWidget(btn)
        left_layout.addLayout(h)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Model Type:"))
        self.model_type_combo = QtWidgets.QComboBox()
        self.model_type_combo.addItems(get_model_types())
        self.model_type_combo.currentTextChanged.connect(self.update_model_name_combo)
        h.addWidget(self.model_type_combo)
        h.addWidget(QtWidgets.QLabel("Model Name:"))
        self.model_name_combo = QtWidgets.QComboBox()
        self.update_model_name_combo()
        h.addWidget(self.model_name_combo)
        left_layout.addLayout(h)

        gb_param = QtWidgets.QGroupBox("Training Hyperparameters")
        grid = QtWidgets.QGridLayout()

        grid.addWidget(QtWidgets.QLabel("Learning Rate:"), 0, 0)
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(1e-8, 1.0);
        self.lr_spin.setDecimals(6);
        self.lr_spin.setValue(3e-5);
        self.lr_spin.setSingleStep(1e-5)
        grid.addWidget(self.lr_spin, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Batch Size:"), 0, 2)
        self.bs_spin = QtWidgets.QSpinBox()
        self.bs_spin.setRange(1, 128);
        self.bs_spin.setValue(16)
        grid.addWidget(self.bs_spin, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Epoch:"), 1, 0)
        self.epoch_spin = QtWidgets.QSpinBox()
        self.epoch_spin.setRange(1, 2000);
        self.epoch_spin.setValue(50)
        grid.addWidget(self.epoch_spin, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Dropout:"), 1, 2)
        self.dropout_spin = QtWidgets.QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9);
        self.dropout_spin.setDecimals(2);
        self.dropout_spin.setValue(0.0)
        self.dropout_spin.setSingleStep(0.05);
        self.dropout_spin.setSpecialValueText("Default")
        grid.addWidget(self.dropout_spin, 1, 3)

        gb_param.setLayout(grid)
        left_layout.addWidget(gb_param)

        h = QtWidgets.QHBoxLayout()
        self.resume_edit = QtWidgets.QLineEdit()
        self.resume_edit.setPlaceholderText("Optional: select a .pth checkpoint to resume training")
        h.addWidget(self.resume_edit, 3)
        btn = QtWidgets.QPushButton("Select checkpoint")
        btn.clicked.connect(self.browse_resume_weight)
        h.addWidget(btn)
        left_layout.addLayout(h)

        h = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("Start Training")
        self.train_stop_btn = QtWidgets.QPushButton("Stop Training")
        self.train_stop_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.start_training)
        self.train_stop_btn.clicked.connect(global_state.stop_train)
        h.addWidget(self.train_btn);
        h.addWidget(self.train_stop_btn)
        left_layout.addLayout(h)

        # ==================== Modified: Text-based progress bar ====================
        self.train_progress = QtWidgets.QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setFormat("%v/%m | %p% | %s")  # Custom Formatting
        self.train_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #1976d2;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #42a5f5;
            }
        """)
        left_layout.addWidget(self.train_progress)

        gb_log = QtWidgets.QGroupBox("Training Logs")
        v = QtWidgets.QVBoxLayout()
        self.train_log = QtWidgets.QTextEdit()
        self.train_log.setReadOnly(True);
        self.train_log.setFont(QtGui.QFont("Consolas", 10))
        v.addWidget(self.train_log)
        gb_log.setLayout(v)
        left_layout.addWidget(gb_log, 1)

        splitter.addWidget(left)

        middle = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout(middle)

        title = QtWidgets.QLabel("Inference and Visualization")
        title.setStyleSheet("font-size:18pt;font-weight:bold;color:#1976d2;")
        middle_layout.addWidget(title)

        gb_weight = QtWidgets.QGroupBox("Trained weights (multiple selections allowed)")
        v = QtWidgets.QVBoxLayout()
        self.weight_list = QtWidgets.QListWidget()
        self.weight_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.weight_list.itemSelectionChanged.connect(self.show_weight_info)
        v.addWidget(self.weight_list)

        h = QtWidgets.QHBoxLayout()
        btn = QtWidgets.QPushButton("Refresh Weight List")
        btn.clicked.connect(self.refresh_weights)
        h.addWidget(btn)
        v.addLayout(h)
        gb_weight.setLayout(v)
        middle_layout.addWidget(gb_weight)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Inputting Path:"))
        self.input_path_edit = QtWidgets.QLineEdit()
        h.addWidget(self.input_path_edit, 2)

        btn_img = QtWidgets.QPushButton("Select a single image")
        btn_img.clicked.connect(lambda: self.browse_input(is_file=True))
        h.addWidget(btn_img)

        btn_folder = QtWidgets.QPushButton("Select Folder")
        btn_folder.clicked.connect(lambda: self.browse_input(is_file=False))
        h.addWidget(btn_folder)

        middle_layout.addLayout(h)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Nuclear Analysis Threshold (percentile):"))
        self.percentile_spin = QtWidgets.QDoubleSpinBox()
        self.percentile_spin.setRange(80.0, 99.9);
        self.percentile_spin.setValue(95.0);
        self.percentile_spin.setSingleStep(0.5)
        h.addWidget(self.percentile_spin)
        middle_layout.addLayout(h)

        h = QtWidgets.QHBoxLayout()
        self.infer_btn = QtWidgets.QPushButton("Starting deduction")
        self.infer_stop_btn = QtWidgets.QPushButton("Ending deduction")
        self.infer_stop_btn.setEnabled(False)
        self.infer_btn.clicked.connect(self.start_inference)
        self.infer_stop_btn.clicked.connect(global_state.stop_infer)
        h.addWidget(self.infer_btn);
        h.addWidget(self.infer_stop_btn)
        middle_layout.addLayout(h)

        gb_log = QtWidgets.QGroupBox("Deduction Log")
        v = QtWidgets.QVBoxLayout()
        self.infer_log = QtWidgets.QTextEdit()
        self.infer_log.setReadOnly(True);
        self.infer_log.setFont(QtGui.QFont("Consolas", 10))
        v.addWidget(self.infer_log)
        gb_log.setLayout(v)
        middle_layout.addWidget(gb_log, 1)

        splitter.addWidget(middle)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        title = QtWidgets.QLabel("Weights Information")
        title.setStyleSheet("font-size:18pt;font-weight:bold;color:#1976d2;")
        right_layout.addWidget(title)

        gb_info = QtWidgets.QGroupBox("Selected Weight Details")
        v = QtWidgets.QVBoxLayout()
        self.weight_info_text = QtWidgets.QTextEdit()
        self.weight_info_text.setReadOnly(True)
        self.weight_info_text.setFont(QtGui.QFont("Consolas", 10))
        self.weight_info_text.setPlaceholderText("Select weight to view details")
        v.addWidget(self.weight_info_text)
        gb_info.setLayout(v)
        right_layout.addWidget(gb_info, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2);
        splitter.setStretchFactor(1, 2);
        splitter.setStretchFactor(2, 1)

    def update_model_name_combo(self):
        self.model_name_combo.clear()
        self.model_name_combo.addItems(get_model_names(self.model_type_combo.currentText()))

    def browse_data_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select root directory for dataset")
        if d: self.data_root_edit.setText(d)

    def browse_resume_weight(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Weights selected for continued training", cfg.weight_dir, "PyTorch Weight (*.pth)")
        if f: self.resume_edit.setText(f)

    def browse_input(self, is_file=True):
        try:
            if is_file:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select a single image", "",
                                                                "image file (*.png *.jpg *.jpeg)")
            else:
                path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Image Folder")

            if path:
                self.input_path_edit.setText(path)
                if is_file and path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.uploaded_pil = Image.open(path).convert("RGB")
                else:
                    self.uploaded_pil = None
        except Exception as e:
            append_log_with_limit(self.infer_log, f"Error occurred while browsing files: {str(e)}")

    def refresh_weights(self):
        try:
            current_selection = [item.text() for item in self.weight_list.selectedItems()]

            self.weight_list.clear()
            self.weight_map.clear()

            if os.path.exists(cfg.weight_dir):
                os.listdir(cfg.weight_dir)

            items = scan_weights()
            if not items:
                append_log_with_limit(self.infer_log, "⚠️ No weight files found. Please check the checkpoints folder")
                self.weight_info_text.setText("No weight files found")
                return

            for label, path in items:
                self.weight_list.addItem(label)
                self.weight_map[label] = path

            for i in range(self.weight_list.count()):
                item = self.weight_list.item(i)
                if item.text() in current_selection:
                    item.setSelected(True)

            append_log_with_limit(self.infer_log, f"✅ Weight list has been refreshed. Total loaded: {len(items)}")

            pth_count = len([1 for _, p in items if p.endswith('.pth')])
            pkl_count = len([1 for _, p in items if p.endswith('.pkl')])
            self.weight_info_text.setText(
                f"Weights\nPyTorch Models: {pth_count}\nML Models: {pkl_count} \nOverall: {len(items)}")

        except Exception as e:
            error_msg = f"❌ Failed to refresh the weight list: {str(e)}\n{traceback.format_exc()}"
            append_log_with_limit(self.infer_log, error_msg)
            self.weight_info_text.setText(error_msg)

    def show_weight_info(self):
        try:
            selected_items = self.weight_list.selectedItems()
            if not selected_items:
                self.weight_info_text.setText("No weights selected")
                return

            info_text = ""
            for i, item in enumerate(selected_items):
                label = item.text()
                path = self.weight_map[label]

                info_text += f"=== Weight {i + 1}/{len(selected_items)} ===\n"
                info_text += f"Display Name: {label}\n"
                info_text += f"File path: {os.path.basename(path)}\n"

                try:
                    if path.endswith(".pth"):
                        ckpt = torch.load(path, map_location="cpu")
                        info_text += f"Model Type: {ckpt.get('model_type', 'Unknown')}\n"
                        info_text += f"Model Name: {ckpt.get('model_name', 'Unknown')}\n"
                        info_text += f"Training Session: {ckpt.get('epoch', 'Unknown')}\n"
                        info_text += f"Verification Accuracy Rate: {ckpt.get('acc', 0):.5f}\n"
                        info_text += f"Optimal CAM Layer: {ckpt.get('best_cam_layer_name', 'Not set')}\n"
                        if "dropout_rate" in ckpt:
                            info_text += f"Dropout rate: {ckpt['dropout_rate']}\n"
                        else:
                            info_text += f"Dropout rate: Default\n"
                        if "optimizer_state_dict" in ckpt:
                            opt_state = ckpt["optimizer_state_dict"]
                            if "param_groups" in opt_state and len(opt_state["param_groups"]) > 0:
                                info_text += f"Learning rate: {opt_state['param_groups'][0].get('lr', 'Unknown')}\n"
                    elif path.endswith(".pkl"):
                        info_text += f"Model Type: ml\n"
                        info_text += f"Model Name: {os.path.basename(path).replace('.pkl', '')}\n"
                        info_text += f"Note: ML model weights, no training iteration information\n"
                except Exception as e:
                    info_text += f"❌ Failed to read: {str(e)}\n"

                info_text += "\n"

            self.weight_info_text.setText(info_text.strip())
        except Exception as e:
            self.weight_info_text.setText(f"Error occurred while displaying weight information: {str(e)}")

    def on_train_error(self, error_msg):
        append_log_with_limit(self.train_log, f"\n{'=' * 50}\n❌ Training error:\n{error_msg}\n{'=' * 50}\n")
        self.on_train_finished()
        self.train_btn.setEnabled(False)
        QtCore.QTimer.singleShot(3000, lambda: self.train_btn.setEnabled(True))

    def on_infer_error(self, error_msg):
        append_log_with_limit(self.infer_log, f"\n{'=' * 50}\n❌ Reasoning Error:\n{error_msg}\n{'=' * 50}\n")
        self.on_infer_finished()
        self.infer_btn.setEnabled(False)
        QtCore.QTimer.singleShot(3000, lambda: self.infer_btn.setEnabled(True))

    def on_train_finished(self):
        self.train_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)
        self.train_progress.setValue(0)
        # Reset progress bar text
        self.train_progress.setFormat("Ready | Click to begin training")
        append_log_with_limit(self.train_log, "Training task completed. Status reset")
        QtCore.QTimer.singleShot(500, self.refresh_weights)
        append_log_with_limit(self.train_log, "Weight list will automatically refresh in 0.5 seconds...")

    def on_infer_finished(self):
        self.infer_btn.setEnabled(True)
        self.infer_stop_btn.setEnabled(False)
        append_log_with_limit(self.infer_log, "Deduction task completed. Status reset")

    # ==================== Progress bar updating function ====================
    def update_train_progress(self, epoch, total_epoch, status, time_info):
        """Update training progress bar display"""
        # Calculate overall progress
        overall_progress = int((epoch / total_epoch) * 100) if total_epoch > 0 else 0

        # Set progress value
        self.train_progress.setValue(overall_progress)

        # Build Display Text
        if "完成" in status:
            # Training Completion Status
            self.train_progress.setFormat(f"✅ Training completed | {epoch}/{total_epoch} | {time_info}")
        elif "中止" in status:
            # User interrupt status
            self.train_progress.setFormat(f"⏹️ Has been discontinued | {epoch}/{total_epoch}")
        elif "出错" in status:
            # Error Status
            self.train_progress.setFormat(f"❌ Training error | {epoch}/{total_epoch}")
        else:
            # Normal training status
            self.train_progress.setFormat(f"{epoch}/{total_epoch} | {overall_progress}% | {status} | {time_info}")

        # Update progress bar color (optional)
        if "Completed" in status:
            self.train_progress.setStyleSheet("""
                QProgressBar { border: 2px solid #4CAF50; border-radius: 5px; text-align: center; height: 25px; }
                QProgressBar::chunk { background-color: #66BB6A; }
            """)
        elif "Terminated" in status or "Error" in status:
            self.train_progress.setStyleSheet("""
                QProgressBar { border: 2px solid #F44336; border-radius: 5px; text-align: center; height: 25px; }
                QProgressBar::chunk { background-color: #EF5350; }
            """)
        else:
            self.train_progress.setStyleSheet("""
                QProgressBar { border: 2px solid #1976d2; border-radius: 5px; text-align: center; height: 25px; }
                QProgressBar::chunk { background-color: #42a5f5; }
            """)

    def start_training(self):
        try:
            data_root = self.data_root_edit.text().strip()
            if not data_root or not os.path.exists(os.path.join(data_root, "train")):
                append_log_with_limit(self.train_log, "❌ Invalid dataset path! Please verify train/test folder exists")
                return

            dropout_rate = self.dropout_spin.value()
            if dropout_rate == 0.0:
                dropout_rate = None

            params = {
                "data_root": data_root,
                "model_type": self.model_type_combo.currentText(),
                "model_name": self.model_name_combo.currentText(),
                "lr": self.lr_spin.value(),
                "batch_size": int(self.bs_spin.value()),
                "max_epoch": int(self.epoch_spin.value()),
                "resume_path": self.resume_edit.text().strip() or None,
                "dropout_rate": dropout_rate
            }

            if params["resume_path"] and not os.path.exists(params["resume_path"]):
                params["resume_path"] = None

            config_str = f"Starting Training → {params['model_type']}/{params['model_name']}"
            if dropout_rate is not None:
                config_str += f" (Dropout={dropout_rate})"
            append_log_with_limit(self.train_log, config_str)
            append_log_with_limit(self.train_log,
                                  f"Parameters: lr={params['lr']}, bs={params['batch_size']}, epochs={params['max_epoch']}")

            self.train_btn.setEnabled(False)
            self.train_stop_btn.setEnabled(True)
            # Reset the progress bar to its initial state
            self.train_progress.setValue(0)
            self.train_progress.setFormat(f"0/{params['max_epoch']} | 0% | Initializing...")

            self.train_worker = TrainWorker(params, self.train_signals)
            self.train_worker.start()

        except Exception as e:
            append_log_with_limit(self.train_log, f"Error occurred during training startup: {traceback.format_exc()}")
            self.on_train_finished()

    def start_inference(self):
        try:
            items = self.weight_list.selectedItems()
            if not items:
                append_log_with_limit(self.infer_log, "❌ Please select at least one weight")
                return

            labels = [it.text() for it in items]
            path = self.input_path_edit.text().strip()
            if not path:
                append_log_with_limit(self.infer_log, "❌ Please enter the image path")
                return

            self.infer_btn.setEnabled(False)
            self.infer_stop_btn.setEnabled(True)

            self.infer_worker = InferenceWorker(labels, self.weight_map, path, self.uploaded_pil,
                                                self.percentile_spin.value(), self.infer_signals,
                                                self.data_root_edit.text())
            self.infer_worker.start()

        except Exception as e:
            append_log_with_limit(self.infer_log, f"Error occurred during reasoning initiation: {traceback.format_exc()}")
            self.on_infer_finished()