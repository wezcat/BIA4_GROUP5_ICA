from dataclasses import dataclass
import os
import torch

@dataclass
class Config:
    data_root: str = "./data/BreaKHis_400X"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: int = 224
    weight_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    html_report_dir: str = "./html_reports"
    temp_dir: str = "./temp_infer_uploads"
    max_log_lines: int = 150

cfg = Config()
for d in [cfg.weight_dir, cfg.log_dir, cfg.html_report_dir, cfg.temp_dir]:
    os.makedirs(d, exist_ok=True)