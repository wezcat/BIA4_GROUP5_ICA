# analysis/pseudo_nuclear.py

from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import label, regionprops


def pseudo_nuclear_analysis(cam_map: np.ndarray, percentile_thr: float = 95.0) -> dict:
    """
    A lightweight, heuristic 'nuclear-like' analysis based on the CAM heatmap.
    Thresholds the CAM by percentile, runs morphological closing, then measures connected components.
    """
    thr = np.percentile(cam_map, percentile_thr)
    binary = (cam_map > thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    labeled = label(binary // 255)
    props = regionprops(labeled)
    areas = [p.area for p in props]

    return {
        "nucleus_count": len(props),
        "nucleus_density_per_million_pixels": round(len(props) / (cam_map.size / 1e6), 3) if cam_map.size else 0,
        "mean_nucleus_area_pixels": round(float(np.mean(areas)), 2) if areas else 0,
        "large_nucleus_ratio_percent": round(float(np.mean(np.array(areas) > 50) * 100), 2) if areas else 0,
        "analysis_threshold_percentile": percentile_thr,
    }
