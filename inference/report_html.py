import os
import datetime
import numpy as np
import cv2
from PIL import Image


def generate_professional_html_report(
    report_dir, img_path, original_filename, individual_results, mean_prob,
    ensemble_pred, nuclear_analysis, cam_map
):
    """
    Generate a professional, academic-style HTML analysis report.

    If CAM is not available (e.g., ML .pkl models / empty CAM / all-blue CAM),
    the report will show "CAM not available" instead of the CAM panels.

    :param img_path: The actual image path used for processing (may be a temporary file)
    :param original_filename: The original image filename (for display)
    """
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = os.path.splitext(original_filename)[0]
    safe_name = base_name.replace(" ", "_").replace("/", "_")

    report_name = f"report_{safe_name}_{timestamp}.html"
    report_path = os.path.join(report_dir, report_name)

    try:
        # 1) Load original image (RGB)
        original_img = Image.open(img_path).convert("RGB")

        # -----------------------------
        # Decide whether CAM is usable
        # -----------------------------
        has_cam = False
        cam_width, cam_height = original_img.size  # fallback to original size

        if cam_map is not None and isinstance(cam_map, np.ndarray) and cam_map.size > 0:
            cam_max = float(np.nanmax(cam_map))
            cam_min = float(np.nanmin(cam_map))
            cam_range = cam_max - cam_min

            # Heuristic: if CAM is almost constant / near-zero, treat as not available.
            # This catches the "all-blue" maps (almost uniform).
            if np.isfinite(cam_range) and cam_range > 1e-6 and cam_max > 1e-6:
                has_cam = True

        # -----------------------------
        # Prepare images for report
        # -----------------------------
        # Always save a resized "original" so the UI is consistent.
        # If CAM exists: resize original to CAM size.
        # If no CAM: keep original size.
        if has_cam:
            # Normalize CAM map
            cam_norm = (cam_map - np.nanmin(cam_map)) / (np.nanmax(cam_map) - np.nanmin(cam_map) + 1e-8)
            if cam_norm.size > 0:
                cam_norm = cv2.GaussianBlur(cam_norm.astype(np.float32), (5, 5), 0)
                cam_norm = np.clip(cam_norm, 0, 1)

            cam_colored = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)

            cam_height, cam_width = cam_colored.shape[0], cam_colored.shape[1]
            original_img_resized = original_img.resize((cam_width, cam_height), Image.LANCZOS)

            # Save resized original image
            original_file = os.path.join(report_dir, f"orig_{safe_name}_{timestamp}.png")
            original_img_resized.save(original_file, quality=95, optimize=True)

            # Save CAM image
            cam_file = os.path.join(report_dir, f"cam_{safe_name}_{timestamp}.png")
            cv2.imwrite(cam_file, cam_colored)

            # Overlay
            orig_cv = cv2.cvtColor(np.array(original_img_resized), cv2.COLOR_RGB2BGR)
            if len(cam_colored.shape) == 2:
                cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_GRAY2BGR)

            overlay = cv2.addWeighted(orig_cv, 0.6, cam_colored, 0.4, 0)
            overlay_file = os.path.join(report_dir, f"overlay_{safe_name}_{timestamp}.png")
            cv2.imwrite(overlay_file, overlay)

            rel_img = os.path.relpath(original_file, report_dir)
            rel_cam = os.path.relpath(cam_file, report_dir)
            rel_overlay = os.path.relpath(overlay_file, report_dir)

        else:
            # No CAM: only save original image (no cam/overlay files)
            original_file = os.path.join(report_dir, f"orig_{safe_name}_{timestamp}.png")
            original_img.save(original_file, quality=95, optimize=True)

            rel_img = os.path.relpath(original_file, report_dir)
            rel_cam = ""       # unused
            rel_overlay = ""   # unused

            # Make nuclear analysis explicit
            nuclear_analysis = {
                "nuclear_analysis": "Skipped (CAM not available)"
            }

        # -----------------------------
        # Build HTML
        # -----------------------------
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>BreastCAD-Pro Pathology Analysis Report</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                    background: linear-gradient(135deg, #f5f7fa 0%, #e4e7f1 100%);
                    color: #2c3e50;
                    line-height: 1.6;
                    padding: 30px;
                }}

                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
                    overflow: hidden;
                }}

                .header {{
                    background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%);
                    color: white;
                    padding: 40px 50px;
                    text-align: center;
                    border-bottom: 4px solid #ff9800;
                }}

                .header h1 {{
                    font-size: 32px;
                    font-weight: 600;
                    margin-bottom: 10px;
                    letter-spacing: 0.5px;
                }}

                .header .subtitle {{
                    font-size: 16px;
                    opacity: 0.9;
                    font-weight: 300;
                }}

                .content {{
                    padding: 40px 50px;
                }}

                .section {{
                    margin-bottom: 40px;
                    padding: 25px;
                    background: #f8fbff;
                    border-left: 5px solid #1976d2;
                    border-radius: 8px;
                }}

                .section h2 {{
                    color: #1976d2;
                    font-size: 22px;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #e3f2fd;
                }}

                .visualization-box {{
                    background: white;
                    border: 2px solid #e3f2fd;
                    border-radius: 10px;
                    padding: 30px;
                    margin: 20px 0;
                    box-shadow: 0 4px 15px rgba(25, 118, 210, 0.1);
                }}

                .visualization-box h3 {{
                    color: #1976d2;
                    font-size: 20px;
                    text-align: center;
                    margin-bottom: 25px;
                    font-weight: 600;
                }}

                .image-grid {{
                    display: flex;
                    justify-content: space-between;
                    gap: 20px;
                    flex-wrap: wrap;
                }}

                .image-item {{
                    flex: 1;
                    min-width: 300px;
                    text-align: center;
                }}

                .image-item h4 {{
                    color: #2c3e50;
                    font-size: 16px;
                    margin-bottom: 10px;
                    font-weight: 600;
                }}

                .image-item p {{
                    font-size: 13px;
                    color: #7f8c8d;
                    margin-bottom: 15px;
                    min-height: 40px;
                }}

                .image-item img {{
                    width: 100%;
                    height: 320px;
                    object-fit: contain;
                    border: 1px solid #e0e0e0;
                    border-radius: 5px;
                    padding: 10px;
                    background: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}

                .cam-na {{
                    text-align: center;
                    padding: 40px;
                    color: #777;
                    background: #fbfcff;
                    border: 1px dashed #cfd8dc;
                    border-radius: 10px;
                }}

                .cam-na h4 {{
                    font-size: 18px;
                    color: #455a64;
                    margin-bottom: 10px;
                }}

                .diagnosis-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                    margin-bottom: 20px;
                }}

                .diagnosis-main {{
                    font-size: 24px;
                    font-weight: 700;
                    padding: 15px;
                    border-radius: 6px;
                    text-align: center;
                    margin-bottom: 15px;
                }}

                .pred-benign {{
                    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
                    color: #2e7d32;
                    border: 1px solid #a5d6a7;
                }}

                .pred-malignant {{
                    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
                    color: #c62828;
                    border: 1px solid #ef9a9a;
                }}

                .model-table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                }}

                .model-table th {{
                    background: #1976d2;
                    color: white;
                    padding: 15px;
                    text-align: center;
                    font-weight: 600;
                }}

                .model-table td {{
                    padding: 12px 15px;
                    text-align: center;
                    border-bottom: 1px solid #ecf0f1;
                }}

                .model-table tr:hover {{
                    background: #f5f7fa;
                }}

                .nuclear-analysis {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 6px;
                    border: 1px solid #e9ecef;
                }}

                .nuclear-analysis ul {{
                    list-style-type: none;
                    padding-left: 0;
                }}

                .nuclear-analysis li {{
                    padding: 8px 0;
                    border-bottom: 1px solid #e9ecef;
                }}

                .nuclear-analysis li:last-child {{
                    border-bottom: none;
                }}

                .nuclear-analysis li b {{
                    color: #1976d2;
                    font-weight: 600;
                }}

                .footer {{
                    text-align: center;
                    padding: 20px;
                    font-size: 12px;
                    color: #95a5a6;
                    background: #f8f9fa;
                    border-top: 1px solid #e9ecef;
                }}

                @media (max-width: 768px) {{
                    .image-grid {{
                        flex-direction: column;
                    }}

                    .image-item {{
                        min-width: 100%;
                    }}

                    .content {{
                        padding: 20px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>BreastCAD-Pro Breast Pathology AI Analysis Report</h1>
                    <div class="subtitle">
                        Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
                        Input image: {original_filename}
                    </div>
                </div>

                <div class="content">
                    <div class="section">
                        <h2> Ensemble Diagnosis</h2>
                        <div class="diagnosis-card">
                            <div class="diagnosis-main {'pred-malignant' if ensemble_pred in ['恶性','malignant','Malignant'] else 'pred-benign'}">
                                Final prediction: {ensemble_pred}
                            </div>
                            <p style="text-align: center; font-size: 18px;">
                                <strong>Malignancy probability: {mean_prob:.2f}%</strong> |
                                Benign probability: {100 - mean_prob:.2f}%
                            </p>
                        </div>
                    </div>

                    <div class="section">
                        <h2> Visualization</h2>
                        <div class="visualization-box">
                            <h3>Model Attention Visualization (Image size: {cam_width}×{cam_height})</h3>
        """

        if has_cam:
            html += f"""
                            <div class="image-grid">
                                <div class="image-item">
                                    <h4>Original Histopathology Image</h4>
                                    <p>Input H&amp;E-stained histopathology image<br>(standardized for analysis)</p>
                                    <img src="{rel_img}" alt="Original image">
                                </div>
                                <div class="image-item">
                                    <h4>Grad-CAM Heatmap</h4>
                                    <p>Regions contributing most to the model decision<br>(red = high activation, blue = low activation)</p>
                                    <img src="{rel_cam}" alt="Grad-CAM heatmap">
                                </div>
                                <div class="image-item">
                                    <h4>Overlay</h4>
                                    <p>Overlay of original image and heatmap<br>(opacity: 60% original + 40% CAM)</p>
                                    <img src="{rel_overlay}" alt="Overlay image">
                                </div>
                            </div>
            """
        else:
            html += f"""
                            <div class="cam-na">
                                <h4>CAM not available</h4>
                                <p>
                                    This model is a non-differentiable machine learning classifier (e.g., RF/SVM/XGBoost),
                                    or the CAM map is not informative (near-constant / all-blue).<br>
                                    Gradient-based visual explanations (Grad-CAM) are only applicable to neural networks.
                                </p>
                                <div style="margin-top:20px;">
                                    <img src="{rel_img}" alt="Original image" style="max-width: 520px; width: 100%; height: auto;">
                                </div>
                            </div>
            """

        html += """
                        </div>
                    </div>

                    <div class="section">
                        <h2> Per-model Predictions</h2>
                        <table class="model-table">
                            <thead>
                                <tr>
                                    <th>Model Type/Name</th>
                                    <th>Validation Accuracy</th>
                                    <th>Prediction</th>
                                    <th>Malignancy Probability</th>
                                </tr>
                            </thead>
                            <tbody>"""

        for r in individual_results:
            pred_val = r.get("pred", "")
            is_malig = pred_val in ["恶性", "malignant", "Malignant"]
            color = "pred-malignant" if is_malig else "pred-benign"
            acc_str = f"{r['acc']:.4f}" if r.get("acc", 0) > 0 else "N/A"

            html += f"""
                                <tr>
                                    <td>{r.get('model', '')}</td>
                                    <td>{acc_str}</td>
                                    <td class="{color}" style="font-weight:600;">{pred_val}</td>
                                    <td>{float(r.get('prob', 0.0)):.2f}%</td>
                                </tr>"""

        html += """
                            </tbody>
                        </table>
                    </div>

                    <div class="section">
                        <h2> Nuclear Morphology Analysis</h2>
                        <div class="nuclear-analysis">
                            <ul>"""

        for k, v in (nuclear_analysis or {}).items():
            html += f"""
                                <li><b>{k}:</b> {v}</li>"""

        html += """
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="footer">
                    <p>This report is automatically generated by BreastCAD-Pro and is intended for research use only.
                       It does not replace professional medical diagnosis.</p>
                    <p>Powered by: PyTorch + pytorch-grad-cam | Generated at: """ + datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S") + """</p>
                </div>
            </div>
        </body>
        </html>"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        return report_path

    except Exception as e:
        raise RuntimeError(f"Failed to generate report: {str(e)}")
