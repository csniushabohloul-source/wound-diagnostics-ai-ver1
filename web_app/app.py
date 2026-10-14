import os
import sys
import io
import base64
import time
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.config import NUM_CLASSES, CHECKPOINTS_DIR, TISSUE_CLASSES, CLASS_COLORS_RGB
from src.data.preprocessor import WoundImagePreprocessor
from src.data.mask_encoder import WoundMaskEncoder
from src.models.lightweight_unet import CostEffectiveMultiTaskWoundUNet
from src.preventive.trajectory_model import WoundTrajectoryModel
from src.preventive.clinical_decision_support import ClinicalDecisionSupportEngine

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max

# Initialize Model & Preprocessors
device = "cpu"
model = CostEffectiveMultiTaskWoundUNet(in_channels=3, num_classes=NUM_CLASSES, base_channels=32)
ckpt_path = CHECKPOINTS_DIR / "best_cost_effective_unet.pt"
if ckpt_path.exists():
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained model weights from {ckpt_path}")
    except Exception as e:
        print(f"Warning loading checkpoint: {e}")
model.to(device)
model.eval()

preprocessor = WoundImagePreprocessor(target_size=(256, 256), blur_threshold=70.0)
mask_encoder = WoundMaskEncoder()


def array_to_base64_png(arr_rgb: np.ndarray) -> str:
    """Converts RGB numpy array to base64 encoded PNG string."""
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', bgr)
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/samples', methods=['GET'])
def get_samples():
    samples_dir = PROJECT_ROOT / "web_app" / "static" / "samples"
    samples = []
    if samples_dir.exists():
        for f in sorted(samples_dir.glob("*.png")):
            # Extract patient code if in name (e.g. 0001_P20T_1.png)
            parts = f.stem.split("_")
            label = f.stem
            patient = "Patient_20" if "P20" in f.stem else "Patient_10" if "P10" in f.stem else "Patient_34" if "P34" in f.stem else "Patient"
            samples.append({
                "filename": f.name,
                "label": label,
                "patient": patient,
                "url": f"/static/samples/{f.name}"
            })
    return jsonify({"samples": samples})


@app.route('/api/predict', methods=['POST'])
def predict():
    start_time = time.perf_counter()
    
    # 1. Read input image (either uploaded file or selected sample)
    img_bgr = None
    patient_id = "Patient_Unknown"

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        in_memory = file.read()
        nparr = np.frombuffer(in_memory, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        patient_id = request.form.get("patient_id", "Patient_Uploaded")
    elif 'sample_name' in request.json:
        sample_name = request.json['sample_name']
        sample_path = PROJECT_ROOT / "web_app" / "static" / "samples" / sample_name
        if sample_path.exists():
            img_bgr = cv2.imread(str(sample_path))
            patient_id = "Patient_20" if "P20" in sample_name else "Patient_10" if "P10" in sample_name else "Patient_34"

    if img_bgr is None:
        return jsonify({"error": "No valid image provided"}), 400

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Quality Assessment
    quality = preprocessor.assess_quality(img_rgb)

    # 3. Preprocessing (CLAHE and resize with padding)
    enhanced_rgb = preprocessor.apply_clahe(img_rgb)
    resized_input = preprocessor.resize_with_aspect_ratio(enhanced_rgb, is_mask=False)

    # 4. Model Inference
    # Convert HWC [0, 255] to CHW [0, 1] tensor
    inp_tensor = torch.from_numpy(np.transpose(resized_input.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
    
    inference_start = time.perf_counter()
    with torch.no_grad():
        logits = model(inp_tensor)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    inference_time_ms = round((time.perf_counter() - inference_start) * 1000, 1)

    # 5. Mask Decoding and Tissue Analytics
    mask_rgb = mask_encoder.class_indices_to_rgb(preds)
    stats = mask_encoder.compute_tissue_composition(preds)

    # 6. Longitudinal & Preventive Decision Support
    mock_forecast = {
        "healing_status": "PROGRESSING" if stats["healing_index"] > 0.4 else "STALLED_OR_EXPANDING",
        "estimated_days_to_closure": int(28 * (1.0 - stats["healing_index"])) if stats["healing_index"] > 0.4 else None
    }
    
    risk_score = WoundTrajectoryModel.compute_biofilm_deterioration_score(
        slough_ratio=stats.get("slough_ratio", 0.0),
        necrotic_ratio=stats.get("necrotic_ratio", 0.0),
        delta_slough=0.05,
        delta_area_pct=0.0
    )

    cdss = ClinicalDecisionSupportEngine.evaluate_clinical_status(
        patient_id=patient_id,
        current_stats=stats,
        trajectory_analysis=mock_forecast,
        risk_score=risk_score
    )

    total_time_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return jsonify({
        "success": True,
        "patient_id": patient_id,
        "latency": {
            "inference_ms": inference_time_ms,
            "total_ms": total_time_ms,
            "model_params": "0.38M (Cost-Effective)",
            "model_size_mb": "1.46 MB"
        },
        "quality": quality,
        "images": {
            "original": array_to_base64_png(resized_input),
            "mask": array_to_base64_png(mask_rgb)
        },
        "tissue_stats": stats,
        "cdss": cdss
    })


if __name__ == '__main__':
    print("Starting Cost-Effective Wound Diagnostics Telemedicine Web App on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
