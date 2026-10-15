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
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


# Dynamic Model Loader
device = "cpu"
model = None
model_meta = {}


def get_model():
    global model, model_meta
    ckpt_path = CHECKPOINTS_DIR / "best_cost_effective_unet.pt"
    if not ckpt_path.exists():
        if model is None:
            model = CostEffectiveMultiTaskWoundUNet(in_channels=3, num_classes=NUM_CLASSES, base_channels=24, multi_task=True)
            model.to(device).eval()
        return model, model_meta

    mtime = ckpt_path.stat().st_mtime
    if model is None or model_meta.get("mtime") != mtime:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            base_channels = ckpt.get("base_channels", 24)
            multi_task = ckpt.get("multi_task", True)
            new_model = CostEffectiveMultiTaskWoundUNet(
                in_channels=3,
                num_classes=NUM_CLASSES,
                base_channels=base_channels,
                multi_task=multi_task
            )
            new_model.load_state_dict(ckpt["model_state_dict"])
            new_model.to(device).eval()
            model = new_model
            model_meta = {
                "mtime": mtime,
                "multi_task": multi_task,
                "base_channels": base_channels,
                "val_boundary_dice": ckpt.get("val_boundary_dice", 0.0),
                "val_tissue_dice": ckpt.get("val_tissue_dice", 0.0)
            }
            print(f"Loaded trained checkpoint: base_channels={base_channels}, multi_task={multi_task}, boundary_dice={model_meta['val_boundary_dice']*100:.2f}%")
        except Exception as e:
            print(f"Warning loading checkpoint: {e}")
            if model is None:
                model = CostEffectiveMultiTaskWoundUNet(in_channels=3, num_classes=NUM_CLASSES, base_channels=24, multi_task=True)
                model.to(device).eval()
    return model, model_meta


# Pre-warm model
get_model()

preprocessor = WoundImagePreprocessor(target_size=(256, 256), blur_threshold=70.0)
mask_encoder = WoundMaskEncoder()


def array_to_base64_png(arr: np.ndarray) -> str:
    """Converts RGB (3-channel) or RGBA (4-channel) numpy array to base64 encoded PNG string."""
    if arr.ndim == 3 and arr.shape[2] == 4:
        bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        _, buffer = cv2.imencode('.png', bgra)
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
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
            name = f.stem
            if "FUSeg" in name:
                patient = "FUSeg Diabetic Cohort"
                label = f"FUSeg Ulcer #{name.split('_')[-1]}"
            elif "Medetec" in name:
                patient = "Medetec Clinical Archive"
                label = "Medetec Ulcer" if "Foot" in name else "Medetec Pressure Sore"
            elif "P20" in name:
                patient = "Patient 20 (Longitudinal)"
                label = f"Visit #{name.split('_')[-1]}"
            elif "P10" in name:
                patient = "Patient 10 (Longitudinal)"
                label = f"Visit #{name.split('_')[-1]}"
            elif "P34" in name:
                patient = "Patient 34"
                label = f"Visit #{name.split('_')[-1]}"
            else:
                patient = "Clinical Sample"
                label = name

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
    
    try:
        # 1. Read input image (either uploaded file or selected sample)
        img_bgr = None
        patient_id = "Patient_Unknown"

        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            in_memory = file.read()
            nparr = np.frombuffer(in_memory, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            patient_id = request.form.get("patient_id", "Patient_Uploaded")
        elif request.is_json and request.json and 'image_base64' in request.json:
            b64_str = request.json['image_base64']
            if ',' in b64_str:
                b64_str = b64_str.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            patient_id = request.json.get("patient_id", "Patient_Uploaded")
        elif request.is_json and request.json and 'sample_name' in request.json:
            sample_name = request.json['sample_name']
            sample_path = PROJECT_ROOT / "web_app" / "static" / "samples" / sample_name
            if sample_path.exists():
                img_bgr = cv2.imread(str(sample_path))
                if "FUSeg" in sample_name:
                    patient_id = "Patient_FUSeg"
                elif "Medetec" in sample_name:
                    patient_id = "Patient_Medetec"
                else:
                    patient_id = "Patient_20" if "P20" in sample_name else "Patient_10" if "P10" in sample_name else "Patient_34"

        if img_bgr is None:
            return jsonify({"error": "تصویر معتبری ارسال نشده است یا فرمت فایل پشتیبانی نمی‌شود.", "success": False}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Quality Assessment
        quality = preprocessor.assess_quality(img_rgb)

        # 3. Preprocessing (CLAHE and resize with padding)
        enhanced_rgb = preprocessor.apply_clahe(img_rgb)
        resized_input = preprocessor.resize_with_aspect_ratio(enhanced_rgb, is_mask=False)

        # 4. Multi-Task Model Inference
        curr_model, meta = get_model()
        model_input = cv2.resize(resized_input, (160, 160), interpolation=cv2.INTER_LINEAR)
        inp_tensor = torch.from_numpy(np.transpose(model_input.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
        
        inference_start = time.perf_counter()
        with torch.no_grad():
            outputs = curr_model(inp_tensor)
            if isinstance(outputs, dict):
                boundary_logits = outputs["boundary"]
                tissue_logits = outputs["tissue"]
                prob_bound = torch.sigmoid(boundary_logits).squeeze().cpu().numpy()
                tissue_probs = torch.softmax(tissue_logits, dim=1).squeeze().cpu().numpy()

                # Robust Gated Wound Prediction:
                # 1. Wound boundary detection threshold
                is_wound = (prob_bound > 0.35)
                # 2. Tissue decomposition inside detected wound
                tissue_pred = np.argmax(tissue_probs[1:], axis=0) + 1 # classes 1 to 5
                
                preds_160 = np.zeros_like(prob_bound, dtype=np.uint8)
                preds_160[is_wound] = tissue_pred[is_wound]
                
                # Morphological cleanup of small isolated noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                preds_160 = cv2.morphologyEx(preds_160, cv2.MORPH_OPEN, kernel)
                
                preds = cv2.resize(preds_160, (resized_input.shape[1], resized_input.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                preds_raw = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                preds = cv2.resize(preds_raw, (resized_input.shape[1], resized_input.shape[0]), interpolation=cv2.INTER_NEAREST)

        inference_time_ms = round((time.perf_counter() - inference_start) * 1000, 1)

        # 5. Mask Decoding and Tissue Analytics (both standard RGB and transparent RGBA overlay)
        mask_rgb = mask_encoder.class_indices_to_rgb(preds)
        mask_rgba = mask_encoder.class_indices_to_rgba(preds)
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
    except Exception as e:
        print(f"Error during predict: {e}")
        return jsonify({"error": f"خطا در پردازش تصویر: {str(e)}", "success": False}), 500

    return jsonify({
        "success": True,
        "patient_id": patient_id,
        "latency": {
            "inference_ms": inference_time_ms,
            "total_ms": total_time_ms,
            "model_params": "0.87M (Edge Multi-Task)",
            "model_size_mb": "3.32 MB"
        },
        "quality": quality,
        "images": {
            "original": array_to_base64_png(resized_input),
            "mask": array_to_base64_png(mask_rgb),
            "overlay_mask": array_to_base64_png(mask_rgba)
        },
        "tissue_stats": stats,
        "cdss": cdss
    })


if __name__ == '__main__':
    print("Starting Cost-Effective Wound Diagnostics Telemedicine Web App on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
