import os
import sys
from pathlib import Path

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from src.config import PROJECT_ROOT, RESULTS_DIR, TISSUE_CLASSES
from src.data.mask_encoder import WoundMaskEncoder
from src.preventive.trajectory_model import WoundTrajectoryModel
from src.preventive.clinical_decision_support import ClinicalDecisionSupportEngine


def run_preventive_care_pipeline():
    """
    Executes Phase 4: Longitudinal Trajectory Analytics & Preventive Care Early Warning Engine.
    """
    preventive_out = RESULTS_DIR / "preventive"
    preventive_out.mkdir(parents=True, exist_ok=True)
    mask_encoder = WoundMaskEncoder()

    print("=" * 60)
    print("Executing Phase 4: Preventive Care & Longitudinal Decision Engine")
    print("=" * 60)

    # 1. Load available masks and group by patient
    mask_files = sorted(list(PROJECT_ROOT.glob("data/raw/lutseg/LUTSeg/examples/Masks_RGB/**/*.png")))
    if not mask_files:
        mask_files = sorted(list(PROJECT_ROOT.glob("data/raw/lutseg/full_dataset/Masks/**/*.png")))

    patient_visits = defaultdict(list)
    for p in mask_files:
        p_id = p.parent.name # e.g. Patient_10
        t_id = p.stem        # e.g. P10T_1
        mask_bgr = cv2.imread(str(p))
        if mask_bgr is None:
            continue
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        indices = mask_encoder.rgb_to_class_indices(mask_rgb)
        stats = mask_encoder.compute_tissue_composition(indices)
        patient_visits[p_id].append({
            "timestamp": t_id,
            "stats": stats,
            "mask_indices": indices
        })

    print(f"Loaded {len(patient_visits)} patients with longitudinal assessments.")

    # 2. Longitudinal Trajectory Modeling & Clinical Evaluation
    patient_summaries = []
    plot_data = {}

    for p_id, visits in sorted(patient_visits.items()):
        # Sort visits by timestamp name
        visits = sorted(visits, key=lambda x: x["timestamp"])
        areas = [v["stats"]["wound_pixels"] for v in visits]
        his = [v["stats"]["healing_index"] for v in visits]
        sloughs = [v["stats"].get("slough_ratio", 0.0) for v in visits]
        necrotics = [v["stats"].get("necrotic_ratio", 0.0) for v in visits]

        # Forecast and trajectory
        forecast = WoundTrajectoryModel.forecast_closure(areas)
        
        # Calculate acute changes between latest two visits
        delta_slough = sloughs[-1] - sloughs[-2] if len(sloughs) >= 2 else 0.0
        delta_area_pct = ((areas[-1] - areas[-2]) / max(areas[-2], 1)) * 100.0 if len(areas) >= 2 else 0.0

        # Biofilm Deterioration Score
        risk_score = WoundTrajectoryModel.compute_biofilm_deterioration_score(
            sloughs[-1], necrotics[-1], delta_slough, delta_area_pct
        )

        # Clinical Decision Support System (CDSS)
        cdss = ClinicalDecisionSupportEngine.evaluate_clinical_status(
            patient_id=p_id,
            current_stats=visits[-1]["stats"],
            trajectory_analysis=forecast,
            risk_score=risk_score
        )

        patient_summaries.append({
            "patient_id": p_id,
            "num_visits": len(visits),
            "latest_area": areas[-1],
            "healing_index": round(his[-1] * 100, 1),
            "slough_pct": round(sloughs[-1] * 100, 1),
            "necrotic_pct": round(necrotics[-1] * 100, 1),
            "risk_score": risk_score,
            "cdss": cdss,
            "forecast": forecast
        })

        plot_data[p_id] = {
            "timestamps": [v["timestamp"].split("_")[-1] for v in visits],
            "areas": areas,
            "his": [h * 100 for h in his],
            "sloughs": [s * 100 for s in sloughs],
            "necrotics": [n * 100 for n in necrotics]
        }

    # 3. Generate Multi-Patient Longitudinal Trajectory Visualization
    fig, (ax_hi, ax_area, ax_risk) = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    colors = {"Patient_10": "#27ae60", "Patient_20": "#e74c3c", "Patient_34": "#f39c12"}

    for p_id, p_info in plot_data.items():
        c = colors.get(p_id, "#34495e")
        # 1. Healing Index curve
        ax_hi.plot(p_info["timestamps"], p_info["his"], marker='o', linewidth=2.5, color=c, label=p_id)
        # 2. Total Wound Area curve
        ax_area.plot(p_info["timestamps"], p_info["areas"], marker='s', linewidth=2.5, color=c, label=p_id)

    ax_hi.set_title("Healing Index Trajectory (HI %)", fontsize=11, fontweight="bold")
    ax_hi.set_xlabel("Timepoint (T1, T2, ...)", fontsize=10)
    ax_hi.set_ylabel("Granulation + Epithelial (%)", fontsize=10)
    ax_hi.set_ylim(-5, 105)
    ax_hi.grid(True, linestyle="--", alpha=0.5)
    ax_hi.legend()

    ax_area.set_title("Wound Surface Area Trajectory (Pixels)", fontsize=11, fontweight="bold")
    ax_area.set_xlabel("Timepoint (T1, T2, ...)", fontsize=10)
    ax_area.set_ylabel("Total Labeled Pixels", fontsize=10)
    ax_area.grid(True, linestyle="--", alpha=0.5)
    ax_area.legend()

    # 3. Risk Score Bar Chart
    p_names = [p["patient_id"] for p in patient_summaries]
    r_scores = [p["risk_score"] for p in patient_summaries]
    bar_colors = ["#f1c40f" if s < 40 else "#e67e22" if s < 65 else "#e74c3c" for s in r_scores]

    ax_risk.bar(p_names, r_scores, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax_risk.axhline(65, color="red", linestyle="--", label="Critical Threshold (>=65)")
    ax_risk.axhline(40, color="orange", linestyle=":", label="High Risk Threshold (>=40)")
    ax_risk.set_title("Biofilm & Deterioration Risk Score (BDRS)", fontsize=11, fontweight="bold")
    ax_risk.set_ylabel("Risk Score [0-100]", fontsize=10)
    ax_risk.set_ylim(0, 100)
    ax_risk.grid(axis='y', linestyle="--", alpha=0.5)
    ax_risk.legend(fontsize=8.5)

    plt.tight_layout()
    chart_path = preventive_out / "healing_trajectory_projections.png"
    plt.savefig(str(chart_path))
    plt.close()
    print(f"Longitudinal trajectory projection chart saved to: {chart_path}")

    # 4. Generate Clinical Telemedicine Physician Note for High-Risk Patient (e.g. Patient_20)
    p20 = next((p for p in patient_summaries if p["patient_id"] == "Patient_20"), patient_summaries[0])
    doc_note_path = preventive_out / "Sample_Telemedicine_Alert_Patient_20.md"
    with open(doc_note_path, "w", encoding="utf-8") as f:
        f.write("# هشدار پزشکی از راه دور و تصمیم‌یار بالینی (Telemedicine Clinical Alert)\n\n")
        f.write(f"**شناسه بیمار:** {p20['patient_id']}  \n")
        f.write(f"**سطح خطر ارزیابی‌شده:** {p20['cdss']['color_badge']}  \n")
        f.write(f"**امتیاز خطر عفونت و بیوفیلم (BDRS):** {p20['risk_score']} از ۱۰۰  \n")
        f.write(f"**شاخص التیام فعلی (HI):** {p20['healing_index']}%  \n")
        f.write(f"**ترکیب بافت‌های بستر زخم:** اسلاف {p20['slough_pct']}٪ | نکروز {p20['necrotic_pct']}٪  \n")
        f.write(f"**فوریت مداخله بالینی:** {p20['cdss']['clinical_urgency']}  \n\n")

        f.write("---\n\n")
        f.write("## توصیه‌های درمانی ساختاریافته بر مبنای چارچوب بالینی TIME:\n\n")
        for key, rec in p20["cdss"]["time_framework_recommendations"].items():
            f.write(f"- **{key} (مدیریت بافت و عفونت):** {rec}\n")

        f.write("\n---\n\n")
        f.write("## راهنمای هوشمند ارسال‌شده به بیمار در اپلیکیشن موبایل:\n")
        f.write(f"> {p20['cdss']['patient_instructions']}\n")

    print(f"Sample doctor alert note saved to: {doc_note_path}")

    # 5. Summary Table for All Patients
    summary_table_path = preventive_out / "patient_summary_table.md"
    with open(summary_table_path, "w", encoding="utf-8") as f:
        f.write("# جدول پایش طولی و هشدارهای پیشگیرانه بیماران (Longitudinal Preventive Summary)\n\n")
        f.write("| کد بیمار | تعداد ویزیت | مساحت زخم (px) | شاخص التیام (HI) | بافت اسلاف | بافت نکروز | امتیاز ریسک (BDRS) | سطح هشدار | پیش‌بینی زمان بسته شدن |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for p in patient_summaries:
            closure_str = f"~{p['forecast']['estimated_days_to_closure']} روز" if p['forecast'].get('estimated_days_to_closure') else "متوقف / نیازمند مداخله"
            f.write(
                f"| {p['patient_id']} | {p['num_visits']} | {p['latest_area']:,} | "
                f"{p['healing_index']}% | {p['slough_pct']}% | {p['necrotic_pct']}% | "
                f"{p['risk_score']} | **{p['cdss']['risk_tier']}** | {closure_str} |\n"
            )

    print(f"Patient summary table saved to: {summary_table_path}")
    print("=" * 60)
    print("Phase 4 successfully completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_preventive_care_pipeline()
