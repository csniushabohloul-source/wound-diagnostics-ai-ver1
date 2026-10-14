from typing import Dict, Any, List
from datetime import datetime


class ClinicalDecisionSupportEngine:
    """
    Evidence-based Clinical Decision Support System (CDSS) for chronic wound care.
    Translates pixel-level AI tissue segmentations and longitudinal trajectories into
    actionable clinical guidelines according to the international TIME framework:
    - T: Tissue Non-Viable (Debridement)
    - I: Infection & Inflammation (Antimicrobial management)
    - M: Moisture Imbalance (Dressing selection)
    - E: Edge Non-Advancing (Epidermal advancement)
    """

    @classmethod
    def evaluate_clinical_status(
        cls,
        patient_id: str,
        current_stats: Dict[str, Any],
        trajectory_analysis: Dict[str, Any],
        risk_score: float
    ) -> Dict[str, Any]:
        """
        Determines risk category and produces structured clinical guidelines.
        """
        slough_ratio = current_stats.get("slough_ratio", 0.0)
        necrotic_ratio = current_stats.get("necrotic_ratio", 0.0)
        healing_index = current_stats.get("healing_index", 0.0)
        healing_status = trajectory_analysis.get("healing_status", "UNKNOWN")

        # Risk Classification
        if necrotic_ratio > 0.15 or risk_score >= 65.0:
            tier = "CRITICAL"
            color_badge = "🔴 قرمز - بحرانی (Critical Risk)"
            urgency = "URGENT (24-48 hours)"
        elif slough_ratio > 0.30 or necrotic_ratio > 0.05 or risk_score >= 40.0:
            tier = "HIGH"
            color_badge = "🟠 نارنجی - ریسک بالا (High Risk / Biofilm Threat)"
            urgency = "MODERATE (2-4 days)"
        elif healing_status == "STALLED_OR_EXPANDING" or slough_ratio > 0.15 or risk_score >= 20.0:
            tier = "MODERATE"
            color_badge = "🟡 زرد - نیازمند پایش (Moderate Risk / Stalled)"
            urgency = "ROUTINE (5-7 days)"
        else:
            tier = "LOW"
            color_badge = "🟢 سبز - روند مطلوب (Low Risk / Normal Healing)"
            urgency = "STANDARD (7-14 days)"

        # Clinical Action Items (TIME Framework)
        time_actions = {}
        
        # T: Tissue
        if necrotic_ratio > 0.05:
            time_actions["Tissue"] = "Heavy necrotic eschar observed. Sharp or enzymatic debridement required to expose viable granulation tissue."
        elif slough_ratio > 0.20:
            time_actions["Tissue"] = "Fibrinous slough layer covering wound bed. Autolytic debridement or hydrogel cleansing recommended."
        else:
            time_actions["Tissue"] = "Healthy granulation tissue dominant. Protect fragile capillary buds from mechanical trauma."

        # I: Infection
        if risk_score >= 50.0:
            time_actions["Infection"] = "High probability of critical colonization / biofilm. Apply topical antimicrobial dressing (Silver / Cadexomer Iodine). Order bacterial culture if systemic signs present."
        elif slough_ratio > 0.25:
            time_actions["Infection"] = "Moderate bioburden risk. Cleanse with antiseptic wound irrigation solution (PHMB or hypochlorous acid)."
        else:
            time_actions["Infection"] = "Low infection signs. No systemic antimicrobials required."

        # M: Moisture
        if tier in ["CRITICAL", "HIGH"]:
            time_actions["Moisture"] = "Manage potential heavy exudate with absorbent polyurethane foam or calcium alginate dressing."
        else:
            time_actions["Moisture"] = "Maintain optimal physiological moisture balance using hydrocolloid or silicone-bordered foam."

        # E: Edge
        if healing_index >= 0.70:
            time_actions["Edge"] = "Advancing epithelial margins noted. Encourage surface re-epithelialization; avoid aggressive tape stripping."
        else:
            time_actions["Edge"] = "Undermined or rolled margins (epibole) possible. Re-assess offloading in diabetic foot ulcers and pressure relief in bedsores."

        # Telemedicine Physician Summary
        summary = {
            "patient_id": patient_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "risk_tier": tier,
            "color_badge": color_badge,
            "clinical_urgency": urgency,
            "risk_score_100": risk_score,
            "healing_index_pct": round(healing_index * 100, 1),
            "trajectory_status": healing_status,
            "estimated_closure_days": trajectory_analysis.get("estimated_days_to_closure"),
            "time_framework_recommendations": time_actions,
            "patient_instructions": cls._generate_patient_instructions(tier)
        }

        return summary

    @staticmethod
    def _generate_patient_instructions(tier: str) -> str:
        if tier == "CRITICAL":
            return "فوری: لطفاً ظرف ۲۴ ساعت آینده به درمانگاه تخصصی یا پزشک خود مراجعه فرمایید. پانسمان را خشک نگه دارید و از راه رفتن روی پا خودداری کنید."
        elif tier == "HIGH":
            return "هشدار: بافت ترشحی و زرد رنگ زخم افزایش یافته است. طبق دستور پزشک پانسمان آنتی‌باکتریال را تعویض کرده و طی ۳ روز آینده ویزیت شوید."
        elif tier == "MODERATE":
            return "توجه: روند التیام زخم کمی کند شده است. محل زخم را تمیز نگه داشته و از فشار آوردن به ناحیه خودداری کنید. عکس بعدی را ۵ روز دیگر ارسال کنید."
        else:
            return "عالی: زخم شما با بافت قرمز و سالم در حال بسته شدن است. روال فعلی پانسمان را ادامه دهید و ویزیت بعدی را یک هفته دیگر ثبت فرمایید."
