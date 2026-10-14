import numpy as np
from typing import List, Dict, Any, Optional


class WoundTrajectoryModel:
    """
    Mathematical formulation of chronic wound healing trajectories
    and longitudinal dynamics based on the clinical TIME framework.
    """

    @staticmethod
    def compute_healing_index(granulation_px: int, epithelial_px: int, total_wound_px: int) -> float:
        """
        Healing Index HI(t):
        Ratio of viable reparative tissue (granulation + epithelialization) to total wound bed.
        HI(t) in [0.0, 1.0]. Higher is better.
        """
        if total_wound_px <= 0:
            return 1.0 # Closed wound
        return float(np.clip((granulation_px + epithelial_px) / total_wound_px, 0.0, 1.0))

    @staticmethod
    def compute_area_reduction_rate(initial_area: float, current_area: float) -> float:
        """
        Area Reduction Rate (ARR):
        Standard clinical benchmark: ARR = ((A_0 - A_t) / A_0) * 100%.
        Clinical rule: An ARR >= 50% at 4 weeks is a proven predictor of 12-week complete closure.
        """
        if initial_area <= 0:
            return 0.0
        return float(((initial_area - current_area) / initial_area) * 100.0)

    @staticmethod
    def compute_biofilm_deterioration_score(
        slough_ratio: float,
        necrotic_ratio: float,
        delta_slough: float = 0.0,
        delta_area_pct: float = 0.0
    ) -> float:
        """
        Biofilm & Deterioration Risk Score (BDRS) in [0, 100]:
        Combines non-viable tissue ratios with acute temporal velocity.
        Weights:
        - Slough Ratio (Fibrin/biofilm substrate): 35%
        - Necrotic Ratio (Hypoxic/dead tissue): 35%
        - Acute Slough Expansion (dSlough/dt): 15%
        - Wound Expansion (dArea/dt): 15%
        """
        base_risk = (slough_ratio * 35.0) + (necrotic_ratio * 50.0)
        temporal_risk = max(0.0, delta_slough * 100.0 * 0.75) + max(0.0, (delta_area_pct / 100.0) * 25.0)
        total_score = np.clip(base_risk + temporal_risk, 0.0, 100.0)
        return float(round(total_score, 1))

    @staticmethod
    def forecast_closure(
        visit_areas: List[float],
        visit_intervals_days: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Forecasts expected days to closure using linear healing velocity regression.
        """
        n = len(visit_areas)
        if n < 2:
            return {"forecast_available": False, "reason": "Requires at least 2 longitudinal visits."}

        # If days between visits not provided, assume 7-day intervals
        if visit_intervals_days is None:
            time_points = np.array([i * 7 for i in range(n)], dtype=np.float32)
        else:
            time_points = np.array(np.cumsum([0] + visit_intervals_days), dtype=np.float32)

        areas = np.array(visit_areas, dtype=np.float32)

        # Linear regression: Area(t) = m * t + b
        m, b = np.polyfit(time_points, areas, 1)

        # Healing velocity (pixels reduced per day)
        velocity_per_day = -m

        if velocity_per_day <= 0:
            # Wound is expanding or stalled
            return {
                "forecast_available": True,
                "healing_status": "STALLED_OR_EXPANDING",
                "velocity_per_day": round(float(velocity_per_day), 2),
                "estimated_days_to_closure": None,
                "message": "Negative healing velocity. Wound is not trending toward closure without intervention."
            }

        # Projected days when Area(t) reaches 0
        current_area = areas[-1]
        days_remaining = current_area / velocity_per_day

        return {
            "forecast_available": True,
            "healing_status": "PROGRESSING",
            "velocity_per_day": round(float(velocity_per_day), 2),
            "estimated_days_to_closure": int(np.ceil(days_remaining)),
            "message": f"Positive healing velocity. Projected complete closure in ~{int(np.ceil(days_remaining))} days."
        }
