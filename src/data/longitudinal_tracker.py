import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


class LongitudinalWoundTracker:
    """
    Longitudinal healing trajectory and preventive early warning module.
    Analyzes temporal progression of wound surface area and tissue dynamics
    between visits (T_1, T_2, ..., T_k).
    """

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.history: List[Dict[str, Any]] = []

    def add_visit(self, timestamp_id: str, composition_stats: Dict[str, Any], quality_info: Dict[str, Any] = None):
        """Adds a clinical assessment record for a timepoint (e.g. T_1, T_2)."""
        record = {
            "timestamp": timestamp_id,
            "wound_pixels": composition_stats["wound_pixels"],
            "healing_index": composition_stats["healing_index"],
            "slough_ratio": composition_stats.get("slough_ratio", 0.0),
            "necrotic_ratio": composition_stats.get("necrotic_ratio", 0.0),
            "tissues": composition_stats["tissues"],
            "quality": quality_info or {}
        }
        self.history.append(record)

    def evaluate_preventive_risk(self) -> Dict[str, Any]:
        """
        Evaluates clinical risks and generates early warnings:
        - Infection / Deterioration Alert: If Slough or Necrotic tissue rises abruptly.
        - Stalled Wound Alert: If healing index plateaus or decreases.
        - Favorable Healing: If granulation/epithelial coverage is expanding.
        """
        if not self.history:
            return {"status": "NO_DATA", "recommendation": "No visits recorded."}

        latest = self.history[-1]
        alerts = []
        severity = "NORMAL"

        # Check latest tissue severity
        if latest["necrotic_ratio"] > 0.08:
            severity = "HIGH"
            alerts.append(f"Significant necrotic tissue present ({latest['necrotic_ratio']*100:.1f}%). Debridement recommended.")

        if latest["slough_ratio"] > 0.35:
            if severity != "HIGH":
                severity = "MODERATE"
            alerts.append(f"High fibrinous slough ({latest['slough_ratio']*100:.1f}%). Risk of biofilm and delayed closure.")

        # Temporal trajectory analysis if >= 2 visits
        if len(self.history) >= 2:
            prev = self.history[-2]
            delta_area = (latest["wound_pixels"] - prev["wound_pixels"]) / max(prev["wound_pixels"], 1)
            delta_hi = latest["healing_index"] - prev["healing_index"]
            delta_slough = latest["slough_ratio"] - prev["slough_ratio"]

            if delta_slough > 0.12:
                severity = "HIGH"
                alerts.append(f"Acute increase in slough (+{delta_slough*100:.1f}%). Early warning: Potential wound infection.")

            if delta_area > 0.15:
                severity = "HIGH"
                alerts.append(f"Wound enlargement observed (+{delta_area*100:.1f}% area expansion).")
            elif delta_area < -0.10 and delta_hi >= 0:
                alerts.append(f"Positive healing progression: {abs(delta_area)*100:.1f}% reduction in wound area.")

            if abs(delta_area) < 0.05 and abs(delta_hi) < 0.05:
                if severity == "NORMAL":
                    severity = "MONITOR"
                alerts.append("Wound healing appears stalled (chronicity indicator). Consider dressing reassessment.")

        recommendation = "Continue standard preventive dressing protocol."
        if severity == "HIGH":
            recommendation = "URGENT: Schedule immediate in-person clinical review, microbial swab, or debridement."
        elif severity == "MODERATE":
            recommendation = "MODERATE: Clean wound bed, monitor exudate, re-evaluate in 3-5 days."

        return {
            "patient_id": self.patient_id,
            "num_visits": len(self.history),
            "severity": severity,
            "alerts": alerts,
            "recommendation": recommendation,
            "latest_healing_index": latest["healing_index"]
        }

    def plot_trajectory(self, output_path: Path):
        """Generates and saves a publication-quality longitudinal trajectory chart."""
        if len(self.history) < 2:
            return

        timestamps = [h["timestamp"] for h in self.history]
        healing_indices = [h["healing_index"] * 100 for h in self.history]
        slough_pcts = [h["slough_ratio"] * 100 for h in self.history]
        necrotic_pcts = [h["necrotic_ratio"] * 100 for h in self.history]
        wound_areas = [h["wound_pixels"] for h in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

        # Plot 1: Healing Index & Critical Tissues
        ax1.plot(timestamps, healing_indices, marker='o', linewidth=2.5, color='#2ecc71', label='Healing Index % (Granulation + Epithelial)')
        ax1.plot(timestamps, slough_pcts, marker='s', linewidth=2, color='#f1c40f', linestyle='--', label='Slough % (Fibrin)')
        ax1.plot(timestamps, necrotic_pcts, marker='^', linewidth=2, color='#e74c3c', linestyle=':', label='Necrotic % (Eschar)')
        ax1.set_title(f"Patient {self.patient_id} - Tissue Dynamics Across Visits", fontsize=11, fontweight='bold', pad=10)
        ax1.set_xlabel("Clinical Visit / Timepoint", fontsize=10)
        ax1.set_ylabel("Composition % of Wound", fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='best', fontsize=8.5)
        ax1.set_ylim(-5, 105)

        # Plot 2: Total Wound Area Trajectory
        ax2.plot(timestamps, wound_areas, marker='D', linewidth=2.5, color='#3498db')
        ax2.fill_between(timestamps, wound_areas, color='#3498db', alpha=0.15)
        ax2.set_title(f"Patient {self.patient_id} - Total Wound Area Trajectory", fontsize=11, fontweight='bold', pad=10)
        ax2.set_xlabel("Clinical Visit / Timepoint", fontsize=10)
        ax2.set_ylabel("Wound Pixels (Area)", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path))
        plt.close()
