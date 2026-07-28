import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfidenceMetric:
    """
    Composite meta-sensor implementing a scientifically backed behavioral confidence index.
    
    In behavioral psychology, confidence is projected through three primary channels:
    1. Kinesic Openness (Spine uprightness, shoulder symmetry, absence of defensive/slouching statics).
    2. Gaze Poise (Consistent, non-hostile eye contact; absence of high-frequency gaze shifting).
    3. Gesture Purpose (Deliberate, expressive gestures; absence of both extreme rigidity/panic 
       and rapid/fidgety movements).
       
    This sensor uses a multi-factor non-linear equation incorporating behavioral modifiers:
    - Slouching Penalty (defeatist static state): -15 pts
    - High-frequency Fidgeting Penalty (anxiety marker): up to -20 pts
    - Shifty Gaze Penalty (nervous tracking): up to -15 pts
    - Active Eye Poise Bonus (engaged connection): +5 pts
    - Symmetric Pose Poise Bonus: +5 pts
    """

    def __init__(self):
        self._sub_weights = {
            "posture": 0.30,
            "eye_contact": 0.40,
            "body_language": 0.30,
        }

    def update_sub_weights(self, weights: Dict[str, float]):
        for k, v in weights.items():
            if k in self._sub_weights:
                self._sub_weights[k] = max(0.0, min(1.0, v))
        # Normalize
        total = sum(self._sub_weights.values())
        if total > 0:
            self._sub_weights = {k: v / total for k, v in self._sub_weights.items()}

    def calculate(self, posture: Dict[str, Any], eye_contact: Dict[str, Any], body_language: Dict[str, Any]) -> Dict[str, Any]:
        p_score = posture.get("score", 50.0)
        e_score = eye_contact.get("score", 50.0)
        b_score = body_language.get("score", 50.0)

        # 1. Base Weighted Average
        base_confidence = (
            (p_score * self._sub_weights["posture"]) +
            (e_score * self._sub_weights["eye_contact"]) +
            (b_score * self._sub_weights["body_language"])
        )

        modifiers = []
        modified_score = base_confidence

        p_details = posture.get("details", {})
        e_details = eye_contact.get("details", {})
        b_details = body_language.get("details", {})

        # --- Posture Modifiers ---
        if p_details.get("detected"):
            # Slouching Penalty (-15)
            if p_details.get("is_slouching"):
                modified_score -= 15.0
                modifiers.append({"type": "Slouching Penalty", "effect": -15.0})
            
            # Shoulder Symmetry Bonus (+5)
            shoulder_align = p_details.get("shoulder_alignment", 100.0)
            if shoulder_align > 90.0:
                modified_score += 5.0
                modifiers.append({"type": "Symmetric Poise Bonus", "effect": 5.0})

        # --- Eye Gaze Modifiers ---
        if e_details.get("detected") and e_details.get("iris_available"):
            # Eye contact bonus (+5)
            if e_details.get("is_making_contact"):
                modified_score += 5.0
                modifiers.append({"type": "Gaze Poise Bonus", "effect": 5.0})
                
            # Gaze Shiftyness Penalty (up to -15)
            deviation = e_details.get("deviation", 0.0)
            if deviation > 0.12:
                penalty = min(15.0, (deviation - 0.12) * 100.0)
                modified_score -= penalty
                modifiers.append({"type": "Shifty Gaze Penalty", "effect": -round(penalty, 1)})

        # --- Gesture/Kinesics Modifiers ---
        if b_details.get("detected"):
            # Fidgeting Penalty (high hand velocity -> anxiety) (up to -20)
            hand_activity = b_details.get("hand_activity_raw", 0.0)
            if hand_activity > 0.035:
                # Scale penalty from 0.035 to 0.1
                penalty = min(20.0, (hand_activity - 0.035) * 300.0)
                modified_score -= penalty
                modifiers.append({"type": "Fidgeting Penalty", "effect": -round(penalty, 1)})

        # Bound score between 0.0 and 100.0
        final_score = max(0.0, min(100.0, modified_score))

        # Calculate composure
        deviation = e_details.get("deviation", 0.0) if (e_details.get("detected") and e_details.get("iris_available")) else 0.0
        gaze_penalty = min(25.0, deviation * 120.0)
        hand_activity = b_details.get("hand_activity_raw", 0.0) if b_details.get("detected") else 0.0
        fidget_penalty_comp = min(25.0, hand_activity * 250.0)
        slouch_penalty = 15.0 if p_details.get("is_slouching") else 0.0

        # Base composure combines eye poise, body stability, and upright posture
        composure_val = max(0.0, min(100.0, (e_score * 0.5 + p_score * 0.3 + b_score * 0.2) - gaze_penalty - fidget_penalty_comp - slouch_penalty))

        # Calculate stress resilience (Feature 33)
        slouch_penalty = 20.0 if p_details.get("is_slouching") else 0.0
        fidget_penalty_stress = min(30.0, hand_activity * 400.0)
        stress_res_val = max(0.0, min(100.0, (p_score + b_score) / 2.0 - slouch_penalty - fidget_penalty_stress))

        return {
            "sensor_type": "confidence",
            "score": round(final_score, 1),
            "details": {
                "base_weighted": round(base_confidence, 1),
                "composure": round(composure_val, 1),
                "stress_resilience": round(stress_res_val, 1),
                "modifiers": modifiers,
                "sub_weights": dict(self._sub_weights),
                "input_scores": {
                    "posture": p_score,
                    "eye_contact": e_score,
                    "body_language": b_score,
                },
            },
        }
