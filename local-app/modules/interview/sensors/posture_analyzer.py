import logging
import os
import numpy as np
import math
from typing import Dict, Any

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models")

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed. PostureAnalyzer will return mock data.")

# Pose landmark indices (same as legacy PoseLandmark enum)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_EAR = 7
NOSE = 0


class PostureAnalyzer:
    """
    Real-time posture analysis using MediaPipe Tasks API PoseLandmarker.
    Evaluates shoulder alignment, spine angle, and slouch detection.
    Returns a normalized score 0-100.
    """

    def __init__(self):
        self._landmarker = None
        if MEDIAPIPE_AVAILABLE:
            model_path = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
            if os.path.exists(model_path):
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._landmarker = PoseLandmarker.create_from_options(options)
            else:
                logger.error(
                    f"Pose model file 'pose_landmarker_lite.task' not found at {model_path}. "
                    "Please download it from https://storage.googleapis.com/mediapipe-models/ "
                    f"and place it in '{MODELS_DIR}'."
                )
        self._history: list = []
        self._max_history = 30

    def _angle_between(self, a, b, c) -> float:
        """Calculate angle at point b given three landmarks (NormalizedLandmark)."""
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if not MEDIAPIPE_AVAILABLE or self._landmarker is None:
            return {
                "sensor_type": "posture",
                "score": 75.0,
                "details": {
                    "mock": True,
                    "is_slouching": False,
                    "posture_status": "Optimal"
                }
            }

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = self._landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                self._history.append(50.0)
                if len(self._history) > self._max_history:
                    self._history.pop(0)
                smoothed = sum(self._history) / len(self._history)
                return {
                    "sensor_type": "posture",
                    "score": round(smoothed, 1),
                    "details": {
                        "detected": False,
                        "is_slouching": False,
                        "posture_status": "Not Detected"
                    }
                }

            lm = result.pose_landmarks[0]

            # --- Shoulder alignment ---
            shoulder_diff = abs(lm[LEFT_SHOULDER].y - lm[RIGHT_SHOULDER].y)
            shoulder_score = max(0.0, 100.0 - shoulder_diff * 500.0)

            # --- Spine angle (ear-shoulder-hip) ---
            spine_angle = self._angle_between(lm[LEFT_EAR], lm[LEFT_SHOULDER], lm[LEFT_HIP])
            spine_score = max(0.0, min(100.0, (spine_angle - 90.0) * (100.0 / 90.0)))

            # --- Head forward tilt ---
            shoulder_mid_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2.0
            head_forward = abs(lm[NOSE].x - shoulder_mid_x)
            head_score = max(0.0, 100.0 - head_forward * 300.0)

            raw_score = (shoulder_score * 0.35) + (spine_score * 0.45) + (head_score * 0.20)
            raw_score = max(0.0, min(100.0, raw_score))

            is_slouching = spine_angle < 140.0 or raw_score < 60.0
            posture_status = "Slouching" if is_slouching else ("Optimal" if raw_score >= 80 else "Fair")

            self._history.append(raw_score)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            smoothed = sum(self._history) / len(self._history)

            return {
                "sensor_type": "posture",
                "score": round(smoothed, 1),
                "details": {
                    "detected": True,
                    "shoulder_alignment": round(shoulder_score, 1),
                    "spine_angle": round(spine_angle, 1),
                    "spine_score": round(spine_score, 1),
                    "head_alignment": round(head_score, 1),
                    "is_slouching": is_slouching,
                    "posture_status": posture_status,
                    "raw_score": round(raw_score, 1),
                },
            }
        except Exception as e:
            logger.error(f"Error in posture_analyzer: {e}")
            return {
                "sensor_type": "posture",
                "score": 0.0,
                "details": {"detected": False, "error": str(e), "is_slouching": False, "posture_status": "Error"}
            }

    def release(self):
        if self._landmarker:
            self._landmarker.close()
