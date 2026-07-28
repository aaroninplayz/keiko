import logging
import os
import numpy as np
from typing import Dict, Any
from collections import deque

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

# Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


class AttireAnalyzer:
    """
    Attire professionalism heuristic using MediaPipe Tasks API PoseLandmarker + color analysis.
    Uses upper-body region crop and HSV color variance as a proxy for formality.
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
                )
                self._landmarker = PoseLandmarker.create_from_options(options)
            else:
                logger.error(
                    f"Pose model file 'pose_landmarker_lite.task' not found at {model_path}. "
                    "Please download it from https://storage.googleapis.com/mediapipe-models/ "
                    f"and place it in '{MODELS_DIR}'."
                )
        self._score_history = deque(maxlen=30)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if not MEDIAPIPE_AVAILABLE or self._landmarker is None:
            return {"sensor_type": "attire", "score": 80.0, "details": {"mock": True}}

        try:
            import cv2
            h, w, _ = frame.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = self._landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return {"sensor_type": "attire", "score": 0.0, "details": {"detected": False}}

            lm = result.pose_landmarks[0]

            # Get torso bounding box
            x_min = int(min(lm[LEFT_SHOULDER].x, lm[RIGHT_SHOULDER].x, lm[LEFT_HIP].x, lm[RIGHT_HIP].x) * w)
            x_max = int(max(lm[LEFT_SHOULDER].x, lm[RIGHT_SHOULDER].x, lm[LEFT_HIP].x, lm[RIGHT_HIP].x) * w)
            y_min = int(min(lm[LEFT_SHOULDER].y, lm[RIGHT_SHOULDER].y) * h)
            y_max = int(max(lm[LEFT_HIP].y, lm[RIGHT_HIP].y) * h)

            pad = 10
            x_min = max(0, x_min - pad)
            x_max = min(w, x_max + pad)
            y_min = max(0, y_min - pad)
            y_max = min(h, y_max + pad)

            if x_max <= x_min or y_max <= y_min:
                return {"sensor_type": "attire", "score": 50.0, "details": {"crop_failed": True}}

            # Need BGR for cv2.cvtColor — input frame was converted to RGB for MediaPipe,
            # but our orchestrator passes BGR frames. Let's work with the original frame.
            torso_crop = frame[y_min:y_max, x_min:x_max]
            # Convert RGB to HSV (frame is already RGB from orchestrator)
            hsv = cv2.cvtColor(torso_crop, cv2.COLOR_RGB2HSV)

            sat_channel = hsv[:, :, 1].astype(float)
            sat_std = float(np.std(sat_channel))
            sat_score = max(0, min(100, 100 - sat_std * 1.5))

            val_channel = hsv[:, :, 2].astype(float)
            val_std = float(np.std(val_channel))
            brightness_score = max(0, min(100, 100 - val_std * 1.2))

            hue_channel = hsv[:, :, 0].astype(float)
            hue_bins = np.histogram(hue_channel, bins=12, range=(0, 180))[0]
            dominant_bins = np.sum(hue_bins > (hue_bins.sum() * 0.05))
            color_score = max(0, min(100, 120 - dominant_bins * 15))

            raw_score = (sat_score * 0.40) + (brightness_score * 0.30) + (color_score * 0.30)
            raw_score = max(0, min(100, raw_score))

            self._score_history.append(raw_score)
            smoothed = sum(self._score_history) / len(self._score_history)

            return {
                "sensor_type": "attire",
                "score": round(smoothed, 1),
                "details": {
                    "detected": True,
                    "saturation_uniformity": round(sat_score, 1),
                    "brightness_consistency": round(brightness_score, 1),
                    "color_simplicity": round(color_score, 1),
                    "dominant_colors": int(dominant_bins),
                },
            }
        except Exception as e:
            logger.error(f"Error in attire_analyzer: {e}")
            return {"sensor_type": "attire", "score": 0.0, "details": {"detected": False, "error": str(e)}}

    def release(self):
        if self._landmarker:
            self._landmarker.close()
