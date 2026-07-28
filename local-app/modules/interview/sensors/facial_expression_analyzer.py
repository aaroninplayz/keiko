import logging
import os
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models")

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed. FacialExpressionAnalyzer will return mock data.")


class FacialExpressionAnalyzer:
    """
    Real-time facial expression analysis using MediaPipe FaceLandmarker blendshapes.
    Classifies expressions into: neutral, happy, confused, nervous.
    """
    def __init__(self):
        self._landmarker = None
        if MEDIAPIPE_AVAILABLE:
            model_path = os.path.join(MODELS_DIR, "face_landmarker.task")
            if os.path.exists(model_path):
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                )
                self._landmarker = FaceLandmarker.create_from_options(options)
            else:
                logger.error(
                    f"Face model file 'face_landmarker.task' not found at {model_path}. "
                    "Please download it from https://storage.googleapis.com/mediapipe-models/ "
                    f"and place it in '{MODELS_DIR}'."
                )

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if not MEDIAPIPE_AVAILABLE or self._landmarker is None:
            return {
                "sensor_type": "facial_expression",
                "score": 75.0,
                "details": {
                    "mock": True,
                    "primary": "neutral",
                    "scores": {"neutral": 1.0, "happy": 0.0, "confused": 0.0, "nervous": 0.0}
                }
            }

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = self._landmarker.detect(mp_image)

            if not result.face_blendshapes or len(result.face_blendshapes) == 0:
                return {
                    "sensor_type": "facial_expression",
                    "score": 0.0,
                    "details": {"detected": False, "primary": "neutral", "scores": {}}
                }

            blendshapes = result.face_blendshapes[0]
            scores = {"neutral": 0.5, "happy": 0.0, "confused": 0.0, "nervous": 0.0}

            # Map MediaPipe blendshape list to categories
            shape_map = {c.category_name: c.score for c in blendshapes}

            smile = (shape_map.get("mouthSmileLeft", 0.0) + shape_map.get("mouthSmileRight", 0.0)) / 2.0
            brow_down = (shape_map.get("browDownLeft", 0.0) + shape_map.get("browDownRight", 0.0)) / 2.0
            mouth_stretch = (shape_map.get("mouthStretchLeft", 0.0) + shape_map.get("mouthStretchRight", 0.0)) / 2.0

            scores["happy"] = round(smile, 2)
            scores["confused"] = round(brow_down, 2)
            scores["nervous"] = round(mouth_stretch, 2)
            
            max_other = max(scores["happy"], scores["confused"], scores["nervous"])
            scores["neutral"] = round(max(0.0, 1.0 - max_other), 2)

            primary = "neutral"
            if max_other > 0.3:
                if scores["happy"] == max_other:
                    primary = "happy"
                elif scores["confused"] == max_other:
                    primary = "confused"
                elif scores["nervous"] == max_other:
                    primary = "nervous"

            base_score = 100.0
            if primary == "nervous":
                base_score -= 30.0 * scores["nervous"]
            elif primary == "confused":
                base_score -= 15.0 * scores["confused"]

            return {
                "sensor_type": "facial_expression",
                "score": round(base_score, 1),
                "details": {
                    "detected": True,
                    "primary": primary,
                    "scores": scores,
                    "mock": False
                }
            }
        except Exception as e:
            logger.error(f"Error in facial_expression_analyzer: {e}")
            return {
                "sensor_type": "facial_expression",
                "score": 0.0,
                "details": {"detected": False, "error": str(e), "primary": "neutral", "scores": {}}
            }

    def release(self):
        if self._landmarker:
            self._landmarker.close()
