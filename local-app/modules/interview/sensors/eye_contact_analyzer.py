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
    logger.warning("MediaPipe not installed. EyeContactAnalyzer will return mock data.")

# Iris and eye landmark indices (same as legacy Face Mesh)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


class EyeContactAnalyzer:
    """
    Real-time eye contact detection using MediaPipe Tasks API FaceLandmarker.
    Uses iris landmarks to compute gaze direction.
    Returns a score 0-100 (100 = perfect eye contact).
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
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self._landmarker = FaceLandmarker.create_from_options(options)
            else:
                logger.error(
                    f"Face model file 'face_landmarker.task' not found at {model_path}. "
                    "Please download it from https://storage.googleapis.com/mediapipe-models/ "
                    f"and place it in '{MODELS_DIR}'."
                )
        self._history: list = []
        self._max_history = 60  # Track window of last 60 frames (~6-10 seconds)
        self._total_contact_frames = 0
        self._total_evaluated_frames = 0

    def _iris_ratio(self, landmarks, iris_indices, inner_idx, outer_idx) -> float:
        iris_x = sum(landmarks[i].x for i in iris_indices) / len(iris_indices)
        inner_x = landmarks[inner_idx].x
        outer_x = landmarks[outer_idx].x
        denom = abs(outer_x - inner_x) + 1e-8
        ratio = (iris_x - min(inner_x, outer_x)) / denom
        return ratio

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if not MEDIAPIPE_AVAILABLE or self._landmarker is None:
            return {
                "sensor_type": "eye_contact",
                "score": 75.0,
                "details": {
                    "mock": True,
                    "eye_contact_percentage": 75.0,
                    "is_making_contact": True
                }
            }

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = self._landmarker.detect(mp_image)

            if not result.face_landmarks or len(result.face_landmarks) == 0:
                self._history.append(0.0)
                if len(self._history) > self._max_history:
                    self._history.pop(0)
                score_pct = (sum(1 for s in self._history if s > 50) / max(1, len(self._history))) * 100.0
                return {
                    "sensor_type": "eye_contact",
                    "score": round(score_pct, 1),
                    "details": {
                        "detected": False,
                        "eye_contact_percentage": round(score_pct, 1),
                        "is_making_contact": False
                    }
                }

            landmarks = result.face_landmarks[0]

            # Check if iris landmarks are present
            if len(landmarks) < 478:
                self._history.append(60.0)
                if len(self._history) > self._max_history:
                    self._history.pop(0)
                score_pct = round(sum(self._history) / len(self._history), 1)
                return {
                    "sensor_type": "eye_contact",
                    "score": score_pct,
                    "details": {
                        "detected": True,
                        "iris_available": False,
                        "eye_contact_percentage": score_pct,
                        "is_making_contact": True
                    }
                }

            left_ratio = self._iris_ratio(landmarks, LEFT_IRIS, LEFT_EYE_INNER, LEFT_EYE_OUTER)
            right_ratio = self._iris_ratio(landmarks, RIGHT_IRIS, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
            avg_ratio = (left_ratio + right_ratio) / 2.0

            # Gaze boundaries for direct camera contact
            if 0.38 <= avg_ratio <= 0.62:
                deviation = 0.0
            elif avg_ratio < 0.38:
                deviation = 0.38 - avg_ratio
            else:
                deviation = avg_ratio - 0.62

            # Instantaneous contact score for this frame
            frame_score = max(0.0, min(100.0, (1.0 - deviation * 4.0) * 100.0))

            # Vertical pitch deviation check
            left_eye_center_y = sum(landmarks[i].y for i in LEFT_IRIS) / len(LEFT_IRIS)
            right_eye_center_y = sum(landmarks[i].y for i in RIGHT_IRIS) / len(RIGHT_IRIS)
            eye_mid_y = (left_eye_center_y + right_eye_center_y) / 2.0
            nose_y = landmarks[1].y
            vertical_deviation = abs(nose_y - eye_mid_y)
            if vertical_deviation > 0.22:
                frame_score *= 0.5

            is_contact = frame_score >= 60.0

            self._total_evaluated_frames += 1
            if is_contact:
                self._total_contact_frames += 1

            self._history.append(100.0 if is_contact else 0.0)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            # Rolling window percentage of time looking directly at camera
            window_contact_pct = (sum(self._history) / len(self._history)) if self._history else 0.0

            return {
                "sensor_type": "eye_contact",
                "score": round(window_contact_pct, 1),
                "details": {
                    "detected": True,
                    "iris_available": True,
                    "horizontal_ratio": round(avg_ratio, 3),
                    "deviation": round(deviation, 3),
                    "is_making_contact": is_contact,
                    "eye_contact_percentage": round(window_contact_pct, 1),
                    "cumulative_contact_pct": round((self._total_contact_frames / max(1, self._total_evaluated_frames)) * 100.0, 1),
                    "raw_score": round(frame_score, 1),
                },
            }
        except Exception as e:
            logger.error(f"Error in eye_contact_analyzer: {e}")
            return {
                "sensor_type": "eye_contact",
                "score": 0.0,
                "details": {"detected": False, "error": str(e), "eye_contact_percentage": 0.0}
            }

    def release(self):
        if self._landmarker:
            self._landmarker.close()

