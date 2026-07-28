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
        self._max_history = 20

    def _iris_ratio(self, landmarks, iris_indices, inner_idx, outer_idx) -> float:
        iris_x = sum(landmarks[i].x for i in iris_indices) / len(iris_indices)
        inner_x = landmarks[inner_idx].x
        outer_x = landmarks[outer_idx].x
        denom = abs(outer_x - inner_x) + 1e-8
        ratio = (iris_x - min(inner_x, outer_x)) / denom
        return ratio

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if not MEDIAPIPE_AVAILABLE or self._landmarker is None:
            return {"sensor_type": "eye_contact", "score": 70.0, "details": {"mock": True}}

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = self._landmarker.detect(mp_image)

            if not result.face_landmarks or len(result.face_landmarks) == 0:
                return {"sensor_type": "eye_contact", "score": 0.0, "details": {"detected": False}}

            landmarks = result.face_landmarks[0]

            # Check if we have enough landmarks for iris detection
            if len(landmarks) < 478:
                # No iris landmarks available, fall back to basic face detection
                return {"sensor_type": "eye_contact", "score": 60.0, "details": {"detected": True, "iris_available": False}}

            left_ratio = self._iris_ratio(landmarks, LEFT_IRIS, LEFT_EYE_INNER, LEFT_EYE_OUTER)
            right_ratio = self._iris_ratio(landmarks, RIGHT_IRIS, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
            avg_ratio = (left_ratio + right_ratio) / 2.0

            # Define a generous "okay" zone representing gaze pointing anywhere near the camera (0.38 - 0.62)
            if 0.38 <= avg_ratio <= 0.62:
                deviation = 0.0
            elif avg_ratio < 0.38:
                deviation = 0.38 - avg_ratio
            else:
                deviation = avg_ratio - 0.62

            # Smoothly degrade score only when the pupil shifts outside the generous proximity boundary
            raw_score = max(0, min(100, (1.0 - deviation * 4.0) * 100))

            # Vertical gaze check - also make it generous!
            nose = landmarks[1]
            left_eye_center_y = sum(landmarks[i].y for i in LEFT_IRIS) / len(LEFT_IRIS)
            right_eye_center_y = sum(landmarks[i].y for i in RIGHT_IRIS) / len(RIGHT_IRIS)
            eye_mid_y = (left_eye_center_y + right_eye_center_y) / 2.0

            vertical_deviation = abs(nose.y - eye_mid_y)
            # Only penalize extreme vertical head tilts or looking far up/down
            if vertical_deviation > 0.22:
                raw_score *= 0.5

            is_contact = raw_score > 60

            self._history.append(raw_score)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            smoothed = sum(self._history) / len(self._history)

            return {
                "sensor_type": "eye_contact",
                "score": round(smoothed, 1),
                "details": {
                    "detected": True,
                    "iris_available": True,
                    "horizontal_ratio": round(avg_ratio, 3),
                    "deviation": round(deviation, 3),
                    "is_making_contact": is_contact,
                    "raw_score": round(raw_score, 1),
                },
            }
        except Exception as e:
            logger.error(f"Error in eye_contact_analyzer: {e}")
            return {"sensor_type": "eye_contact", "score": 0.0, "details": {"detected": False, "error": str(e)}}

    def release(self):
        if self._landmarker:
            self._landmarker.close()
