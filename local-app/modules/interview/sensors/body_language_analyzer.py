import logging
import os
import numpy as np
import math
from typing import Dict, Any
from collections import deque

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models")

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HolisticLandmarker,
        HolisticLandmarkerOptions,
        RunningMode,
    )
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed. BodyLanguageAnalyzer will return mock data.")

# Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
NOSE = 0


class BodyLanguageAnalyzer:
    """
    Analyzes body language using MediaPipe Tasks API HolisticLandmarker.
    Tracks: gesture frequency, hand movement, head nodding, shoulder openness.
    Returns an engagement/openness score 0-100.
    """

    def __init__(self):
        self._landmarker = None
        if MEDIAPIPE_AVAILABLE:
            model_path = os.path.join(MODELS_DIR, "holistic_landmarker.task")
            if os.path.exists(model_path):
                options = HolisticLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.IMAGE,
                    min_pose_detection_confidence=0.5,
                )
                self._landmarker = HolisticLandmarker.create_from_options(options)
            else:
                logger.error(
                    f"Holistic model file 'holistic_landmarker.task' not found at {model_path}. "
                    "Please download it from https://storage.googleapis.com/mediapipe-models/ "
                    f"and place it in '{MODELS_DIR}'."
                )
        self._prev_wrists = None
        self._hand_movement_history = deque(maxlen=30)
        self._head_y_history = deque(maxlen=20)
        self._score_history = deque(maxlen=30)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if not MEDIAPIPE_AVAILABLE or self._landmarker is None:
            return {"sensor_type": "body_language", "score": 65.0, "details": {"mock": True}}

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = self._landmarker.detect(mp_image)

            gesture_score = 50.0
            openness_score = 50.0
            head_nod_score = 50.0
            hand_activity = 0.0

            has_pose = bool(result.pose_landmarks and len(result.pose_landmarks) > 0)

            if has_pose:
                pose = result.pose_landmarks[0]

                # --- Shoulder openness ---
                shoulder_width = abs(pose[LEFT_SHOULDER].x - pose[RIGHT_SHOULDER].x)
                openness_score = min(100, shoulder_width * 400)

                # --- Head nodding ---
                self._head_y_history.append(pose[NOSE].y)
                if len(self._head_y_history) > 5:
                    y_vals = list(self._head_y_history)
                    y_diffs = [abs(y_vals[i] - y_vals[i - 1]) for i in range(1, len(y_vals))]
                    avg_movement = sum(y_diffs) / len(y_diffs)
                    if 0.002 < avg_movement < 0.015:
                        head_nod_score = 85.0
                    elif avg_movement < 0.002:
                        head_nod_score = 40.0
                    else:
                        head_nod_score = 55.0

            # --- Hand gesture tracking ---
            has_hands = bool(
                (result.left_hand_landmarks and len(result.left_hand_landmarks) > 0) or
                (result.right_hand_landmarks and len(result.right_hand_landmarks) > 0)
            )

            if has_hands:
                current_wrists = {}
                if result.left_hand_landmarks and len(result.left_hand_landmarks) > 0:
                    w = result.left_hand_landmarks[0][0]  # wrist is index 0
                    current_wrists["left"] = (w.x, w.y)
                if result.right_hand_landmarks and len(result.right_hand_landmarks) > 0:
                    w = result.right_hand_landmarks[0][0]
                    current_wrists["right"] = (w.x, w.y)

                if self._prev_wrists:
                    total_movement = 0.0
                    count = 0
                    for hand_key in current_wrists:
                        if hand_key in self._prev_wrists:
                            dx = abs(current_wrists[hand_key][0] - self._prev_wrists[hand_key][0])
                            dy = abs(current_wrists[hand_key][1] - self._prev_wrists[hand_key][1])
                            total_movement += math.sqrt(dx**2 + dy**2)
                            count += 1
                    if count > 0:
                        hand_activity = total_movement / count

                self._prev_wrists = current_wrists
                self._hand_movement_history.append(hand_activity)

                avg_hand = sum(self._hand_movement_history) / len(self._hand_movement_history) if self._hand_movement_history else 0
                if 0.005 < avg_hand < 0.05:
                    gesture_score = 85.0
                elif avg_hand >= 0.05:
                    gesture_score = 50.0
                else:
                    gesture_score = 55.0
            else:
                gesture_score = 40.0
                self._prev_wrists = None

            raw_score = (openness_score * 0.30) + (gesture_score * 0.35) + (head_nod_score * 0.35)
            raw_score = max(0, min(100, raw_score))

            self._score_history.append(raw_score)
            smoothed = sum(self._score_history) / len(self._score_history)

            return {
                "sensor_type": "body_language",
                "score": round(smoothed, 1),
                "details": {
                    "detected": has_pose,
                    "openness": round(openness_score, 1),
                    "gesture_activity": round(gesture_score, 1),
                    "head_engagement": round(head_nod_score, 1),
                    "hand_activity_raw": round(hand_activity, 4),
                },
            }
        except Exception as e:
            logger.error(f"Error in body_language_analyzer: {e}")
            return {"sensor_type": "body_language", "score": 0.0, "details": {"detected": False, "error": str(e)}}

    def release(self):
        if self._landmarker:
            self._landmarker.close()
