import asyncio
import logging
import time
import base64
import numpy as np
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from fastapi import WebSocket

from .sensors.posture_analyzer import PostureAnalyzer
from .sensors.eye_contact_analyzer import EyeContactAnalyzer
from .sensors.body_language_analyzer import BodyLanguageAnalyzer
from .sensors.attire_analyzer import AttireAnalyzer
from .sensors.confidence_metric import ConfidenceMetric
from .sensors.weight_config import WeightConfig, global_weights
from .sensors.facial_expression_analyzer import FacialExpressionAnalyzer
from .sensors.voice_analyzer import VoiceAnalyzer
from .sensors.engagement_tracker import EngagementTracker
from .sensors.professional_presence import ProfessionalPresenceEvaluator

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound MediaPipe processing
_executor = ThreadPoolExecutor(max_workers=2)


def _decode_frame(frame_data: str) -> Optional[np.ndarray]:
    """Decode a base64-encoded JPEG/PNG frame into a numpy array."""
    try:
        import cv2
        # Strip data URL prefix if present
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]
        raw = base64.b64decode(frame_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        logger.error(f"Frame decode error: {e}")
        return None


def _sanitize_for_json(obj):
    """
    Recursively convert numpy scalars/arrays to native Python types.
    Without this, json.dumps() inside websocket.send_json() crashes with
    'Object of type float32 is not JSON serializable', which kills the WebSocket.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class InterviewOrchestrator:
    """
    Central controller for the interview lifecycle.
    Manages all sensors, processes frames, and streams results back to the client.
    """

    def __init__(self):
        self.active_sessions: Dict[str, WebSocket] = {}
        self.session_weights: Dict[str, WeightConfig] = {}
        self.latest_session_metrics: Dict[str, Dict[str, Any]] = {}

        # Lazy initialized sensors
        self._posture_analyzer = None
        self._eye_contact_analyzer = None
        self._body_language_analyzer = None
        self._attire_analyzer = None
        self._confidence_metric = None
        self._facial_expression_analyzer = None
        self._voice_analyzer = None
        self._engagement_tracker = None
        self._professional_presence_evaluator = None

        # Frame throttle: process every Nth frame
        self._frame_counters: Dict[str, int] = {}
        self._process_every_n = 3  # analyze every 3rd frame for performance

    @property
    def posture_analyzer(self):
        if self._posture_analyzer is None:
            self._posture_analyzer = PostureAnalyzer()
        return self._posture_analyzer

    @property
    def eye_contact_analyzer(self):
        if self._eye_contact_analyzer is None:
            self._eye_contact_analyzer = EyeContactAnalyzer()
        return self._eye_contact_analyzer

    @property
    def body_language_analyzer(self):
        if self._body_language_analyzer is None:
            self._body_language_analyzer = BodyLanguageAnalyzer()
        return self._body_language_analyzer

    @property
    def attire_analyzer(self):
        if self._attire_analyzer is None:
            self._attire_analyzer = AttireAnalyzer()
        return self._attire_analyzer

    @property
    def confidence_metric(self):
        if self._confidence_metric is None:
            self._confidence_metric = ConfidenceMetric()
        return self._confidence_metric

    @property
    def facial_expression_analyzer(self):
        if self._facial_expression_analyzer is None:
            self._facial_expression_analyzer = FacialExpressionAnalyzer()
        return self._facial_expression_analyzer

    @property
    def voice_analyzer(self):
        if self._voice_analyzer is None:
            self._voice_analyzer = VoiceAnalyzer()
        return self._voice_analyzer

    @property
    def engagement_tracker(self):
        if self._engagement_tracker is None:
            self._engagement_tracker = EngagementTracker()
        return self._engagement_tracker

    @property
    def professional_presence_evaluator(self):
        if self._professional_presence_evaluator is None:
            self._professional_presence_evaluator = ProfessionalPresenceEvaluator()
        return self._professional_presence_evaluator

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_sessions[session_id] = websocket
        self.session_weights[session_id] = WeightConfig()
        self.latest_session_metrics[session_id] = {}
        self._frame_counters[session_id] = 0
        logger.info(f"Session {session_id} connected")

    def disconnect(self, session_id: str):
        self.active_sessions.pop(session_id, None)
        self.session_weights.pop(session_id, None)
        self.latest_session_metrics.pop(session_id, None)
        self._frame_counters.pop(session_id, None)
        if hasattr(self, "_audio_buffers"):
            self._audio_buffers.pop(session_id, None)
        if hasattr(self, "_audio_buffer_locks"):
            self._audio_buffer_locks.pop(session_id, None)
        if hasattr(self, "_latest_voice_metrics"):
            self._latest_voice_metrics.pop(session_id, None)
        logger.info(f"Session {session_id} disconnected")

    def get_latest_metrics(self, session_id: str) -> Dict[str, Any]:
        return self.latest_session_metrics.get(session_id, {})

    def get_weights(self, session_id: str) -> Dict[str, float]:
        wc = self.session_weights.get(session_id, global_weights)
        return wc.weights

    def update_weights(self, session_id: str, updates: Dict[str, float]):
        wc = self.session_weights.get(session_id)
        if wc:
            wc.update_weights(updates)

    def _run_sensors(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run all sensors synchronously (called in thread pool).
        Each sensor processes the same frame independently.
        """
        posture = self.posture_analyzer.process_frame(frame)
        eye_contact = self.eye_contact_analyzer.process_frame(frame)
        body_lang = self.body_language_analyzer.process_frame(frame)
        attire = self.attire_analyzer.process_frame(frame)
        facial_expression = self.facial_expression_analyzer.process_frame(frame)

        # Confidence is a meta-sensor using other sensor outputs
        confidence = self.confidence_metric.calculate(
            posture=posture,
            eye_contact=eye_contact,
            body_language=body_lang
        )

        return {
            "posture": posture,
            "eye_contact": eye_contact,
            "body_language": body_lang,
            "attire": attire,
            "confidence": confidence,
            "facial_expression": facial_expression,
        }

    async def process_frame(self, session_id: str, frame_data: str) -> Optional[Dict[str, Any]]:
        """
        Process a base64-encoded video frame through all sensors.
        Returns metrics dict or None if frame was skipped (throttle).
        """
        # Throttle
        self._frame_counters[session_id] = self._frame_counters.get(session_id, 0) + 1
        if self._frame_counters[session_id] % self._process_every_n != 0:
            return None

        frame = _decode_frame(frame_data)
        if frame is None:
            return None

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(_executor, self._run_sensors, frame)
        except Exception as e:
            logger.error(f"Sensor processing error: {e}")
            return None

        # Include voice metrics in the sensor results
        voice_telemetry = getattr(self, "_latest_voice_metrics", {}).get(session_id, {
            "pace": 70.0,
            "modulation": 70.0,
            "clarity": 70.0,
            "fluency": 70.0,
            "wpm": 0.0,
            "filler_count": 0
        })
        results["voice"] = {
            "sensor_type": "voice",
            "score": round(sum([voice_telemetry["pace"], voice_telemetry["modulation"], voice_telemetry["clarity"], voice_telemetry["fluency"]]) / 4.0, 1),
            "details": voice_telemetry
        }

        # Calculate composite sensors (Engagement & Professional Presence)
        latency = getattr(self, "_latencies", {}).get(session_id, None)
        
        results["engagement"] = self.engagement_tracker.calculate(
            eye_contact=results["eye_contact"],
            body_language=results["body_language"],
            facial_expression=results["facial_expression"],
            latency=latency
        )
        
        results["professional_presence"] = self.professional_presence_evaluator.calculate(
            posture=results["posture"],
            attire=results["attire"],
            eye_contact=results["eye_contact"],
            facial_expression=results["facial_expression"]
        )

        # Calculate composure & resilience scores
        posture_details = results.get("posture", {}).get("details", {})
        eye_details = results.get("eye_contact", {}).get("details", {})
        voice_details = results.get("voice", {}).get("details", {})
        facial_details = results.get("facial_expression", {}).get("details", {})
        facial_scores = facial_details.get("scores", {})

        is_slouching = posture_details.get("is_slouching", False)
        deviation = eye_details.get("deviation", 0.0)
        wpm = voice_details.get("wpm", 0.0)
        fluency = voice_details.get("fluency", 70.0)
        nervous = facial_details.get("nervous", facial_scores.get("nervous", 0.0))
        confused = facial_details.get("confused", facial_scores.get("confused", 0.0))
        filler_count = voice_details.get("filler_count", 0)

        composure_score = 100.0
        if is_slouching:
            composure_score -= 10.0
        if deviation > 0.15:
            composure_score -= 15.0
        if wpm > 170.0 or wpm < 90.0:
            composure_score -= 15.0
        composure_score -= 20.0 * nervous
        composure_score -= 10.0 * confused
        if fluency < 70.0:
            composure_score -= 15.0
        composure_score = max(0.0, min(100.0, composure_score))

        stress_score = 0.0
        if nervous > 0.2:
            stress_score += nervous * 40.0
        if deviation > 0.12:
            stress_score += deviation * 150.0
        if is_slouching:
            stress_score += 15.0
        if wpm > 160.0:
            stress_score += (wpm - 160.0) * 0.5
        if filler_count > 3:
            stress_score += filler_count * 2.0
        resilience_score = max(0.0, min(100.0, 100.0 - stress_score))

        if "confidence" not in results:
            results["confidence"] = {"sensor_type": "confidence", "score": 70.0, "details": {}}
        elif "details" not in results["confidence"] or results["confidence"]["details"] is None:
            results["confidence"]["details"] = {}

        results["confidence"]["details"]["composure"] = round(composure_score, 1)
        results["confidence"]["details"]["stress_resilience"] = round(resilience_score, 1)

        # Compute weighted overall score
        wc = self.session_weights.get(session_id, global_weights)
        scores = {k: v["score"] for k, v in results.items()}
        weighted_overall = wc.compute_weighted_score(scores)

        payload = {
            "type": "metrics_update",
            "timestamp": time.time(),
            "sensors": results,
            "weighted_overall": weighted_overall,
            "weights": wc.weights,
        }

        # Sanitize all numpy types to native Python types so json.dumps won't crash
        sanitized_payload = _sanitize_for_json(payload)
        self.latest_session_metrics[session_id] = sanitized_payload
        return sanitized_payload

    def release_all(self):
        """Release all MediaPipe resources."""
        self.posture_analyzer.release()
        self.eye_contact_analyzer.release()
        self.body_language_analyzer.release()
        self.attire_analyzer.release()
        self.facial_expression_analyzer.release()

    def _get_audio_lock(self, session_id: str):
        if not hasattr(self, "_audio_buffer_locks"):
            self._audio_buffer_locks = {}
        if session_id not in self._audio_buffer_locks:
            import threading
            self._audio_buffer_locks[session_id] = threading.Lock()
        return self._audio_buffer_locks[session_id]

    async def process_audio_chunk(self, session_id: str, chunk_data: str) -> Dict[str, Any]:
        """Decode base64 audio chunk and append to the session's audio buffer."""
        try:
            raw_audio = base64.b64decode(chunk_data)
        except Exception:
            raw_audio = b""
        
        if not hasattr(self, "_audio_buffers"):
            self._audio_buffers = {}
            
        lock = self._get_audio_lock(session_id)
        with lock:
            if session_id not in self._audio_buffers:
                self._audio_buffers[session_id] = bytearray()
                
            # Limit buffer to 15MB for safety
            if len(self._audio_buffers[session_id]) + len(raw_audio) > 15 * 1024 * 1024:
                logger.warning(f"Audio buffer limit exceeded for session {session_id}")
                return {
                    "type": "audio_ack",
                    "session_id": session_id,
                    "buffer_size": len(self._audio_buffers[session_id]),
                    "overflow": True,
                    "message": "Audio buffer limit exceeded (15 MB). Chunk dropped."
                }
            else:
                self._audio_buffers[session_id].extend(raw_audio)
                
            buffer_len = len(self._audio_buffers[session_id])
            
        # Record response latency if it's the first chunk for the current question
        if hasattr(self, "_question_start_times") and session_id in self._question_start_times:
            start_time = self._question_start_times.pop(session_id)
            latency = time.time() - start_time
            if not hasattr(self, "_latencies"):
                self._latencies = {}
            self._latencies[session_id] = latency

        return {
            "type": "audio_ack",
            "session_id": session_id,
            "buffer_size": buffer_len
        }

    async def transcribe_audio(self, session_id: str) -> str:
        """Transcribe audio using Whisper with fallback to API."""
        if not hasattr(self, "_audio_buffers"):
            return ""
        
        lock = self._get_audio_lock(session_id)
        with lock:
            if session_id not in self._audio_buffers:
                return ""
            buffer_data = bytes(self._audio_buffers[session_id])
            self._audio_buffers[session_id].clear()
            
        if not buffer_data:
            return ""

        try:
            # VoiceAnalyzer.transcribe is CPU-bound or performs HTTP requests, so run in executor
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(_executor, self.voice_analyzer.transcribe, buffer_data)
            
            if isinstance(res, dict):
                text = res.get("text", "")
                
                # Perform speech metrics analysis and save to telemetry
                if not hasattr(self, "_latest_voice_metrics"):
                    self._latest_voice_metrics = {}
                
                # 16-bit mono 16000Hz PCM has 32000 bytes per second (16000 * 2)
                audio_len = len(buffer_data) / 32000.0
                telemetry = self.voice_analyzer.analyze_speech(text, audio_len=audio_len, pcm_bytes=buffer_data)
                self._latest_voice_metrics[session_id] = telemetry
                
                return text
            else:
                return str(res)
        except Exception as e:
            logger.error(f"Transcription error in transcribe_audio: {e}", exc_info=True)
            return ""


orchestrator = InterviewOrchestrator()
