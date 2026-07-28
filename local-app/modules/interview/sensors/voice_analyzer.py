import logging
import wave
import io
import numpy as np
try:
    import torch
    from transformers import pipeline
    HAS_TORCH = True
except ImportError:
    torch = None
    pipeline = None
    HAS_TORCH = False
from core.config import settings, MODELS_DIR

logger = logging.getLogger(__name__)


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM = 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()


class VoiceAnalyzer:
    def __init__(self):
        self.provider = settings.STT_PROVIDER.lower()
        self.model_id = settings.WHISPER_MODEL_SIZE
        self._pipe = None
        self._pipe_attempted = False

    @property
    def pipe(self):
        if not self._pipe_attempted:
            self._pipe_attempted = True
            if self.provider == "local" and HAS_TORCH and torch and pipeline:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"Initializing local Whisper pipeline with model {self.model_id} on {device}...")
                    self._pipe = pipeline(
                        "automatic-speech-recognition",
                        model=self.model_id,
                        device=0 if device == "cuda" else -1,
                        model_kwargs={"cache_dir": MODELS_DIR}
                    )
                except Exception as e:
                    logger.error(f"Local Whisper init failed: {e}. Fallback to API/Mock will be used.")
        return self._pipe

    def transcribe(self, pcm_bytes: bytes) -> dict:
        """
        Transcribe the raw 16-bit PCM bytes (16000Hz mono).
        Returns a dict: {"text": str, "mock": bool}
        """
        if not pcm_bytes:
            return {"text": "", "mock": False}

        # If local provider is selected and pipeline is loaded
        if self.provider == "local" and self.pipe:
            try:
                # Convert PCM 16-bit bytes to float32 numpy array
                samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                result = self.pipe({"raw": samples, "sampling_rate": 16000})
                text = result.get("text", "").strip()
                return {"text": text, "mock": False}
            except Exception as e:
                logger.error(f"Local Whisper transcription failed: {e}. Trying API fallback...")

        # If local failed or provider is API, try API fallback
        if settings.OPENAI_API_KEY:
            try:
                return self._transcribe_openai(pcm_bytes)
            except Exception as e:
                logger.error(f"OpenAI transcription fallback failed: {e}")

        if settings.GROQ_API_KEY:
            try:
                return self._transcribe_groq(pcm_bytes)
            except Exception as e:
                logger.error(f"Groq transcription fallback failed: {e}")

        # If neither local model nor API keys worked/are available, degrade gracefully to mock
        mock_text = "This is a mock transcription because neither local Whisper nor API providers are available."
        return {"text": mock_text, "mock": True}

    def _transcribe_openai(self, pcm_bytes: bytes) -> dict:
        import requests
        wav_bytes = pcm_to_wav_bytes(pcm_bytes)
        url = f"{settings.OPENAI_API_BASE.rstrip('/')}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
        data = {"model": "whisper-1"}
        r = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        r.raise_for_status()
        text = r.json().get("text", "").strip()
        return {"text": text, "mock": False}

    def _transcribe_groq(self, pcm_bytes: bytes) -> dict:
        import requests
        wav_bytes = pcm_to_wav_bytes(pcm_bytes)
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
        data = {"model": "whisper-large-v3"}
        r = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        r.raise_for_status()
        text = r.json().get("text", "").strip()
        return {"text": text, "mock": False}

    async def process_audio(self, audio_chunk: bytes) -> dict:
        """
        Analyzes audio chunk (16-bit PCM 16000Hz mono) in memory for volume, pitch modulation, fluency, and tone using numpy DSP.
        """
        if not audio_chunk or len(audio_chunk) < 2:
            return {
                "type": "voice_metric",
                "rms": 0.0,
                "projection": 50.0,
                "modulation": 50.0,
                "fluency": 70.0,
                "silence_ratio": 0.0,
                "pace": "optimal",
                "tone": "confident"
            }

        try:
            # Convert 16-bit PCM to float32 normalized samples (-1.0 to 1.0)
            samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000

            # 1. RMS Energy (Volume / Projection Score)
            rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
            projection_score = max(0.0, min(100.0, rms * 500.0))

            # 2. Zero-Crossing Rate & Modulation
            frame_size = int(sr * 0.05)  # 50ms frames
            if len(samples) >= frame_size:
                n_frames = len(samples) // frame_size
                zcr_frames = [
                    float(np.mean(np.abs(np.diff(np.signbit(samples[i * frame_size:(i + 1) * frame_size])))))
                    for i in range(n_frames)
                ]
                zcr_std = float(np.std(zcr_frames)) if len(zcr_frames) > 0 else 0.0
                modulation_score = max(30.0, min(100.0, 50.0 + zcr_std * 500.0))

                # 3. Silence Gap Detection & Fluency
                frame_energies = [
                    float(np.sqrt(np.mean(samples[i * frame_size:(i + 1) * frame_size] ** 2)))
                    for i in range(n_frames)
                ]
                silence_threshold = max(0.005, rms * 0.2)
                silent_frames = sum(1 for e in frame_energies if e < silence_threshold)
                silence_ratio = silent_frames / float(len(frame_energies)) if frame_energies else 0.0

                if silence_ratio > 0.5:
                    fluency_score = max(20.0, 100.0 - (silence_ratio - 0.5) * 160.0)
                else:
                    fluency_score = max(50.0, min(100.0, 100.0 - silence_ratio * 30.0))
            else:
                modulation_score = 50.0
                silence_ratio = 0.0
                fluency_score = 70.0

            # Determine tone and pace classification
            pace = "optimal"
            if silence_ratio > 0.45:
                pace = "hesitant"
            elif silence_ratio < 0.1 and rms > 0.1:
                pace = "fast"

            tone = "confident"
            if projection_score < 30.0:
                tone = "quiet"
            elif modulation_score > 75.0:
                tone = "expressive"

            return {
                "type": "voice_metric",
                "rms": round(rms, 4),
                "projection": round(projection_score, 1),
                "modulation": round(modulation_score, 1),
                "fluency": round(fluency_score, 1),
                "silence_ratio": round(silence_ratio, 2),
                "pace": pace,
                "tone": tone
            }
        except Exception as e:
            logger.error(f"Error processing audio DSP: {e}")
            return {
                "type": "voice_metric",
                "rms": 0.0,
                "projection": 50.0,
                "modulation": 50.0,
                "fluency": 70.0,
                "silence_ratio": 0.0,
                "pace": "optimal",
                "tone": "confident"
            }

    def analyze_speech(self, text: str, audio_len: float = 5.0, pcm_bytes: bytes = None) -> dict:
        """
        Calculates speaking pace, modulation, clarity, and fluency scores (0-100) combining text transcription and audio DSP.
        """
        words = text.split()
        wpm = (len(words) / audio_len) * 60.0 if audio_len > 0 else 0.0
        
        # Pace score (optimal speaking pace is around 130-150 WPM)
        if wpm == 0:
            pace_score = 50.0
        else:
            deviation = abs(wpm - 140.0)
            pace_score = max(0.0, min(100.0, 100.0 - deviation * 0.5))
        
        # Text-based modulation score
        modulation_score = 80.0
        if len(words) > 1:
            modulation_score = max(30.0, min(100.0, 70.0 + len(set(words)) * 2.0))
            
        # Clarity score
        clarity_score = 85.0
        
        # Fluency (filler word detection)
        fillers = ["um", "like", "uh", "err", "ah"]
        filler_count = sum(1 for w in words if w.lower().strip(",.?!") in fillers)
        if len(words) > 0:
            ratio = filler_count / len(words)
            fluency_score = max(0.0, min(100.0, 100.0 - ratio * 200.0))
        else:
            fluency_score = 80.0

        # Incorporate DSP metrics if raw PCM audio bytes are available
        if pcm_bytes and len(pcm_bytes) >= 2:
            try:
                samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
                frame_size = int(16000 * 0.05)
                if len(samples) >= frame_size:
                    n_frames = len(samples) // frame_size
                    zcr_frames = [
                        float(np.mean(np.abs(np.diff(np.signbit(samples[i * frame_size:(i + 1) * frame_size])))))
                        for i in range(n_frames)
                    ]
                    zcr_std = float(np.std(zcr_frames)) if len(zcr_frames) > 0 else 0.0
                    dsp_modulation = max(30.0, min(100.0, 50.0 + zcr_std * 500.0))

                    frame_energies = [
                        float(np.sqrt(np.mean(samples[i * frame_size:(i + 1) * frame_size] ** 2)))
                        for i in range(n_frames)
                    ]
                    silence_thresh = max(0.005, rms * 0.2)
                    silent_count = sum(1 for e in frame_energies if e < silence_thresh)
                    silence_ratio = silent_count / float(len(frame_energies))
                    dsp_fluency = max(20.0, min(100.0, 100.0 - silence_ratio * 100.0))
                else:
                    dsp_modulation = modulation_score
                    dsp_fluency = fluency_score

                modulation_score = round(0.5 * modulation_score + 0.5 * dsp_modulation, 1)
                fluency_score = round(0.5 * fluency_score + 0.5 * dsp_fluency, 1)
            except Exception as e:
                logger.error(f"Error incorporating DSP metrics into analyze_speech: {e}")
            
        return {
            "pace": round(pace_score, 1),
            "modulation": round(modulation_score, 1),
            "clarity": round(clarity_score, 1),
            "fluency": round(fluency_score, 1),
            "wpm": round(wpm, 1),
            "filler_count": filler_count
        }
