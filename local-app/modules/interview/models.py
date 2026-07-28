from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ResumeContext(BaseModel):
    parsed_text: str
    skills: List[str]
    experience_years: int


class JobDescriptionContext(BaseModel):
    title: str
    requirements: List[str]


class WeightConfigSchema(BaseModel):
    """Schema for reading/updating sensor weights via the API."""
    posture: float = Field(0.15, ge=0.0, le=1.0)
    eye_contact: float = Field(0.15, ge=0.0, le=1.0)
    body_language: float = Field(0.15, ge=0.0, le=1.0)
    attire: float = Field(0.10, ge=0.0, le=1.0)
    confidence: float = Field(0.10, ge=0.0, le=1.0)
    facial_expression: float = Field(0.10, ge=0.0, le=1.0)
    voice: float = Field(0.15, ge=0.0, le=1.0)
    engagement: float = Field(0.05, ge=0.0, le=1.0)
    professional_presence: float = Field(0.05, ge=0.0, le=1.0)


class SensorMetricSnapshot(BaseModel):
    """A single real-time metric reading from a sensor."""
    sensor_type: str
    score: float = Field(..., ge=0.0, le=100.0)
    details: Dict[str, Any] = {}
    timestamp: float = 0.0


class MetricsUpdate(BaseModel):
    """Full bundle of all sensor readings sent to the frontend in real-time."""
    posture: SensorMetricSnapshot
    eye_contact: SensorMetricSnapshot
    attire: SensorMetricSnapshot
    body_language: SensorMetricSnapshot
    confidence: SensorMetricSnapshot
    facial_expression: SensorMetricSnapshot
    voice: SensorMetricSnapshot
    engagement: SensorMetricSnapshot
    professional_presence: SensorMetricSnapshot
    weighted_overall: float = 0.0
    weights: Dict[str, float] = {}


class InterviewEvaluation(BaseModel):
    overall_score: float
    attire_score: float
    posture_score: float
    eye_contact_score: float
    body_language_score: float
    confidence_score: float
    feedback: str
    weights_used: Dict[str, float] = {}


# --- Interview Session API Models ---

class InterviewType(str, Enum):
    HR = 'HR'
    Tech = 'Tech'
    Situational = 'Situational'

class DifficultyMode(str, Enum):
    Beginner = 'Beginner'
    Intermediate = 'Intermediate'
    Advanced = 'Advanced'
    Adaptive = 'Adaptive'

class DurationType(str, Enum):
    questions = 'questions'
    minutes = 'minutes'
    unlimited = 'unlimited'

class StartSessionRequest(BaseModel):
    interview_type: InterviewType
    difficulty_mode: DifficultyMode
    duration_type: DurationType
    duration_value: int = Field(..., gt=0)
    session_id: Optional[str] = None
    candidate_context: Optional[Dict[str, Any]] = None
    jd_text: Optional[str] = None

class SubmitAnswerRequest(BaseModel):
    session_id: str
    answer: str
