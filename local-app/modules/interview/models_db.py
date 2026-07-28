import datetime
from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship
from core.database import Base

class InterviewSession(Base):
    """
    SQLAlchemy model representing the state of an interview session.
    """
    __tablename__ = "interview_sessions"

    # Primary session identifier
    session_id = Column(String(36), primary_key=True, index=True)
    
    # Session configurations
    interview_type = Column(String(50), nullable=False)  # HR, Tech, Situational
    difficulty_mode = Column(String(50), nullable=False) # Beginner, Intermediate, Advanced, Adaptive
    duration_type = Column(String(20), nullable=False)   # questions, minutes
    duration_value = Column(Integer, nullable=False)     # Limit value (e.g., 3 questions or 15 mins)
    
    # Dynamic lifecycle/state fields
    phase = Column(String(20), nullable=False, default="Warmup") # Warmup, Main, Concluding, Completed
    status = Column(String(20), nullable=False, default="active") # active, completed
    current_index = Column(Integer, nullable=False, default=0)
    current_question = Column(Text, nullable=True)
    next_question_feedback = Column(Text, nullable=True)
    
    # State & Metrics Trackers
    interviewer_interest = Column(Float, nullable=False, default=50.0) # 0-100
    job_fit_score = Column(Float, nullable=False, default=50.0)        # 0-100
    conversation_summary = Column(Text, nullable=True)                 # Compressed conversation history (every 5 turns)
    
    # Cached context objects (enables full session recovery without local files)
    resume_profile = Column(JSON, nullable=True)     # Extracted candidate profile details
    jd_profile = Column(JSON, nullable=True)         # Extracted job description details
    match_results = Column(JSON, nullable=True)      # Match alignment and skill gaps
    candidate_profile = Column(JSON, nullable=True)  # Compiled normalized candidate profile
    
    # Compiled final evaluation report
    final_report = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), onupdate=lambda: datetime.datetime.now(datetime.timezone.utc))

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")
    history = relationship(
        "ConversationHistory",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ConversationHistory.timestamp"
    )

    @property
    def type(self) -> str:
        """Alias for interview_type to match conceptual naming."""
        return self.interview_type

    @property
    def difficulty(self) -> str:
        """Alias for difficulty_mode to match conceptual naming."""
        return self.difficulty_mode


class ConversationHistory(Base):
    """
    SQLAlchemy model representing a single Q&A and evaluation history record.
    """
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("interview_sessions.session_id", ondelete="CASCADE"), nullable=False)
    
    # Content fields
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    
    # Evaluation results
    evaluation_score = Column(Float, nullable=True) # maps to accuracy_score from CentralEvaluator
    feedback = Column(Text, nullable=True)
    quality_tier = Column(String(50), nullable=True)
    word_count = Column(Integer, nullable=True)
    matched_keywords = Column(JSON, nullable=True)  # List of strings (e.g. ["Python", "Fastapi"])
    
    # Sensor metrics
    posture_score = Column(Float, nullable=True)
    eye_contact_score = Column(Float, nullable=True)
    emotions = Column(JSON, nullable=True)          # List of strings (e.g. ["neutral", "nervous"])
    metrics_raw = Column(JSON, nullable=True)       # Store full metrics dictionary
    
    # Time tracking
    timestamp = Column(Float, nullable=False)       # Epoch float as used by sensors/evaluator

    # Relationships
    session = relationship("InterviewSession", back_populates="history")


class ResumeRecord(Base):
    __tablename__ = "resume_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_name = Column(String(100), nullable=False)
    role_applied = Column(String(100), nullable=False)
    email = Column(String(100), nullable=True)
    phone = Column(String(50), nullable=True)
    overall_score = Column(Float, nullable=False, default=75.0)
    skills_match_pct = Column(Float, nullable=False, default=80.0)
    experience_level = Column(String(50), nullable=False, default="Mid-Level")
    status = Column(String(50), nullable=False, default="Parsed")
    uploaded_at = Column(String(50), nullable=False, default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    score_breakdown = Column(JSON, nullable=True)
    skills_tags = Column(JSON, nullable=True)
    experience_timeline = Column(JSON, nullable=True)
    red_flags = Column(JSON, nullable=True)
    ai_summary = Column(Text, nullable=True)
    source = Column(String(50), nullable=False, default="Direct Upload")


class JobRecord(Base):
    __tablename__ = "job_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_title = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False)
    quality_score = Column(Float, nullable=False, default=85.0)
    required_skills_count = Column(Integer, nullable=False, default=8)
    status = Column(String(50), nullable=False, default="Active")
    seniority = Column(String(50), nullable=False, default="Senior")
    last_updated = Column(String(50), nullable=False, default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    candidates_matched_count = Column(Integer, nullable=False, default=12)
    score_breakdown = Column(JSON, nullable=True)
    must_have_skills = Column(JSON, nullable=True)
    nice_to_have_skills = Column(JSON, nullable=True)
    inclusivity_analysis = Column(JSON, nullable=True)
    market_comparison = Column(JSON, nullable=True)
    ai_suggestions = Column(JSON, nullable=True)


class VideoRecord(Base):
    __tablename__ = "video_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_name = Column(String(100), nullable=False)
    interview_type = Column(String(50), nullable=False, default="Technical")
    duration_minutes = Column(Integer, nullable=False, default=30)
    video_score = Column(Float, nullable=False, default=82.0)
    engagement_rate = Column(Float, nullable=False, default=88.0)
    eye_contact_pct = Column(Float, nullable=False, default=85.0)
    status = Column(String(50), nullable=False, default="Analyzed")
    recorded_at = Column(String(50), nullable=False, default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    flagged_behaviors_count = Column(Integer, nullable=False, default=0)
    body_language = Column(JSON, nullable=True)
    communication = Column(JSON, nullable=True)
    visual_presentation = Column(JSON, nullable=True)
    ai_insights = Column(JSON, nullable=True)


class AudioRecord(Base):
    __tablename__ = "audio_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_name = Column(String(100), nullable=False)
    duration_minutes = Column(Integer, nullable=False, default=25)
    audio_score = Column(Float, nullable=False, default=84.0)
    clarity_score = Column(Float, nullable=False, default=88.0)
    sentiment = Column(String(50), nullable=False, default="Confident")
    speaking_rate_wpm = Column(Float, nullable=False, default=140.0)
    filler_word_count = Column(Integer, nullable=False, default=3)
    status = Column(String(50), nullable=False, default="Processed")
    recorded_at = Column(String(50), nullable=False, default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    speech_analysis = Column(JSON, nullable=True)
    tone_sentiment = Column(JSON, nullable=True)
    fluency = Column(JSON, nullable=True)
    content_quality = Column(JSON, nullable=True)
    ai_insights = Column(JSON, nullable=True)


class CandidateEvaluationRecord(Base):
    __tablename__ = "candidate_evaluation_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_name = Column(String(100), nullable=False)
    position_applied = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False, default="Engineering")
    resume_score = Column(Float, nullable=False, default=80.0)
    jd_match_score = Column(Float, nullable=False, default=85.0)
    video_score = Column(Float, nullable=False, default=82.0)
    audio_score = Column(Float, nullable=False, default=84.0)
    final_composite_score = Column(Float, nullable=False, default=83.0)
    recommendation = Column(String(50), nullable=False, default="Strongly Recommend")
    status = Column(String(50), nullable=False, default="Complete")
    evaluated_at = Column(String(50), nullable=False, default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    executive_summary = Column(Text, nullable=True)
    weight_config = Column(JSON, nullable=True)
    quadrant_scores = Column(JSON, nullable=True)


from core.database import engine
from sqlalchemy import inspect, text

def ensure_db_schema():
    try:
        inspector = inspect(engine)
        if "interview_sessions" in inspector.get_table_names():
            columns = [c["name"] for c in inspector.get_columns("interview_sessions")]
            with engine.begin() as conn:
                if "interviewer_interest" not in columns:
                    conn.execute(text("ALTER TABLE interview_sessions ADD COLUMN interviewer_interest FLOAT DEFAULT 50.0"))
                if "job_fit_score" not in columns:
                    conn.execute(text("ALTER TABLE interview_sessions ADD COLUMN job_fit_score FLOAT DEFAULT 50.0"))
                if "conversation_summary" not in columns:
                    conn.execute(text("ALTER TABLE interview_sessions ADD COLUMN conversation_summary TEXT"))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not auto-migrate DB schema: {e}")

ensure_db_schema()


