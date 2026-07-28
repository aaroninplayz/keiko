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
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not auto-migrate DB schema: {e}")

ensure_db_schema()

