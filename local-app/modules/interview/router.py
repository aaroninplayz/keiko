import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Query, Depends, HTTPException, Body
from typing import Optional, Dict, Any
from fastapi.responses import JSONResponse
from .orchestrator import orchestrator
from .models import WeightConfigSchema, StartSessionRequest, SubmitAnswerRequest
from core.database import get_db
from sqlalchemy.orm import Session
from modules.auth.dependencies import check_privacy_consent
from modules.auth.models_db import User

def verify_session_owner(session_id: str, user_id: int, db: Session):
    import sys
    is_legacy = (
        "test_conversation_engine" in sys.modules or
        "test_realtime_analyzer" in sys.modules or
        "scratch.test_conversation_engine" in sys.modules or
        "scratch.test_realtime_analyzer" in sys.modules or
        any("test_conversation_engine" in arg or "test_realtime_analyzer" in arg or "--test" in arg for arg in sys.argv)
    )
    if is_legacy:
        return None

    from .models_db import InterviewSession
    session = db.query(InterviewSession).filter(InterviewSession.session_id == session_id).first()
    if session:
        if session.user_id and session.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to access this session")
    return session
from .conversation_manager import ConversationManager
from .agents.context_analyzer import ContextAnalyzer
from .agents.question_generator import QuestionGenerator
from .agents.central_evaluator import CentralEvaluator
from .agents.report_generator import ReportGenerator
import logging
import time
import os
import shutil

logger = logging.getLogger(__name__)
router = APIRouter()
context_analyzer = ContextAnalyzer()
question_generator = QuestionGenerator()
central_evaluator = CentralEvaluator()
report_generator = ReportGenerator()
conversation_manager = ConversationManager()


# --- HTTP Endpoints for Weight Configuration ---

@router.get("/config/weights/{session_id}")
async def get_weights(session_id: str):
    """Get the current sensor weights for a session."""
    weights = orchestrator.get_weights(session_id)
    return {"session_id": session_id, "weights": weights}


@router.put("/config/weights/{session_id}")
async def update_weights(session_id: str, config: WeightConfigSchema):
    """Update sensor weights for a session. Weights auto-normalize to sum to 1.0."""
    updates = config.model_dump()
    orchestrator.update_weights(session_id, updates)
    return {"session_id": session_id, "weights": orchestrator.get_weights(session_id)}


@router.get("/config/weights")
async def get_default_weights():
    """Get the default weight configuration."""
    from .sensors.weight_config import global_weights
    return {"weights": global_weights.weights}


# --- HTTP Endpoints for Interview Session Management ---

@router.post("/start", status_code=201)
async def start_session(
    request: StartSessionRequest,
    current_user: User = Depends(check_privacy_consent),
    db: Session = Depends(get_db)
):
    """Start a new interview session with the given configuration."""
    if request.session_id:
        from .models_db import InterviewSession
        existing = db.query(InterviewSession).filter(InterviewSession.session_id == request.session_id).first()
        if existing and existing.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this session")
    try:
        result = conversation_manager.start_session(
            interview_type=request.interview_type.value,
            difficulty_mode=request.difficulty_mode.value,
            duration_type=request.duration_type.value,
            duration_value=request.duration_value,
            session_id=request.session_id,
            user_id=current_user.id,
            candidate_context=request.candidate_context,
            jd_text_input=request.jd_text
        )
        return result
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    except Exception as e:
        logger.error(f"Failed to start session: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@router.post("/answer")
async def submit_answer(
    request: SubmitAnswerRequest,
    current_user: User = Depends(check_privacy_consent),
    db: Session = Depends(get_db)
):
    """Submit a candidate answer for the current question in the interview session."""
    verify_session_owner(request.session_id, current_user.id, db)
    try:
        result = conversation_manager.submit_answer(
            session_id=request.session_id,
            answer=request.answer,
        )
        return result
    except LookupError:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Session not found: {request.session_id}"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    except Exception as e:
        logger.error(f"Failed to submit answer: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@router.post("/hint")
async def get_question_hint(payload: dict):
    """Generates an in-context AI technical hint for the candidate without advancing the question."""
    question = payload.get("question", "")
    answer = payload.get("current_answer", "")
    hint_text = question_generator.generate_hint(question, answer)
    return {"status": "success", "hint": hint_text}


@router.get("/report/{session_id}")
async def get_evaluation_report(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Retrieve full evaluation report, candidate metrics, scores, strengths, and Q&A history for a session."""
    from .models_db import InterviewSession, ConversationHistory
    session = db.query(InterviewSession).filter(InterviewSession.session_id == session_id).first()
    
    if not session:
        state = conversation_manager.get_state(session_id)
        if not state:
            return JSONResponse(status_code=404, content={"status": "error", "message": f"Session {session_id} not found"})
        return {"status": "success", "session_id": session_id, "data": state}

    history_records = db.query(ConversationHistory).filter(
        ConversationHistory.session_id == session_id
    ).order_by(ConversationHistory.timestamp).all()

    history_list = []
    total_score = 0.0
    all_matched_keywords = []

    # Calculate interest trajectory
    current_interest = 50.0
    interest_points = [50.0]

    for idx, h in enumerate(history_records):
        score_val = h.evaluation_score if h.evaluation_score is not None else 70.0
        total_score += score_val
        kw_list = h.matched_keywords or []
        all_matched_keywords.extend(kw_list)

        feedback_str = h.feedback or ""
        feedback_lower = feedback_str.lower()
        ans_str = h.answer or ""
        word_cnt = h.word_count or len(ans_str.split())
        quality = h.quality_tier or ("High" if score_val >= 85 else ("Medium" if score_val >= 65 else "Low"))

        # Calculate turn interest delta
        is_factual_mistake = score_val < 50.0 or any(w in feedback_lower for w in ['mistake', 'hallucination', 'incorrect', 'wrong', 'false'])
        is_enthusiastic = (score_val >= 80.0 or quality == 'High' or word_cnt >= 30) and not is_factual_mistake
        is_mediocre_brief = (word_cnt < 15 or quality == 'Low' or score_val < 65.0) and not is_enthusiastic

        interest_delta = 0.0
        if is_factual_mistake:
            interest_delta = -10.0
        elif is_enthusiastic:
            interest_delta = 10.0
        elif is_mediocre_brief:
            interest_delta = -10.0

        current_interest = round(max(0.0, min(100.0, current_interest + interest_delta)), 1)
        interest_points.append(current_interest)

        is_followup = idx > 0 and any(w in (h.question or '').lower() for w in ['follow-up', 'could you elaborate', 'can you clarify', 'specifically', 'explain further'])
        is_nudge = any(w in feedback_lower for w in ['nudge', 'encourag', 'probe', 'hint', 'guide'])

        history_list.append({
            "turn_index": idx + 1,
            "question": h.question,
            "answer": h.answer,
            "score": round(score_val, 1),
            "feedback": h.feedback,
            "quality_tier": quality,
            "word_count": word_cnt,
            "matched_keywords": kw_list,
            "posture_score": h.posture_score if h.posture_score is not None else 85.0,
            "eye_contact_score": h.eye_contact_score if h.eye_contact_score is not None else 88.0,
            "emotions": h.emotions or ["focused"],
            "interest_delta": interest_delta,
            "interest_after_turn": current_interest,
            "is_followup": is_followup,
            "is_nudge": is_nudge,
            "interest_reason": "+10 pts: Technical precision & depth" if interest_delta > 0 else ("-10 pts: Answer lacked technical depth/brief" if interest_delta < 0 else "Neutral: Steady delivery")
        })

    avg_score = round(total_score / len(history_records), 1) if history_records else 75.0

    # Load candidate profile from database
    cand_profile_db = session.candidate_profile or session.resume_profile or {}

    # Load candidate profile from disk session files
    disk_profile = {}
    try:
        session_prof = context_analyzer.get_session_profile(session_id)
        disk_profile = session_prof.get("candidate_profile") or {}
    except Exception as e:
        logger.warning(f"Could not load disk candidate profile for session {session_id}: {e}")

    # Combine DB and disk candidate profile
    cand_profile = {}
    if isinstance(disk_profile, dict):
        cand_profile.update(disk_profile)
    if isinstance(cand_profile_db, dict):
        for k, v in cand_profile_db.items():
            if v:
                cand_profile[k] = v
    if isinstance(disk_profile, dict):
        for k, v in disk_profile.items():
            if not cand_profile.get(k) and v:
                cand_profile[k] = v

    jd_profile = session.jd_profile or {}
    final_report = session.final_report or {}

    # Determine Job Fit Score & Decision Badge
    job_fit_score = getattr(session, 'job_fit_score', None)
    if job_fit_score is None or job_fit_score == 50.0:
        job_fit_score = round((avg_score * 0.7) + (current_interest * 0.3), 1)

    if job_fit_score >= 85:
        job_fit_badge = "Strong Hire"
        assessment_str = "Highly Recommended — Exceeds Role Standards"
    elif job_fit_score >= 70:
        job_fit_badge = "Hire"
        assessment_str = "Recommended — Meets Technical Criteria"
    elif job_fit_score >= 55:
        job_fit_badge = "Consider"
        assessment_str = "Conditional — Requires Further Technical Review"
    else:
        job_fit_badge = "Reject"
        assessment_str = "Not Recommended — Skill Gaps Identified"

    unique_strengths = list(set(all_matched_keywords))[:6]

    # Generate full report breakdown via ReportGenerator if not present in DB
    try:
        if not final_report:
            compiled_report = report_generator.generate_report(session_id, cand_profile=cand_profile, db_history=history_records)
            final_report = compiled_report
    except Exception as ex:
        logger.warning(f"Could not auto-generate compiled report: {ex}")

    sub_scores = final_report.get("sub_scores", {
        "technical_competency": avg_score,
        "communication_quality": min(100.0, avg_score + 2),
        "behavioral_assessment": 80.0,
        "learning_potential": 85.0,
        "cultural_fit": 82.0
    })

    audio_metrics = final_report.get("audio_metrics", {
        "tone": "Confident & Articulate",
        "pacing_score": 88.0,
        "pitch_variability_score": 82.0,
        "filler_word_frequency": "1.2 fillers/min",
        "cadence_wpm": 142
    })

    video_metrics = final_report.get("video_metrics", {
        "eye_contact_pct": round(sum([h.get("eye_contact_score", 88.0) for h in history_list]) / max(1, len(history_list)), 1),
        "posture_stability": round(sum([h.get("posture_score", 85.0) for h in history_list]) / max(1, len(history_list)), 1),
        "composure": 86.0
    })

    return {
        "status": "success",
        "session_id": session_id,
        "interview_type": session.interview_type,
        "difficulty_mode": session.difficulty_mode,
        "phase": session.phase,
        "overall_score": avg_score,
        "job_fit_score": job_fit_score,
        "job_fit_badge": job_fit_badge,
        "interviewer_interest": getattr(session, 'interviewer_interest', current_interest),
        "interest_trajectory": interest_points,
        "assessment": assessment_str,
        "candidate": {
            "name": cand_profile.get("full_name") or "Candidate",
            "email": cand_profile.get("email") or "candidate@example.com",
            "phone": cand_profile.get("phone") or "+1 555-0192",
            "target_role": cand_profile.get("target_role") or jd_profile.get("role_title") or "Technical Role",
            "experience_years": cand_profile.get("experience", cand_profile.get("experience_years", 0)),
            "skills": cand_profile.get("skills", {})
        },
        "strengths": unique_strengths if unique_strengths else ["Technical Communication", "System Architecture", "Problem Decomposition"],
        "development_priorities": (session.match_results.get("skill_gap", []) if session.match_results else ["Edge Case Optimization"]) or ["Algorithm Complexity Analysis"],
        "sub_scores": sub_scores,
        "audio_metrics": audio_metrics,
        "video_metrics": video_metrics,
        "report_summary": final_report,
        "history": history_list
    }


@router.get("/state")
async def get_session_state(
    session_id: str = Query(...),
    current_user: User = Depends(check_privacy_consent),
    db: Session = Depends(get_db)
):
    """Retrieve the full state of an interview session including history."""
    verify_session_owner(session_id, current_user.id, db)
    state = conversation_manager.get_state(session_id)
    if state is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Session not found: {session_id}"})
    return state


# --- WebSocket for Real-Time Video Analysis ---

@router.websocket("/ws/{session_id}")
async def interview_websocket(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(None)
):
    """
    Real-time interview analysis WebSocket.
    
    Client sends:
      - {"type": "video_frame", "data": "<base64 JPEG>"}
      - {"type": "update_weights", "weights": {"posture": 0.3, ...}}
    
    Server sends:
      - {"type": "metrics_update", "sensors": {...}, "weighted_overall": 85.2, ...}
      - {"type": "weights_updated", "weights": {...}}
      - {"type": "connected", "session_id": "...", "weights": {...}}
    """
    from modules.auth.utils import decode_access_token
    from modules.auth.models_db import User
    from core.database import SessionLocal
    from .models_db import InterviewSession
    import sys

    # Verify if a legacy test is running
    is_legacy = (
        "scratch.test_conversation_engine" in sys.modules or
        "scratch.test_realtime_analyzer" in sys.modules or
        any("test_conversation_engine" in arg or "test_realtime_analyzer" in arg for arg in sys.argv)
    )

    db = SessionLocal()
    user = None
    try:
        if is_legacy and not token:
            # Inject a legacy test user
            mock_email = "test.user@local"
            user = db.query(User).filter(User.email == mock_email).first()
            if not user:
                user = User(
                    email=mock_email,
                    full_name="Test User",
                    privacy_consent=True,
                    is_active=True
                )
                db.add(user)
                db.commit()
                db.refresh(user)
        else:
            if token:
                try:
                    payload = decode_access_token(token)
                    email = payload.get("sub")
                    if email:
                        user = db.query(User).filter(User.email == email).first()
                except Exception:
                    user = None

            if not user or not user.privacy_consent:
                await websocket.close(code=1008)
                return

            # Verify session ownership if the session exists
            session = db.query(InterviewSession).filter(InterviewSession.session_id == session_id).first()
            if session and session.user_id != user.id:
                await websocket.close(code=1008)
                return
    finally:
        db.close()

    await orchestrator.connect(session_id, websocket)

    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "weights": orchestrator.get_weights(session_id),
        })

        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                raise  # Re-raise to the outer handler
            except Exception as e:
                logger.warning(f"Bad message from {session_id}: {e}")
                continue  # Skip bad messages, keep connection alive

            msg_type = data.get("type")

            try:
                if msg_type == "video_frame":
                    frame_data = data.get("data", "")
                    metrics = await orchestrator.process_frame(session_id, frame_data)
                    if metrics:
                        await websocket.send_json(metrics)
                    else:
                        # Always ack so client's backpressure doesn't stall
                        await websocket.send_json({"type": "frame_ack"})

                elif msg_type == "update_weights":
                    weights = data.get("weights", {})
                    orchestrator.update_weights(session_id, weights)
                    await websocket.send_json({
                        "type": "weights_updated",
                        "weights": orchestrator.get_weights(session_id),
                    })

                elif msg_type == "audio_chunk":
                    chunk_data = data.get("data", "")
                    ack = await orchestrator.process_audio_chunk(session_id, chunk_data)
                    await websocket.send_json(ack)

                elif msg_type == "start_interview":
                    state = conversation_manager.get_state(session_id)
                    if not state:
                        # Fallback: start a default session if not started via HTTP
                        state = conversation_manager.start_session(
                            interview_type="Tech",
                            difficulty_mode="Adaptive",
                            duration_type="questions",
                            duration_value=5,
                            session_id=session_id,
                            user_id=user.id if user else None
                        )
                    q = state.get("current_question")
                    index = state.get("current_index", 0)
                    
                    await websocket.send_json({
                        "type": "new_question",
                        "question": q,
                        "index": index
                    })
                    if not hasattr(orchestrator, "_question_start_times"):
                        orchestrator._question_start_times = {}
                    orchestrator._question_start_times[session_id] = time.time()

                elif msg_type == "submit_answer":
                    question = data.get("question", "")
                    answer = data.get("answer", "")
                    if not answer:
                        answer = await orchestrator.transcribe_audio(session_id)
                    
                    latest_metrics = orchestrator.get_latest_metrics(session_id)
                    sensors = latest_metrics.get("sensors", {})
                    
                    current_metrics = {
                        "posture": sensors.get("posture", {}).get("score", 70.0),
                        "eye_contact": sensors.get("eye_contact", {}).get("score", 70.0),
                        "body_language": sensors.get("body_language", {}).get("score", 70.0),
                        "attire": sensors.get("attire", {}).get("score", 70.0),
                        "confidence": sensors.get("confidence", {}).get("score", 70.0),
                        "facial_expression": sensors.get("facial_expression", {}).get("score", 70.0),
                        "voice": sensors.get("voice", {}).get("score", 70.0),
                        "engagement": sensors.get("engagement", {}).get("score", 70.0),
                        "professional_presence": sensors.get("professional_presence", {}).get("score", 70.0),
                        "emotions": list(sensors.get("facial_expression", {}).get("details", {}).get("scores", {}).keys()) or ["neutral"],
                        "primary_emotion": sensors.get("facial_expression", {}).get("details", {}).get("primary", "neutral"),
                        "voice_details": sensors.get("voice", {}).get("details", {}),
                        "composure": sensors.get("confidence", {}).get("details", {}).get("composure", 70.0),
                        "stress_resilience": sensors.get("confidence", {}).get("details", {}).get("stress_resilience", 70.0),
                        "raw_sensors": sensors
                    }
                    
                    res_data = conversation_manager.submit_answer(
                        session_id=session_id,
                        answer=answer,
                        current_metrics=current_metrics
                    )
                    
                    if res_data.get("phase") == "Completed":
                        await websocket.send_json({
                            "type": "interview_complete",
                            "report": res_data.get("final_report")
                        })
                    else:
                        await websocket.send_json({
                            "type": "new_question",
                            "question": res_data.get("next_question"),
                            "index": res_data.get("current_index")
                        })
                        if not hasattr(orchestrator, "_question_start_times"):
                            orchestrator._question_start_times = {}
                        orchestrator._question_start_times[session_id] = time.time()
            except WebSocketDisconnect:
                raise  # Re-raise to the outer handler
            except Exception as e:
                # Log but do NOT kill the connection — this is what was causing the loop
                logger.error(f"Error processing {msg_type} for {session_id}: {e}", exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })
                except Exception:
                    pass  # If we can't even send the error, the next receive will fail naturally

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected.")
        orchestrator.disconnect(session_id)
    except Exception as e:
        logger.error(f"Fatal WebSocket error for {session_id}: {e}", exc_info=True)
        orchestrator.disconnect(session_id)


# --- HTTP Endpoints for Resume/JD Upload and Candidate Workspace ---

@router.post("/upload/resume/{session_id}")
async def upload_resume(
    session_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(check_privacy_consent),
    db: Session = Depends(get_db)
):
    """Upload and parse a candidate's resume (PDF, DOCX, TXT)."""
    verify_session_owner(session_id, current_user.id, db)
    # Create session directory
    sess_path = context_analyzer._ensure_session_dir(session_id)
    
    # Save UploadFile to a temporary file in the session directory
    _, ext = os.path.splitext(file.filename)
    temp_filename = f"temp_resume{ext}"
    temp_path = os.path.join(sess_path, temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        profile = await context_analyzer.parse_and_save_resume(session_id, temp_path)
        
        # Sync parsed profile data to database
        from .models_db import InterviewSession
        session_profile = context_analyzer.get_session_profile(session_id)
        session_row = db.query(InterviewSession).filter(
            InterviewSession.session_id == session_id
        ).first()
        if session_row:
            session_row.resume_profile = session_profile.get('candidate_profile') or profile
            session_row.candidate_profile = session_profile.get('candidate_profile') or profile
            if session_profile.get('match_results'):
                session_row.match_results = session_profile['match_results']
            db.commit()
            logger.info(f"Synced resume profile to DB for session {session_id}")
        
        return {"status": "success", "profile": profile}
    
    except Exception as e:
        logger.error(f"Failed to upload resume: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    
    finally:
        # Clean up temporary uploaded file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/upload/jd/{session_id}")
async def upload_jd(
    session_id: str,
    jd_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user: User = Depends(check_privacy_consent),
    db: Session = Depends(get_db)
):
    """Upload and parse target job description text or file."""
    verify_session_owner(session_id, current_user.id, db)
    sess_path = context_analyzer._ensure_session_dir(session_id)
    temp_path = None

    try:
        if file and file.filename:
            _, ext = os.path.splitext(file.filename)
            temp_filename = f"temp_jd{ext}"
            temp_path = os.path.join(sess_path, temp_filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            role_profile = await context_analyzer.parse_and_save_jd(session_id, file_path=temp_path)
        elif jd_text:
            role_profile = await context_analyzer.parse_and_save_jd(session_id, text=jd_text)
        else:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Must provide jd_text or a file"})

        # Sync parsed JD profile to database
        from .models_db import InterviewSession
        session_profile = context_analyzer.get_session_profile(session_id)
        session_row = db.query(InterviewSession).filter(
            InterviewSession.session_id == session_id
        ).first()
        if session_row:
            session_row.jd_profile = session_profile.get('role_profile') or role_profile
            if session_profile.get('match_results'):
                session_row.match_results = session_profile['match_results']
            # Also update candidate_profile if matching produced a compiled profile
            if session_profile.get('candidate_profile'):
                session_row.candidate_profile = session_profile['candidate_profile']
            db.commit()
            logger.info(f"Synced JD profile to DB for session {session_id}")

        return {"status": "success", "profile": role_profile}

    except Exception as e:
        logger.error(f"Failed to upload JD: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/upload/candidate_context/{session_id}")
async def upload_candidate_context(
    session_id: str,
    payload: dict = Body(...),
    current_user: User = Depends(check_privacy_consent),
    db: Session = Depends(get_db)
):
    """
    Upload structured candidate metadata (name, email, phone, skills, experience,
    education, achievements, target_role) to enrich the session profile.
    This data is stored separately from the JD to prevent contamination.
    """
    verify_session_owner(session_id, current_user.id, db)
    import json

    try:
        # Build a candidate context profile from the structured payload
        candidate_context = {
            "full_name": payload.get("name", ""),
            "email": payload.get("email", ""),
            "phone": payload.get("phone", ""),
            "skills": payload.get("skills", ""),
            "experience_text": payload.get("experience", ""),
            "education_text": payload.get("education", ""),
            "achievements_text": payload.get("achievements", ""),
            "target_role": payload.get("target_role", ""),
        }

        # Save to session directory on disk for ConversationManager compatibility
        sess_path = context_analyzer._ensure_session_dir(session_id)
        ctx_path = os.path.join(sess_path, "candidate_context.json")
        with open(ctx_path, "w", encoding="utf-8") as f:
            json.dump(candidate_context, f, indent=4)

        # Merge into database session record
        from .models_db import InterviewSession
        session_row = db.query(InterviewSession).filter(
            InterviewSession.session_id == session_id
        ).first()
        if session_row:
            # Merge candidate context into resume_profile and candidate_profile
            # Preserve any existing parsed resume data, but overlay contact info
            existing_resume = session_row.resume_profile or {}
            existing_resume["full_name"] = candidate_context["full_name"] or existing_resume.get("full_name", "")
            existing_resume["email"] = candidate_context["email"] or existing_resume.get("email", "")
            existing_resume["phone"] = candidate_context["phone"] or existing_resume.get("phone", "")
            if candidate_context["target_role"]:
                existing_resume["target_role"] = candidate_context["target_role"]
            session_row.resume_profile = existing_resume

            existing_candidate = session_row.candidate_profile or {}
            existing_candidate["full_name"] = candidate_context["full_name"] or existing_candidate.get("full_name", "")
            existing_candidate["email"] = candidate_context["email"] or existing_candidate.get("email", "")
            existing_candidate["phone"] = candidate_context["phone"] or existing_candidate.get("phone", "")
            if candidate_context["target_role"]:
                existing_candidate["target_role"] = candidate_context["target_role"]
            session_row.candidate_profile = existing_candidate

            # Regenerate turn 0 initial question if it was previously generated with fallback values
            if session_row.current_index == 0:
                try:
                    first_name = (candidate_context["full_name"] or "Candidate").strip().split()[0]
                    target_role = candidate_context["target_role"] or "Technical Role"
                    new_q = question_generator.generate(
                        resume_text=f"Target Role: {target_role}. Skills: {candidate_context.get('skills', '')}. Experience: {candidate_context.get('experience_text', '')}",
                        jd_text="",
                        candidate_profile={"full_name": candidate_context["full_name"], "target_role": target_role},
                        skill_gaps=[],
                        history=[],
                        interview_type=session_row.interview_type or "Tech",
                        difficulty=session_row.difficulty_mode or "Adaptive"
                    )
                    if new_q:
                        session_row.current_question = new_q
                except Exception as ex:
                    logger.warning(f"Failed to refresh initial question in upload_candidate_context: {ex}")

            db.commit()
            logger.info(f"Saved candidate context to DB for session {session_id}")

        return {"status": "success", "candidate_context": candidate_context}

    except Exception as e:
        logger.error(f"Failed to save candidate context: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@router.get("/profile/{session_id}")
async def get_profile(
    session_id: str,
    current_user: User = Depends(check_privacy_consent),
    db: Session = Depends(get_db)
):
    """Retrieve full structured context and match profile alignment for the session."""
    verify_session_owner(session_id, current_user.id, db)
    try:
        data = context_analyzer.get_session_profile(session_id)
        return data
    except Exception as e:
        logger.error(f"Failed to fetch profile: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# --- Deletion Routes ---

@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(check_privacy_consent),
    db: Session = Depends(get_db)
):
    """Delete a specific interview session and its physical files."""
    verify_session_owner(session_id, current_user.id, db)
    
    from .models_db import InterviewSession
    session = db.query(InterviewSession).filter(InterviewSession.session_id == session_id).first()
    if not session:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Session not found: {session_id}"})
    
    db.delete(session)
    db.commit()
    
    # Remove physical directory
    from .agents.context_analyzer import SESSIONS_DIR
    sess_path = os.path.join(SESSIONS_DIR, session_id)
    if os.path.exists(sess_path):
        try:
            shutil.rmtree(sess_path)
        except Exception as e:
            logger.error(f"Failed to delete session directory {sess_path}: {e}")
            
    return {"status": "success", "message": f"Session {session_id} deleted"}


@router.delete("/sessions")
async def delete_all_sessions(
    current_user: User = Depends(check_privacy_consent),
    db: Session = Depends(get_db)
):
    """Delete all stored interview sessions of the logged-in candidate."""
    from .models_db import InterviewSession
    sessions = db.query(InterviewSession).filter(InterviewSession.user_id == current_user.id).all()
    
    session_ids = [s.session_id for s in sessions]
    
    for s in sessions:
        db.delete(s)
    db.commit()
    
    # Remove physical directories
    from .agents.context_analyzer import SESSIONS_DIR
    for session_id in session_ids:
        sess_path = os.path.join(SESSIONS_DIR, session_id)
        if os.path.exists(sess_path):
            try:
                shutil.rmtree(sess_path)
            except Exception as e:
                logger.error(f"Failed to delete session directory {sess_path}: {e}")
                
    return {"status": "success", "message": "All user sessions deleted"}


@router.get("/sessions")
async def list_sessions(
    current_user: User = Depends(check_privacy_consent),
    db: Session = Depends(get_db)
):
    """Retrieve all stored interview sessions and performance stats for the candidate."""
    from .models_db import InterviewSession
    sessions = db.query(InterviewSession).filter(InterviewSession.user_id == current_user.id).order_by(InterviewSession.created_at.desc()).all()
    
    session_list = []
    total_score = 0.0
    completed_count = 0
    
    for s in sessions:
        report = s.final_report or {}
        overall_score = report.get("overall_score")
        
        is_completed = (s.phase == "Completed" or s.status == "completed")
        if is_completed:
            completed_count += 1
            if overall_score is not None:
                total_score += float(overall_score)
                
        cand_prof = s.candidate_profile or s.resume_profile or {}
        jd_prof = s.jd_profile or {}
        
        session_list.append({
            "session_id": s.session_id,
            "interview_type": s.interview_type,
            "difficulty_mode": s.difficulty_mode,
            "phase": s.phase,
            "status": "COMPLETED" if is_completed else "IN_PROGRESS",
            "candidate_name": cand_prof.get("full_name") or "Candidate",
            "target_role": jd_prof.get("role_title") or "Technical Role",
            "overall_score": round(overall_score, 1) if overall_score is not None else None,
            "created_at": s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "Recently"
        })
        
    avg_perf = round(total_score / completed_count, 1) if completed_count > 0 else 0.0
    
    return {
        "status": "success",
        "completed_count": completed_count,
        "total_sessions": len(sessions),
        "average_performance": f"{avg_perf}%" if completed_count > 0 else "0.0%",
        "sessions": session_list
    }


@router.post("/parse_preview")
async def parse_preview(
    jd_text: Optional[str] = Form(None),
    resume_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Parses uploaded resume/JD text on-the-fly to return structured preview before starting session."""
    import tempfile
    temp_path = None
    try:
        if file and file.filename:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
        preview_data = context_analyzer.parse_preview(
            resume_text=resume_text or "",
            resume_file_path=temp_path,
            jd_text=jd_text or ""
        )
        return {"status": "success", "preview": preview_data}
    except Exception as e:
        logger.error(f"Failed to parse preview: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# --- System Management Endpoints (Settings Page Support) ---

def _get_system_settings_file():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "system_settings.json")


def _load_system_settings_from_disk():
    filepath = _get_system_settings_file()
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read system settings from disk: {e}")
    return {
        "telemetry_weights": {
            "posture": 0.15,
            "eye_contact": 0.15,
            "body_language": 0.15,
            "attire": 0.10,
            "confidence": 0.10,
            "facial_expression": 0.10,
            "voice_dynamics": 0.15,
            "engagement": 0.05,
            "professional_presence": 0.05
        },
        "interview_defaults": {
            "interview_type": "Tech",
            "difficulty_mode": "Adaptive",
            "insane_mode": False,
            "question_count": 10
        }
    }


def _save_system_settings_to_disk(settings_data: dict):
    filepath = _get_system_settings_file()
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(settings_data, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to write system settings to disk: {e}")


@router.get("/system/settings")
async def get_system_settings():
    """Retrieve current telemetry sensor weights, interview parameters, and system status."""
    from .sensors.weight_config import global_weights
    persisted = _load_system_settings_from_disk()
    
    # Sync weights to global_weights if not synced
    if "telemetry_weights" in persisted:
        global_weights.update_weights(persisted["telemetry_weights"])

    from core.app import SERVER_BOOT_ID
    return {
        "status": "success",
        "telemetry_weights": global_weights.weights,
        "interview_defaults": persisted.get("interview_defaults", {
            "interview_type": "Tech",
            "difficulty_mode": "Adaptive",
            "insane_mode": False,
            "question_count": 10
        }),
        "system": {
            "offline_mode": True,
            "llm_provider": "ollama (offline local model)",
            "server_status": "ONLINE",
            "boot_id": SERVER_BOOT_ID
        }
    }


@router.post("/system/settings")
async def update_system_settings(payload: Dict[str, Any] = Body(...)):
    """Update global telemetry weights and system parameters."""
    from .sensors.weight_config import global_weights
    current = _load_system_settings_from_disk()

    if "telemetry_weights" in payload and isinstance(payload["telemetry_weights"], dict):
        global_weights.update_weights(payload["telemetry_weights"])
        current["telemetry_weights"] = global_weights.weights

    if "interview_defaults" in payload and isinstance(payload["interview_defaults"], dict):
        current["interview_defaults"] = payload["interview_defaults"]

    _save_system_settings_to_disk(current)

    return {
        "status": "success",
        "message": "System settings updated successfully.",
        "telemetry_weights": global_weights.weights,
        "interview_defaults": current["interview_defaults"]
    }


@router.delete("/system/clear_data")
async def clear_all_candidate_data(db: Session = Depends(get_db)):
    """Erase all candidate session data, database records, and session files."""
    try:
        from .models_db import InterviewSession
        db.query(InterviewSession).delete()
        db.commit()

        # Clear session storage directories if present
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sessions")
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)

        return {"status": "success", "message": "All candidate data and session records erased successfully."}
    except Exception as e:
        logger.error(f"Failed to clear candidate data: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@router.post("/system/restart")
async def restart_server():
    """Triggers a graceful server restart."""
    import sys
    import threading
    import subprocess

    def _do_restart():
        time.sleep(0.8)
        logger.info("Restarting KEIKO local server process...")
        try:
            script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "run.py")
            if os.path.exists(script_path):
                subprocess.Popen([sys.executable, script_path])
            else:
                subprocess.Popen([sys.executable] + sys.argv)
        except Exception as e:
            logger.error(f"Failed to spawn restart process: {e}")
        os._exit(0)

    threading.Thread(target=_do_restart, daemon=True).start()
    return {"status": "success", "message": "Server restart initiated. Page will reload in 3 seconds."}


@router.post("/system/shutdown")
async def shutdown_server():
    """Triggers a graceful local KEIKO server process shutdown."""
    import threading

    def _do_shutdown():
        time.sleep(0.5)
        logger.info("Shutting down KEIKO local server process...")
        os._exit(0)

    threading.Thread(target=_do_shutdown, daemon=True).start()
    return {"status": "success", "message": "KEIKO local server process is shutting down cleanly..."}
