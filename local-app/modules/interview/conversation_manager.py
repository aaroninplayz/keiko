import uuid
import time
import logging
from typing import Dict, Any, Optional, List
from core.database import SessionLocal
from .models_db import InterviewSession, ConversationHistory
from .agents.question_generator import QuestionGenerator
from .agents.central_evaluator import CentralEvaluator
from .agents.report_generator import ReportGenerator
from .agents.context_analyzer import ContextAnalyzer, SESSIONS_DIR

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Central state machine managing the interview conversation lifecycle.
    Handles session creation, answer submission, phase transitions,
    adaptive difficulty, and database persistence.
    """

    VALID_INTERVIEW_TYPES = {'HR', 'Tech', 'Situational'}
    VALID_DIFFICULTY_MODES = {'Beginner', 'Intermediate', 'Advanced', 'Adaptive'}
    VALID_DURATION_TYPES = {'questions', 'minutes', 'unlimited'}
    DIFFICULTY_LEVELS = ['Beginner', 'Intermediate', 'Advanced']

    def __init__(self):
        self.question_generator = QuestionGenerator()
        self.central_evaluator = CentralEvaluator()
        self.report_generator = ReportGenerator()
        self.context_analyzer = ContextAnalyzer()
        self.sessions = {}

    def start_session(
        self,
        interview_type: str,
        difficulty_mode: str,
        duration_type: str,
        duration_value: int,
        session_id: Optional[str] = None,
        user_id: Optional[int] = None,
        candidate_context: Optional[Dict[str, Any]] = None,
        jd_text_input: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a new interview session, generates the first question,
        and persists the session to the database.
        """
        # Validate inputs
        if interview_type not in self.VALID_INTERVIEW_TYPES:
            raise ValueError(f"Invalid interview_type: {interview_type}")
        if difficulty_mode not in self.VALID_DIFFICULTY_MODES:
            raise ValueError(f"Invalid difficulty_mode: {difficulty_mode}")
        if duration_type not in self.VALID_DURATION_TYPES:
            raise ValueError(f"Invalid duration_type: {duration_type}")
        if duration_value <= 0:
            raise ValueError(f"duration_value must be positive, got: {duration_value}")

        if not session_id:
            session_id = str(uuid.uuid4())

        # Determine effective difficulty for Adaptive mode
        effective_difficulty = 'Beginner' if difficulty_mode == 'Adaptive' else difficulty_mode

        # If candidate_context was passed, save it to disk for this session immediately
        import os, json
        from .agents.context_analyzer import SESSIONS_DIR
        sess_dir = os.path.join(SESSIONS_DIR, session_id)
        os.makedirs(sess_dir, exist_ok=True)

        if candidate_context:
            ctx_path = os.path.join(sess_dir, "candidate_context.json")
            normalized_ctx = {
                "full_name": candidate_context.get("name") or candidate_context.get("full_name") or "",
                "email": candidate_context.get("email") or "",
                "phone": candidate_context.get("phone") or "",
                "skills": candidate_context.get("skills") or "",
                "experience_text": candidate_context.get("experience") or candidate_context.get("experience_text") or "",
                "education_text": candidate_context.get("education") or candidate_context.get("education_text") or "",
                "achievements_text": candidate_context.get("achievements") or candidate_context.get("achievements_text") or "",
                "target_role": candidate_context.get("target_role") or "",
            }
            try:
                with open(ctx_path, "w", encoding="utf-8") as f:
                    json.dump(normalized_ctx, f, indent=4)
            except Exception as e:
                logger.warning(f"Could not save candidate_context.json in start_session: {e}")

        # Try to load resume/JD context if available
        profile_info = {}
        candidate_profile = {}
        skill_gaps = []
        resume_text = ""
        jd_text = jd_text_input or ""
        try:
            profile_info = self.context_analyzer.get_session_profile(session_id)
            candidate_profile = profile_info.get('candidate_profile') or {}
            match_results = profile_info.get('match_results') or {}
            skill_gaps = match_results.get('skill_gap') or match_results.get('skill_gaps') or []

            role_profile = profile_info.get('role_profile') or {}
            if role_profile and not jd_text:
                req_skills = ", ".join(role_profile.get('required_skills', []))
                domain = role_profile.get('industry_domain', '')
                exp = role_profile.get('required_experience_years', 0)
                jd_text = f"Domain: {domain}. Required Experience: {exp} years. Required Skills: {req_skills}."

            if candidate_profile:
                parts = []
                exp = candidate_profile.get('experience') or candidate_profile.get('experience_years', 0)
                if exp:
                    parts.append(f"Experience: {exp} years")
                
                skills_list = []
                for cat, skills_in_cat in candidate_profile.get('skills', {}).items():
                    for s in skills_in_cat:
                        if isinstance(s, dict):
                            skills_list.append(s.get('name', ''))
                        else:
                            skills_list.append(str(s))
                if skills_list:
                    parts.append(f"Skills: {', '.join(skills_list)}")

                work = candidate_profile.get('work_history') or candidate_profile.get('work_experience') or []
                if work:
                    parts.append(f"Work Experience: {'; '.join(str(w) for w in work[:3])}")

                projs = candidate_profile.get('projects') or candidate_profile.get('project_expertise') or []
                if projs:
                    proj_strs = []
                    for p in projs[:3]:
                        if isinstance(p, dict):
                            proj_strs.append(f"{p.get('name', '')}: {p.get('description', '')}")
                        else:
                            proj_strs.append(str(p))
                    parts.append(f"Projects: {'; '.join(proj_strs)}")

                edu = candidate_profile.get('education') or []
                if edu:
                    parts.append(f"Education: {'; '.join(str(e) for e in edu[:2])}")

                resume_text = ". ".join(parts)
        except Exception as e:
            logger.warning(f"Could not load profile context for {session_id}: {e}")

        # Ensure candidate details from candidate_context payload override empty profile values
        if candidate_context:
            if candidate_context.get("name") or candidate_context.get("full_name"):
                candidate_profile["full_name"] = candidate_context.get("name") or candidate_context.get("full_name")
            if candidate_context.get("target_role"):
                candidate_profile["target_role"] = candidate_context.get("target_role")
            if candidate_context.get("email"):
                candidate_profile["email"] = candidate_context.get("email")
            if candidate_context.get("skills") and not candidate_profile.get("skills"):
                candidate_profile["skills_text"] = candidate_context.get("skills")

            # Build resume_text from structured candidate context if resume_text is empty
            if not resume_text:
                ctx_parts = []
                if candidate_profile.get("target_role"):
                    ctx_parts.append(f"Target Role: {candidate_profile['target_role']}")
                if candidate_context.get("skills"):
                    ctx_parts.append(f"Skills: {candidate_context['skills']}")
                if candidate_context.get("experience"):
                    ctx_parts.append(f"Experience: {candidate_context['experience']}")
                if candidate_context.get("education"):
                    ctx_parts.append(f"Education: {candidate_context['education']}")
                if candidate_context.get("achievements"):
                    ctx_parts.append(f"Achievements: {candidate_context['achievements']}")
                resume_text = ". ".join(ctx_parts)

        # Ensure candidate full_name is resolved from disk or user account if missing
        if not candidate_profile.get('full_name'):
            if user_id:
                try:
                    db_temp = SessionLocal()
                    from modules.auth.models_db import User
                    user_record = db_temp.query(User).filter(User.id == user_id).first()
                    if user_record:
                        c_name = getattr(user_record, 'full_name', None) or getattr(user_record, 'username', None)
                        if c_name:
                            candidate_profile['full_name'] = c_name
                    db_temp.close()
                except Exception as ex:
                    logger.warning(f"Could not load user record for candidate name: {ex}")

            if not candidate_profile.get('full_name'):
                context_file = os.path.join(SESSIONS_DIR, session_id, "candidate_context.json")
                if os.path.exists(context_file):
                    try:
                        with open(context_file, "r", encoding="utf-8") as f:
                            ctx_data = json.load(f)
                            if ctx_data.get('full_name'):
                                candidate_profile['full_name'] = ctx_data['full_name']
                            if ctx_data.get('target_role') and not candidate_profile.get('target_role'):
                                candidate_profile['target_role'] = ctx_data['target_role']
                    except Exception as ex:
                        logger.warning(f"Could not load candidate_context file: {ex}")

        # Generate first question
        first_question = self.question_generator.generate(
            resume_text=resume_text,
            jd_text=jd_text,
            candidate_profile=candidate_profile,
            skill_gaps=skill_gaps,
            history=[],
            evaluator_feedback=None,
            interview_type=interview_type,
            difficulty=effective_difficulty
        )

        # Persist to database
        db = SessionLocal()
        try:
            db.query(ConversationHistory).filter(
                ConversationHistory.session_id == session_id
            ).delete()
            existing = db.query(InterviewSession).filter(
                InterviewSession.session_id == session_id
            ).first()
            if existing:
                db.delete(existing)
                db.flush()

            session_obj = InterviewSession(
                session_id=session_id,
                interview_type=interview_type,
                difficulty_mode=difficulty_mode,
                duration_type=duration_type,
                duration_value=duration_value,
                phase='Warmup',
                status='active',
                current_index=0,
                current_question=first_question,
                interviewer_interest=50.0,
                job_fit_score=50.0,
                resume_profile=profile_info.get('candidate_profile'),
                jd_profile=profile_info.get('role_profile'),
                match_results=profile_info.get('match_results'),
                candidate_profile=candidate_profile or None,
                user_id=user_id,
            )
            db.add(session_obj)
            db.commit()
            db.refresh(session_obj)

            return {
                'session_id': session_id,
                'interview_type': interview_type,
                'difficulty_mode': difficulty_mode,
                'duration_type': duration_type,
                'duration_value': duration_value,
                'phase': 'Warmup',
                'status': 'active',
                'current_question': first_question,
                'current_index': 0,
                'interviewer_interest': 50.0,
                'job_fit_score': 50.0,
                'history': [],
            }
        finally:
            db.close()

    def submit_answer(self, session_id: str, answer: str, current_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes a candidate answer: evaluates it, persists history,
        manages phase transitions, and generates the next question.
        """
        db = SessionLocal()
        try:
            session = db.query(InterviewSession).filter(
                InterviewSession.session_id == session_id
            ).first()

            if not session:
                raise LookupError(f"Session not found: {session_id}")

            if session.phase == 'Completed' or session.status == 'completed':
                final_report = session.final_report or {}
                if not final_report:
                    try:
                        history_records = db.query(ConversationHistory).filter(ConversationHistory.session_id == session_id).all()
                        h_dicts = [{'question': h.question, 'answer': h.answer, 'accuracy_score': h.evaluation_score} for h in history_records]
                        final_report = self._generate_final_report(session_id, session, h_dicts)
                    except Exception:
                        pass
                return {
                    'session_id': session_id,
                    'phase': 'Completed',
                    'status': 'completed',
                    'current_index': session.current_index,
                    'next_question': None,
                    'session_completed': True,
                    'interviewer_interest': getattr(session, 'interviewer_interest', 50.0),
                    'job_fit_score': getattr(session, 'job_fit_score', 50.0),
                    'final_report': final_report
                }

            current_question = session.current_question or ''

            # Profanity & Answer Decorum Filter
            import re
            PROFANITY_PATTERNS = [
                r"\bfuck\b", r"\bshit\b", r"\bbitch\b", r"\basshole\b", r"\bcrap\b", 
                r"\bidiot\b", r"\bdumb\b", r"\bhate you\b", r"\bsuck\b", r"\bwtf\b"
            ]
            
            answer_clean = answer.strip().lower()
            has_profanity = any(re.search(pat, answer_clean) for pat in PROFANITY_PATTERNS)
            
            # Flag profanity but DON'T block — let the LLM handle it conversationally
            profanity_context = ""
            if has_profanity:
                profanity_context = (
                    "WARNING: Candidate used unprofessional language in their last response. "
                    "Respond professionally but firmly, acknowledge the behavior, and redirect to the interview."
                )
                logger.warning(f"Profanity detected in session {session_id}: answer contains flagged language")

            history_records = db.query(ConversationHistory).filter(
                ConversationHistory.session_id == session_id
            ).all()

            word_count = len(answer_clean.split())
            is_greeting_pattern = bool(re.match(r"^(hello|hi|hey|my name is [a-z]+|i am [a-z]+)$", answer_clean))
            is_greeting_only = (word_count <= 3) and is_greeting_pattern

            is_cand_question = "?" in answer or any(re.search(pat, answer_clean) for pat in [
                r"\bwhat (is|are|about|does|do|can|tech|stack|culture|role)\b",
                r"\bhow (do|does|is|are|can)\b",
                r"\bcan you (tell|explain|share|elaborate)\b",
                r"\bcould you (tell|explain|share|elaborate)\b",
                r"\bdo you (have|use|offer|work)\b",
                r"\bis there\b",
                r"\bwhat's\b",
                r"\btell me (about|more)\b"
            ])
            if len(history_records) >= 2 and not is_cand_question and (word_count < 3 and is_greeting_only):
                redirection_msg = f"Your answer was too brief or off-topic. Could you elaborate on your experience regarding: '{current_question}'?"
                return {
                    "session_id": session_id,
                    "next_question": redirection_msg,
                    "is_redirection": True,
                    "interviewer_interest": getattr(session, 'interviewer_interest', 50.0),
                    "job_fit_score": getattr(session, 'job_fit_score', 50.0),
                    "status": "active"
                }

            # Evaluate the answer
            eval_result = self.central_evaluator.evaluate_answer(
                session_id=session_id,
                question=current_question,
                answer=answer,
                current_metrics=current_metrics or {'posture': 50.0, 'eye_contact': 50.0, 'emotions': ['neutral']}
            )

            score = eval_result.get('evaluation_score', 70.0)
            quality = eval_result.get('quality_tier', 'Medium')
            feedback = eval_result.get('next_question_feedback', '')
            matched_keywords = eval_result.get('matched_keywords', [])

            def get_metric_val(key: str, default_val: float) -> float:
                val = eval_result.get(key)
                if val is not None:
                    return float(val)
                if current_metrics and key in current_metrics:
                    cm_val = current_metrics[key]
                    if isinstance(cm_val, dict):
                        return float(cm_val.get("score", default_val))
                    return float(cm_val)
                return default_val

            # Compile comprehensive metrics for storage
            raw_metrics = {
                **(current_metrics or {}),
                "technical_competency": get_metric_val("technical_competency", score),
                "communication_quality": get_metric_val("communication_quality", score),
                "behavioral_assessment": get_metric_val("behavioral_assessment", score),
                "learning_potential": get_metric_val("learning_potential", score),
                "cultural_fit": get_metric_val("cultural_fit", score)
            }

            # Penalize behavioral and cultural scores when profanity is detected
            if has_profanity:
                raw_metrics["behavioral_assessment"] = min(raw_metrics.get("behavioral_assessment", 70.0), 25.0)
                raw_metrics["cultural_fit"] = min(raw_metrics.get("cultural_fit", 70.0), 25.0)
                # Prepend profanity context to evaluator feedback for question generation
                feedback = profanity_context + " " + feedback if feedback else profanity_context

            def extract_score(metrics, key, default=None):
                if not metrics:
                    return default
                val = metrics.get(key, default)
                if isinstance(val, dict):
                    return val.get("score", default)
                return val

            # Persist ConversationHistory row
            history_entry = ConversationHistory(
                session_id=session_id,
                question=current_question,
                answer=answer,
                evaluation_score=score,
                feedback=feedback,
                quality_tier=quality,
                word_count=len(answer.split()),
                matched_keywords=matched_keywords,
                posture_score=extract_score(current_metrics, 'posture', None),
                eye_contact_score=extract_score(current_metrics, 'eye_contact', None),
                emotions=current_metrics.get('emotions', ['neutral']) if current_metrics else ['neutral'],
                metrics_raw=raw_metrics,
                timestamp=time.time()
            )
            db.add(history_entry)

            # Update session index
            session.current_index += 1
            session.next_question_feedback = feedback

            try:
                db.commit()
            except Exception:
                db.rollback()
                session = db.query(InterviewSession).filter(
                    InterviewSession.session_id == session_id
                ).first()
                if session:
                    session.current_index += 1
                    session.next_question_feedback = feedback
                    db.add(history_entry)
                    db.commit()

            session = db.query(InterviewSession).filter(
                InterviewSession.session_id == session_id
            ).first() or self.sessions.get(session_id)
            if not session:
                raise LookupError(f"Session not found: {session_id}")

            history_records = db.query(ConversationHistory).filter(
                ConversationHistory.session_id == session_id
            ).order_by(ConversationHistory.timestamp).all()
            history_dicts = [
                {
                    'question': h.question,
                    'answer': h.answer,
                    'accuracy_score': h.evaluation_score,
                    'words': h.word_count,
                    'quality_tier': h.quality_tier,
                    'technical_competency': h.metrics_raw.get('technical_competency', h.evaluation_score) if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else h.evaluation_score,
                    'communication_quality': h.metrics_raw.get('communication_quality', h.evaluation_score) if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else h.evaluation_score,
                    'behavioral_assessment': h.metrics_raw.get('behavioral_assessment', h.evaluation_score) if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else h.evaluation_score,
                    'learning_potential': h.metrics_raw.get('learning_potential', 70.0) if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else 70.0,
                    'cultural_fit': h.metrics_raw.get('cultural_fit', 70.0) if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else 70.0,
                    'metrics': h.metrics_raw if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else {
                        'posture': h.posture_score if h.posture_score is not None else 70.0,
                        'eye_contact': h.eye_contact_score if h.eye_contact_score is not None else 70.0,
                        'emotions': h.emotions if h.emotions is not None else ['neutral'],
                        'body_language': 70.0,
                        'attire': 70.0,
                        'confidence': 70.0,
                        'facial_expression': 70.0,
                        'voice': 70.0,
                        'engagement': 70.0,
                        'professional_presence': 70.0
                    }
                }
                for h in history_records
            ]

            # Determine effective difficulty (for Adaptive mode)
            effective_difficulty = self._get_effective_difficulty(session, history_dicts)

            # Dynamic interviewer interest adjustment
            current_interest = getattr(session, 'interviewer_interest', 50.0)
            if current_interest is None:
                current_interest = 50.0

            feedback_lower = (feedback or '').lower()
            is_factual_mistake = score < 50.0 or any(w in feedback_lower for w in ['mistake', 'hallucination', 'incorrect', 'wrong', 'false'])
            is_enthusiastic = (score >= 80.0 or quality == 'High' or len(answer.split()) >= 30) and not is_factual_mistake
            is_mediocre_brief = (len(answer.split()) < 15 or quality == 'Low' or score < 65.0) and not is_enthusiastic

            interest_delta = 0.0
            if is_factual_mistake:
                interest_delta = -10.0
            elif is_enthusiastic:
                interest_delta = 10.0
            elif is_mediocre_brief:
                interest_delta = -10.0

            new_interest = round(max(0.0, min(100.0, current_interest + interest_delta)), 1)
            session.interviewer_interest = new_interest

            # Dynamic job fit score calculation synthesizing ALL collected metrics
            scores = [h.get('accuracy_score', 70.0) for h in history_dicts if h.get('accuracy_score') is not None]
            avg_acc = sum(scores) / len(scores) if scores else score

            tech_comp = raw_metrics.get('technical_competency', score)
            comm_qual = raw_metrics.get('communication_quality', score)
            behav_assess = raw_metrics.get('behavioral_assessment', score)
            learn_pot = raw_metrics.get('learning_potential', score)
            cult_fit = raw_metrics.get('cultural_fit', score)
            match_score = (session.match_results or {}).get('role_alignment_score', 75.0) or 75.0

            eye_score = extract_score(raw_metrics, 'eye_contact', 70.0)
            posture_score = extract_score(raw_metrics, 'posture', 70.0)
            conf_score = extract_score(raw_metrics, 'confidence', 70.0)
            voice_score = extract_score(raw_metrics, 'voice', 70.0)
            sensor_avg = (eye_score + posture_score + conf_score + voice_score) / 4.0

            job_fit_score = round(max(0.0, min(100.0, 
                (avg_acc * 0.20) + 
                (tech_comp * 0.15) + 
                (behav_assess * 0.15) + 
                (comm_qual * 0.15) + 
                (cult_fit * 0.10) + 
                (match_score * 0.15) + 
                (sensor_avg * 0.10)
            )), 1)
            session.job_fit_score = job_fit_score

            # Context compression: Every 5 questions, generate/update compressed summary
            if len(history_dicts) > 0 and len(history_dicts) % 5 == 0:
                try:
                    compressed_summary = self.question_generator.summarize_history(history_dicts)
                    session.conversation_summary = compressed_summary
                    db.commit()
                except Exception as ex:
                    logger.warning(f"Could not generate compressed conversation summary: {ex}")

            # Phase transition logic
            next_question = None
            final_report = None
            resume_text, jd_text = self._get_profile_summaries(session)
            conv_summary = getattr(session, 'conversation_summary', None)

            # Early exit on low interest applies ONLY to unlimited or adaptive mode, NOT fixed question count
            if new_interest < 25.0 and (session.duration_type == 'unlimited' or session.difficulty_mode == 'Adaptive'):
                session.phase = 'Concluding'
                next_question = "Thank you for your time today. We have gathered sufficient details for our evaluation and will be in touch."

            if session.phase == 'Concluding':
                # The candidate answered the concluding question -> Complete
                session.phase = 'Completed'
                session.status = 'completed'
                next_question = None

                # Generate final report
                try:
                    final_report = self._generate_final_report(session_id, session, history_dicts)
                    session.final_report = final_report
                except Exception as e:
                    logger.error(f"Failed to generate final report: {e}")
                    final_report = {'error': str(e)}

            elif session.duration_type == 'questions' and session.current_index >= session.duration_value:
                # HARD QUESTION COUNT LIMIT REACHED (e.g. 5, 10, 50, 100 questions) -> transition to Concluding
                session.phase = 'Concluding'
                next_question = self.question_generator.generate_concluding(session.interview_type)

            elif session.duration_type == 'minutes':
                # Check elapsed time against duration limit
                elapsed_minutes = (time.time() - session.created_at.timestamp()) / 60.0 if session.created_at else 0
                if elapsed_minutes >= session.duration_value:
                    session.phase = 'Concluding'
                    next_question = self.question_generator.generate_concluding(session.interview_type)
                else:
                    # Normal progression within time limit
                    if session.phase == 'Warmup':
                        session.phase = 'Main'

                    candidate_profile = session.candidate_profile or {}
                    match_results = session.match_results or {}
                    skill_gaps = match_results.get('skill_gap') or match_results.get('skill_gaps') or []

                    next_question = self.question_generator.generate(
                        resume_text=resume_text,
                        jd_text=jd_text,
                        candidate_profile=candidate_profile,
                        skill_gaps=skill_gaps,
                        history=history_dicts,
                        evaluator_feedback=feedback,
                        interview_type=session.interview_type,
                        difficulty=effective_difficulty,
                        sensor_data=current_metrics,
                        interviewer_interest=new_interest,
                        conversation_summary=conv_summary,
                        window_size=3
                    )

            elif session.duration_type == 'unlimited':
                # Insane Mode (Unlimited AI evaluation length guided by Evaluator Interest)
                num_questions = len(history_dicts)
                scores = [h.get('accuracy_score', 70.0) for h in history_dicts if h.get('accuracy_score') is not None]
                avg_score = sum(scores) / len(scores) if scores else 70.0

                # Assess whether candidate has been sufficiently evaluated (minimum 5 questions) based on interest & performance
                if num_questions >= 5 and avg_score >= 82.0 and new_interest >= 75.0:
                    # Candidate clearly qualified & high interest -> transition to Concluding
                    session.phase = 'Concluding'
                    next_question = "Thank you! You have demonstrated exceptional technical depth across all our evaluation areas. Before we wrap up, do you have any final questions for our team?"
                elif num_questions >= 5 and (avg_score < 55.0 or new_interest < 30.0):
                    # Candidate clearly unqualified or low interest -> transition to Concluding
                    session.phase = 'Concluding'
                    next_question = "Thank you very much for your time today. We have gathered sufficient evaluation details for your profile and our team will be in touch regarding next steps."
                else:
                    # Continue asking dynamically generated questions
                    if session.phase == 'Warmup':
                        session.phase = 'Main'

                    candidate_profile = session.candidate_profile or {}
                    match_results = session.match_results or {}
                    skill_gaps = match_results.get('skill_gap') or match_results.get('skill_gaps') or []

                    next_question = self.question_generator.generate(
                        resume_text=resume_text,
                        jd_text=jd_text,
                        candidate_profile=candidate_profile,
                        skill_gaps=skill_gaps,
                        history=history_dicts,
                        evaluator_feedback=feedback,
                        interview_type=session.interview_type,
                        difficulty=effective_difficulty,
                        sensor_data=current_metrics,
                        interviewer_interest=new_interest,
                        conversation_summary=conv_summary,
                        window_size=3
                    )

            else:
                # Normal progression (fallback)
                if session.phase == 'Warmup':
                    session.phase = 'Main'

                candidate_profile = session.candidate_profile or {}
                match_results = session.match_results or {}
                skill_gaps = match_results.get('skill_gap') or match_results.get('skill_gaps') or []

                next_question = self.question_generator.generate(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    candidate_profile=candidate_profile,
                    skill_gaps=skill_gaps,
                    history=history_dicts,
                    evaluator_feedback=feedback,
                    interview_type=session.interview_type,
                    difficulty=effective_difficulty,
                    sensor_data=current_metrics,
                    interviewer_interest=new_interest,
                    conversation_summary=conv_summary,
                    window_size=3
                )

            # Interest-Driven Interviewer Behaviors for next question
            if session.phase not in ['Concluding', 'Completed'] and next_question:
                is_imperfect_answer = (score < 60.0 or quality == 'Low' or len(answer.split()) < 15 or is_factual_mistake) and quality != 'Candidate Inquiry'
                if is_imperfect_answer:
                    if current_interest >= 70.0:
                        # High Interest (>= 70): Gently nudge candidate to give them a chance to self-correct
                        nudge = "Are you sure about that? Would you like to rethink that specific part?"
                        if nudge not in next_question:
                            next_question = f"{nudge} {next_question}"
                    else:
                        # Low/Normal Interest (< 70): Respond with a neutral "Okay, let's move on"
                        neutral_phrase = "Okay, let's move on."
                        if neutral_phrase not in next_question:
                            next_question = f"{neutral_phrase} {next_question}"

            session.current_question = next_question
            db.commit()

            return {
                'session_id': session_id,
                'phase': session.phase,
                'status': session.status,
                'current_index': session.current_index,
                'next_question': next_question,
                'interviewer_interest': new_interest,
                'job_fit_score': job_fit_score,
                'evaluation': {
                    'score': score,
                    'quality': quality,
                    'feedback': feedback,
                    'interviewer_interest_delta': interest_delta,
                },
                'next_question_feedback': feedback,
                'final_report': final_report,
            }
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full session state including history from the database.
        Returns None if session not found.
        """
        db = SessionLocal()
        try:
            session = db.query(InterviewSession).filter(
                InterviewSession.session_id == session_id
            ).first()

            if not session:
                return None

            history_records = db.query(ConversationHistory).filter(
                ConversationHistory.session_id == session_id
            ).order_by(ConversationHistory.timestamp).all()

            history = [
                {
                    'question': h.question,
                    'answer': h.answer,
                    'evaluation_score': h.evaluation_score,
                    'quality_tier': h.quality_tier,
                    'feedback': h.feedback,
                    'word_count': h.word_count,
                    'matched_keywords': h.matched_keywords,
                    'technical_competency': h.metrics_raw.get('technical_competency', h.evaluation_score) if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else h.evaluation_score,
                    'communication_quality': h.metrics_raw.get('communication_quality', h.evaluation_score) if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else h.evaluation_score,
                    'behavioral_assessment': h.metrics_raw.get('behavioral_assessment', h.evaluation_score) if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else h.evaluation_score,
                    'learning_potential': h.metrics_raw.get('learning_potential', 70.0) if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else 70.0,
                    'cultural_fit': h.metrics_raw.get('cultural_fit', 70.0) if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else 70.0,
                    'metrics': h.metrics_raw if (h.metrics_raw and isinstance(h.metrics_raw, dict)) else {
                        'posture': h.posture_score if h.posture_score is not None else 70.0,
                        'eye_contact': h.eye_contact_score if h.eye_contact_score is not None else 70.0,
                        'emotions': h.emotions if h.emotions is not None else ['neutral'],
                        'body_language': 70.0,
                        'attire': 70.0,
                        'confidence': 70.0,
                        'facial_expression': 70.0,
                        'voice': 70.0,
                        'engagement': 70.0,
                        'professional_presence': 70.0
                    }
                }
                for h in history_records
            ]

            return {
                'session_id': session.session_id,
                'interview_type': session.interview_type,
                'difficulty_mode': session.difficulty_mode,
                'duration_type': session.duration_type,
                'duration_value': session.duration_value,
                'phase': session.phase,
                'status': session.status,
                'current_index': session.current_index,
                'current_question': session.current_question,
                'interviewer_interest': getattr(session, 'interviewer_interest', 50.0),
                'job_fit_score': getattr(session, 'job_fit_score', 50.0),
                'next_question_feedback': session.next_question_feedback,
                'conversation_summary': getattr(session, 'conversation_summary', None),
                'history': history,
                'final_report': session.final_report,
            }
        finally:
            db.close()

    def _get_profile_summaries(self, session) -> tuple:
        resume_text = ""
        jd_text = ""
        try:
            role_profile = session.jd_profile or {}
            if role_profile:
                req_skills = ", ".join(role_profile.get('required_skills', []))
                domain = role_profile.get('industry_domain', '')
                exp = role_profile.get('required_experience_years', 0)
                jd_text = f"Domain: {domain}. Required Experience: {exp} years. Required Skills: {req_skills}."
            
            candidate_profile = session.candidate_profile or {}
            if candidate_profile:
                skills_list = []
                for cat, skills_in_cat in candidate_profile.get('skills', {}).items():
                    for s in skills_in_cat:
                        if isinstance(s, dict):
                            skills_list.append(s.get('name', ''))
                        else:
                            skills_list.append(str(s))
                skills_str = ", ".join(skills_list)
                exp = candidate_profile.get('experience', 0)
                resume_text = f"Experience: {exp} years. Skills: {skills_str}."
        except Exception as e:
            logger.warning(f"Error building profile summaries: {e}")
        return resume_text, jd_text

    def _get_effective_difficulty(self, session, history_dicts: List[Dict]) -> str:
        """
        For Adaptive mode or Insane mode, compute dynamic difficulty based on candidate score
        and live interviewer interest telemetry.
        For fixed modes (Beginner, Intermediate, Advanced), return the configured difficulty.
        """
        if not session:
            return 'Intermediate'
        if session.difficulty_mode != 'Adaptive' and session.duration_type != 'unlimited':
            return session.difficulty_mode

        if not history_dicts:
            return 'Beginner'

        scores = [h.get('accuracy_score', 70.0) for h in history_dicts if h.get('accuracy_score') is not None]
        if not scores:
            return 'Beginner'

        recent_scores = scores[-3:]
        avg_score = sum(recent_scores) / len(recent_scores)
        interest = getattr(session, 'interviewer_interest', 50.0) or 50.0

        # Combine performance score and interviewer interest telemetry
        combined_metric = (avg_score * 0.7) + (interest * 0.3)

        if combined_metric >= 76.0:
            return 'Advanced'
        elif combined_metric >= 55.0:
            return 'Intermediate'
        else:
            return 'Beginner'

    def _generate_final_report(
        self,
        session_id: str,
        session,
        history_dicts: List[Dict],
    ) -> Dict[str, Any]:
        """
        Generates the final recruiter report from DB history.
        """
        candidate_profile = session.candidate_profile or {}
        match_results = session.match_results or {}

        from .orchestrator import orchestrator
        weights = orchestrator.get_weights(session_id)

        report = self.report_generator.generate_report(
            session_id=session_id,
            cand_profile=candidate_profile,
            match_results=match_results,
            db_history=history_dicts,
            weights=weights
        )
        report['interviewer_interest'] = getattr(session, 'interviewer_interest', 50.0)
        report['job_fit_score'] = getattr(session, 'job_fit_score', 50.0)
        return report
