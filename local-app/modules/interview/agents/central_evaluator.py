import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# Base path for session profile cache
SESSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
    "data", 
    "sessions"
)

class CentralEvaluator:
    """
    Central Evaluator Agent. Tracks state across the interview, processes transcribed candidate answers,
    aggregates real-time sensor metrics (eye contact, posture, emotions), grades responses, and provides
    dynamic guidelines back to the QuestionGenerator for the next question.
    """

    def __init__(self):
        self._llm_client = LLMClient()

    def _ensure_session_dir(self, session_id: str) -> str:
        sess_path = os.path.join(SESSIONS_DIR, session_id)
        os.makedirs(sess_path, exist_ok=True)
        return sess_path

    def load_interview_state(self, session_id: str) -> Dict[str, Any]:
        """
        Loads the interview progress state for a session.
        """
        sess_path = self._ensure_session_dir(session_id)
        state_path = os.path.join(sess_path, "interview_state.json")
        
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading interview state for {session_id}: {e}")

        # Default state
        return {
            "session_id": session_id,
            "current_index": 0,
            "history": [],
            "sensor_history": [],
            "next_question_feedback": None
        }

    def save_interview_state(self, session_id: str, state: Dict[str, Any]):
        """
        Saves the interview progress state.
        """
        sess_path = self._ensure_session_dir(session_id)
        state_path = os.path.join(sess_path, "interview_state.json")
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write interview state for {session_id}: {e}")

    def evaluate_answer(
        self,
        session_id: str,
        question: str,
        answer: str,
        current_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates the candidate's transcribed answer, updates the session's interview state,
        and outputs matching feedback for the QuestionGenerator.
        """
        logger.info(f"Evaluating answer for session {session_id}")
        
        # Load state
        state = self.load_interview_state(session_id)
        
        score = None
        quality = None
        feedback = None
        tech_keywords_matched = []
        learning_potential_score = None
        cultural_fit_score = None

        # Tier 1: Try External LLM API
        active_provider = self._llm_client.detect_provider()
        if active_provider:
            try:
                logger.info(f"Evaluating answer via external LLM client ({active_provider})...")
                system_message = (
                    "You are an expert technical interviewer and evaluator. Your task is to evaluate the candidate's answer "
                    "to a specific question in the context of the interview. "
                    "Analyze the answer for completeness, accuracy, and technical competence.\n"
                    "Output your evaluation in strict JSON format with the following keys:\n"
                    '- "score": a numeric score from 0 to 100\n'
                    '- "quality": a short string representation of quality, e.g., "Low", "Medium", "High"\n'
                    '- "feedback": constructive feedback and guidelines for the next question.\n'
                    '- "keywords": a list of matched technical keywords mentioned in the answer.\n'
                    '- "technical_competency": a numeric score from 0 to 100 assessing precision and depth.\n'
                    '- "communication_quality": a numeric score from 0 to 100 assessing structure, vocabulary, and clarity.\n'
                    '- "behavioral_assessment": a numeric score from 0 to 100 assessing soft skills, ownership, and mindset.\n'
                    '- "learning_potential": a numeric score from 0 to 100 assessing adaptability, curiosity, and response to feedback.\n'
                    '- "cultural_fit": a numeric score from 0 to 100 assessing alignment with standard team values, collaboration, and mission.'
                )
                
                user_message = (
                    f"Question Asked: {question}\n"
                    f"Candidate's Answer: {answer}\n"
                    f"Please evaluate the response and provide JSON output."
                )
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                
                llm_response = self._llm_client.complete(messages, provider=active_provider, temperature=0.1)
                if llm_response:
                    clean_res = llm_response.strip()
                    if "```json" in clean_res:
                        match = re.search(r'```json\s*(.*?)\s*```', clean_res, re.DOTALL)
                        if match:
                            clean_res = match.group(1)
                    elif "```" in clean_res:
                        match = re.search(r'```\s*(.*?)\s*```', clean_res, re.DOTALL)
                        if match:
                            clean_res = match.group(1)
                    
                    data = json.loads(clean_res)
                    score = float(data.get("score", 70.0))
                    quality = str(data.get("quality", "Medium"))
                    feedback = str(data.get("feedback", ""))
                    tech_keywords_matched = list(data.get("keywords", []))
                    technical_competency_score = float(data.get("technical_competency", score))
                    communication_quality_score = float(data.get("communication_quality", score))
                    behavioral_assessment_score = float(data.get("behavioral_assessment", score))
                    learning_potential_score = data.get("learning_potential")
                    if learning_potential_score is not None:
                        learning_potential_score = float(learning_potential_score)
                    cultural_fit_score = data.get("cultural_fit")
                    if cultural_fit_score is not None:
                        cultural_fit_score = float(cultural_fit_score)
            except Exception as e:
                logger.error(f"External LLM evaluation failed: {e}. Falling back to NLP heuristics.")

        # Tier 2: Local Heuristic Fallback
        if score is None:
            word_count = len(answer.split())
            for word in ["python", "fastapi", "docker", "kubernetes", "sql", "git", "cloud", "aws", "gcp", "postgres"]:
                if word in answer.lower():
                    tech_keywords_matched.append(word.title())

            # Simple completeness heuristics
            if word_count < 10:
                quality = "Low (Extremely brief)"
                score = 40.0
                feedback = "The candidate answered very briefly. Instruct QuestionGenerator to probe further on this specific topic."
            elif word_count < 25:
                quality = "Medium"
                score = 70.0
                feedback = "The answer was concise but could be elaborated. Suggest probing for architectural choices."
            else:
                quality = "High"
                score = 90.0
                feedback = f"Great response! Verified keywords: {', '.join(tech_keywords_matched)}. Proceed to the next topic."

            # Adjust score if technical keywords were required but missing
            if score > 50.0 and len(tech_keywords_matched) == 0 and any(t in question.lower() for t in ["python", "fastapi", "docker", "kubernetes", "postgres"]):
                score -= 15.0
                feedback += " The candidate answered but did not mention technical details. Probe for implementation specifics."

            # 1. Communication Quality
            vocab_richness = len(set(answer.lower().split())) / max(1.0, word_count)
            voice_details = current_metrics.get("voice_details", {})
            filler_count = voice_details.get("filler_count", 0)
            fluency_score = voice_details.get("fluency", 80.0)
            
            comm_score = min(100.0, max(0.0, 60.0 + (word_count * 0.8) + (vocab_richness * 30.0) - (filler_count * 3.0)))
            if fluency_score < 60.0:
                comm_score -= 10.0
            communication_quality_score = round(max(10.0, min(100.0, comm_score)), 1)

            # 2. Technical Competency
            tech_score_base = 50.0
            if tech_keywords_matched:
                tech_score_base += len(tech_keywords_matched) * 8.0
            if score < 50.0:
                tech_score_base -= 20.0
            technical_competency_score = round(max(10.0, min(100.0, tech_score_base)), 1)

            # 3. Behavioral Assessment
            behavioral_score_base = 70.0
            behavioral_words = ["team", "collaboration", "lead", "responsibility", "ownership", "learn", "grow", "conflict", "helped", "shared", "growth"]
            matched_behavioral = [w for w in behavioral_words if w in answer.lower()]
            behavioral_score_base += len(matched_behavioral) * 5.0
            primary_emotion = current_metrics.get("primary_emotion", "neutral")
            if primary_emotion in ["nervous", "anxious"]:
                behavioral_score_base -= 5.0
            behavioral_assessment_score = round(max(10.0, min(100.0, behavioral_score_base)), 1)

        # Helper to extract scores when nested in dicts (common in tests/raw updates)
        def extract_score(metrics, key, default=70.0):
            if not metrics:
                return default
            val = metrics.get(key, default)
            if isinstance(val, dict):
                return val.get("score", default)
            return val

        # Compute local heuristics for learning potential and cultural fit if not populated by LLM
        if learning_potential_score is None:
            learning_potential_score = extract_score(current_metrics, "learning_potential", None)
        if learning_potential_score is None:
            word_count = len(answer.split())
            lp_score_base = 75.0
            lp_keywords = ["learn", "grow", "adapt", "feedback", "improve", "adaptability", "correct", "learned"]
            if any(k in answer.lower() for k in lp_keywords):
                lp_score_base += 5.0
            
            prev_feedback = state.get("next_question_feedback", "") or ""
            prev_feedback_lower = prev_feedback.lower()
            requested_details = any(phrase in prev_feedback_lower for phrase in ["detail", "specific", "technical", "probe", "architectural", "elaboration", "elaborate"])
            asked_elaboration = any(phrase in prev_feedback_lower for phrase in ["elaborate", "elaboration", "probe", "brief", "detail", "specific"])
            
            if requested_details and word_count > 20:
                lp_score_base += 10.0
            if asked_elaboration and word_count < 10:
                lp_score_base -= 15.0
            
            learning_potential_score = round(max(0.0, min(100.0, lp_score_base)), 1)

        if cultural_fit_score is None:
            cultural_fit_score = extract_score(current_metrics, "cultural_fit", None)
        if cultural_fit_score is None:
            cf_score_base = 75.0
            cf_keywords = ["culture", "value", "team", "collaborate", "align", "mission", "passion", "excited", "grow", "work ethic", "transparency"]
            cf_matches = sum(1 for k in cf_keywords if k in answer.lower())
            cf_score_base += min(20.0, cf_matches * 4.0)
            
            cultural_fit_score = round(max(0.0, min(100.0, cf_score_base)), 1)

        # Integrate Sensor Metrics
        emotions = current_metrics.get("emotions", ["neutral"])
        posture_score = extract_score(current_metrics, "posture", 70.0)
        eye_contact_score = extract_score(current_metrics, "eye_contact", 70.0)
        body_language_score = extract_score(current_metrics, "body_language", 70.0)
        attire_score = extract_score(current_metrics, "attire", 70.0)
        confidence_score = extract_score(current_metrics, "confidence", 70.0)
        facial_expression_score = extract_score(current_metrics, "facial_expression", 70.0)
        voice_score = extract_score(current_metrics, "voice", 70.0)
        engagement_score = extract_score(current_metrics, "engagement", 70.0)
        professional_presence_score = extract_score(current_metrics, "professional_presence", 70.0)

        # Check emotional state/confidence
        is_nervous = "nervous" in emotions or "anxious" in emotions or current_metrics.get("primary_emotion") in ["nervous", "anxious"]
        if is_nervous:
            feedback += " Note: Candidate appears slightly nervous. Suggest adjusting question difficulty or asking an encouraging follow-up."

        # Compile evaluation details
        single_eval = {
            "question": question,
            "answer": answer,
            "quality_tier": quality,
            "accuracy_score": score,
            "words": len(answer.split()),
            "matched_keywords": tech_keywords_matched,
            "technical_competency": technical_competency_score,
            "communication_quality": communication_quality_score,
            "behavioral_assessment": behavioral_assessment_score,
            "learning_potential": learning_potential_score,
            "cultural_fit": cultural_fit_score,
            "metrics": {
                "posture": posture_score,
                "eye_contact": eye_contact_score,
                "body_language": body_language_score,
                "attire": attire_score,
                "confidence": confidence_score,
                "facial_expression": facial_expression_score,
                "voice": voice_score,
                "engagement": engagement_score,
                "professional_presence": professional_presence_score,
                "emotions": emotions
            }
        }

        # Save to historical session states
        state["history"].append(single_eval)
        state["current_index"] += 1
        state["next_question_feedback"] = feedback
        
        # Save state back to file
        self.save_interview_state(session_id, state)

        return {
            "evaluation_score": score,
            "quality_tier": quality,
            "next_question_feedback": feedback,
            "matched_keywords": tech_keywords_matched,
            "technical_competency": technical_competency_score,
            "communication_quality": communication_quality_score,
            "behavioral_assessment": behavioral_assessment_score,
            "learning_potential": learning_potential_score,
            "cultural_fit": cultural_fit_score
        }
