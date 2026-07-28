import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Base path for session profile cache
SESSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
    "data", 
    "sessions"
)

class ReportGenerator:
    """
    Report Generation Agent. Aggregates candidate profile details, semantic matches,
    transcribed interview question histories, and sensor signals into a professional report.
    """

    def __init__(self):
        pass

    def _calculate_recommendations(
        self, 
        match_results: Dict[str, Any], 
        sub_scores: Dict[str, Any],
        zero_weight_sensors: Optional[set] = None
    ) -> Dict[str, Any]:
        """
        Calculate personalized, actionable coaching/learning recommendations based on the candidate's performance.
        Excludes any sensor that has zero weight.
        """
        if zero_weight_sensors is None:
            zero_weight_sensors = set()

        recommendations = {
            "technical_learning_paths": {},
            "communication_advice": None,
            "presentation_advice": [],
            "custom_practice_questions": []
        }
        
        # 1. Analyze skill gaps from match results
        skill_gaps = match_results.get("skill_gap", []) if match_results else []
        for gap in skill_gaps:
            recommendations["technical_learning_paths"][gap] = {
                "action_items": [
                    f"Read official documentation and explore advanced features of {gap}.",
                    f"Build a small end-to-end prototype using {gap} to gain hands-on proficiency.",
                    f"Review open-source examples showing {gap} integrated in real-world applications."
                ],
                "suggested_resources": [
                    f"Official {gap} Documentation",
                    f"Production-ready design patterns for {gap}",
                    f"Community forums and tutorials for {gap}"
                ]
            }
            # Custom practice questions based on identified gaps
            recommendations["custom_practice_questions"].append(
                f"How would you explain the architecture and key benefits of using {gap} in a production environment?"
            )
            recommendations["custom_practice_questions"].append(
                f"Describe a challenge you might face when integrating {gap} into an existing software stack, and how you would address it."
            )
            
        if not skill_gaps:
            recommendations["custom_practice_questions"].append(
                "Describe a complex technical challenge you solved recently and the trade-offs you considered."
            )
            
        # 2. Analyze communication quality (if not zero weight)
        comm_val = sub_scores.get("communication_quality", 70.0)
        comm_score = comm_val if isinstance(comm_val, (int, float)) else 70.0
        if comm_score < 75.0 and "communication_quality" not in zero_weight_sensors:
            recommendations["communication_advice"] = (
                "Your communication score was below 75. We suggest using the STAR method "
                "(Situation, Task, Action, Result) to structure your behavioral responses. "
                "Furthermore, focus on explaining deeper trade-off rationales behind your technical decisions."
            )
            
        # 3. Analyze posture and eye contact (if not zero weight)
        posture_val = sub_scores.get("posture", 70.0)
        posture_score = posture_val if isinstance(posture_val, (int, float)) else 70.0
        eye_val = sub_scores.get("eye_contact", 70.0)
        eye_score = eye_val if isinstance(eye_val, (int, float)) else 70.0

        if posture_score < 75.0 and "posture" not in zero_weight_sensors:
            recommendations["presentation_advice"].append(
                "Maintain correct posture alignment. Ensure you sit straight and keep your shoulders symmetrical."
            )
        if eye_score < 75.0 and "eye_contact" not in zero_weight_sensors:
            recommendations["presentation_advice"].append(
                "Keep a steady gaze on the camera rather than the screen to project direct eye contact with the interviewer."
            )
            
        return recommendations

    def generate_report(
        self,
        session_id: str,
        cand_profile: Optional[Dict[str, Any]] = None,
        match_results: Optional[Dict[str, Any]] = None,
        db_history: Optional[List[Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Compiles the candidate interview history and generates a comprehensive evaluation report.
        Saves report to recruiter_report.json. Uses configured sensor weights for final scoring
        and handles zero-weight sensors gracefully.
        """
        logger.info(f"Compiling final recruiter report for session {session_id}")
        sess_path = os.path.join(SESSIONS_DIR, session_id)
        
        # Load weights from global default if not provided
        if weights is None:
            from ..sensors.weight_config import global_weights
            weights = global_weights.weights

        # Load candidate profile files if not provided or incomplete
        if cand_profile is None or not cand_profile.get("full_name"):
            disk_profile = {}
            for p_name in ("candidate_context.json", "candidate_profile.json", "resume_profile.json"):
                p_path = os.path.join(sess_path, p_name)
                if os.path.exists(p_path):
                    try:
                        with open(p_path, "r", encoding="utf-8") as f:
                            p_data = json.load(f)
                            if isinstance(p_data, dict):
                                for k, v in p_data.items():
                                    if not disk_profile.get(k) and v:
                                        disk_profile[k] = v
                    except Exception as e:
                        logger.warning(f"Could not load {p_name} for report candidate profile: {e}")
            if cand_profile is None:
                cand_profile = disk_profile
            else:
                for k, v in disk_profile.items():
                    if not cand_profile.get(k) and v:
                        cand_profile[k] = v
        
        if match_results is None:
            match_results = {}
            match_path = os.path.join(sess_path, "match_results.json")
            if os.path.exists(match_path):
                with open(match_path, "r", encoding="utf-8") as f:
                    match_results = json.load(f)

        if db_history is not None:
            history = db_history
        else:
            interview_state = {}
            state_path = os.path.join(sess_path, "interview_state.json")
            if os.path.exists(state_path):
                with open(state_path, "r", encoding="utf-8") as f:
                    interview_state = json.load(f)
            history = interview_state.get("history", [])

        # Helper to compute metric average score from history
        def get_dim_score(key: str, default: float = 70.0) -> float:
            if not history:
                return default
            scores = []
            for h in history:
                val = None
                if key in h and h[key] is not None:
                    val = h[key]
                elif "metrics" in h and isinstance(h["metrics"], dict) and key in h["metrics"]:
                    val = h["metrics"][key]

                if isinstance(val, dict):
                    val = val.get("score", default)

                if isinstance(val, (int, float)):
                    if val <= 1.0:
                        val = val * 100.0
                    scores.append(float(val))
                else:
                    scores.append(default)
            return sum(scores) / len(scores) if scores else default

        # Complete dictionary of evaluateable dimensions
        sensor_scores = {
            "technical_competency": get_dim_score("technical_competency", 75.0),
            "communication_quality": get_dim_score("communication_quality", 75.0),
            "behavioral_assessment": get_dim_score("behavioral_assessment", 75.0),
            "learning_potential": get_dim_score("learning_potential", 75.0),
            "cultural_fit": get_dim_score("cultural_fit", 75.0),
            "posture": get_dim_score("posture", 80.0),
            "eye_contact": get_dim_score("eye_contact", 80.0),
            "body_language": get_dim_score("body_language", 80.0),
            "attire": get_dim_score("attire", 80.0),
            "confidence": get_dim_score("confidence", 80.0),
            "facial_expression": get_dim_score("facial_expression", 80.0),
            "voice": get_dim_score("voice", 80.0),
            "engagement": get_dim_score("engagement", 80.0),
            "professional_presence": get_dim_score("professional_presence", 80.0),
        }

        # Calculate Overall Score based on configured weights
        # Zero-weight sensors are completely excluded from numerator and denominator
        weighted_score_sum = 0.0
        total_active_weight = 0.0
        sub_scores: Dict[str, Any] = {}
        zero_weight_sensors = set()

        for sensor_name, raw_score in sensor_scores.items():
            # Check configured weight for sensor
            w = weights.get(sensor_name)
            if w is None:
                # Default weight allocations for Q&A assessment areas if not explicitly in weights map
                default_qna_weights = {
                    "technical_competency": 0.30,
                    "communication_quality": 0.20,
                    "behavioral_assessment": 0.15,
                    "learning_potential": 0.10,
                    "cultural_fit": 0.05,
                }
                w = default_qna_weights.get(sensor_name, 0.10)

            if w > 0:
                weighted_score_sum += raw_score * w
                total_active_weight += w
                sub_scores[sensor_name] = round(raw_score, 1)
            else:
                zero_weight_sensors.add(sensor_name)
                sub_scores[sensor_name] = "N/A — Not evaluated (configured weight: 0)"

        if total_active_weight > 0:
            overall_score = round(weighted_score_sum / total_active_weight, 1)
        else:
            overall_score = 100.0

        # Retrieve resume-JD match rating and compute combined role alignment score
        resume_jd_score = match_results.get("role_alignment_score") if match_results else 75.0
        if resume_jd_score is None:
            resume_jd_score = 75.0
        role_alignment_score = round((resume_jd_score * 0.4) + (overall_score * 0.6), 1)

        # 2. Extract Behavioral / Emotional Summary
        emotions_encountered = set()
        for h in history:
            for emotion in h.get("metrics", {}).get("emotions", []):
                emotions_encountered.add(str(emotion).lower())
        
        if "nervous" in emotions_encountered or "anxious" in emotions_encountered:
            behavior_summary = "The candidate demonstrated initial nervous signals but remained composed and communicative throughout the session."
        else:
            behavior_summary = "The candidate maintained a professional, calm demeanor with highly stable engagement metrics."

        # 3. Assemble Strengths & Improvements (exclude zero-weight sensors)
        strengths = list(match_results.get("strengths", []))
        strengths.append("Completed the full interview session.")
        
        comm_score = sensor_scores["communication_quality"]
        posture_score = sensor_scores["posture"]
        presence_score = sensor_scores["professional_presence"]
        eye_score = sensor_scores["eye_contact"]

        if comm_score >= 80.0 and "communication_quality" not in zero_weight_sensors:
            strengths.append("Excellent communication skills with structured, detailed answers.")
        if posture_score >= 85.0 and "posture" not in zero_weight_sensors:
            strengths.append("Very strong posture alignment and positive non-verbal cues.")
        if presence_score >= 85.0 and "professional_presence" not in zero_weight_sensors:
            strengths.append("Exceptional professional presence and composure.")

        improvements = []
        if match_results.get("skill_gap"):
            gaps = ", ".join(match_results.get("skill_gap"))
            improvements.append(f"Strengthen alignment on missing stack technologies: {gaps}.")
        if comm_score < 65.0 and "communication_quality" not in zero_weight_sensors:
            improvements.append("Elaborate further on engineering choices (answers were slightly brief).")
        if eye_score < 75.0 and "eye_contact" not in zero_weight_sensors:
            improvements.append("Maintain more direct camera gaze/eye contact during verbal explanations.")
        if posture_score < 75.0 and "posture" not in zero_weight_sensors:
            improvements.append("Improve posture symmetry and try to avoid slouching.")

        # 4. Extract explicit Audio & Video Metrics from history telemetry
        tones = []
        pacing_scores = []
        pitch_variabilities = []
        filler_counts = []
        eye_contact_pcts = []
        posture_stabilities = []
        composure_scores = []

        for h in history:
            m = h.get("metrics", {}) if isinstance(h.get("metrics"), dict) else {}
            # Audio
            v_details = m.get("voice_details", {}) if isinstance(m.get("voice_details"), dict) else {}
            if "tone" in v_details:
                tones.append(v_details["tone"])
            elif "tone" in m:
                tones.append(m["tone"])
            
            pacing = v_details.get("pacing", v_details.get("fluency", m.get("voice", 80.0)))
            pacing_scores.append(float(pacing) if isinstance(pacing, (int, float)) else 80.0)
            
            pitch = v_details.get("pitch_variability", v_details.get("pitch", 75.0))
            pitch_variabilities.append(float(pitch) if isinstance(pitch, (int, float)) else 75.0)

            filler = v_details.get("filler_count", m.get("filler_count", 0))
            filler_counts.append(float(filler) if isinstance(filler, (int, float)) else 0.0)

            # Video
            eye = m.get("eye_contact", h.get("eye_contact_score", 80.0))
            if isinstance(eye, dict):
                eye = eye.get("score", 80.0)
            eye_contact_pcts.append(float(eye) if isinstance(eye, (int, float)) else 80.0)

            posture = m.get("posture", h.get("posture_score", 80.0))
            if isinstance(posture, dict):
                posture = posture.get("score", 80.0)
            posture_stabilities.append(float(posture) if isinstance(posture, (int, float)) else 80.0)

            comp = m.get("composure", m.get("confidence", 80.0))
            if isinstance(comp, dict):
                comp = comp.get("score", 80.0)
            composure_scores.append(float(comp) if isinstance(comp, (int, float)) else 80.0)

        dominant_tone = max(set(tones), key=tones.count) if tones else "Confident"
        avg_pacing = round(sum(pacing_scores) / len(pacing_scores), 1) if pacing_scores else 80.0
        avg_pitch = round(sum(pitch_variabilities) / len(pitch_variabilities), 1) if pitch_variabilities else 75.0
        avg_fillers = round(sum(filler_counts) / len(filler_counts), 1) if filler_counts else 0.0

        avg_eye_contact = round(sum(eye_contact_pcts) / len(eye_contact_pcts), 1) if eye_contact_pcts else 80.0
        avg_posture = round(sum(posture_stabilities) / len(posture_stabilities), 1) if posture_stabilities else 80.0
        avg_composure = round(sum(composure_scores) / len(composure_scores), 1) if composure_scores else 80.0

        audio_metrics = {
            "tone": dominant_tone,
            "pacing_score": avg_pacing,
            "pitch_variability_score": avg_pitch,
            "filler_word_frequency": f"{avg_fillers} fillers/response"
        }

        video_metrics = {
            "eye_contact_pct": avg_eye_contact,
            "posture_stability": avg_posture,
            "composure": avg_composure
        }

        # 5. Final report object
        report = {
            "session_id": session_id,
            "overall_score": overall_score,
            "role_alignment_score": role_alignment_score,
            "weights_used": weights,
            "sub_scores": sub_scores,
            "audio_metrics": audio_metrics,
            "video_metrics": video_metrics,
            "candidate_recommendations": self._calculate_recommendations(
                match_results, 
                sub_scores, 
                zero_weight_sensors=zero_weight_sensors
            ),
            "experience_years": cand_profile.get("experience", cand_profile.get("experience_years", 0)),
            "career_level": cand_profile.get("career_level", "Mid-level"),
            "specialization": cand_profile.get("specialization", []),
            "candidate_profile": cand_profile,
            "strengths": strengths,
            "areas_of_improvement": improvements,
            "behavior_summary": behavior_summary,
            "qa_history": history
        }

        # Save to recruiter_report.json
        report_path = os.path.join(sess_path, "recruiter_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        logger.info(f"Saved recruiter report for session {session_id} to {report_path}")
        return report
