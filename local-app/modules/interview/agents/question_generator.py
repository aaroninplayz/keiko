import re
import random
import logging
from typing import Dict, Any, List, Optional
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

QUESTION_TEMPLATES = {
    'HR': {
        'Beginner': [
            'Tell me about yourself and what motivates you in your career.',
            'What are your greatest strengths as a team member?',
            'Why are you interested in this role?',
            'Describe your ideal work environment.',
            'What are your career goals for the next 2-3 years?',
            'How do you handle constructive criticism?',
        ],
        'Intermediate': [
            'Can you describe a conflict with a coworker and how you resolved it?',
            'Tell me about a time when you had to give difficult feedback to a colleague.',
            'How do you handle competing priorities under tight deadlines?',
            'Describe a situation where you had to adapt to a major organizational change.',
            'What strategies do you use to maintain work-life balance while delivering results?',
            'How do you approach building relationships across different teams?',
        ],
        'Advanced': [
            'Describe how you have influenced company culture or team dynamics in a leadership role.',
            'Tell me about a strategic decision you made that had significant business impact.',
            'How do you approach talent development and mentoring within your team?',
            'Describe a time you navigated organizational politics to achieve a critical objective.',
            'How do you balance short-term deliverables with long-term vision?',
            'Tell me about a time you had to champion a significant change initiative.',
        ],
    },
    'Tech': {
        'Beginner': [
            'What programming languages are you most comfortable with and why?',
            'Can you explain the difference between a list and a dictionary in Python?',
            'How do you approach debugging a simple bug in your code?',
            'What is version control, and how have you used Git in your projects?',
            'Describe a small project you built and the technologies you chose.',
            'What do you understand about APIs and how they work?',
        ],
        'Intermediate': [
            'Walk me through how you would design a REST API for a user management system.',
            'Explain how you handle database migrations in a production environment.',
            'How do you ensure code quality through testing strategies?',
            'Describe your approach to optimizing a slow database query.',
            'How do you handle authentication and authorization in web applications?',
            'Explain the trade-offs between SQL and NoSQL databases for different use cases.',
        ],
        'Advanced': [
            'How would you architect a distributed system that handles millions of concurrent requests?',
            'Explain your approach to designing a fault-tolerant microservices architecture.',
            'How do you handle consistency vs availability tradeoffs in distributed databases?',
            'Describe your strategy for zero-downtime deployments in a large-scale production system.',
            'How would you design a real-time event processing pipeline?',
            'Explain how you would implement observability and monitoring across a complex system.',
        ],
    },
    'Situational': {
        'Beginner': [
            'Tell me about a time you had to learn something new quickly for a project.',
            'Describe a situation where you made a mistake at work and how you handled it.',
            'Can you share an experience where you received constructive criticism?',
            'Tell me about a time you worked with someone whose style was different from yours.',
            'Describe a situation where you had to ask for help.',
            'Tell me about a time when you went above and beyond your responsibilities.',
        ],
        'Intermediate': [
            'Describe a time you led a project through unexpected challenges.',
            'Tell me about a situation where you had to make a critical decision with incomplete information.',
            'How did you handle a scenario where a key dependency or team member was unavailable?',
            'Describe a time when you identified and resolved a systemic issue in your workflow.',
            'Tell me about a project where requirements changed midway through development.',
            'Describe a time you had to convince a skeptical stakeholder to support your approach.',
        ],
        'Advanced': [
            'Describe a situation where you had to rescue a failing project and turn it around.',
            'Tell me about a time you had to make an unpopular decision for the greater good of the team.',
            'How did you handle a critical production incident that affected thousands of users?',
            'Describe how you managed stakeholder expectations during a high-risk delivery.',
            'Tell me about a time you had to balance technical debt with business deadlines.',
            'Describe a complex cross-team negotiation you led to achieve a shared goal.',
        ],
    },
}

CONCLUDING_TEMPLATES = [
    'Thank you for your detailed responses today. Before we wrap up, do you have any questions about the role or our team?',
    'I appreciate you sharing your experience with us. Is there anything else you would like to highlight or ask about this position?',
    'We are coming to the end of our conversation. Do you have any final questions for me about the role, the team, or the company?',
]

SIGNOFF_TEMPLATES = [
    'Thank you for your time today. We will review your responses and our team will be in touch regarding the next steps. We appreciate your interest and wish you all the best.',
    'It was a pleasure speaking with you. We will be evaluating all candidates and will reach out soon with an update. Thank you again for your interest.',
]

class QuestionGenerator:
    """
    Hybrid LLM/NLP Question Generator. Ingests full resume text, job description text,
    normalized Candidate Profile, skill gaps, interview history, and evaluator guidelines.
    Uses a small local instruct model if available, falling back to a deterministic NLP template engine.
    """

    def __init__(self):
        self._llm_client = LLMClient()
        self._generator = None
        self._local_llm_attempted = False

    @property
    def generator(self):
        if not self._local_llm_attempted:
            self._load_local_llm()
        return self._generator

    def _load_local_llm(self):
        if self._local_llm_attempted:
            return
        self._local_llm_attempted = True
        try:
            from transformers import pipeline
            # Use Qwen2.5-0.5B-Instruct: tiny, lightweight instruct model (~950MB) running cleanly on CPU
            logger.info("Attempting to load local instruction LLM 'Qwen/Qwen2.5-0.5B-Instruct'...")
            self._generator = pipeline(
                "text-generation", 
                model="Qwen/Qwen2.5-0.5B-Instruct", 
                device=-1, # Force CPU execution
                max_new_tokens=100
            )
            logger.info("Local instruction LLM successfully loaded.")
        except Exception as e:
            logger.warning(f"Could not load local LLM, falling back to robust NLP template generator: {e}")
            self._generator = None

    def generate(
        self,
        resume_text: str,
        jd_text: str,
        candidate_profile: Dict[str, Any],
        skill_gaps: List[str],
        history: List[Dict[str, str]],
        evaluator_feedback: Optional[str] = None,
        interview_type: str = 'Tech',
        difficulty: str = 'Intermediate',
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates the next interview question using multi-stage context, 10-Q phase thresholding,
        live emotion/composure ingestion, and resume discrepancy cross-referencing.
        """
        turn_count = len(history) + 1
        raw_name = candidate_profile.get('full_name') or candidate_profile.get('name') or 'Candidate'
        first_name = raw_name.strip().split()[0] if raw_name and raw_name != 'Candidate' else 'Candidate'
        target_role = candidate_profile.get('target_role') or 'Target Role'

        # Turn 1: Onboarding with natural pleasantries
        if turn_count == 1:
            return f"Hello {first_name}! Welcome to your {interview_type} interview today for the {target_role} position. How are you doing, and could you briefly introduce yourself and walk me through your background?"

        # Stage/Phase Thresholding
        phase_label = "Phase 1: Onboarding & Background" if turn_count <= 10 else ("Phase 2: Technical Deep Dive" if turn_count <= 20 else "Phase 3: System Architecture")

        # Tier 1: Try LLM Client (OpenAI, Gemini, Anthropic, Groq, or local Ollama qwen2.5:1.5b)
        active_provider = self._llm_client.detect_provider()
        if active_provider:
            try:
                logger.info(f"Generating question via LLM provider ({active_provider}) for Turn {turn_count} ({phase_label})...")
                
                last_turn_str = ""
                if history:
                    last_turn = history[-1]
                    last_turn_str = f"Last Interviewer Question: {last_turn.get('question')}\nCandidate Answer: {last_turn.get('answer')[:350]}\n"

                # Extract live emotion & sensor telemetry
                sensor_info = ""
                if sensor_data:
                    comp = sensor_data.get('composure', 75)
                    eye = sensor_data.get('eye_contact', 80)
                    tone = sensor_data.get('tone', 'confident')
                    sensor_info = f"Candidate Live Composure: {comp}%, Eye Contact: {eye}%, Tone: {tone}.\n"
                    if comp < 50:
                        sensor_info += "STRESS DETECTED: Keep question encouraging, concise, and clear.\n"
                    elif comp > 85:
                        sensor_info += "HIGH CONFIDENCE: Escalate technical challenge.\n"

                system_message = (
                    "You are Keiko, a professional and personable AI technical interviewer.\n\n"
                    f"Candidate's first name: {first_name}\n"
                    f"Current Phase: {phase_label} | Mode: {interview_type} | Level: {difficulty}\n"
                    f"{sensor_info}\n"
                    "ALWAYS:\n"
                    "1. Start with a brief, natural acknowledgment of the candidate's last answer (1-2 sentences).\n"
                    "2. Then ask your next question.\n"
                    "3. Be conversational but professional.\n"
                    "4. If the candidate asks you something, respond naturally.\n"
                    "5. If the candidate is rude or unprofessional, calmly redirect while noting it.\n"
                    f"6. Use the candidate's name ({first_name}) occasionally.\n"
                    "7. Keep total response under 60 words.\n"
                    "8. Cross-reference answers with their resume — if they claim something contradictory, politely ask for clarification.\n\n"
                    "NEVER:\n"
                    "1. Skip acknowledging the candidate's answer.\n"
                    "2. Be robotic or formulaic.\n"
                    "3. Repeat questions that have already been asked.\n"
                    "4. Ignore what the candidate said."
                )
                
                user_message = (
                    f"Candidate Name: {first_name}\n"
                    f"Target Role & JD Context: {jd_text[:1500]}\n"
                    f"Candidate Resume Context: {resume_text[:1500]}\n"
                    f"Priority Skill Gaps: {', '.join(skill_gaps[:3]) if skill_gaps else 'None identified'}\n"
                    f"{last_turn_str}"
                    f"Evaluator Guidance: {evaluator_feedback or 'Ask next relevant question for current phase.'}\n\n"
                    "Generate your response (1-sentence conversational acknowledgment + next question):"
                )

                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                
                question = self._llm_client.complete(messages, provider=active_provider, temperature=0.7, max_tokens=150)
                if question:
                    question = re.sub(r'[\r\n]+', ' ', question).strip()
                    question = question.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                    question = re.sub(r"^(Interviewer|Keiko|Question):\s*", "", question, flags=re.IGNORECASE).strip()
                    if len(question) > 10:
                        return question
            except Exception as e:
                logger.error(f"LLM question generation failed: {e}. Falling back to templates.")

        # Turns 1-3 serve as semi-fixed natural onboarding starters using candidate's name
        if turn_count == 1:
            return f"Hello {first_name}! Welcome to your {interview_type} interview today for the {target_role} position. How are you doing, and could you briefly introduce yourself and walk me through your background?"
        elif turn_count == 2:
            if skill_gaps:
                target_gap = skill_gaps[0]
                return f"Thank you for sharing that overview, {first_name}! Looking at the job requirements, I see {target_gap} is a key component. Could you tell me about your experience with {target_gap}?"
            return f"Thank you for sharing that overview, {first_name}! Could you walk me through a key project or role from your resume that best demonstrates your core capabilities?"
        elif turn_count == 3:
            if len(skill_gaps) > 1:
                target_gap = skill_gaps[1]
                return f"Got it, {first_name}. Could you also describe your familiarity or experience with {target_gap}?"
            return "What specific programming languages, frameworks, or tools did you rely on most heavily in that role, and why were they selected?"

        # Turn 4+: Rely exclusively on LLM or Tier 3 NLP Template Generator (no fixed elif chain)
        return self._generate_nlp_fallback(candidate_profile, skill_gaps, history, evaluator_feedback, interview_type, difficulty)

    def generate_hint(self, question: str, candidate_answer: str, resume_text: str = "") -> str:
        """
        Generates a concise 1-sentence AI hint / clarification for the candidate without changing the main question.
        """
        active_provider = self._llm_client.detect_provider()
        if active_provider:
            try:
                prompt = (
                    f"Interviewer Question: {question}\n"
                    f"Candidate Current Draft: {candidate_answer[:200]}\n"
                    "Provide a helpful 1-sentence technical hint to guide the candidate's answer. NO salutations, under 25 words."
                )
                res = self._llm_client.complete([{"role": "user", "content": prompt}], provider=active_provider, temperature=0.5, max_tokens=60)
                if res:
                    clean = re.sub(r'[\r\n]+', ' ', res).strip()
                    return clean
            except Exception as e:
                logger.warning(f"AI hint generation failed: {e}")

        return "Focus on your specific individual contribution, key architectural choices, and measurable results achieved."

    def _build_prompt(
        self,
        resume_text: str,
        jd_text: str,
        profile: Dict[str, Any],
        gaps: List[str],
        history: List[Dict[str, str]],
        feedback: Optional[str],
        interview_type: str = 'Tech',
        difficulty: str = 'Intermediate'
    ) -> str:
        history_str = ""
        for h in history:
            history_str += f"Interviewer: {h.get('question')}\nCandidate: {h.get('answer')}\n"

        prompt = (
            "<|im_start|>system\n"
            "You are Keiko, a professional and personable AI technical interviewer.\n"
            f"Interview Type: {interview_type}\n"
            f"Difficulty Level: {difficulty}\n"
            "ALWAYS:\n"
            "1. Start with a brief, natural acknowledgment of the candidate's last answer (1-2 sentences).\n"
            "2. Then ask your next question.\n"
            "3. Be conversational but professional.\n"
            "4. Keep total response under 60 words.\n"
            "NEVER:\n"
            "1. Skip acknowledging the candidate's answer.\n"
            "2. Be robotic or formulaic.\n"
            "3. Repeat questions.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Candidate Experience Level: {profile.get('career_level')}\n"
            f"Candidate Resume Context: {resume_text[:1500]}\n"
            f"Job Description: {jd_text[:1500]}\n"
            f"Skill Gaps Identified: {', '.join(gaps)}\n"
            f"Interview History:\n{history_str}\n"
            f"Evaluator Guidance: {feedback or 'First question, ask a main job description question.'}\n\n"
            "Generate your response (acknowledgment + next question):\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt

    def _generate_nlp_fallback(
        self,
        profile: Dict[str, Any],
        gaps: List[str],
        history: List[Dict[str, str]],
        feedback: Optional[str],
        interview_type: str = 'Tech',
        difficulty: str = 'Intermediate'
    ) -> str:
        """
        High-fidelity template engine. Synthesizes contextual probing and main questions.
        Uses candidate name, interview type, skill gaps, and difficulty to select from comprehensive template banks.
        """
        q_count = len(history)
        raw_name = profile.get('full_name') or profile.get('name') or 'Candidate'
        first_name = raw_name.strip().split()[0] if raw_name and raw_name != 'Candidate' else 'Candidate'

        # Conversational acknowledgment prefix if candidate answered a previous question
        ack_prefix = ""
        if history:
            ack_options = [
                f"Thank you for sharing that overview, {first_name}.",
                f"That makes sense, {first_name}.",
                "Thanks for explaining that.",
                "Appreciate the context.",
                f"Got it, {first_name}."
            ]
            ack_prefix = random.choice(ack_options) + " "

        # 1. Handle Evaluator Feedback (Probing / Adjustments)
        if feedback:
            lower_fb = feedback.lower()
            # If evaluator instructs to probe a specific topic or skill
            for gap in gaps:
                if gap.lower() in lower_fb:
                    return f"{ack_prefix}I noticed a gap in {gap} on your resume. Could you describe your familiarity with {gap} or any experience learning similar technologies?"

            # If evaluator wants details on projects
            if "project" in lower_fb and profile.get("project_expertise"):
                proj = profile["project_expertise"][0]
                return f"{ack_prefix}You worked on the project '{proj}'. Could you detail your specific engineering contributions and the technical stack you used?"

            # When probing is requested AND skill gaps exist, target the gaps first
            if "probe" in lower_fb and gaps:
                asked_topics = set()
                for h in history:
                    q_lower = h.get('question', '').lower()
                    for gap in gaps:
                        if gap.lower() in q_lower:
                            asked_topics.add(gap)
                unasked_gaps = [g for g in gaps if g not in asked_topics]
                if unasked_gaps:
                    target_gap = unasked_gaps[0]
                    return f"{ack_prefix}Looking at the job requirements, I see {target_gap} is a key component. Can you walk me through your experience with {target_gap} or related tools?"

            # General probing fallback based on feedback
            if "probe" in lower_fb:
                cand_skills = []
                for cat, skills in profile.get("skills", {}).items():
                    for s in skills:
                        if isinstance(s, dict):
                            cand_skills.append(s.get('name', ''))
                        else:
                            cand_skills.append(str(s))
                if cand_skills:
                    return f"{ack_prefix}Can you explain a challenging problem you solved using {cand_skills[0]} and how you optimized your solution?"

        # 2. Skill Gap Targeting: If gaps exist and haven't been addressed yet, ask about them
        if gaps and q_count > 0:
            asked_topics = set()
            for h in history:
                q_lower = h.get('question', '').lower()
                for gap in gaps:
                    if gap.lower() in q_lower:
                        asked_topics.add(gap)
            unasked_gaps = [g for g in gaps if g not in asked_topics]
            if unasked_gaps:
                target_gap = unasked_gaps[0]
                return f"{ack_prefix}Looking at the job requirements, I see {target_gap} is a key component. Can you walk me through your experience with {target_gap} or related tools?"

        # 3. Dynamic Template-based question selection avoiding duplicates
        templates = QUESTION_TEMPLATES.get(interview_type, QUESTION_TEMPLATES['Tech'])
        questions = templates.get(difficulty, templates['Intermediate'])
        
        # Filter out already asked questions
        history_questions = {h.get('question', '').strip().lower() for h in history}
        available_questions = [q for q in questions if q.strip().lower() not in history_questions]
        
        if not available_questions:
            available_questions = questions

        index = q_count % len(available_questions)
        selected_q = available_questions[index]
        return f"{ack_prefix}{selected_q}"

    def generate_concluding(self, interview_type: str = 'Tech') -> str:
        """Returns a random concluding question to wrap up the interview."""
        return random.choice(CONCLUDING_TEMPLATES)

    def generate_signoff(self) -> str:
        """Returns a random sign-off message to end the interview."""
        return random.choice(SIGNOFF_TEMPLATES)
