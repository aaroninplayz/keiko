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
    Hybrid LLM/NLP Question Generator.
    Supports distinct HR, Technical, and Situational interview personas.
    Ingests live sensor tracking (eye contact, posture, composure, tone, confidence),
    uses context compression (summarizes history every 5 questions),
    and enforces a sliding window context with strict non-repetition rules.
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

    def summarize_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Compresses preceding Q&A conversation history into a concise 2-3 sentence summary.
        Called every 5 questions to keep prompt context compact.
        """
        if not history:
            return ""

        active_provider = self._llm_client.detect_provider()
        formatted_turns = []
        for i, turn in enumerate(history, start=1):
            q = turn.get('question', '')
            a = turn.get('answer', '')[:250]
            formatted_turns.append(f"Turn {i} - Q: {q} | A: {a}")

        transcript_text = "\n".join(formatted_turns)
        prompt = (
            "Summarize the following interview conversation history into a concise 2-3 sentence overview. "
            "Focus on candidate highlights, demonstrated capabilities, and general performance trajectory. "
            "Keep it under 60 words:\n\n"
            f"{transcript_text}"
        )

        if active_provider:
            try:
                res = self._llm_client.complete([{"role": "user", "content": prompt}], provider=active_provider, temperature=0.3, max_tokens=100)
                if res:
                    clean = re.sub(r'[\r\n]+', ' ', res).strip()
                    return clean
            except Exception as e:
                logger.warning(f"Context compression via LLM failed: {e}")

        # Fallback summarizer if LLM is unavailable
        num_turns = len(history)
        avg_score = sum(float(h.get('accuracy_score', 70.0) or 70.0) for h in history) / max(1, num_turns)
        return f"Candidate completed {num_turns} turns with an average score of {avg_score:.1f}%. Key responses covered background, technical/behavioral concepts, and problem solving."

    def _build_system_prompt(
        self,
        interview_type: str,
        difficulty: str,
        first_name: str,
        phase_label: str,
        sensor_info: str,
        interviewer_interest: float,
        asked_questions: List[str]
    ) -> str:
        """
        Constructs explicit system prompt rules customized for interview type (HR, Tech, Situational)
        and difficulty tier (Beginner, Intermediate, Advanced, Adaptive).
        Rules are included on EVERY single prompt request.
        """
        # Persona & Type Directives
        if interview_type == 'HR':
            persona = (
                "You are Keiko, an experienced Human Resources (HR) Talent Acquisition Lead.\n"
                "FOCUS AREAS: Soft skills, motivation, company culture fit, team collaboration, leadership, work environment, career aspirations, and communication.\n"
                "STRICTLY FORBIDDEN: DO NOT ask any coding syntax, software engineering theory, system design, framework, or technical CS questions."
            )
        elif interview_type == 'Situational':
            persona = (
                "You are Keiko, a Senior Operations Manager conducting a Situational & Scenario Interview.\n"
                "FOCUS AREAS: Scenario-based hypothetical workplace decision-making under pressure, crisis management, priority trade-offs, conflict resolution, and adaptability to unexpected changes.\n"
                "STRICTLY FORBIDDEN: DO NOT ask standard coding or CS theory questions."
            )
        else:
            persona = (
                "You are Keiko, a Senior Technical Architect conducting a Technical Engineering Interview.\n"
                "FOCUS AREAS: Code architecture, system design, data structures, APIs, database trade-offs, debugging, performance optimization, and engineering practices.\n"
                "STRICTLY FORBIDDEN: DO NOT ask generic HR behavioral questions."
            )

        # Difficulty Directives
        if difficulty == 'Beginner':
            diff_text = "DIFFICULTY LEVEL: Beginner (Ask entry-level, foundational questions focusing on core concepts)."
        elif difficulty == 'Advanced':
            diff_text = "DIFFICULTY LEVEL: Advanced (Ask senior/architect-level questions covering high scale, edge cases, system trade-offs, and resilience)."
        elif difficulty == 'Adaptive':
            diff_text = "DIFFICULTY LEVEL: Adaptive (Dynamically adjust complexity based on candidate score and interest score)."
        else:
            diff_text = "DIFFICULTY LEVEL: Intermediate (Ask mid-level industry standard questions with trade-off analysis)."

        # Interest Level Guidance
        interest_guidance = ""
        if interviewer_interest >= 75.0:
            interest_guidance = f"INTERVIEWER INTEREST: HIGH ({interviewer_interest:.1f}/100). Candidate is performing exceptionally well. Offer encouraging tone and dive deeper into complex aspects."
        elif interviewer_interest <= 40.0:
            interest_guidance = f"INTERVIEWER INTEREST: LOW ({interviewer_interest:.1f}/100). Candidate gave brief or off-target response. Ask a clear, encouraging question to help candidate clarify."
        else:
            interest_guidance = f"INTERVIEWER INTEREST: NORMAL ({interviewer_interest:.1f}/100). Maintain steady professional engagement."

        # Asked Questions List for Non-Repetition
        asked_str = ""
        if asked_questions:
            formatted_q = "\n".join(f"- {q}" for q in asked_questions)
            asked_str = f"\nALREADY ASKED QUESTIONS (DO NOT REPEAT):\n{formatted_q}\n"

        system_prompt = (
            f"{persona}\n\n"
            f"Candidate Name: {first_name}\n"
            f"Current Phase: {phase_label}\n"
            f"{diff_text}\n"
            f"{interest_guidance}\n"
            f"{sensor_info}\n"
            f"{asked_str}\n"
            "MAIN MANDATORY RULES (ENFORCE ON EVERY SINGLE QUESTION):\n"
            "1. STRICT NON-REPETITION: DO NOT ask any question that is semantically or lexically similar to any question in the Already Asked Questions list above.\n"
            "2. CONVERSATIONAL HR/INTERVIEWER TONE: Start with a brief, natural acknowledgment of the candidate's last answer (1-2 sentences).\n"
            "3. CANDIDATE QUESTIONS: If the candidate asked a question back (about role, tech stack, team, or culture), DIRECTLY ANSWER IT in 1-2 friendly sentences BEFORE asking your next question.\n"
            "4. TARGETED NEXT QUESTION: Ask your next question matching the exact interview type and difficulty level.\n"
            "5. LENGTH LIMIT: Keep total response concise (under 70 words).\n"
            "6. PERSONALIZATION: Use the candidate's name occasionally and cross-reference resume context when relevant."
        )

        return system_prompt

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
        sensor_data: Optional[Dict[str, Any]] = None,
        interviewer_interest: float = 50.0,
        conversation_summary: Optional[str] = None,
        window_size: int = 3
    ) -> str:
        """
        Generates the next interview question using multi-stage context,
        sliding window memory (last 3 turns), 5-question context compression,
        and live sensor tracking telemetry (eye contact, posture, composure, tone, emotions).
        """
        turn_count = len(history) + 1
        raw_name = candidate_profile.get('full_name') or candidate_profile.get('name') or 'Candidate'
        first_name = raw_name.strip().split()[0] if raw_name and raw_name != 'Candidate' else 'Candidate'
        target_role = candidate_profile.get('target_role') or 'Target Role'

        # Extract all previously asked questions to enforce non-repetition
        asked_questions = [h.get('question', '') for h in history if h.get('question')]

        # Turn 1: Onboarding with natural pleasantries
        if turn_count == 1:
            return f"Hello {first_name}! Welcome to your {interview_type} interview today for the {target_role} position. How are you doing, and could you briefly introduce yourself and walk me through your background?"

        # Phase thresholding
        phase_label = "Phase 1: Onboarding & Background" if turn_count <= 5 else ("Phase 2: Deep Dive" if turn_count <= 15 else "Phase 3: Advanced Architecture & Leadership")

        # Tier 1: Try LLM Client
        active_provider = self._llm_client.detect_provider()
        if active_provider:
            try:
                logger.info(f"Generating question via LLM ({active_provider}) for Turn {turn_count} ({interview_type}/{difficulty})...")

                # Extract live emotion & sensor telemetry
                sensor_info = ""
                if sensor_data:
                    eye_obj = sensor_data.get('eye_contact', {})
                    eye = eye_obj.get('score', 80.0) if isinstance(eye_obj, dict) else float(eye_obj)

                    posture_obj = sensor_data.get('posture', {})
                    posture = posture_obj.get('score', 80.0) if isinstance(posture_obj, dict) else float(posture_obj)

                    conf_obj = sensor_data.get('confidence', {})
                    conf_details = conf_obj.get('details', {}) if isinstance(conf_obj, dict) else {}
                    comp = conf_details.get('composure', sensor_data.get('composure', 75.0))

                    tone = sensor_data.get('tone', 'confident')
                    sensor_info = f"LIVE TRACKING PARAMETERS -> Eye Contact: {eye:.1f}%, Posture: {posture:.1f}%, Composure: {comp:.1f}%, Tone: {tone}.\n"
                    if comp < 50 or eye < 50:
                        sensor_info += "SENSOR TELEMETRY ALERT: Stress or gaze deviation detected. Keep question clear, encouraging, and focused.\n"
                    elif comp > 85 and eye > 80:
                        sensor_info += "SENSOR TELEMETRY ALERT: High confidence & eye contact detected. Candidate is engaged.\n"

                system_message = self._build_system_prompt(
                    interview_type=interview_type,
                    difficulty=difficulty,
                    first_name=first_name,
                    phase_label=phase_label,
                    sensor_info=sensor_info,
                    interviewer_interest=interviewer_interest,
                    asked_questions=asked_questions
                )

                # Context Assembly: Sliding Window (last N turns) + Compressed Summary for older turns
                window_history = history[-window_size:] if len(history) > window_size else history
                
                recent_turns_str = ""
                for h in window_history:
                    recent_turns_str += f"Interviewer Question: {h.get('question')}\nCandidate Answer: {h.get('answer', '')[:350]}\n"

                summary_str = f"COMPRESSED PREVIOUS CONVERSATION SUMMARY: {conversation_summary}\n" if conversation_summary else ""

                user_message = (
                    f"Candidate Name: {first_name}\n"
                    f"Target Role: {target_role}\n"
                    f"Job Context: {jd_text[:1200]}\n"
                    f"Resume Context: {resume_text[:1200]}\n"
                    f"Skill Gaps: {', '.join(skill_gaps[:3]) if skill_gaps else 'None'}\n"
                    f"{summary_str}"
                    f"SLIDING WINDOW RECENT TURNS:\n{recent_turns_str}\n"
                    f"Evaluator Feedback: {evaluator_feedback or 'Ask next relevant question for phase.'}\n\n"
                    "Generate response (conversational acknowledgment + next unasked question):"
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
                    
                    if len(question) > 10 and not self._is_question_repeated(question, history):
                        return question
            except Exception as e:
                logger.error(f"LLM question generation failed: {e}. Falling back to NLP template generator.")

        # Fallback NLP generator
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
                    "Provide a helpful 1-sentence hint to guide the candidate's answer. NO salutations, under 25 words."
                )
                res = self._llm_client.complete([{"role": "user", "content": prompt}], provider=active_provider, temperature=0.5, max_tokens=60)
                if res:
                    clean = re.sub(r'[\r\n]+', ' ', res).strip()
                    return clean
            except Exception as e:
                logger.warning(f"AI hint generation failed: {e}")

        return "Focus on your specific individual contribution, key choices, and measurable results achieved."

    def _is_question_repeated(self, candidate_q: str, history: List[Dict[str, str]]) -> bool:
        """
        Strictly checks if candidate_q is semantically or lexically similar to any question in history.
        """
        if not candidate_q or not history:
            return False

        cand_clean = re.sub(r'[^a-zA-Z0-9\s]', '', candidate_q.lower()).strip()
        cand_words = set(w for w in cand_clean.split() if len(w) > 3)

        for turn in history:
            prev_q = turn.get('question', '')
            if not prev_q:
                continue
            prev_clean = re.sub(r'[^a-zA-Z0-9\s]', '', prev_q.lower()).strip()

            if cand_clean == prev_clean:
                return True
            if len(cand_clean) > 15 and (cand_clean in prev_clean or prev_clean in cand_clean):
                return True

            prev_words = set(w for w in prev_clean.split() if len(w) > 3)
            if cand_words and prev_words:
                intersection = cand_words.intersection(prev_words)
                union = cand_words.union(prev_words)
                jaccard = len(intersection) / len(union) if union else 0.0
                if jaccard >= 0.45:
                    return True

        return False

    def _get_next_unasked_question(
        self,
        profile: Dict[str, Any],
        gaps: List[str],
        history: List[Dict[str, str]],
        interview_type: str = 'Tech',
        difficulty: str = 'Intermediate'
    ) -> str:
        templates = QUESTION_TEMPLATES.get(interview_type, QUESTION_TEMPLATES['Tech'])
        questions = templates.get(difficulty, templates.get('Intermediate', []))

        for q in questions:
            if not self._is_question_repeated(q, history):
                return q

        for diff_tier in ['Intermediate', 'Beginner', 'Advanced']:
            for q in templates.get(diff_tier, []):
                if not self._is_question_repeated(q, history):
                    return q

        target_role = profile.get('target_role') or 'this role'
        if interview_type == 'HR':
            return f"What specific qualities or values do you believe are most important for succeeding in the {target_role} position?"
        elif interview_type == 'Situational':
            return f"Describe how you handle unexpected project changes or conflicting stakeholder priorities in a fast-paced environment."
        else:
            return f"Walk me through a challenging technical problem you solved in your recent work and the key architectural trade-offs involved."

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
        Template generator fallback ensuring unasked question selection matching interview type.
        """
        raw_name = profile.get('full_name') or profile.get('name') or 'Candidate'
        first_name = raw_name.strip().split()[0] if raw_name and raw_name != 'Candidate' else 'Candidate'

        last_answer = history[-1].get('answer', '') if history else ''
        is_cand_q = False
        if feedback and "CANDIDATE_ASKED_QUESTION" in feedback:
            is_cand_q = True
        elif last_answer:
            answer_clean = last_answer.strip().lower()
            if "?" in last_answer or any(re.search(pat, answer_clean) for pat in [
                r"\bwhat (is|are|about|does|do|can|tech|stack|culture|role)\b",
                r"\bhow (do|does|is|are|can)\b",
                r"\bcan you (tell|explain|share|elaborate)\b",
                r"\bcould you (tell|explain|share|elaborate)\b",
                r"\bdo you (have|use|offer|work)\b",
                r"\bis there\b",
                r"\bwhat's\b",
                r"\btell me (about|more)\b"
            ]):
                is_cand_q = True

        if is_cand_q and last_answer:
            ans_clean = last_answer.lower()
            if any(k in ans_clean for k in ["culture", "work environment", "team", "values", "people"]):
                response = "Our company culture is centered around collaboration, open communication, continuous learning, and work-life balance."
            elif any(k in ans_clean for k in ["tech", "stack", "tools", "technology", "language", "framework"]):
                response = "Our team utilizes modern, scalable technologies and industry best practices focusing on solid architecture, testing, and continuous delivery."
            elif any(k in ans_clean for k in ["role", "responsibility", "day to day", "expectation"]):
                response = f"In this {profile.get('target_role', 'position')}, you will work closely with cross-functional teams, driving key initiatives from concept to delivery."
            else:
                response = "In our organization, we prioritize transparency, employee growth, and strong cross-functional teamwork."

            next_q = self._get_next_unasked_question(profile, gaps, history, interview_type, difficulty)
            return f"{response} {next_q}"

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

        next_q = self._get_next_unasked_question(profile, gaps, history, interview_type, difficulty)
        return f"{ack_prefix}{next_q}"

    def generate_concluding(self, interview_type: str = 'Tech') -> str:
        """Returns a random concluding question to wrap up the interview."""
        return random.choice(CONCLUDING_TEMPLATES)

    def generate_signoff(self) -> str:
        """Returns a random sign-off message to end the interview."""
        return random.choice(SIGNOFF_TEMPLATES)
