import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class MatchingEngine:
    """
    Computes semantic similarity alignment between Candidate Profiles and Role Profiles,
    identifying matches, missing skills (Skill Gaps), and overall Role Alignment Scores.
    Uses sentence-transformers to run local text vector similarity checks.
    """

    _shared_model = None
    _model_loaded = False

    def __init__(self):
        pass

    @property
    def _model(self):
        if not MatchingEngine._model_loaded:
            self._load_model()
        return MatchingEngine._shared_model

    def _load_model(self):
        if MatchingEngine._model_loaded:
            return
        MatchingEngine._model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer
            # Uses a fast, lightweight CPU-friendly embedding model (~90MB download size)
            logger.info("Initializing local SentenceTransformer 'all-MiniLM-L6-v2'...")
            MatchingEngine._shared_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer model successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            MatchingEngine._shared_model = None

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Computes cosine similarity between text_a and text_b.
        Returns a float between 0.0 and 1.0.
        """
        if not self._model:
            # Fallback to simple Jaccard similarity if model load failed
            a_words = set(text_a.lower().split())
            b_words = set(text_b.lower().split())
            intersection = a_words.intersection(b_words)
            union = a_words.union(b_words)
            return len(intersection) / len(union) if union else 0.0

        try:
            embeddings = self._model.encode([text_a, text_b], show_progress_bar=False)
            emb_a = embeddings[0]
            emb_b = embeddings[1]
            dot = np.dot(emb_a, emb_b)
            norm_a = np.linalg.norm(emb_a)
            norm_b = np.linalg.norm(emb_b)
            similarity = dot / (norm_a * norm_b + 1e-8)
            return float(max(0.0, min(1.0, similarity)))
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def align_profiles(self, candidate_profile: Dict[str, Any], role_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aligns a Candidate Profile against a Role Profile.
        Calculates: matched skills, skill gap (missing), experience alignment, and Role Alignment Score.
        """
        cand_skills = []
        # Flatten all skill category lists from Candidate Profile
        for cat, skills in candidate_profile.get("skills", {}).items():
            for s in skills:
                cand_skills.append(s["name"])

        jd_skills = role_profile.get("required_skills", [])
        
        matched_skills = []
        missing_skills = [] # The Skill Gap
        transferable_skills = []

        # Find matching skills based on semantic similarity
        # Threshold 0.70 represents very strong semantic proximity
        # Threshold 0.50-0.69 represents a potential transferable skill
        for jd_s in jd_skills:
            best_score = 0.0
            best_match = ""

            for cand_s in cand_skills:
                # Check exact case-insensitive match first to save encoding latency
                if jd_s.lower() == cand_s.lower():
                    best_score = 1.0
                    best_match = cand_s
                    break

                score = self.compute_similarity(jd_s, cand_s)
                if score > best_score:
                    best_score = score
                    best_match = cand_s

            if best_score >= 0.70:
                matched_skills.append({
                    "jd_skill": jd_s,
                    "candidate_skill": best_match,
                    "similarity": round(best_score, 2)
                })
            elif 0.50 <= best_score < 0.70:
                transferable_skills.append({
                    "jd_skill": jd_s,
                    "candidate_skill": best_match,
                    "similarity": round(best_score, 2)
                })
            else:
                missing_skills.append(jd_s)

        # Calculate Experience Alignment
        cand_exp = candidate_profile.get("experience_years", 0)
        jd_exp = role_profile.get("required_experience_years", 2)
        if cand_exp >= jd_exp:
            exp_score = 100.0
        else:
            exp_score = round((cand_exp / max(1, jd_exp)) * 100, 1)

        # Calculate Skill Match Score
        if jd_skills:
            # Transferable skills count as half a match
            matches_count = len(matched_skills) + (len(transferable_skills) * 0.5)
            skill_match_score = round((matches_count / len(jd_skills)) * 100, 1)
        else:
            skill_match_score = 100.0

        # Calculate Overall Role Alignment Score
        # 60% skills alignment, 40% experience level alignment
        role_alignment_score = round((skill_match_score * 0.6) + (exp_score * 0.4), 1)

        # Compile strengths & weaknesses summary
        strengths = []
        weaknesses = []

        if exp_score >= 100.0:
            strengths.append(f"Candidate meets or exceeds required experience levels ({cand_exp} yrs vs {jd_exp} yrs).")
        else:
            weaknesses.append(f"Candidate's experience level is slightly below target ({cand_exp} yrs vs {jd_exp} yrs).")

        if len(matched_skills) > 0.5 * len(jd_skills) if jd_skills else True:
            strengths.append("Candidate demonstrates a strong core match on target technologies.")
        else:
            weaknesses.append("Key required technical competencies are missing or undocumented in the resume.")

        # Special check for domain alignment
        cand_domains = candidate_profile.get("domain_expertise", [])
        jd_domain = role_profile.get("industry_domain", "")
        domain_match = any(self.compute_similarity(d, jd_domain) >= 0.65 for d in cand_domains)
        if domain_match:
            strengths.append(f"Strong industry sector alignment with {jd_domain} expectations.")

        return {
            "role_alignment_score": role_alignment_score,
            "skill_match_score": skill_match_score,
            "experience_score": exp_score,
            "matched_skills": matched_skills,
            "transferable_skills": transferable_skills,
            "skill_gap": missing_skills, # Competencies missing from candidate profile
            "strengths": strengths,
            "weaknesses": weaknesses,
        }
