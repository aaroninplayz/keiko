import os
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# List of common security/injection phrases to sanitize
INJECTION_PATTERNS = [
    r"(?i)ignore\s+(?:all\s+)?previous\s+instructions",
    r"(?i)system\s+prompt\s+override",
    r"(?i)override\s+(?:all\s+)?settings",
]

class JobDescriptionIntelligenceAgent:
    """
    Parses and extracts expected requirements, responsibilities, skills, and organizational 
    expectations from a pasted job description or uploaded document to build a structured Role Profile.
    """

    def __init__(self):
        # Known technical keywords
        self.tech_keywords = [
            "python", "javascript", "typescript", "go", "golang", "java", "c++", "c#", "c", "ruby", "rust", 
            "fastapi", "django", "flask", "react", "angular", "vue", "next.js", "express", 
            "docker", "kubernetes", "git", "github", "aws", "gcp", "azure", "postgresql", "mysql", "mongodb", "redis",
            "sql", "html", "css", "html/css", "vanilla css", "tailwind", "tailwind css", "pytorch", "opencv",
            "machine learning", "tensorflow", "ci/cd", "rest api", "graphql", "ui/ux", "ui/ux design"
        ]

    def sanitize_text(self, text: str) -> str:
        """Cleans text while preserving line breaks."""
        if not text:
            return ""
        clean = re.sub(r"<[^>]*>", " ", text)
        for pattern in INJECTION_PATTERNS:
            clean = re.sub(pattern, "[Sanitized Malicious Intent Phrase]", clean)
        clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", clean)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in clean.splitlines()]
        clean_text = "\n".join([l for l in lines if l])
        return clean_text.strip()

    def parse_file(self, file_path: str) -> str:
        """Parses text from PDF, DOCX, or TXT format safely."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Job description file not found: {file_path}")

        _, ext = os.path.splitext(file_path.lower())
        raw_text = ""

        try:
            if ext == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                pages_text = []
                for page in reader.pages[:10]:
                    txt = page.extract_text()
                    if txt:
                        pages_text.append(txt)
                raw_text = "\n".join(pages_text)

            elif ext == ".docx":
                import docx
                doc = docx.Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs[:500] if p.text]
                raw_text = "\n".join(paragraphs)

            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read(100000)

            return self.sanitize_text(raw_text)

        except Exception as e:
            logger.error(f"Error parsing job description file {file_path}: {e}")
            raise ValueError(f"Failed to read job description file: {str(e)}")

    def extract_required_experience(self, text: str) -> int:
        """
        Extracts expected minimum years of experience.
        """
        match = re.search(r"(?:minimum|min|at least|required)?\s*(\d+)\+?\s*(?:-\s*\d+)?\s*years?(?:\s+of)?\s*(?:relevant)?\s*experience", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 1

    def extract_role_title(self, text: str) -> str:
        """Extracts target role title cleanly."""
        roles_regex = r"(?i)\b(Python Developer|Python Backend Engineer|Backend Engineer|Full Stack Engineer|Full Stack Developer|Frontend Engineer|Frontend Developer|Software Engineer|AI/ML Engineer|Data Scientist|DevOps Engineer|Mobile Developer|QA Engineer|System Architect)\b"
        m = re.search(roles_regex, text)
        if m:
            return m.group(1).title()
        
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines[:3]:
            cleaned = re.sub(r"(?i)^(we are looking for a|job title:|role:|position:)\s*", "", line).strip()
            if len(cleaned) < 50 and any(term in cleaned.lower() for term in ["developer", "engineer", "intern", "architect", "lead", "designer", "analyst"]):
                return cleaned.title()
        return "Software Engineer"

    def extract_role_profile(self, text: str) -> Dict[str, Any]:
        """
        Builds a structured Role Profile from JD text.
        """
        sanitized = self.sanitize_text(text)
        profile = {
            "role_title": self.extract_role_title(sanitized),
            "required_skills": [],
            "required_experience_years": 1,
            "responsibilities": [],
            "preferred_qualifications": [],
            "soft_skills": [],
            "industry_domain": "Software Development",
        }

        profile["required_experience_years"] = self.extract_required_experience(sanitized)

        # 2. Required Skills
        for skill in self.tech_keywords:
            pattern = rf"(?i)\b{re.escape(skill)}\b"
            if re.search(pattern, sanitized):
                name = skill.replace("\\", "").title()
                if name.lower() in ["gcp", "aws", "html", "css", "sql", "ci/cd", "rest api"]:
                    name = name.upper()
                elif name.lower() == "fastapi":
                    name = "FastAPI"
                elif name.lower() == "pytorch":
                    name = "PyTorch"
                elif name.lower() == "opencv":
                    name = "OpenCV"
                elif name.lower() == "ui/ux":
                    name = "UI/UX Design"
                profile["required_skills"].append(name)

        # 3. Soft Skills
        soft_skills_list = ["communication", "collaboration", "leadership", "adaptability", "team player", "agile", "scrum", "mentoring"]
        for skill in soft_skills_list:
            if re.search(rf"(?i)\b{re.escape(skill)}\b", sanitized):
                profile["soft_skills"].append(skill.title())

        # 4. Parsing Responsibilities & Qualifications from text sections
        lines = sanitized.splitlines()
        current_section = None
        buffers = {
            "responsibilities": [],
            "preferred": [],
        }

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            lower_line = line_str.lower()
            if any(k in lower_line for k in ["responsibilities", "what you will do", "duties", "role description"]):
                current_section = "responsibilities"
                continue
            elif any(k in lower_line for k in ["preferred", "nice to have", "plus", "bonus", "qualifications", "requirements"]):
                if any(p in lower_line for p in ["preferred", "nice to have", "plus", "bonus"]):
                    current_section = "preferred"
                else:
                    current_section = "requirements"
                continue

            if current_section == "responsibilities" and len(buffers["responsibilities"]) < 10:
                buffers["responsibilities"].append(line_str)
            elif current_section == "preferred" and len(buffers["preferred"]) < 8:
                buffers["preferred"].append(line_str)

        for line in buffers["responsibilities"]:
            profile["responsibilities"].append(line.lstrip("-*• ").strip())

        for line in buffers["preferred"]:
            profile["preferred_qualifications"].append(line.lstrip("-*• ").strip())

        if not profile["responsibilities"]:
            action_verbs = ["design", "develop", "maintain", "build", "collaborate", "lead", "manage", "optimize", "write"]
            for line in lines[:30]:
                if any(line.strip().lower().startswith(v) for v in action_verbs):
                    profile["responsibilities"].append(line.strip())

        if any(w in sanitized.lower() for w in ["ai", "machine learning", "pytorch", "model", "opencv"]):
            profile["industry_domain"] = "Artificial Intelligence"
        elif any(w in sanitized.lower() for w in ["cloud", "devops", "aws", "kubernetes"]):
            profile["industry_domain"] = "Cloud & Infrastructure"

        return profile
