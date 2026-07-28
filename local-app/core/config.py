import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

# Set HF_HOME and SENTENCE_TRANSFORMERS_HOME globally prior to ML packages loading
MODELS_DIR = str(Path(__file__).resolve().parent.parent / "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODELS_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

class Settings(BaseSettings):
    PROJECT_NAME: str = "Modular Web App"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Base Database Configuration
    DATABASE_URL: str = "sqlite:///./app.db"
    
    # Modules to exclude from loading
    DISABLED_MODULES: List[str] = []

    # External LLM Providers Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o"

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-pro"

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20240620"

    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama3-8b-8192"

    # Local Ollama / Lightweight Parameter Model Settings
    # Recommended minimum model size: qwen2.5:7b or llama3:8b (smaller 1.5b models may struggle with structured output and reasoning)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"

    # Speech-to-Text Configuration
    STT_PROVIDER: str = "local"
    WHISPER_MODEL_SIZE: str = "openai/whisper-tiny"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
