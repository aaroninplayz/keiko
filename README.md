<p align="center">
  <h1 align="center">KEIKO</h1>
  <p align="center"><strong>AI-Powered Real-Time Interview Intelligence System</strong></p>
  <p align="center">
    Multi-sensor biometric evaluation · Resume &amp; JD matching · Adaptive question generation · Comprehensive recruiter reports
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick%20Start-blue?style=for-the-badge" alt="Quick Start"></a>
    <a href="#-architecture"><img src="https://img.shields.io/badge/Architecture-purple?style=for-the-badge" alt="Architecture"></a>
    <a href="#-api-reference"><img src="https://img.shields.io/badge/API%20Reference-green?style=for-the-badge" alt="API Reference"></a>
    <a href="#-sensor-array"><img src="https://img.shields.io/badge/Sensor%20Array-orange?style=for-the-badge" alt="Sensor Array"></a>
  </p>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Sensor Array](#-sensor-array)
- [AI Agent Pipeline](#-ai-agent-pipeline)
- [API Reference](#-api-reference)
- [WebSocket Protocol](#-websocket-protocol)
- [Marketing Site (Vercel Deployment)](#-marketing-site)
- [Vision System](#-vision-system)
- [Tech Stack](#-tech-stack)
- [Environment Variables](#-environment-variables)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

**KEIKO** (Knowledge-Enhanced Interview Knowledge Orchestrator) is a real-time AI interview analysis platform that combines computer vision, speech analysis, and natural language processing to provide comprehensive candidate evaluation during live interviews.

The system streams webcam video and microphone audio through a WebSocket connection to a local FastAPI server, where **9 independent biometric sensors** analyze the candidate's performance across multiple dimensions — from posture and eye contact to voice modulation and professional presence. An AI evaluation engine then generates detailed recruiter reports with actionable coaching recommendations.

### How It Works

```
┌─────────────────┐     WebSocket      ┌─────────────────────────────────────┐
│   Browser UI    │ ◄──────────────►   │         FastAPI Backend             │
│                 │   base64 frames    │                                     │
│  • Webcam Feed  │   + audio chunks   │  ┌──────────┐   ┌───────────────┐  │
│  • Mic Stream   │ ──────────────►    │  │  Vision   │   │    Audio      │  │
│  • Config Panel │                    │  │  Sensors  │   │    Pipeline   │  │
│  • Live Metrics │ ◄──────────────    │  │ (9 total) │   │  (Whisper)    │  │
│                 │   telemetry JSON   │  └─────┬─────┘   └──────┬────────┘  │
└─────────────────┘                    │        │                │           │
                                       │  ┌─────▼────────────────▼────────┐  │
                                       │  │      Orchestrator Engine      │  │
                                       │  │  • Weighted score fusion      │  │
                                       │  │  • Composite metric calc      │  │
                                       │  │  • Real-time normalization    │  │
                                       │  └──────────────┬────────────────┘  │
                                       │                 │                   │
                                       │  ┌──────────────▼────────────────┐  │
                                       │  │       AI Agent Pipeline       │  │
                                       │  │  • Central Evaluator          │  │
                                       │  │  • Question Generator         │  │
                                       │  │  • Report Generator           │  │
                                       │  │  • Resume/JD Intelligence     │  │
                                       │  └───────────────────────────────┘  │
                                       └─────────────────────────────────────┘
```

---

## ✨ Key Features

| Category | Features |
|:---------|:---------|
| **🎥 Visual Analysis** | Real-time posture tracking, eye contact measurement, body language assessment, attire formality detection, facial expression/emotion classification |
| **🎤 Audio Analysis** | Speech-to-text via Whisper, speaking pace (WPM), voice modulation, clarity scoring, fluency/hesitation detection, filler word counting |
| **🧠 AI Evaluation** | Multi-LLM support (OpenAI, Gemini, Anthropic, Groq), technical competency grading, behavioral assessment, adaptive question generation |
| **📊 Scoring Engine** | 9 independently weighted sensors, real-time score normalization, composite metrics (Composure, Engagement, Professional Presence, Stress/Resilience) |
| **📝 Reports** | Comprehensive recruiter reports with dimensional breakdowns, personalized coaching recommendations, technical learning paths |
| **🔗 Resume Intelligence** | PDF/DOCX parsing, skill extraction, semantic matching against job descriptions using sentence-transformers |
| **⚡ Real-Time** | Full-duplex WebSocket streaming, frame throttling (every 3rd frame), threaded sensor execution, sub-second telemetry updates |
| **🔧 Configurable** | Dynamic weight adjustment during live sessions, multi-provider LLM switching, local-first architecture with cloud API fallback |

---

## 🏗 Architecture

KEIKO is split into two independently deployable components:

### `marketing-site/` — Public Website (Serverless / Vercel)
Static HTML marketing pages showcasing features, documentation, architecture diagrams, and a smart download page that detects whether the local backend is running.

### `local-app/` — Backend Application (Local Machine / Docker)
FastAPI-powered backend with WebSocket streaming, computer vision pipeline, speech analysis, and AI evaluation agents. Runs locally on the user's machine for privacy-first processing.

```
keiko/
├── marketing-site/          # Public website (deploy to Vercel)
│   ├── index.html           # Landing page
│   ├── about.html           # Intelligence board / team
│   ├── features.html        # Product features showcase
│   ├── architecture.html    # System architecture diagrams
│   ├── documentation.html   # Docs (changelogs, FAQ, open-source tabs)
│   ├── download.html        # Smart download hub (health polling)
│   ├── interview-journey.html
│   ├── audio-intelligence.html
│   ├── video-intelligence.html
│   ├── job-intelligence.html
│   ├── resume-intelligence.html
│   ├── interview-intelligence.html
│   └── js/
│       └── particles-config.js
│
├── local-app/               # Backend application (run locally)
│   ├── main.py              # Entry point (Uvicorn server)
│   ├── requirements.txt     # Python dependencies
│   ├── setup_and_run.bat    # Windows quick-launch script
│   │
│   ├── core/                # Application foundation
│   │   ├── app.py           # FastAPI factory + CORS + auto-browser
│   │   ├── config.py        # Pydantic settings (LLM keys, DB, models)
│   │   ├── database.py      # SQLAlchemy engine + session factory
│   │   └── registry.py      # Dynamic module discovery & loading
│   │
│   ├── modules/
│   │   ├── auth/            # Authentication (bypassed → user@local.app)
│   │   │   ├── dependencies.py  # Auto-user injection (no tokens required)
│   │   │   ├── models_db.py     # User SQLAlchemy model
│   │   │   ├── router.py        # Auth endpoints (legacy, bypassed)
│   │   │   └── utils.py         # JWT helpers (legacy)
│   │   │
│   │   ├── interview/       # Core interview module
│   │   │   ├── router.py        # REST + WebSocket endpoints
│   │   │   ├── orchestrator.py  # Multi-threaded sensor orchestrator
│   │   │   ├── models.py        # Pydantic request/response schemas
│   │   │   ├── models_db.py     # Interview DB models
│   │   │   ├── conversation_manager.py  # LLM conversation handler
│   │   │   │
│   │   │   ├── agents/          # AI evaluation agents
│   │   │   │   ├── central_evaluator.py    # Response grading engine
│   │   │   │   ├── question_generator.py   # Adaptive question AI
│   │   │   │   ├── report_generator.py     # Recruiter report builder
│   │   │   │   ├── resume_intelligence.py  # Resume parser & skill extractor
│   │   │   │   ├── job_intelligence.py     # JD parser & requirement mapper
│   │   │   │   ├── matching_engine.py      # Semantic resume↔JD matcher
│   │   │   │   ├── context_analyzer.py     # Conversation context tracker
│   │   │   │   └── llm_client.py           # Multi-vendor LLM abstraction
│   │   │   │
│   │   │   └── sensors/         # Biometric sensor array
│   │   │       ├── posture_analyzer.py
│   │   │       ├── eye_contact_analyzer.py
│   │   │       ├── body_language_analyzer.py
│   │   │       ├── attire_analyzer.py
│   │   │       ├── facial_expression_analyzer.py
│   │   │       ├── voice_analyzer.py
│   │   │       ├── confidence_metric.py
│   │   │       ├── engagement_tracker.py
│   │   │       ├── professional_presence.py
│   │   │       └── weight_config.py
│   │   │
│   │   └── example/         # Example module template
│   │
│   ├── models/              # ML model weights directory
│   │   ├── *.pth            # PyTorch weights (gitignored)
│   │   ├── *.task           # MediaPipe task files (gitignored)
│   │   └── *_labels.txt     # Model label definitions
│   │
│   ├── static/              # Local app web interface
│   │   ├── dashboard.html          # Main dashboard (Stitch UI)
│   │   ├── interview-setup.html    # Single-page interview config
│   │   ├── interview.html          # Live interview chamber
│   │   ├── active-interview-chamber.html
│   │   ├── evaluation-report.html  # Post-interview report
│   │   ├── evaluation-engine.html  # Evaluation dashboard
│   │   ├── admin.html              # Admin panel
│   │   ├── css/style.css
│   │   └── js/
│   │       ├── app.js              # Main app logic + session handling
│   │       ├── admin.js
│   │       └── particles-config.js
│   │
│   └── vision_system/       # Standalone vision training tools
│       ├── cam_feed.py      # Webcam Flask server
│       ├── detectors.py     # PyTorch classifier inference
│       ├── aggregator.py    # Multi-detector fusion
│       ├── train_detector.py    # Model training pipeline
│       ├── collect_data.py      # Training data collector
│       ├── dashboard_helper.py  # CLI management dashboard
│       ├── download_datasets.py # Dataset downloader
│       ├── config.yaml          # Vision model configuration
│       ├── requirements.txt
│       └── run_vision.bat       # Vision system launcher
│
├── .gitignore
├── PROJECT.md               # Internal architecture reference
├── INTERFACES_SPEC.md       # WebSocket protocol specification
├── TEST_INFRA.md            # Test infrastructure documentation
└── TEST_READY.md            # Test readiness checklist
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (3.11 or 3.13 recommended)
- **pip** (or any Python package manager)
- A webcam and microphone (for live interview analysis)
- *(Optional)* An LLM API key for AI-powered evaluation (see [Configuration](#-configuration))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/aaroninplayz/keiko.git
cd keiko/local-app

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Create a .env file for LLM API keys
# See "Environment Variables" section below

# 6. Start the server
python main.py
```

The server will start on `http://localhost:8000` and **automatically open the dashboard** in your default browser.

### Windows Quick Launch

Double-click `setup_and_run.bat` — it will create a virtual environment, install dependencies, and launch the server automatically.

### Endpoints After Launch

| URL | Description |
|:----|:------------|
| `http://localhost:8000/static/dashboard.html` | Main dashboard |
| `http://localhost:8000/static/interview-setup.html` | Interview configuration |
| `http://localhost:8000/static/interview.html?session_id=<id>` | Live interview chamber |
| `http://localhost:8000/static/admin.html` | Admin panel |
| `http://localhost:8000/health` | Health check endpoint |
| `http://localhost:8000/docs` | Interactive API documentation (Swagger) |

---

## ⚙ Configuration

### LLM Provider Setup

KEIKO supports **4 LLM providers** with automatic fallback. Configure one or more by setting API keys:

| Provider | Environment Variable | Default Model | Priority |
|:---------|:--------------------|:--------------|:---------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` | 1st (highest) |
| Google Gemini | `GEMINI_API_KEY` | `gemini-1.5-pro` | 2nd |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20240620` | 3rd |
| Groq | `GROQ_API_KEY` | `llama3-8b-8192` | 4th |

> **Note**: If no API key is configured, KEIKO will still function with deterministic (non-LLM) evaluation scoring. LLM integration enhances question generation, response grading, and report narratives.

### Configuration File

All settings are managed via Pydantic in `local-app/core/config.py` and can be overridden with environment variables or a `.env` file:

```ini
# .env (place in local-app/ directory)

# LLM Providers (set at least one for AI features)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Custom model selections (optional)
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-1.5-pro
ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
GROQ_MODEL=llama3-8b-8192

# OpenAI-compatible endpoint (for local models like Ollama)
OPENAI_API_BASE=http://localhost:11434/v1

# Speech-to-Text
STT_PROVIDER=local                    # "local" uses Whisper, "api" uses OpenAI/Groq
WHISPER_MODEL_SIZE=openai/whisper-tiny

# Database
DATABASE_URL=sqlite:///./app.db

# Browser auto-open on startup
KEIKO_AUTO_OPEN=true

# JWT (only needed if re-enabling authentication)
JWT_SECRET_KEY=your-secret-key-here

# Disable specific modules
DISABLED_MODULES=["example"]
```

---

## 📡 Sensor Array

KEIKO deploys **9 independent biometric sensors**, each producing a real-time score from 0 to 100. Scores are fused into a weighted composite using configurable weights that normalize to 1.0.

### Visual Sensors

| # | Sensor | Module | Technology | What It Measures | Default Weight |
|:-:|:-------|:-------|:-----------|:-----------------|:---------------|
| 1 | **Posture** | `posture_analyzer.py` | MediaPipe PoseLandmarker | Shoulder alignment, spine inclination, head position, slouch detection | `0.15` |
| 2 | **Eye Contact** | `eye_contact_analyzer.py` | MediaPipe FaceLandmarker (Iris) | Pupil position, gaze deviation, camera contact percentage, gaze stability | `0.15` |
| 3 | **Body Language** | `body_language_analyzer.py` | MediaPipe HolisticLandmarker | Hand gesture frequency, head nodding, shoulder openness, physical engagement | `0.15` |
| 4 | **Attire** | `attire_analyzer.py` | MediaPipe Pose + OpenCV HSV | Upper-body color variance, brightness uniformity, formality classification (formal vs. casual) | `0.10` |
| 5 | **Facial Expression** | `facial_expression_analyzer.py` | MediaPipe FaceLandmarker Blendshapes | Emotion classification (neutral, happy, confused, nervous), expressiveness | `0.10` |

### Audio Sensors

| # | Sensor | Module | Technology | What It Measures | Default Weight |
|:-:|:-------|:-------|:-----------|:-----------------|:---------------|
| 6 | **Voice** | `voice_analyzer.py` | HuggingFace Whisper + Audio Analysis | Speaking pace (WPM), voice modulation, speech clarity, fluency, hesitation, filler words | `0.15` |

### Composite Meta-Sensors

| # | Sensor | Module | Combines | Default Weight |
|:-:|:-------|:-------|:---------|:---------------|
| 7 | **Confidence** | `confidence_metric.py` | Posture openness + Eye contact stability + Gesture purposefulness | `0.10` |
| 8 | **Engagement** | `engagement_tracker.py` | Gaze consistency + Head movement + Facial expressiveness + Response latency | `0.05` |
| 9 | **Professional Presence** | `professional_presence.py` | Posture alignment + Attire score + Eye contact poise + Facial composure | `0.05` |

### Weight Configuration

Weights are dynamically adjustable during a live session via the REST API or the interview setup page. All weights auto-normalize to sum to `1.0`:

```python
# Default weights (from weight_config.py)
DEFAULT_WEIGHTS = {
    "posture": 0.15,
    "eye_contact": 0.15,
    "body_language": 0.15,
    "attire": 0.10,
    "confidence": 0.10,
    "facial_expression": 0.10,
    "voice": 0.15,
    "engagement": 0.05,
    "professional_presence": 0.05,
}
# Sum = 1.0
```

---

## 🤖 AI Agent Pipeline

KEIKO's intelligence layer consists of specialized AI agents that work together:

### Resume Intelligence (`resume_intelligence.py`)
Parses uploaded resumes (PDF/DOCX) and extracts structured candidate profiles:
- **Technical skills** — programming languages, frameworks, databases, cloud platforms
- **Work experience** — roles, durations, responsibilities
- **Education** — degrees, institutions, certifications
- **Soft skills** — leadership, teamwork, communication indicators

### Job Intelligence (`job_intelligence.py`)
Parses job descriptions and extracts structured requirements:
- Required/preferred technical skills
- Experience level requirements
- Role responsibilities and expectations
- Technology stack mapping

### Matching Engine (`matching_engine.py`)
Performs **semantic similarity matching** between candidate profiles and job requirements using `sentence-transformers` (all-MiniLM-L6-v2):
- Skill-by-skill gap analysis
- Experience alignment scoring
- Overall match percentage with dimensional breakdowns

### Central Evaluator (`central_evaluator.py`)
The core evaluation engine that grades candidate responses:
- **LLM-powered grading** — sends interview Q&A + context to the active LLM provider
- **Deterministic fallback** — rule-based scoring when no LLM is configured
- **Multi-dimensional scoring** — technical accuracy, communication quality, behavioral indicators
- **Session state persistence** — saves evaluation state to `data/sessions/{session_id}/interview_state.json`

### Question Generator (`question_generator.py`)
Generates adaptive interview questions based on:
- Job description requirements
- Resume analysis results
- Previous answer performance
- Identified knowledge gaps
- Configured difficulty progression

### Report Generator (`report_generator.py`)
Produces comprehensive recruiter evaluation reports:
- **Overall verdict** with confidence level
- **Dimensional breakdowns** — Technical, Communication, Behavioral, Presence
- **Coaching recommendations**:
  - Technical learning paths for skill gaps
  - STAR method tips (when communication score < 75)
  - Posture/eye-contact advice (when presentation score < 75)
  - Custom practice questions for weak areas

---

## 📚 API Reference

### REST Endpoints

All endpoints are prefixed with `/api/v1/interview`.

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/start` | Start a new interview session |
| `POST` | `/answer` | Submit a candidate answer for evaluation |
| `POST` | `/upload/resume/{session_id}` | Upload candidate resume (PDF/DOCX) |
| `POST` | `/upload/jd/{session_id}` | Upload job description text |
| `GET` | `/state` | Get current session status and Q&A history |
| `GET` | `/config/weights` | Get global default sensor weights |
| `GET` | `/config/weights/{session_id}` | Get session-specific weights |
| `PUT` | `/config/weights/{session_id}` | Update session weights (auto-normalizes) |
| `GET` | `/profile/{session_id}` | Get candidate profile for a session |
| `GET` | `/health` | Health check (`{"status": "ok", "version": "0.1.0"}`) |

### Start a Session

```bash
curl -X POST http://localhost:8000/api/v1/interview/start \
  -H "Content-Type: application/json" \
  -d '{"mode": "real_time"}'
```

**Response:**
```json
{
  "session_id": "sess_a1b2c3d4e5f6",
  "status": "active",
  "message": "Interview session started"
}
```

### Update Sensor Weights

```bash
curl -X PUT http://localhost:8000/api/v1/interview/config/weights/sess_a1b2c3d4e5f6 \
  -H "Content-Type: application/json" \
  -d '{
    "posture": 0.20,
    "eye_contact": 0.20,
    "voice": 0.20,
    "body_language": 0.10,
    "attire": 0.05,
    "confidence": 0.10,
    "facial_expression": 0.05,
    "engagement": 0.05,
    "professional_presence": 0.05
  }'
```

> Weights are automatically normalized to sum to `1.0`.

---

## 🔌 WebSocket Protocol

### Connection

```
ws://localhost:8000/api/v1/interview/ws/{session_id}
```

On connection, the server sends a handshake confirming the session and current weight configuration.

### Client → Server Messages

#### Video Frame
```json
{
  "type": "video_frame",
  "data": "<base64-encoded JPEG or PNG>",
  "timestamp": 1700000000.123
}
```
> Recommended: 5–10 FPS. The orchestrator processes every 3rd frame to balance quality vs. performance.

#### Audio Chunk
```json
{
  "type": "audio_chunk",
  "data": "<base64-encoded 16kHz mono 16-bit PCM>",
  "timestamp": 1700000000.456
}
```
> Audio buffer limit: 15 MB per session. Call the transcribe endpoint to process.

#### Update Weights (Live)
```json
{
  "type": "update_weights",
  "weights": {
    "posture": 0.20,
    "eye_contact": 0.20,
    "voice": 0.20
  }
}
```

### Server → Client Messages

#### Metrics Update
```json
{
  "type": "metrics_update",
  "data": {
    "overall_score": 78.5,
    "weights": { ... },
    "sensors": {
      "posture": { "score": 85.0, "label": "good", "details": { ... } },
      "eye_contact": { "score": 72.0, "gaze_deviation": 0.15 },
      "body_language": { "score": 68.0, "gesture_frequency": 3.2 },
      "attire": { "score": 90.0, "classification": "formal" },
      "facial_expression": { "score": 75.0, "emotion": "neutral" },
      "voice": { "score": 80.0, "wpm": 145, "clarity": 82.0 },
      "confidence": { "score": 76.0 },
      "engagement": { "score": 71.0 },
      "professional_presence": { "score": 82.0 }
    },
    "composites": {
      "composure": 79.0,
      "stress_resilience": 74.0,
      "engagement": 71.0,
      "professional_presence": 82.0
    }
  }
}
```

---

## 🌐 Marketing Site

The `marketing-site/` directory contains the public-facing website. It is designed for **serverless deployment** on platforms like Vercel.

### Deploy to Vercel

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → **Add New** → **Project**
3. Import the `KEIKO` repository
4. Set **Root Directory** to `marketing-site`
5. Framework Preset: **Other** (plain HTML)
6. Click **Deploy**

### Smart Download Page (`download.html`)

The download page includes a built-in health check that polls `http://localhost:8000/health` every 4 seconds:

- **🟢 Local Node Online** — Shows a "Launch Web Dashboard" button linking directly to the local app
- **🔴 Local Node Offline** — Shows download buttons for macOS, Windows, and Linux installers plus a setup guide

### Pages

| Page | Description |
|:-----|:------------|
| `index.html` | Landing page with hero, features overview, and CTAs |
| `about.html` | Intelligence board, team information |
| `features.html` | Detailed product feature showcase |
| `architecture.html` | System architecture diagrams and technical deep-dive |
| `documentation.html` | Unified docs hub with tabs: Changelogs, FAQ, Open Source |
| `download.html` | Smart download hub with local node detection |
| `interview-journey.html` | Visual walkthrough of the interview flow |
| `audio-intelligence.html` | Audio analysis feature details |
| `video-intelligence.html` | Video analysis feature details |
| `job-intelligence.html` | Job description intelligence features |
| `resume-intelligence.html` | Resume parsing and analysis features |
| `interview-intelligence.html` | Interview evaluation intelligence features |

---

## 👁 Vision System

The `local-app/vision_system/` directory contains standalone tools for training and managing the computer vision classifiers.

### Vision Models

| Detector | Architecture | Labels | Weight |
|:---------|:-------------|:-------|:-------|
| Posture | ResNet-18 | `good`, `slouched` | `0.20` |
| Eye Contact | ResNet-18 | `focused`, `distracted` | `0.25` |
| Attire | MobileNet V3 | `formal`, `casual` | `0.15` |
| Confidence | ResNet-18 | `confident`, `nervous` | `0.20` |
| Emotions | ResNet-18 | `confident`, `stressed`, `neutral` | `0.20` |

### Training Pipeline

```bash
cd local-app

# Launch the interactive vision management dashboard
python vision_system/dashboard_helper.py

# Or use the batch script (Windows)
model_dashboard.bat
```

The vision dashboard provides options to:
1. **Start Live Dashboard Server** — Real-time webcam analysis at `http://127.0.0.1:5000`
2. **Collect Training Data** — Webcam image grabber with label assignment
3. **Train Classifiers** — PyTorch training pipeline with CUDA support
4. **Install Dependencies** — Auto-install vision requirements
5. **Reset Model Weights** — Selective or bulk weight deletion

### Configuration (`config.yaml`)

```yaml
aggregation_mode: "weighted_rubric"   # or "meta_classifier"

detectors:
  posture:
    weight: 0.20
    threshold: 0.50
    model_path: "models/posture.pth"
    architecture: "resnet18"
    labels: ["good", "slouched"]
  
  eye_contact:
    weight: 0.25
    threshold: 0.50
    model_path: "models/eye_contact.pth"
    architecture: "resnet18"
    labels: ["focused", "distracted"]
  # ... (attire, confidence, emotions)
```

---

## 🛠 Tech Stack

### Backend
| Technology | Purpose |
|:-----------|:--------|
| **FastAPI** | Async web framework with WebSocket support |
| **Uvicorn** | ASGI server with hot reload |
| **SQLAlchemy** | ORM and database management |
| **SQLite** | Default embedded database |
| **Pydantic** | Data validation and settings management |

### Computer Vision & ML
| Technology | Purpose |
|:-----------|:--------|
| **MediaPipe** | Pose, face, hand, and holistic landmark detection |
| **OpenCV** | Image processing and color analysis |
| **PyTorch** | Deep learning model training and inference |
| **TorchVision** | Pre-trained ResNet/MobileNet architectures |
| **HuggingFace Transformers** | Whisper speech-to-text models |
| **Sentence-Transformers** | Semantic similarity for resume/JD matching |

### Frontend
| Technology | Purpose |
|:-----------|:--------|
| **HTML5 + CSS3 + JavaScript** | UI with "Stitch" glassmorphism aesthetic |
| **WebSocket API** | Real-time bi-directional communication |
| **Canvas API** | Webcam video rendering |
| **Web Audio API** | Microphone audio capture |

---

## 🔐 Environment Variables

| Variable | Required | Default | Description |
|:---------|:---------|:--------|:------------|
| `OPENAI_API_KEY` | No | `None` | OpenAI API key for GPT models |
| `OPENAI_API_BASE` | No | `https://api.openai.com/v1` | OpenAI-compatible endpoint (for Ollama, etc.) |
| `OPENAI_MODEL` | No | `gpt-4o` | OpenAI model name |
| `GEMINI_API_KEY` | No | `None` | Google Gemini API key |
| `GEMINI_MODEL` | No | `gemini-1.5-pro` | Gemini model name |
| `ANTHROPIC_API_KEY` | No | `None` | Anthropic API key |
| `ANTHROPIC_MODEL` | No | `claude-3-5-sonnet-20240620` | Anthropic model name |
| `GROQ_API_KEY` | No | `None` | Groq API key |
| `GROQ_MODEL` | No | `llama3-8b-8192` | Groq model name |
| `DATABASE_URL` | No | `sqlite:///./app.db` | Database connection string |
| `STT_PROVIDER` | No | `local` | Speech-to-text provider (`local` or `api`) |
| `WHISPER_MODEL_SIZE` | No | `openai/whisper-tiny` | HuggingFace Whisper model ID |
| `KEIKO_AUTO_OPEN` | No | `true` | Auto-open browser on server start |
| `JWT_SECRET_KEY` | No | *(placeholder)* | JWT signing key (legacy, auth bypassed) |
| `DISABLED_MODULES` | No | `[]` | JSON array of module names to skip |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** this repository
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m "Add amazing feature"`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/KEIKO.git
cd KEIKO/local-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Start development server (with hot reload)
python main.py
```

---

## 📄 License

This project is open source. See the repository for license details.

---

<p align="center">
  Built with ❤️ by the KEIKO team
</p>
