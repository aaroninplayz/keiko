import os
import sys
import time
import threading
import urllib.request
import webview

# 1. Set working directory to local-app/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 2. Set up environment variables
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

existing_pythonpath = os.environ.get("PYTHONPATH", "")
if SCRIPT_DIR not in existing_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{SCRIPT_DIR}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else SCRIPT_DIR

# HuggingFace offline variables (only if cached models exist)
models_dir = os.path.join(SCRIPT_DIR, "models")
has_cached_models = os.path.exists(models_dir) and any(
    entry.startswith("models--") and os.path.isdir(os.path.join(models_dir, entry))
    for entry in os.listdir(models_dir)
)
if has_cached_models:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

# Prevent legacy browser-open logic from triggering
os.environ["KEIKO_AUTO_OPEN"] = "false"

# 3. Start FastAPI server in a daemon background thread
def start_backend():
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")

server_thread = threading.Thread(target=start_backend, daemon=True)
server_thread.start()

# 4. Poll health endpoint every 0.5s (timeout 30s)
health_url = "http://127.0.0.1:8000/health"
start_time = time.time()
timeout = 30.0
server_ready = False

print("Launching KEIKO backend server...")
while time.time() - start_time < timeout:
    try:
        with urllib.request.urlopen(health_url, timeout=1) as response:
            if response.status == 200:
                server_ready = True
                break
    except Exception:
        pass
    time.sleep(0.5)

if not server_ready:
    print("Error: KEIKO backend server failed to start within 30 seconds.")
    sys.exit(1)

print("KEIKO backend server ready. Opening native desktop application...")

# 5. Launch PyWebView window
icon_path = os.path.join(SCRIPT_DIR, "static", "assets", "keiko-app-icon.png")
window = webview.create_window(
    'KEIKO — Interview Intelligence',
    'http://127.0.0.1:8000/static/dashboard.html',
    fullscreen=True,
    text_select=True,
    zoomable=True
)

if os.path.exists(icon_path):
    webview.start(icon=icon_path)
else:
    webview.start()
