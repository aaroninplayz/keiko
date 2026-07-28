"""
KEIKO Local Lab - Unified Cross-Platform Launcher & Diagnostic Tool
Handles pre-flight environment checks, test suite execution, and Uvicorn server startup.
"""

import sys
import os
import argparse
import socket
import logging

# Ensure local-app directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# Also add project root if running from inside local-app
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("keiko.launcher")

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a local TCP port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def run_diagnostics():
    """Performs a fast pre-flight check of the environment, models, and port availability."""
    print("=" * 65)
    print("           KEIKO LOCAL LAB - PRE-FLIGHT DIAGNOSTICS          ")
    print("=" * 65)
    
    # 1. Python Environment
    print(f"\n[1] Python Runtime: {sys.executable}")
    print(f"    Python Version:  {sys.version.split()[0]}")
    
    # 2. Key Package Checks
    packages = ["fastapi", "uvicorn", "sqlalchemy", "cv2", "mediapipe", "torch", "transformers", "sentence_transformers"]
    print("\n[2] Dependency Status:")
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"    [OK]   {pkg}")
        except ImportError:
            print(f"    [MISS] {pkg} (Not installed)")
            
    # 3. Model Files Status
    models_dir = os.path.join(SCRIPT_DIR, "models")
    expected_files = [
        "posture.pth", "eye_contact.pth", "confidence.pth", 
        "emotions.pth", "attire.pth", "pose_landmarker_lite.task",
        "face_landmarker.task", "holistic_landmarker.task"
    ]
    print(f"\n[3] Local Model Files ({models_dir}):")
    if os.path.isdir(models_dir):
        for mf in expected_files:
            fpath = os.path.join(models_dir, mf)
            if os.path.exists(fpath):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"    [OK]   {mf:<28} ({size_mb:.1f} MB)")
            else:
                print(f"    [MISS] {mf:<28} (File missing)")
    else:
        print("    [WARN] Models directory missing!")

    # 4. Port Check
    port = 8000
    print(f"\n[4] Network Port Availability:")
    if is_port_in_use(port):
        print(f"    [WARN] Port {port} is currently IN USE.")
        print(f"           Note: Keiko backend might already be running, or another process is on port {port}.")
    else:
        print(f"    [OK]   Port {port} is free and ready.")
        
    print("\n" + "=" * 65)
    print("                     DIAGNOSTIC COMPLETE                     ")
    print("=" * 65 + "\n")

def run_tests():
    """Runs the Keiko E2E test suites."""
    import unittest
    print("Running Keiko Test Suites...")
    
    test_files = [
        os.path.join(PROJECT_ROOT, "scratch", "test_realtime_analyzer.py"),
        os.path.join(PROJECT_ROOT, "scratch", "test_conversation_engine.py"),
        os.path.join(PROJECT_ROOT, "scratch", "test_auth_privacy.py"),
        os.path.join(PROJECT_ROOT, "scratch", "test_pipeline_nlp.py")
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for tf in test_files:
        if os.path.exists(tf):
            dirname, filename = os.path.split(tf)
            if dirname not in sys.path:
                sys.path.insert(0, dirname)
            module_name = os.path.splitext(filename)[0]
            tests = loader.loadTestsFromName(module_name)
            suite.addTests(tests)
        else:
            print(f"Warning: Test file not found at {tf}")
            
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

def start_server(host: str, port: int, reload: bool):
    """Starts the FastAPI application via Uvicorn."""
    import uvicorn
    
    if is_port_in_use(port, host):
        logger.warning(f"Port {port} is already occupied! Checking server...")
        
    print("\n" + "=" * 65)
    print("                  KEIKO LOCAL LAB BACKEND                    ")
    print("=" * 65)
    print(f"  • Dashboard UI:        http://{host}:{port}/static/dashboard.html")
    print(f"  • Interview Setup:     http://{host}:{port}/static/interview-setup.html")
    print(f"  • System Settings:     http://{host}:{port}/static/settings.html")
    print(f"  • System Health API:   http://{host}:{port}/health")
    print("=" * 65 + "\n")
    
    uvicorn.run("main:app", host=host, port=port, reload=reload)

def main():
    parser = argparse.ArgumentParser(description="KEIKO Local Lab Launcher")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload (dev mode)")
    parser.add_argument("--check", action="store_true", help="Run pre-flight diagnostics and exit")
    parser.add_argument("--test", action="store_true", help="Run unit test suite and exit")

    args = parser.parse_args()

    if args.check:
        run_diagnostics()
        return

    if args.test:
        run_tests()
        return

    start_server(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()
