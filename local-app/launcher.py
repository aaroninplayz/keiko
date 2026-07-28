"""
KEIKO Native Desktop Executable Launcher
Launches the KEIKO local server backend and opens default web browser to the Keiko Dashboard.
"""

import sys
import os
import time
import subprocess
import webbrowser

def main():
    print("=" * 65)
    print("                   KEIKO TALENT INTELLIGENCE PLATFORM          ")
    print("============================================================")

    # Determine base directory
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Locate project root and local-app directory
    if os.path.exists(os.path.join(base_dir, "local-app", "run.py")):
        local_app_dir = os.path.join(base_dir, "local-app")
    elif os.path.exists(os.path.join(base_dir, "run.py")):
        local_app_dir = base_dir
    else:
        local_app_dir = os.path.abspath(os.path.join(base_dir, "..", "local-app"))

    run_py = os.path.join(local_app_dir, "run.py")

    # Find Python executable
    python_exe = sys.executable
    possible_pythons = [
        r"P:\Dependencies\keiko_venv\Scripts\python.exe",
        os.path.join(local_app_dir, "venv", "Scripts", "python.exe"),
        os.path.join(base_dir, "venv", "Scripts", "python.exe")
    ]
    for py in possible_pythons:
        if os.path.exists(py):
            python_exe = py
            break

    print(f"[+] Python Executable: {python_exe}")
    print(f"[+] Launcher Script:   {run_py}")

    # Launch browser after a short delay
    def open_browser():
        time.sleep(2.5)
        url = "http://localhost:8000/static/dashboard.html"
        print(f"[+] Opening browser: {url}")
        webbrowser.open(url)

    import threading
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    # Launch server
    cmd = [python_exe, run_py]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[!] KEIKO Server stopped by user.")
    except Exception as e:
        print(f"\n[!] Server process exited: {e}")

if __name__ == "__main__":
    main()
