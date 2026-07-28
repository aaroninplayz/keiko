import logging
import os
import threading
import webbrowser
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.registry import registry
from core.database import Base, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import time

SERVER_BOOT_ID = str(int(time.time()))

def _open_dashboard_browser():
    target_url = "http://localhost:8000/static/dashboard.html?reset=true"
    logger.info(f"Opening browser (Firefox preferred) to {target_url} in fullscreen mode...")
    try:
        import subprocess, sys
        if sys.platform == "win32":
            # Try Firefox kiosk/fullscreen mode first, then default fallback
            try:
                subprocess.Popen(f'start firefox -kiosk "{target_url}"', shell=True)
            except Exception:
                try:
                    subprocess.Popen(f'start firefox -url "{target_url}"', shell=True)
                except Exception:
                    webbrowser.open(target_url)
        else:
            try:
                subprocess.Popen(["firefox", "-kiosk", target_url])
            except Exception:
                webbrowser.open(target_url)
    except Exception as e:
        logger.warning(f"Failed to open browser automatically: {e}")

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )
    app.state.boot_id = SERVER_BOOT_ID

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize Database (Create tables if they don't exist)
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)

    # Load Modules/Plugins
    registry.discover_and_load_modules(
        app=app,
        package_name="modules",
        disabled_modules=settings.DISABLED_MODULES
    )

    # Mount static files for frontend
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
        logger.info(f"Mounted static files from: {static_dir}")
    else:
        logger.warning(f"Static directory not found at: {static_dir}")

    @app.get("/", tags=["System"])
    def root_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/static/dashboard.html")

    @app.get("/health", tags=["System"])
    def health_check():
        return {"status": "ok", "version": settings.VERSION, "boot_id": SERVER_BOOT_ID}

    @app.get("/api/settings", tags=["System"])
    async def get_settings_alias():
        from modules.interview.router import get_system_settings
        return await get_system_settings()

    @app.post("/api/settings", tags=["System"])
    async def update_settings_alias(payload: dict = Body(...)):
        from modules.interview.router import update_system_settings
        return await update_system_settings(payload)

    @app.on_event("startup")
    def startup_event():
        auto_open_env = os.environ.get("KEIKO_AUTO_OPEN", "true").lower()
        should_auto_open = auto_open_env not in ("false", "0", "no", "off")

        if should_auto_open:
            threading.Timer(1.0, _open_dashboard_browser).start()
            logger.info("Auto browser open scheduled for dashboard.html on server start/reset")

    return app

