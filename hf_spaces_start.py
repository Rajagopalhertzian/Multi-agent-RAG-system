"""
hf_spaces_start.py
Starts both FastAPI backend (port 8000, internal) and
Streamlit frontend (port 7860, public) as subprocesses.
HF Spaces exposes only port 7860 externally.
"""
import subprocess
import threading
import time
import sys
import os

os.environ.setdefault("CHROMA_PERSIST_DIR", "/app/data/chroma_db")
os.environ.setdefault("FAISS_INDEX_PATH", "/app/data/faiss_index")


def run_api():
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info",
    ])


def run_frontend():
    env = os.environ.copy()
    env["API_URL"] = "http://localhost:8000"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port", "7860",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ], env=env)


if __name__ == "__main__":
    print("Starting FastAPI backend on port 8000...")
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    print("Waiting for API to be ready...")
    time.sleep(8)

    print("Starting Streamlit frontend on port 7860...")
    run_frontend()
