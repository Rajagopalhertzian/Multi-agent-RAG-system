"""
hf_app.py — Entry point for Hugging Face Spaces deployment.
HF Spaces runs this file directly. It starts both the FastAPI backend
and the Streamlit frontend in a single process using subprocess.
"""
import os
import subprocess
import threading
import time
import sys

def run_api():
    """Start FastAPI backend on port 7860 (HF Spaces default)."""
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "7861",
    ])

def run_frontend():
    """Start Streamlit frontend on port 7860."""
    # Point frontend to local API
    os.environ["API_URL"] = "http://localhost:7861"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port", "7860",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ])

if __name__ == "__main__":
    # Start API in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    # Wait for API to be ready
    time.sleep(5)
    print("API started, launching frontend...")

    # Run frontend in main thread
    run_frontend()
