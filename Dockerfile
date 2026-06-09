FROM python:3.11-slim

# System deps for PDF processing and image libraries
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories (vector DB persists here)
RUN mkdir -p data/chroma_db data/uploads

# Expose single port — HF Spaces uses 7860
EXPOSE 7860

# Start both FastAPI (internal 8000) + Streamlit (7860) via supervisor script
CMD ["python", "hf_spaces_start.py"]
