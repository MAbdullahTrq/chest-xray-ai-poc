# Multi-stage Dockerfile for Chest X-ray AI POC

# Stage 1: Base image with CUDA support (using devel for better compatibility)
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libxi6 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libatk1.0-0 \
    libgtk-3-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific versions for optimal CUDA compatibility
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM base as app

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create necessary directories
RUN mkdir -p ./models ./logs ./uploads

# Pre-download AI models to reduce startup time
RUN python3 -c "import torchxrayvision as xrv; print('Downloading models...'); model = xrv.models.get_model('densenet121-res224-all'); print('Models ready!')"

# Expose ports
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting Chest X-ray AI POC..."\n\
\n\
# Start API server in background\n\
echo "Starting API server on port 8000..."\n\
cd /app\n\
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &\n\
API_PID=$!\n\
\n\
# Start frontend server in background\n\
echo "Starting frontend server on port 3000..."\n\
cd /app/frontend\n\
python3 -m http.server 3000 &\n\
FRONTEND_PID=$!\n\
\n\
echo "Both servers started successfully!"\n\
echo "API: http://localhost:8000"\n\
echo "Frontend: http://localhost:3000"\n\
echo "API Docs: http://localhost:8000/docs"\n\
\n\
# Wait for any process to exit\n\
wait -n\n\
\n\
# Exit with status of process that exited first\n\
exit $?\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]

# Alternative: Development mode
# CMD ["python3", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
