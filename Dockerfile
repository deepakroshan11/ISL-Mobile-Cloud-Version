# ============================================================
#  Dockerfile — Hugging Face Spaces (Docker SDK)
#  Free tier: 16GB RAM, 2 vCPU
#  HF exposes only port 7860 — nginx routes both Flask apps
#
#  /          → Flask 5000 (gesture + emotion)
#  /avatar/   → Flask 5001 (speech-to-sign)
# ============================================================

FROM python:3.10-slim

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
USER root

# ── System deps ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    nginx \
    supervisor \
    curl \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── App files ─────────────────────────────────────────────────
COPY isl_detection.py .
COPY isl_ui_dashboard.py .
COPY launcher.py .
COPY speech_to_sign_avatar.py .
COPY translation.py .
COPY unified_launcher.py .
COPY model.h5 .
COPY templates/ templates/

# ── Configs ───────────────────────────────────────────────────
COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ── Nginx log dir permissions ─────────────────────────────────
RUN mkdir -p /var/log/nginx /var/log/supervisor /var/run \
    && chmod -R 777 /var/log/nginx /var/log/supervisor /var/run \
    && chown -R user:user /app

# ── Environment ───────────────────────────────────────────────
ENV RUN_MODE=CLOUD
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1

# HF Spaces MUST use port 7860
EXPOSE 7860

# ── Start supervisor as root (nginx needs it) ─────────────────
USER root
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
