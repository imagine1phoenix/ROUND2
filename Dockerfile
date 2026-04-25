# ── Verdict: Courtroom AI ──────────────────────────────────────────
# HuggingFace Spaces Dockerfile (Gradio demo on port 7860)
# ──────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user:user . .

# Switch to non-root user
USER user

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Launch the wired demo (uses real VerdictEnvironment + cases.json)
CMD ["python", "demo/app.py"]
