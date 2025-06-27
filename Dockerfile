# Use a single stage based on python:3.11-slim
FROM python:3.11-slim

# Set environment variables for Python and the application
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    TRANSFORMERS_CACHE="/data/models" \
    HF_HOME="/data/models" \
    EMBEDDING_MODEL="intfloat/multilingual-e5-large" \
    DOCS_DIR="/data/docs"

# Install system dependencies for building and running the application
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a non-root user with a known UID/GID
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Create data directories and set permissions
RUN mkdir -p /data/docs /data/models /data/logs && \
    touch /data/logs/app.log && \
    touch /data/docs/.placeholder && \
    chown -R appuser:appuser /data && \
    chmod -R 775 /data

# Create virtual environment directory
RUN mkdir -p /app/.venv && \
    chown -R appuser:appuser /app && \
    chmod -R 775 /app

# Switch to non-root user
USER appuser

# Create and activate a virtual environment
RUN python -m venv /app/.venv

# Copy application code and requirements
COPY --chown=appuser:appuser requirements.txt .
COPY --chown=appuser:appuser . .

# Install dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Fix permissions for data directories as root (in case volume has different ownership)
USER root
RUN chmod -R 775 /data/docs /data/models /data/logs && \
    chown -R appuser:appuser /data
USER appuser

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]