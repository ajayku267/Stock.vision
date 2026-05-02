# Multi-stage Dockerfile for StockVision
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome path for Selenium
ENV CHROME_BIN=/usr/bin/chromium \
    CHROME_DRIVER=/usr/bin/chromedriver

# Create app user
RUN groupadd -r stockvision && useradd -r -g stockvision stockvision

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

ENV ENVIRONMENT=development

# Copy source code
COPY . .

# Set permissions
RUN chown -R stockvision:stockvision /app
USER stockvision

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command for development
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

ENV ENVIRONMENT=production

# Copy source code
COPY . .

# Install production-specific dependencies
RUN pip install gunicorn

# Create non-root user for production
RUN chown -R stockvision:stockvision /app
USER stockvision

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["gunicorn", "backend:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]

# Frontend stage
FROM base as frontend

ENV ENVIRONMENT=frontend

# Copy source code
COPY . .

# Set permissions
RUN chown -R stockvision:stockvision /app
USER stockvision

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Frontend command
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# Celery Worker stage
FROM base as celery-worker

ENV ENVIRONMENT=production

# Copy source code
COPY . .

# Set permissions
RUN chown -R stockvision:stockvision /app
USER stockvision

# Celery worker command
CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info", "--concurrency=4"]

# Celery Beat stage
FROM base as celery-beat

ENV ENVIRONMENT=production

# Copy source code
COPY . .

# Set permissions
RUN chown -R stockvision:stockvision /app
USER stockvision

# Celery beat command
CMD ["celery", "-A", "celery_app", "beat", "--loglevel=info"]
