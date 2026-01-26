FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Export dependencies to requirements.txt and install system-wide
# This leverages Docker layer caching - deps only reinstall when lock file changes
RUN uv export --frozen --no-hashes -o requirements.txt && \
    uv pip install --system -r requirements.txt

# Copy application code
COPY . .

# Install the project itself (without deps, they're already installed)
RUN uv pip install --system --no-deps .

# Create necessary directories
RUN mkdir -p /app/data /app/config /app/uploads

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"]
