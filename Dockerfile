FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install dependencies first (leverage Docker cache)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code and scripts
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY streamlit_app.py .
COPY assets/ assets/

# Install the local package in editable mode
RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Default command to run the dashboard
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
