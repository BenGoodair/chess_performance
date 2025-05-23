# Copy application code
COPY chess_data_async.py .
COPY analyze_results.py .

# Create necessary directories and set permissions
RUN mkdir -p /data /app/Figures && \
    chmod 755 /data /app/Figures

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CONCURRENT_REQUESTS=100
ENV OUTPUT_CSV=/data/daily_chess_stats_GB.csvFROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY chess_data_async.py .

# Create necessary directories and set permissions
RUN mkdir -p /data /app/Figures && \
    chmod 755 /data /app/Figures

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CONCURRENT_REQUESTS=100
ENV OUTPUT_CSV=/data/chess_data_6M_GB.csv

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('https://api.chess.com/pub/country/GB', timeout=5)" || exit 1

# Default command
CMD ["python", "chess_data_async.py"]