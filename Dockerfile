FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    python3-dev \
    gcc \
    git \
    && pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY app /app/

# Expose ports
EXPOSE 5000 8501

# Create a script to run both services
RUN echo '#!/bin/bash\n\
fastapi run main.py --port 5000 & \
streamlit run ui.py\n\
wait\n' > /app/start.sh && chmod +x /app/start.sh

# Run the script
CMD ["/app/start.sh"]