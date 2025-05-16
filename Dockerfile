FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    python3-dev \
    gcc \
    git \
    && pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/

EXPOSE 5000 8501

RUN echo '#!/bin/bash\n\
fastapi run main.py --port 5000 & \
streamlit run ui.py\n\
wait\n' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]