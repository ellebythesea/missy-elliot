FROM python:3.11-slim

WORKDIR /app

# Keep images lean and deterministic
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Reasonable defaults; override at run time as needed
ENV OPENAI_MODEL=gpt-4o \
    OPENAI_TEMPERATURE=0.8

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

