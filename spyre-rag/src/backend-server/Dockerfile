FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV RETRIEVER_SERVICE_URL http://localhost:8084
ENV INFERENCE_SERVICE_URL http://localhost:11434/v1
ENV MODEL_NAME granite3.2:latest

# Create working directory
WORKDIR /app

# Copy application code
COPY main.py ./main.py

# Install Python dependencies
RUN pip install --upgrade pip && pip install \
    flask \
    requests

# Expose server port
EXPOSE 8001

# Start the Flask app
CMD ["python", "main.py"]
