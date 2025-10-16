FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U --extra-index-url https://repo.fury.io/mgiessing --prefer-binary \
    flask \
    milvus-lite \
    langchain \
    docling \
    pymilvus==2.5.11 \
    langchain_milvus \
    cohere \
    ollama \
    joblib \
    scikit-learn

# 🔽 Only copy Python files
COPY *.py ./
COPY api_key.txt ./

EXPOSE 8081

CMD ["python", "flask_retreival.py"]
