# Zepto Support Assistant Container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MOCK_LLM=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY support_assistant/ /app/support_assistant/

# Expose FastAPI port
EXPOSE 7860

# Working directory inside support_assistant
WORKDIR /app/support_assistant

# Run Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
