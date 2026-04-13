FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies (IMPORTANT)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*
# Upgrade pip tools (important for Python 3.12)
RUN pip install --upgrade pip setuptools wheel

RUN apt-get update && apt-get install -y curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws/

# Install Python AWS SDK
RUN pip install boto3


RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "serving.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
