FROM python:3.10-slim

WORKDIR /app

# 필수 시스템 패키지 설치 (예: gcc, libmagic 등)
RUN apt-get update && \
    apt-get install -y gcc git build-essential libmagic1 poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 소스 복사
COPY . .