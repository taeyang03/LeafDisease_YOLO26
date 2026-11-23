FROM python:3.11-slim

WORKDIR /app

# OpenCV / 이미지 처리에 필요한 최소 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 설치 (필요 패키지에 맞게 조정)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 가중치 복사
COPY ./runs/detect/leaf_disease/yolo26m_v2/weights/best.pt ./runs/detect/leaf_disease/yolo26m_v2/weights/best.pt

# 애플리케이션 코드 복사
COPY predict.all.py .
COPY predict.best.py .
COPY predict.mapping.py .

# 기본 실행 명령
CMD ["python", "predict.py"]
