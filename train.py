from ultralytics import YOLO
import os
from dotenv import load_dotenv
from roboflow import Roboflow

# .env 파일에서 환경변수 로드
load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("practice-fgqq4").project("leafdisease_yolo26")
version = project.version(2)
dataset = version.download("yolo26")

# YOLO26 Medium 모델 불러오기 (최초 실행 시 가중치 자동 다운로드)
model = YOLO("yolo26m.pt")

# 학습 실행
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,        # 전체 데이터 100회 반복 학습
    imgsz=512,         # 데이터셋(v2) 해상도(512x512)와 일치
    batch=16,          # VRAM 8GB 기준 배치 사이즈
    device=0,          # 0번 : GPU (RTX 4060)
    patience=20,       # 20 에포크 동안 성능 향상이 없으면 조기 종료
    project="leaf_disease",
    name="yolo26m_v2"
)
