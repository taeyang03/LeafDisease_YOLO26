from collections import Counter
from pathlib import Path
from ultralytics import YOLO

# best.pt 불러오기 (최종 가중치)
model = YOLO("./runs/detect/leaf_disease/yolo26m_v2/weights/best.pt")

# 번호 -> (표시 이름, 실제 폴더명) 매핑
menu_options = {
    "1": ("healthy", Path("./test_images/Healthy")),
    "2": ("rust", Path("./test_images/Rust")),
    "3": ("powndery", Path("./test_images/Powdery")),
}

while True:
  print("\n1. healthy, 2. rust, 3. powndery, 4. exit")
  choice = input("번호를 입력하세요: ").strip()

  if choice == "4":
    print("프로세스를 종료합니다.")
    break

  if choice not in menu_options:
    print(
        f"잘못된 입력입니다: '{choice}'. 1, 2, 3, 4 중 하나를 입력해 주세요."
    )
    continue

  selected_label, target_dir = menu_options[choice]

  if not target_dir.exists():
    print(f"대상 폴더가 존재하지 않습니다: {target_dir}")
    continue

  # 선택한 폴더 경로의 이미지로 예측 실행
  results = model.predict(
      source=str(target_dir),
      conf=0.5,  # 확신도(Confidence) 50% 이상인 결과만 표시
      save=True,  # 바운딩 박스와 질병 이름이 그려진 결과 이미지를 파일로 저장
      verbose=False,
  )

  total_images = len(results)
  box_counts = Counter()
  print("===============================================================")

  # 파일명 <-> 판별 결과(클래스 및 확신도) 매핑 출력
  for result in results:
    filename = Path(result.path).name
    if len(result.boxes) == 0:
      print(f"{filename} -> 미검출 (확신도: 0.0%)")
      continue

    for box in result.boxes:
      class_id = int(box.cls[0])
      class_name = model.names[class_id]  # 'healthy', 'powdery', 'rust' 중 하나
      confidence = float(box.conf[0])  # 예측 확신도 (0.0 ~ 1.0)
      print(f"{filename} -> {class_name} (확신도: {confidence * 100:.1f}%)")
      box_counts[class_name] += 1
    
    print("===============================================================")

  total_boxes = sum(box_counts.values())
  display_classes = ["rust", "healthy", "powndery"]

  summary_parts = [f"총 입력 이미지 : {total_images} (검출 박스 : {total_boxes})"]
  for cls in display_classes:
    cnt = (
        box_counts[cls] if cls in box_counts else box_counts.get("powdery", 0)
    )
    pct = (cnt / total_boxes * 100) if total_boxes > 0 else 0.0
    summary_parts.append(f"{cls} : {cnt} ({pct:.1f}%)")

  print(" \\ ".join(summary_parts))
