from collections import Counter
from ultralytics import YOLO
from pathlib import Path

model = YOLO("./runs/detect/leaf_disease/yolo26m_v2/weights/best.pt")

menu_options = {
    "1": ("healthy", Path("./test_images/Healthy")),
    "2": ("rust", Path("./test_images/Rust")),
    "3": ("powndery", Path("./test_images/Powdery")),
}

while True:
  print("1. healthy, 2. rust, 3. powndery, 4. exit")
  choice = input("번호를 입력하세요: ").strip()

  if choice == "4":
    print("프로세스를 종료합니다.")
    break
 
  if choice not in menu_options:
    raise ValueError(
        f"잘못된 입력입니다: '{choice}'. 1, 2, 3 중 하나를 입력해 주세요."
    )

  selected_label, target_dir = menu_options[choice]

  if not target_dir.exists():
    raise FileNotFoundError(f"대상 폴더가 존재하지 않습니다: {target_dir}")

  # 새로운 데이터(선택한 폴더 경로) 넣고 예측 실행
  results = model.predict(
      source=str(target_dir),
      conf=0.5,  # 확신도(Confidence) 50% 이상인 결과만 표시
      save=True,  # 바운딩 박스와 질병 이름이 그려진 결과 이미지를 파일로 저장
      verbose=False,
  )

  total_images = len(results)
  box_counts = Counter()

  for result in results:
    for box in result.boxes:
      class_id = int(box.cls[0])
      class_name = model.names[class_id]
      confidence = float(box.conf[0])
      print(f"판별 결과: {class_name} (확신도: {confidence * 100:.1f}%)")
      box_counts[class_name] += 1

  total_boxes = sum(box_counts.values())
  display_classes = ["rust", "healthy", "powndery"]

  parts = [f"총 입력 이미지 : {total_images} (검출 박스 : {total_boxes})"]
  for cls in display_classes:
    cnt = box_counts[cls] if cls in box_counts else box_counts.get("powdery", 0)
    pct = (cnt / total_boxes * 100) if total_boxes > 0 else 0.0
    parts.append(f"{cls} : {cnt} ({pct:.1f}%)")

  print(" \\ ".join(parts))
