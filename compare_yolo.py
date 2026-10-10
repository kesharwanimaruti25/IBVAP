from pathlib import Path
import time
import torch
from ultralytics import YOLO

videos = sorted(Path("sample_videos").glob("*"))
videos = [
    p for p in videos
    if p.is_file() and p.suffix.lower() in
    {".mp4", ".mov", ".avi", ".mkv"}
]

if not videos:
    raise SystemExit("No test videos found.")

video = videos[0]
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8s.pt")

print(f"Video: {video.name}")
print(f"Device: {device}")

for size in (640, 960):
    count = 0

    # Warm up before timing.
    model.predict(
        source="sample_videos",
        device=device,
        imgsz=size,
        verbose=False,
    ) if False else None

    cap = __import__("cv2").VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise SystemExit(f"Cannot read {video}")

    model.predict(frame, device=device, imgsz=size, verbose=False)

    start = time.perf_counter()
    for _ in model.predict(
        source=str(video),
        device=device,
        imgsz=size,
        stream=True,
        verbose=False,
    ):
        count += 1

    elapsed = time.perf_counter() - start
    print(
        f"imgsz={size}: {count} frames, "
        f"{elapsed:.2f}s, {count / elapsed:.2f} FPS"
    )
