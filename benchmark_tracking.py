from pathlib import Path
import time
import torch
from ultralytics import YOLO

videos = sorted(
    p for p in Path("sample_videos").glob("*")
    if p.is_file() and p.suffix.lower() in
    {".mp4", ".mov", ".avi", ".mkv"}
)

if not videos:
    raise SystemExit("No videos found in sample_videos/")

video = videos[0]
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8s.pt")

print(f"Video: {video.name}")
print(f"Device: {device}")

for size in (640, 960):
    count = 0

    # Warm up tracking on the first frame.
    import cv2
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise SystemExit("Could not read test video.")

    model.track(
        frame, persist=False, device=device,
        imgsz=size, verbose=False
    )

    start = time.perf_counter()

    for _ in model.track(
        source=str(video),
        persist=True,
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
