from pathlib import Path
import time
import cv2
import torch
from ultralytics import YOLO

video_dir = Path("sample_videos")
extensions = {".mp4", ".mov", ".avi", ".mkv"}

videos = sorted(
    p for p in video_dir.glob("*")
    if p.is_file() and p.suffix.lower() in extensions
) if video_dir.exists() else []

if not videos:
    raise SystemExit("ERROR: No videos found in sample_videos/")

video = videos[0]
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Video: {video}")
print(f"Device: {device}")

cap = cv2.VideoCapture(str(video))
ok, frame = cap.read()
cap.release()

if not ok:
    raise SystemExit(f"ERROR: Could not read {video}")

model = YOLO("yolov8s.pt")

print("Warming up...")
model.predict(frame, device=device, imgsz=640, verbose=False)

frames = 0
start = time.perf_counter()

for _ in model.predict(
    source=str(video),
    device=device,
    imgsz=640,
    stream=True,
    verbose=False,
):
    frames += 1

elapsed = time.perf_counter() - start

print("\n--- RESULTS ---")
print(f"Video: {video.name}")
print(f"Frames: {frames}")
print(f"Elapsed: {elapsed:.2f} seconds")
print(f"Average throughput: {frames / elapsed:.2f} FPS")
