import cv2
import easyocr
import re
import time
from collections import deque, Counter
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]

print("Loading OCR engine (first run downloads models, please wait)...")
reader = easyocr.Reader(['en'], gpu=False)
print("OCR ready.")

SOURCE = 0
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("Could not open video source.")
    exit()

for _ in range(20):
    cap.read()

def clean_plate_text(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned) < 8 or len(cleaned) > 11:
        return None

    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
    if re.match(pattern, cleaned):
        return cleaned

    fixed = cleaned
    fixed = fixed[:2].replace('0', 'O').replace('1', 'I').replace('5', 'S') + fixed[2:]
    if len(fixed) >= 4:
        tail = fixed[-4:].replace('O', '0').replace('I', '1').replace('S', '5').replace('Z', '2')
        fixed = fixed[:-4] + tail

    if re.match(pattern, fixed):
        return fixed
    return None

# ---- Multi-frame consensus ----
CONSENSUS_WINDOW_SECONDS = 6
CONSENSUS_THRESHOLD = 3   # how many times a plate must appear to be "confirmed"
recent_reads = deque()    # stores (timestamp, plate_text)
confirmed_plate = None
confirmed_until = 0

def add_read(plate_text):
    recent_reads.append((time.time(), plate_text))

def get_consensus():
    now = time.time()
    while recent_reads and now - recent_reads[0][0] > CONSENSUS_WINDOW_SECONDS:
        recent_reads.popleft()

    if not recent_reads:
        return None, 0

    counts = Counter(p for _, p in recent_reads)
    best_plate, count = counts.most_common(1)[0]
    return best_plate, count

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    clean_frame = frame.copy()
    results = model(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.4)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.putText(frame, f"Vehicle {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

            if frame_count % 15 != 0:
                continue

            plate_region_y1 = y1 + int((y2 - y1) * 0.4)
            pad = 10
            ry1 = max(0, plate_region_y1)
            ry2 = min(clean_frame.shape[0], y2 + pad)
            rx1 = max(0, x1 - pad)
            rx2 = min(clean_frame.shape[1], x2 + pad)
            roi = clean_frame[ry1:ry2, rx1:rx2]

            if roi.size == 0:
                continue

            try:
                roi_big = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                roi_gray = cv2.cvtColor(roi_big, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                roi_proc = clahe.apply(roi_gray)

                ocr_results = reader.readtext(roi_proc)

                for (bbox, text, ocr_conf) in ocr_results:
                    if ocr_conf < 0.20:
                        continue
                    plate = clean_plate_text(text)
                    if plate:
                        print(f"  [read] {plate} (conf={ocr_conf:.2f})")
                        add_read(plate)
                        cv2.putText(frame, f"reading: {plate}", (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            except Exception as e:
                print("OCR error:", e)

    # ---- Check consensus every frame (cheap, no OCR involved) ----
    best_plate, count = get_consensus()
    if best_plate and count >= CONSENSUS_THRESHOLD:
        if best_plate != confirmed_plate:
            print(f"CONFIRMED PLATE: {best_plate} (seen {count}x in last {CONSENSUS_WINDOW_SECONDS}s)")
        confirmed_plate = best_plate
        confirmed_until = time.time() + 5  # keep showing for 5s after last confirmation

    h, w = frame.shape[:2]
    if confirmed_plate and time.time() < confirmed_until:
        cv2.rectangle(frame, (10, h - 60), (10 + 340, h - 15), (0, 0, 0), -1)
        cv2.putText(frame, f"CONFIRMED: {confirmed_plate}", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("IBVAP - ANPR Module", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
