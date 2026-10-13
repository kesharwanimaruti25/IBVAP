import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import json
import os
import sys
import time
import hashlib
import argparse
from datetime import datetime
from collections import deque
from event_integrity import compute_hash
from fence_geometry import segments_intersect, side_of_line
from detection_timing import activity_clock
from detection_rules import is_sustained_approach, toggle_friendly_track_at
model = YOLO("yolov8s.pt")

TRACK_CLASSES = [0, 2, 3, 5, 7]
CLASS_NAMES = {0: "Person", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
COLORS = {
    0: (0, 255, 0), 2: (255, 128, 0), 3: (0, 165, 255),
    5: (255, 0, 255), 7: (0, 0, 255),
}

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

EVENTS_FILE = os.path.expanduser(os.environ.get("IBVAP_EVENTS_FILE", "~/my-ai-app/events.jsonl"))
SNAPSHOT_DIR = os.path.expanduser(os.environ.get("IBVAP_SNAPSHOT_DIR", "~/my-ai-app/snapshots"))
DETECTION_DEVICE = os.environ.get("IBVAP_DEVICE", "mps")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
events_directory = os.path.dirname(EVENTS_FILE)
if events_directory:
    os.makedirs(events_directory, exist_ok=True)

GENESIS_HASH = "0" * 64

def get_last_hash():
    if not os.path.exists(EVENTS_FILE):
        return GENESIS_HASH
    with open(EVENTS_FILE, "r") as f:
        for line in reversed(f.readlines()):
            if line.strip():
                try:
                    last_event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError("Cannot continue a malformed event log.") from error
                last_hash = last_event.get("hash") if isinstance(last_event, dict) else None
                if not last_hash:
                    raise ValueError("Cannot continue an event log with no final event hash.")
                return last_hash
    return GENESIS_HASH

LAST_HASH = get_last_hash()
NIGHT_BRIGHTNESS_THRESHOLD = 70
LOITER_SECONDS = 15
RUNNING_SPEED_THRESHOLD = 500
CONVERGE_DISTANCE_THRESHOLD = 150
CONVERGE_MIN_CLOSING_DISTANCE = 40
RIDER_PROXIMITY_THRESHOLD = 100  # pixels; person within this of a vehicle = treated as a rider, not a pedestrian
CONVERGE_SUSTAIN_SECONDS = 2
CONVERGE_TRACK_COOLDOWN = 30
SUSPICIOUS_COOLDOWN = 10
COOLDOWN_SECONDS = 8

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def get_video_queue(source=None):
    if source is not None:
        arg = source
        if os.path.isdir(arg):
            files = sorted([
                os.path.join(arg, f) for f in os.listdir(arg)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ])
            if not files:
                print(f"No video files found in folder: {arg}")
                sys.exit(1)
            return files
        elif arg == "0":
            return [0]
        else:
            return [arg]
    else:
        return [0]


def setup_fence(cap, label="", live=False, mirror=False):
    fence_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(fence_points) < 2:
            fence_points.append((x, y))

    ret, first_frame = cap.read()
    if not ret:
        return None

    window_name = f"Set Fence - {label}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    if live:
        print(f"[{label}] Live feed: click START point, then END point (ground/feet level).")
        print("SPACE=confirm and start live detection  R=reset  Q=quit")
    else:
        print(f"[{label}] Click START point, then END point (ground/feet level).")
        print("SPACE=confirm  R=reset  Q=quit everything")

    while True:
        if live:
            ret, frame = cap.read()
            if not ret:
                cv2.destroyWindow(window_name)
                return None
        else:
            frame = first_frame
        if mirror:
            frame = cv2.flip(frame, 1)
        display = frame.copy()
        for pt in fence_points:
            cv2.circle(display, pt, 6, (0, 255, 255), -1)
        if len(fence_points) == 2:
            cv2.line(display, fence_points[0], fence_points[1], (0, 255, 255), 2)
        instruction = (
            f"{label}: LIVE MIRROR · click GROUND points. SPACE=start R=reset Q=quit"
            if live else f"{label}: click GROUND points. SPACE=confirm R=reset Q=quit"
        )
        cv2.putText(display, instruction,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(' ') and len(fence_points) == 2:
            break
        elif key == ord('r'):
            fence_points = []
        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    if not live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return fence_points[0], fence_points[1]


def wait_for_start(cap, fence_a, fence_b, label=""):
    window_name = f"Ready - {label}"
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    while True:
        display = frame.copy()
        cv2.line(display, fence_a, fence_b, (0, 255, 255), 2)
        cv2.putText(display, f"{label}: SPACE=start  N=skip video  Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            cv2.destroyWindow(window_name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return "start"
        elif key == ord('n'):
            cv2.destroyWindow(window_name)
            return "skip"
        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            return "quit"


def put_fitted_text(frame, text, x, baseline, max_width, font_scale, color, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_width = max(1, max_width)
    scale = min(font_scale, max_width / max(1, cv2.getTextSize(text, font, font_scale, thickness)[0][0]))
    size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, frame.shape[1] - size[0] - 1))
    baseline = max(size[1] + 2, min(baseline, frame.shape[0] - 2))
    cv2.putText(frame, text, (x, baseline), font, scale, color, thickness, cv2.LINE_AA)


def draw_track_label(frame, box, text, color):
    x1, y1, x2, _ = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(0.52, max(0.25, (frame.shape[1] - 14) / max(1, cv2.getTextSize(text, font, 0.52, 1)[0][0])))
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding_x, padding_y = 7, 5
    left = max(0, min(x1, frame.shape[1] - text_width - padding_x * 2))
    top = y1 - text_height - baseline - padding_y * 2 - 3
    if top < 0:
        top = min(frame.shape[0] - text_height - baseline - padding_y * 2, y1 + 3)
    right = min(frame.shape[1] - 1, left + text_width + padding_x * 2)
    bottom = min(frame.shape[0] - 1, top + text_height + baseline + padding_y * 2)
    cv2.rectangle(frame, (left, top), (right, bottom), (26, 39, 31), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
    cv2.putText(
        frame, text, (left + padding_x, bottom - padding_y - baseline),
        font, font_scale, (245, 247, 239), thickness, cv2.LINE_AA
    )


def show_end_screen(last_frame, label=""):
    window_name = f"Video Ended - {label}"
    while True:
        display = last_frame.copy()
        h, w = display.shape[:2]
        overlay = display.copy()
        panel_width = max(1, int(w * 0.88))
        panel_height = max(1, min(int(h * 0.26), max(150, int(w * 0.34))))
        left = (w - panel_width) // 2
        top = (h - panel_height) // 2
        right = left + panel_width
        bottom = top + panel_height
        cv2.rectangle(overlay, (left, top), (right, bottom), (22, 35, 29), -1)
        display = cv2.addWeighted(overlay, 0.88, display, 0.12, 0)
        cv2.rectangle(display, (left, top), (right, bottom), (97, 124, 101), 1, cv2.LINE_AA)
        accent_y = top + max(22, int(panel_height * 0.18))
        cv2.line(display, (left + 18, accent_y), (left + 52, accent_y), (126, 184, 137), 3, cv2.LINE_AA)
        text_x = left + 18
        text_width = panel_width - 36
        put_fitted_text(display, "PLAYBACK COMPLETE", text_x, top + int(panel_height * 0.53),
                        text_width, min(1.0, w / 520), (248, 249, 242), 2)
        put_fitted_text(display, "R  Replay     N  Next video     Q  Quit",
                        text_x, top + int(panel_height * 0.76), text_width,
                        min(0.62, w / 760), (202, 216, 202), 1)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('r'):
            cv2.destroyWindow(window_name)
            return "replay"
        elif key == ord('n'):
            cv2.destroyWindow(window_name)
            return "next"
        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            return "quit"


def log_event(track_id, cls_id, direction, conf, is_night, frame, event_type="crossing",
              source_id=None, media_time_seconds=None):
    global LAST_HASH
    timestamp = datetime.now()
    safe_type = event_type.replace(" ", "_")
    filename = f"{safe_type}_{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    snapshot_path = os.path.join(SNAPSHOT_DIR, filename)
    if not cv2.imwrite(snapshot_path, frame):
        raise OSError(f"Could not save event snapshot: {snapshot_path}")
    with open(snapshot_path, "rb") as image:
        snapshot_sha256 = hashlib.sha256(image.read()).hexdigest()

    event = {
        "timestamp": timestamp.isoformat(),
        "track_id": int(track_id),
        "class": CLASS_NAMES.get(cls_id, "Object"),
        "direction": direction,
        "confidence": round(float(conf), 2),
        "night": bool(is_night),
        "type": event_type,
        "snapshot": filename,
        "snapshot_sha256": snapshot_sha256,
        "source": source_id,
        "media_time_seconds": round(float(media_time_seconds), 3) if media_time_seconds is not None else None,
        "prev_hash": LAST_HASH
    }
    event["hash"] = compute_hash(LAST_HASH, event)
    LAST_HASH = event["hash"]

    with open(EVENTS_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    print("EVENT:", event)

def run_detection(cap, fence_a, fence_b, is_video_file, label="", headless=False, mirror=False):
    if is_video_file:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = max(1, int(1000 / video_fps)) if video_fps > 0 else 30
    else:
        frame_delay = 1

    prev_sides = {}
    prev_ground_points = {}
    last_alert_time = {}
    first_seen = {}
    loiter_logged = set()
    position_history = {}
    running_alert_time = {}
    converge_start_time = {}
    converge_track_alert_time = {}
    flash_until = 0
    last_frame = None
    is_night = False
    session_started_at = time.monotonic()
    frame_index = 0

    window_name = f"IBVAP - {label}"
    friendly_tracks = set()
    tracked_person_boxes = []

    def on_mouse(event, x, y, _flags, _parameter):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        track_id, is_friendly = toggle_friendly_track_at(
            x, y, tracked_person_boxes, friendly_tracks
        )
        if track_id is None:
            print("Friendly tagging: click inside a visible person box.")
            return
        state = "marked friendly; routine alerts muted" if is_friendly else "unmarked; alerts restored"
        print(f"Track #{track_id} {state} for this detector session.")

    if not headless:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            if not headless:
                cv2.destroyWindow(window_name)
            return "ended", last_frame

        if mirror:
            frame = cv2.flip(frame, 1)

        frame_index += 1
        h, w = frame.shape[:2]
        now = time.time()
        if is_video_file:
            media_time_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not np.isfinite(media_time_seconds) or media_time_seconds < 0 or (
                media_time_seconds == 0 and frame_index > 1
            ):
                media_time_seconds = frame_index / video_fps if video_fps > 0 else 0.0
        else:
            media_time_seconds = time.monotonic() - session_started_at
        activity_time = activity_clock(is_video_file, media_time_seconds, now)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if is_night:
            is_night = brightness < (NIGHT_BRIGHTNESS_THRESHOLD + 10)
        else:
            is_night = brightness < NIGHT_BRIGHTNESS_THRESHOLD

        if is_night:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        results = model.track(
            frame, classes=TRACK_CLASSES, persist=True, verbose=False,
            device=DETECTION_DEVICE, conf=0.25, imgsz=960, tracker="my_botsort.yaml"
        )

        current_person_positions = []
        current_vehicle_positions = []
        tracked_person_boxes.clear()

        for result in results:
            if result.boxes.id is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            # First pass: collect vehicle ground points so we can detect riders below
            for box, cls_id in zip(boxes, class_ids):
                if cls_id in (2, 3, 5, 7):
                    vx1, vy1, vx2, vy2 = map(int, box)
                    current_vehicle_positions.append(((vx1 + vx2) // 2, vy2))
            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                ground_x = (x1 + x2) // 2
                ground_y = y2
                label_name = CLASS_NAMES.get(cls_id, "Object")
                color = COLORS.get(cls_id, (255, 255, 255))
                is_friendly = cls_id == 0 and track_id in friendly_tracks
                if cls_id == 0:
                    tracked_person_boxes.append((x1, y1, x2, y2, track_id))
                if is_friendly:
                    color = (255, 200, 0)
                is_rider = False
                if cls_id == 0:
                    for vx, vy in current_vehicle_positions:
                        dist_to_vehicle = ((ground_x - vx) ** 2 + (ground_y - vy) ** 2) ** 0.5
                        if dist_to_vehicle < RIDER_PROXIMITY_THRESHOLD:
                            is_rider = True
                            break

                if track_id not in position_history:
                    position_history[track_id] = deque(maxlen=10)
                if is_friendly:
                    first_seen[track_id] = activity_time
                    loiter_logged.discard(track_id)
                    position_history[track_id] = deque(
                        [(activity_time, ground_x, ground_y)], maxlen=10
                    )
                elif track_id not in first_seen:
                    first_seen[track_id] = activity_time
                elapsed = activity_time - first_seen[track_id]

                if (
                    not is_friendly and elapsed > LOITER_SECONDS
                    and track_id not in loiter_logged and not is_rider
                ):
                    log_event(
                        track_id, cls_id, "loitering", conf, is_night, frame,
                        event_type="loitering", source_id=label,
                        media_time_seconds=media_time_seconds
                    )
                    loiter_logged.add(track_id)
                    flash_until = now + 2
                    color = (0, 165, 255)

                if not is_friendly:
                    position_history[track_id].append((activity_time, ground_x, ground_y))
                if (
                    not is_friendly and cls_id == 0 and not is_rider
                    and len(position_history[track_id]) >= 2
                ):
                    t0, x0, y0 = position_history[track_id][0]
                    dt = activity_time - t0
                    if dt > 0.3:
                        dist = ((ground_x - x0) ** 2 + (ground_y - y0) ** 2) ** 0.5
                        speed = dist / dt
                        if speed > RUNNING_SPEED_THRESHOLD:
                            last_run = running_alert_time.get(track_id, float("-inf"))
                            if activity_time - last_run > SUSPICIOUS_COOLDOWN:
                                log_event(
                                    track_id, cls_id, "running", conf, is_night, frame,
                                    event_type="suspicious_running", source_id=label,
                                    media_time_seconds=media_time_seconds
                                )
                                running_alert_time[track_id] = activity_time
                                flash_until = now + 2
                                color = (0, 0, 255)

                if cls_id == 0 and not is_friendly:
                    current_person_positions.append((track_id, ground_x, ground_y))

                current_point = (ground_x, ground_y)
                current_side = side_of_line(current_point, fence_a, fence_b)

                if track_id in prev_sides and not (cls_id == 0 and (is_rider or is_friendly)):
                    prev_side = prev_sides[track_id]
                    previous_point = prev_ground_points.get(track_id)
                    crossed_fence_segment = (
                        previous_point is not None
                        and segments_intersect(previous_point, current_point, fence_a, fence_b)
                    )
                    if prev_side != 0 and current_side != 0 and prev_side != current_side and crossed_fence_segment:
                        last_time = last_alert_time.get(track_id, float("-inf"))
                        if activity_time - last_time > COOLDOWN_SECONDS:

                            direction = "entering" if current_side < prev_side else "exiting"
                            log_event(
                                track_id, cls_id, direction, conf, is_night, frame,
                                event_type="crossing", source_id=label,
                                media_time_seconds=media_time_seconds
                            )
                            last_alert_time[track_id] = activity_time
                            flash_until = now + 2
                            color = (0, 0, 255)

                if current_side != 0:
                    prev_sides[track_id] = current_side
                    prev_ground_points[track_id] = current_point

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (ground_x, ground_y), 4, (0, 255, 255), -1)
                track_label = f"FRIENDLY #{track_id}" if is_friendly else f"{label_name} #{track_id} {conf:.2f}"
                draw_track_label(frame, (x1, y1, x2, y2), track_label, color)

        active_converging_pairs = set()
        for i in range(len(current_person_positions)):
            for j in range(i + 1, len(current_person_positions)):
                id_a, xa, ya = current_person_positions[i]
                id_b, xb, yb = current_person_positions[j]
                dist = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
                pair_key = frozenset({id_a, id_b})

                if dist < CONVERGE_DISTANCE_THRESHOLD:
                    active_converging_pairs.add(pair_key)
                    if pair_key not in converge_start_time:
                        converge_start_time[pair_key] = (activity_time, dist)
                    start_time, start_distance = converge_start_time[pair_key]
                    if is_sustained_approach(
                        start_time, start_distance, activity_time, dist,
                        CONVERGE_SUSTAIN_SECONDS, CONVERGE_MIN_CLOSING_DISTANCE
                    ):
                        last_alert_a = converge_track_alert_time.get(id_a, float("-inf"))
                        last_alert_b = converge_track_alert_time.get(id_b, float("-inf"))
                        if (
                            activity_time - last_alert_a > CONVERGE_TRACK_COOLDOWN
                            and activity_time - last_alert_b > CONVERGE_TRACK_COOLDOWN
                        ):
                            log_event(
                                id_a, 0, "converging", 0.0, is_night, frame,
                                event_type="suspicious_converging", source_id=label,
                                media_time_seconds=media_time_seconds
                            )
                            converge_track_alert_time[id_a] = activity_time
                            converge_track_alert_time[id_b] = activity_time
                            converge_start_time.pop(pair_key, None)
                            flash_until = now + 2
                            cv2.line(frame, (xa, ya), (xb, yb), (0, 0, 255), 2)
                else:
                    converge_start_time.pop(pair_key, None)
        for pair_key in set(converge_start_time) - active_converging_pairs:
            converge_start_time.pop(pair_key, None)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detector.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                fx1 = int(bbox.xmin * w)
                fy1 = int(bbox.ymin * h)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                face_conf = detection.score[0]
                cv2.rectangle(frame, (fx1, fy1), (fx1 + fw, fy1 + fh), (255, 255, 0), 2)
                cv2.putText(frame, f"Face {face_conf:.2f}", (fx1, fy1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.line(frame, fence_a, fence_b, (0, 255, 255), 2)
        cv2.putText(frame, "VIRTUAL FENCE", (fence_a[0], fence_a[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if is_night:
            cv2.putText(frame, "NIGHT MODE ACTIVE", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if now < flash_until:
            cv2.putText(frame, "ALERT", (int(w * 0.4), 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        put_fitted_text(
            frame,
            f"{label} · click person to toggle FRIENDLY ({len(friendly_tracks)}) · N next · Q quit",
            12, h - 18, w - 24, 0.5, (220, 225, 215), 1
        )

        last_frame = frame.copy()
        if headless:
            continue
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return "quit", None
        elif key == ord('n'):
            cv2.destroyWindow(window_name)
            return "next", None


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Run local video analytics with an optional virtual fence.")
    parser.add_argument("source", nargs="?", help="Video path, folder of videos, or camera index (default: 0)")
    parser.add_argument("--source-label", help="Display/source name to record for the selected video.")
    parser.add_argument(
        "--fence", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
        help="Use a preconfigured fence and skip the interactive fence setup."
    )
    parser.add_argument("--headless", action="store_true", help="Process without opening OpenCV windows.")
    args = parser.parse_args(argv)
    if args.headless and not args.fence:
        parser.error("--headless requires --fence.")
    if args.headless and (args.source is None or args.source == "0"):
        parser.error("--headless requires a video file or video folder.")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    video_queue = get_video_queue(args.source)

    for source in video_queue:
        is_video_file = isinstance(source, str)
        label = (args.source_label or os.path.basename(source)) if is_video_file else "Live Camera"

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Could not open source: {source}", file=sys.stderr)
            continue

        if not is_video_file:
            for _ in range(20):
                cap.read()

        if args.fence:
            fence_a = (args.fence[0], args.fence[1])
            fence_b = (args.fence[2], args.fence[3])
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if any(x < 0 or y < 0 or x >= frame_width or y >= frame_height for x, y in (fence_a, fence_b)):
                print(f"Fence coordinates are outside the {frame_width}x{frame_height} video frame.", file=sys.stderr)
                cap.release()
                return 2
            if args.headless:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                status, _ = run_detection(cap, fence_a, fence_b, is_video_file, label=label, headless=True)
                cap.release()
                if status != "ended":
                    return 1
                if hasattr(model, "predictor") and model.predictor is not None:
                    model.predictor = None
                continue
        else:
            live_camera = not is_video_file
            fence = setup_fence(cap, label=label, live=live_camera, mirror=live_camera)
            if fence is None:
                cap.release()
                break
            fence_a, fence_b = fence

        action = "start" if not is_video_file else wait_for_start(cap, fence_a, fence_b, label=label)
        if action == "quit":
            cap.release()
            break
        elif action == "skip":
            cap.release()
            continue

        quit_all = False
        while True:
            status, last_frame = run_detection(
                cap, fence_a, fence_b, is_video_file, label=label, mirror=not is_video_file
            )

            if status == "quit":
                quit_all = True
                break
            elif status == "next":
                break
            elif status == "ended":
                end_action = show_end_screen(last_frame, label=label)
                if end_action == "replay":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                elif end_action == "next":
                    break
                elif end_action == "quit":
                    quit_all = True
                    break

        cap.release()

        if hasattr(model, "predictor") and model.predictor is not None:
            model.predictor = None

        if quit_all:
            break
    cv2.destroyAllWindows()
    print("All videos processed. Exiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
