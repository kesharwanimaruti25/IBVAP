import argparse
import csv
import math
import os
import sys

LABEL_FIELDS = ["video", "event_type", "time_seconds", "direction", "event_class"]
EVENT_KEYS = {
    ord("1"): "crossing",
    ord("2"): "loitering",
    ord("3"): "suspicious_running",
    ord("4"): "suspicious_converging",
}


def prepare_labels_file(output_path, video_path, append=False):
    output_path = os.path.realpath(output_path)
    video_path = os.path.realpath(video_path)
    if output_path == video_path:
        raise ValueError("The labels CSV must not overwrite the source video.")

    if os.path.exists(output_path):
        if not append:
            raise FileExistsError(
                f"Labels file already exists: {output_path}. Use --append to add rows."
            )
        with open(output_path, newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader, [])
        if header != LABEL_FIELDS:
            raise ValueError(
                f"Existing labels file must have this header: {','.join(LABEL_FIELDS)}"
            )
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(LABEL_FIELDS)
    return output_path


def append_label(output_path, video_name, event_type, time_seconds):
    if event_type not in EVENT_KEYS.values():
        raise ValueError(f"Unsupported event type: {event_type}")
    if not video_name.strip():
        raise ValueError("Video name cannot be empty.")
    if not math.isfinite(time_seconds) or time_seconds < 0:
        raise ValueError("Annotation timestamp must be finite and non-negative.")
    row = {
        "video": video_name,
        "event_type": event_type,
        "time_seconds": f"{time_seconds:.3f}",
        "direction": "",
        "event_class": "",
    }
    with open(output_path, "a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=LABEL_FIELDS).writerow(row)
    return row


def seek_to(cap, timestamp_seconds, fps):
    import cv2

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000)
    ok, frame = cap.read()
    if not ok:
        return None, None
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if not math.isfinite(timestamp) or timestamp < 0:
        timestamp = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
    return frame, timestamp


def annotate(video_path, output_path, source_name=None, append=False):
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        print(f"Cannot open video: {video_path}", file=sys.stderr)
        return 2

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        print(f"Video has an invalid frame rate: {video_path}", file=sys.stderr)
        return 2

    ok, frame = capture.read()
    if not ok:
        capture.release()
        print(f"Video contains no readable frames: {video_path}", file=sys.stderr)
        return 2

    try:
        output_path = prepare_labels_file(output_path, video_path, append)
    except (OSError, ValueError) as error:
        capture.release()
        print(f"Cannot prepare labels file: {error}", file=sys.stderr)
        return 2

    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if not math.isfinite(timestamp) or timestamp < 0:
        timestamp = 0.0
    video_name = source_name or os.path.basename(video_path)
    playing = False
    saved_count = 0
    last_annotation = "No annotations added in this session"
    window_name = "IBVAP Ground-Truth Annotator"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            display = frame.copy()
            state = "PLAYING" if playing else "PAUSED"
            help_lines = [
                f"{video_name} | {timestamp:.3f}s | {state} | {saved_count} new labels",
                "SPACE play/pause | A/D seek 1s | [/] step frame",
                "1 crossing | 2 loitering | 3 running | 4 converging | Q quit",
                last_annotation,
            ]
            font_scale = min(0.58, max(0.35, display.shape[1] / 900.0))
            line_height = max(24, int(font_scale * 50))
            for index, text in enumerate(help_lines):
                while text and cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )[0][0] > display.shape[1] - 24:
                    text = text[:-4] + "..." if len(text) > 4 else text[:-1]
                cv2.putText(
                    display,
                    text,
                    (12, 26 + index * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (20, 20, 20),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    text,
                    (12, 26 + index * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(max(1, int(1000 / fps)) if playing else 30) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                playing = not playing
                continue
            if key in EVENT_KEYS:
                playing = False
                try:
                    row = append_label(output_path, video_name, EVENT_KEYS[key], timestamp)
                except OSError as error:
                    print(f"Cannot save annotation: {error}", file=sys.stderr)
                    return 2
                saved_count += 1
                last_annotation = (
                    f"Saved {row['event_type']} at {row['time_seconds']}s; "
                    "confirm this label against the video."
                )
                print(f"Saved {row['video']},{row['event_type']},{row['time_seconds']}")
                continue

            target_time = None
            if key == ord("a"):
                target_time = timestamp - 1.0
            elif key == ord("d"):
                target_time = timestamp + 1.0
            elif key == ord("["):
                target_time = timestamp - 1.0 / fps
            elif key == ord("]"):
                target_time = timestamp + 1.0 / fps

            if target_time is not None:
                playing = False
                next_frame, next_timestamp = seek_to(capture, target_time, fps)
                if next_frame is not None:
                    frame, timestamp = next_frame, next_timestamp
                continue

            if playing:
                ok, next_frame = capture.read()
                if not ok:
                    playing = False
                    continue
                frame = next_frame
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if not math.isfinite(timestamp) or timestamp < 0:
                    timestamp = capture.get(cv2.CAP_PROP_POS_FRAMES) / fps
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Labels saved to {output_path} ({saved_count} added this session).")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create human-reviewed ground-truth event labels for a video."
    )
    parser.add_argument("video", help="Path to an authorized video file.")
    parser.add_argument("--output", required=True, help="CSV output path.")
    parser.add_argument(
        "--source-name",
        help="Video name to store in the CSV; defaults to the source filename.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV with the expected header.",
    )
    args = parser.parse_args(argv)
    return annotate(args.video, args.output, args.source_name, args.append)


if __name__ == "__main__":
    raise SystemExit(main())
