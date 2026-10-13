import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict


REQUIRED_LABEL_COLUMNS = {"video", "event_type", "time_seconds"}


def load_labels(path):
    labels = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_LABEL_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Label CSV is missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, 2):
            try:
                timestamp = float(row["time_seconds"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid time_seconds at CSV row {row_number}.") from error
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(f"time_seconds must be finite and non-negative at CSV row {row_number}.")
            video = os.path.basename((row.get("video") or "").strip())
            event_type = (row.get("event_type") or "").strip()
            if not video or not event_type:
                raise ValueError(f"video and event_type are required at CSV row {row_number}.")
            labels.append({
                "video": video,
                "event_type": event_type,
                "direction": (row.get("direction") or "").strip(),
                "event_class": (row.get("event_class") or "").strip(),
                "time_seconds": timestamp,
            })
    return labels


def load_predictions(path):
    predictions = []
    unscorable = 0
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on event log line {line_number}.") from error
            if not isinstance(event, dict):
                raise ValueError(f"Event log line {line_number} is not a JSON object.")
            video = os.path.basename(str(event.get("source") or "").strip())
            event_type = str(event.get("type") or "").strip()
            try:
                timestamp = float(event["media_time_seconds"])
            except (KeyError, TypeError, ValueError):
                unscorable += 1
                continue
            if not video or not event_type or not math.isfinite(timestamp) or timestamp < 0:
                unscorable += 1
                continue
            predictions.append({
                "video": video,
                "event_type": event_type,
                "direction": str(event.get("direction") or "").strip(),
                "event_class": str(event.get("class") or "").strip(),
                "time_seconds": timestamp,
            })
    return predictions, unscorable


def _metrics(true_positives, false_positives, false_negatives):
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(labels, predictions, tolerance_seconds=2.0):
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("Tolerance must be finite and non-negative.")

    available_predictions = defaultdict(list)
    for index, prediction in enumerate(predictions):
        key = (prediction["video"], prediction["event_type"])
        available_predictions[key].append((index, prediction))

    candidates_by_label = []
    for label in labels:
        candidates = available_predictions.get((label["video"], label["event_type"]), [])
        eligible = [
            (abs(prediction["time_seconds"] - label["time_seconds"]), index)
            for index, prediction in candidates
            if abs(prediction["time_seconds"] - label["time_seconds"]) <= tolerance_seconds
            and (
                not label.get("direction")
                or prediction.get("direction") == label["direction"]
            )
            and (
                not label.get("event_class")
                or prediction.get("event_class") == label["event_class"]
            )
        ]
        candidates_by_label.append([index for _, index in sorted(eligible)])

    prediction_to_label = {}

    def assign_prediction(label_index, visited):
        for prediction_index in candidates_by_label[label_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous_label = prediction_to_label.get(prediction_index)
            if previous_label is None or assign_prediction(previous_label, visited):
                prediction_to_label[prediction_index] = label_index
                return True
        return False

    label_order = sorted(range(len(labels)), key=lambda index: len(candidates_by_label[index]))
    for label_index in label_order:
        assign_prediction(label_index, set())

    by_type_counts = defaultdict(lambda: [0, 0, 0])
    by_video_counts = defaultdict(lambda: [0, 0, 0])
    matched_label_indexes = set(prediction_to_label.values())
    for label_index, label in enumerate(labels):
        event_type = label["event_type"]
        video = label["video"]
        if label_index in matched_label_indexes:
            by_type_counts[event_type][0] += 1
            by_video_counts[video][0] += 1
        else:
            by_type_counts[event_type][2] += 1
            by_video_counts[video][2] += 1

    for index, prediction in enumerate(predictions):
        if index not in prediction_to_label:
            by_type_counts[prediction["event_type"]][1] += 1
            by_video_counts[prediction["video"]][1] += 1

    by_type = {
        event_type: _metrics(*counts)
        for event_type, counts in sorted(by_type_counts.items())
    }
    by_video = {
        video: _metrics(*counts)
        for video, counts in sorted(by_video_counts.items())
    }
    totals = _metrics(
        len(prediction_to_label),
        sum(values[1] for values in by_type_counts.values()),
        sum(values[2] for values in by_type_counts.values()),
    )
    matches = []
    for prediction_index, label_index in sorted(
        ((prediction_index, label_index) for prediction_index, label_index in prediction_to_label.items()),
        key=lambda pair: (labels[pair[1]]["video"], labels[pair[1]]["time_seconds"]),
    ):
        label = labels[label_index]
        prediction = predictions[prediction_index]
        matches.append({
            "label": label,
            "prediction": prediction,
            "time_error_seconds": abs(label["time_seconds"] - prediction["time_seconds"]),
        })

    unmatched_labels = [
        label for index, label in enumerate(labels)
        if index not in matched_label_indexes
    ]
    unmatched_predictions = [
        prediction for index, prediction in enumerate(predictions)
        if index not in prediction_to_label
    ]

    return {
        "overall": totals,
        "by_event_type": by_type,
        "by_video": by_video,
        "matches": matches,
        "false_negatives": unmatched_labels,
        "false_positives": unmatched_predictions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare video-relative detector events with manually labeled events."
    )
    parser.add_argument("--labels", required=True, help="CSV with video,event_type,time_seconds[,direction]")
    parser.add_argument("--events", required=True, help="Detector JSONL event log")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Timestamp matching tolerance in seconds")
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="Allow a header-only labels CSV for an intentionally event-free negative test clip.",
    )
    parser.add_argument(
        "--json-report",
        help="Optional path to save metrics, input/evaluator SHA-256 digests, matches, false positives, and misses.",
    )
    args = parser.parse_args(argv)

    try:
        if args.json_report and os.path.realpath(args.json_report) in {
            os.path.realpath(args.labels),
            os.path.realpath(args.events),
        }:
            raise ValueError("JSON report path must not overwrite either input file.")
        labels = load_labels(args.labels)
        if not labels and not args.allow_empty_labels:
            raise ValueError(
                "The label CSV contains no ground-truth events. "
                "Use --allow-empty-labels only for an intentionally event-free negative test."
            )
        predictions, unscorable = load_predictions(args.events)
        result = evaluate(labels, predictions, args.tolerance)
        report = {
            "schema_version": 1,
            "method": {
                "unit": "event",
                "matching": "one-to-one",
                "tolerance_seconds": args.tolerance,
                "labels_file": os.path.basename(args.labels),
                "labels_sha256": _sha256_file(args.labels),
                "events_file": os.path.basename(args.events),
                "events_sha256": _sha256_file(args.events),
                "evaluator_sha256": _sha256_file(__file__),
            },
            "ground_truth_events": len(labels),
            "scorable_detector_events": len(predictions),
            "unscorable_detector_events": unscorable,
            **result,
        }
        if args.json_report:
            with open(args.json_report, "w", encoding="utf-8") as file:
                json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
                file.write("\n")
    except (OSError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2

    print(f"Ground-truth events: {len(labels)}")
    print(f"Scorable detector events: {len(predictions)}")
    print(f"Detector events excluded (missing video/time metadata): {unscorable}")
    print(f"Matching tolerance: ±{args.tolerance:g}s")
    print("\nMetrics are event-level precision/recall/F1, not frame-level accuracy.")
    print("Overall:")
    _print_metrics(result["overall"])
    for event_type, metrics in result["by_event_type"].items():
        print(f"\nEvent type: {event_type}")
        _print_metrics(metrics)
    for video, metrics in result["by_video"].items():
        print(f"\nVideo: {video}")
        _print_metrics(metrics)
    if args.json_report:
        print(f"\nDetailed JSON report saved to: {args.json_report}")
    return 0


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_metrics(metrics):
    print(
        "  TP={true_positives} FP={false_positives} FN={false_negatives} "
        "precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}".format(**metrics)
    )


if __name__ == "__main__":
    raise SystemExit(main())
