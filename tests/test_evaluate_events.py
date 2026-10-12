import csv
import json
import os
import tempfile
import unittest

from evaluate_events import evaluate, load_labels, load_predictions, main


class EvaluateEventsTests(unittest.TestCase):
    def setUp(self):
        self.labels = [
            {"video": "clip.mp4", "event_type": "crossing", "time_seconds": 10.0, "direction": "entering"},
            {"video": "clip.mp4", "event_type": "crossing", "time_seconds": 20.0, "direction": "exiting"},
        ]

    def test_finds_maximum_one_to_one_matches_within_tolerance(self):
        predictions = [
            {"video": "clip.mp4", "event_type": "crossing", "time_seconds": 10.5, "direction": "entering"},
            {"video": "clip.mp4", "event_type": "crossing", "time_seconds": 10.8, "direction": "entering"},
            {"video": "clip.mp4", "event_type": "crossing", "time_seconds": 19.0, "direction": "exiting"},
        ]

        result = evaluate(self.labels, predictions, tolerance_seconds=1.0)

        self.assertEqual(result["overall"]["true_positives"], 2)
        self.assertEqual(result["overall"]["false_positives"], 1)
        self.assertEqual(result["overall"]["false_negatives"], 0)
        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(len(result["false_positives"]), 1)
        self.assertEqual(result["false_negatives"], [])
        self.assertAlmostEqual(result["matches"][0]["time_error_seconds"], 0.5)
        self.assertEqual(result["by_video"]["clip.mp4"]["false_positives"], 1)
        self.assertEqual(result["by_video"]["clip.mp4"]["false_negatives"], 0)

    def test_json_report_records_source_digests_and_event_level_errors(self):
        with tempfile.TemporaryDirectory() as root:
            labels_path = os.path.join(root, "labels.csv")
            events_path = os.path.join(root, "events.jsonl")
            report_path = os.path.join(root, "report.json")
            with open(labels_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["video", "event_type", "time_seconds", "direction", "event_class"])
                writer.writerow(["clip.mp4", "crossing", "10", "entering", "Person"])
                writer.writerow(["clip.mp4", "crossing", "20", "exiting", "Person"])
            with open(events_path, "w", encoding="utf-8") as file:
                file.write(json.dumps({
                    "source": "clip.mp4",
                    "type": "crossing",
                    "media_time_seconds": 10.5,
                    "direction": "entering",
                    "class": "Person",
                }) + "\n")
                file.write(json.dumps({
                    "source": "clip.mp4",
                    "type": "crossing",
                    "media_time_seconds": 30,
                    "direction": "entering",
                    "class": "Person",
                }) + "\n")

            exit_code = main([
                "--labels", labels_path,
                "--events", events_path,
                "--tolerance", "1",
                "--json-report", report_path,
            ])

            with open(report_path, encoding="utf-8") as file:
                report = json.load(file)

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["method"]["matching"], "one-to-one")
        self.assertEqual(report["method"]["tolerance_seconds"], 1.0)
        self.assertEqual(len(report["method"]["labels_sha256"]), 64)
        self.assertEqual(len(report["method"]["events_sha256"]), 64)
        self.assertEqual(len(report["method"]["evaluator_sha256"]), 64)
        self.assertEqual(report["overall"]["true_positives"], 1)
        self.assertEqual(report["overall"]["false_positives"], 1)
        self.assertEqual(report["overall"]["false_negatives"], 1)
        self.assertEqual(report["by_video"]["clip.mp4"]["f1"], 0.5)
        self.assertEqual(report["matches"][0]["label"]["time_seconds"], 10.0)
        self.assertEqual(report["false_negatives"][0]["time_seconds"], 20.0)
        self.assertEqual(report["false_positives"][0]["time_seconds"], 30.0)

    def test_json_report_cannot_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as root:
            labels_path = os.path.join(root, "labels.csv")
            events_path = os.path.join(root, "events.jsonl")
            with open(labels_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["video", "event_type", "time_seconds"])
                writer.writerow(["clip.mp4", "crossing", "10"])
            with open(events_path, "w", encoding="utf-8") as file:
                file.write("{}\n")

            exit_code = main([
                "--labels", labels_path,
                "--events", events_path,
                "--json-report", events_path,
            ])

            with open(events_path, encoding="utf-8") as file:
                input_contents = file.read()

        self.assertEqual(exit_code, 2)
        self.assertEqual(input_contents, "{}\n")

    def test_requires_matching_video_type_and_optional_direction(self):
        predictions = [
            {"video": "other.mp4", "event_type": "crossing", "time_seconds": 10.0, "direction": "entering"},
            {"video": "clip.mp4", "event_type": "crossing", "time_seconds": 20.0, "direction": "entering"},
            {"video": "clip.mp4", "event_type": "loitering", "time_seconds": 10.0, "direction": ""},
        ]

        result = evaluate(self.labels, predictions, tolerance_seconds=2.0)

        self.assertEqual(result["overall"]["true_positives"], 0)
        self.assertEqual(result["overall"]["false_positives"], 3)
        self.assertEqual(result["overall"]["false_negatives"], 2)

    def test_can_require_matching_event_class(self):
        label = [{
            "video": "clip.mp4", "event_type": "crossing", "time_seconds": 10.0,
            "direction": "entering", "event_class": "Person",
        }]
        predictions = [{
            "video": "clip.mp4", "event_type": "crossing", "time_seconds": 10.0,
            "direction": "entering", "event_class": "Car",
        }]

        result = evaluate(label, predictions)

        self.assertEqual(result["overall"]["true_positives"], 0)
        self.assertEqual(result["overall"]["false_positives"], 1)
        self.assertEqual(result["overall"]["false_negatives"], 1)

    def test_empty_data_has_zero_metrics(self):
        result = evaluate([], [])

        self.assertEqual(result["overall"]["f1"], 0)
        self.assertEqual(result["overall"]["precision"], 0)
        self.assertEqual(result["overall"]["recall"], 0)

    def test_negative_tolerance_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([], [], tolerance_seconds=-1)

    def test_one_prediction_cannot_match_multiple_labels(self):
        labels = [
            {"video": "clip.mp4", "event_type": "crossing", "time_seconds": 10.0},
            {"video": "clip.mp4", "event_type": "crossing", "time_seconds": 10.2},
        ]
        prediction = [
            {"video": "clip.mp4", "event_type": "crossing", "time_seconds": 10.1},
        ]

        result = evaluate(labels, prediction, tolerance_seconds=1.0)

        self.assertEqual(result["overall"]["true_positives"], 1)
        self.assertEqual(result["overall"]["false_negatives"], 1)

    def test_label_csv_requires_valid_columns_and_timestamp(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "labels.csv")
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["video", "event_type", "time_seconds", "direction"])
                writer.writerow(["clip.mp4", "crossing", "3.5", "entering"])

            labels = load_labels(path)

        self.assertEqual(labels[0]["video"], "clip.mp4")
        self.assertEqual(labels[0]["time_seconds"], 3.5)

    def test_predictions_without_video_relative_time_are_counted_unscorable(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "events.jsonl")
            with open(path, "w", encoding="utf-8") as file:
                file.write(json.dumps({"type": "crossing", "source": "clip.mp4"}) + "\n")
                file.write(json.dumps({
                    "type": "crossing",
                    "source": "/videos/clip.mp4",
                    "media_time_seconds": 4.5,
                }) + "\n")

            predictions, unscorable = load_predictions(path)

        self.assertEqual(unscorable, 1)
        self.assertEqual(predictions[0]["video"], "clip.mp4")
        self.assertEqual(predictions[0]["time_seconds"], 4.5)


if __name__ == "__main__":
    unittest.main()
