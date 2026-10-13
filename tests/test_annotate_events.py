import csv
import os
import tempfile
import unittest

from annotate_events import append_label, prepare_labels_file


class AnnotateEventsTests(unittest.TestCase):
    def test_creates_csv_header_and_writes_a_label_without_guessing_optional_fields(self):
        with tempfile.TemporaryDirectory() as root:
            video_path = os.path.join(root, "clip.mp4")
            output_path = os.path.join(root, "labels.csv")
            with open(video_path, "wb") as file:
                file.write(b"video placeholder")

            prepare_labels_file(output_path, video_path)
            row = append_label(output_path, "clip.mp4", "crossing", 3.14159)

            with open(output_path, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(row["time_seconds"], "3.142")
        self.assertEqual(rows[0]["event_type"], "crossing")
        self.assertEqual(rows[0]["direction"], "")
        self.assertEqual(rows[0]["event_class"], "")

    def test_existing_file_requires_explicit_append_and_expected_header(self):
        with tempfile.TemporaryDirectory() as root:
            video_path = os.path.join(root, "clip.mp4")
            output_path = os.path.join(root, "labels.csv")
            with open(video_path, "wb") as file:
                file.write(b"video placeholder")
            with open(output_path, "w", encoding="utf-8") as file:
                file.write("other,data\n")

            with self.assertRaises(FileExistsError):
                prepare_labels_file(output_path, video_path)
            with self.assertRaises(ValueError):
                prepare_labels_file(output_path, video_path, append=True)

    def test_labels_path_cannot_overwrite_source_video(self):
        with tempfile.TemporaryDirectory() as root:
            video_path = os.path.join(root, "clip.mp4")
            with open(video_path, "wb") as file:
                file.write(b"video placeholder")

            with self.assertRaises(ValueError):
                prepare_labels_file(video_path, video_path)


if __name__ == "__main__":
    unittest.main()
