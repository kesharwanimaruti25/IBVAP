import csv
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from annotate_events import annotate, append_label, prepare_labels_file


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

    def test_keyboard_annotation_saves_the_displayed_video_timestamp(self):
        class Frame:
            shape = (360, 640, 3)

            def copy(self):
                return Frame()

        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: {
            5: 25.0,
            0: 4_000.0,
        }.get(prop, 0.0)
        capture.read.return_value = (True, Frame())

        cv2 = types.ModuleType("cv2")
        cv2.CAP_PROP_FPS = 5
        cv2.CAP_PROP_POS_MSEC = 0
        cv2.WINDOW_NORMAL = 0
        cv2.LINE_AA = 0
        cv2.FONT_HERSHEY_SIMPLEX = 0
        cv2.VideoCapture = Mock(return_value=capture)
        cv2.namedWindow = Mock()
        cv2.getTextSize = Mock(return_value=((100, 20), 0))
        cv2.putText = Mock()
        cv2.imshow = Mock()
        cv2.waitKey = Mock(side_effect=[ord("1"), ord("q")])
        cv2.destroyAllWindows = Mock()

        with tempfile.TemporaryDirectory() as root:
            video_path = os.path.join(root, "clip.mp4")
            output_path = os.path.join(root, "labels.csv")
            with open(video_path, "wb") as file:
                file.write(b"video placeholder")

            with patch.dict(sys.modules, {"cv2": cv2}):
                exit_code = annotate(video_path, output_path)

            with open(output_path, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(exit_code, 0)
        self.assertEqual(rows[0]["video"], "clip.mp4")
        self.assertEqual(rows[0]["event_type"], "crossing")
        self.assertEqual(rows[0]["time_seconds"], "4.000")
        self.assertEqual(cv2.waitKey.call_count, 2)
        capture.release.assert_called_once()
        cv2.destroyAllWindows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
