import unittest

from detection_timing import activity_clock


class ActivityClockTests(unittest.TestCase):
    def test_recorded_video_uses_media_time_for_behavioral_thresholds(self):
        self.assertEqual(activity_clock(True, 3.92, 28.0), 3.92)

    def test_live_camera_uses_elapsed_wall_time(self):
        self.assertEqual(activity_clock(False, 4.0, 28.0), 28.0)


if __name__ == "__main__":
    unittest.main()
