import unittest

from detection_rules import is_sustained_approach, toggle_friendly_track_at


class DetectionRulesTests(unittest.TestCase):
    def test_nearby_stationary_people_do_not_count_as_converging(self):
        self.assertFalse(is_sustained_approach(1.0, 100, 5.0, 100, 2.0, 40))

    def test_people_must_close_distance_for_the_required_duration(self):
        self.assertFalse(is_sustained_approach(1.0, 140, 2.5, 90, 2.0, 40))
        self.assertTrue(is_sustained_approach(1.0, 140, 3.1, 90, 2.0, 40))

    def test_small_distance_fluctuations_do_not_count_as_converging(self):
        self.assertFalse(is_sustained_approach(1.0, 100, 4.0, 75, 2.0, 40))

    def test_click_toggles_the_smallest_matching_person_box_as_friendly(self):
        boxes = [(0, 0, 100, 150, 1), (20, 20, 60, 100, 2)]
        friendly_tracks = set()

        self.assertEqual(toggle_friendly_track_at(30, 30, boxes, friendly_tracks), (2, True))
        self.assertIn(2, friendly_tracks)
        self.assertEqual(toggle_friendly_track_at(30, 30, boxes, friendly_tracks), (2, False))
        self.assertNotIn(2, friendly_tracks)

    def test_click_outside_person_boxes_does_not_change_friendly_tracks(self):
        friendly_tracks = {2}

        self.assertEqual(toggle_friendly_track_at(150, 150, [(0, 0, 100, 100, 1)], friendly_tracks), (None, None))
        self.assertEqual(friendly_tracks, {2})


if __name__ == "__main__":
    unittest.main()
