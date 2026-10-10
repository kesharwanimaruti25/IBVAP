import unittest

from fence_geometry import segments_intersect, side_of_line


class FenceGeometryTests(unittest.TestCase):
    def test_side_of_line_reports_opposite_sides(self):
        start = (0, 5)
        end = (10, 5)

        self.assertEqual(side_of_line((4, 2), start, end), -1)
        self.assertEqual(side_of_line((4, 8), start, end), 1)
        self.assertEqual(side_of_line((4, 5), start, end), 0)

    def test_crossing_through_finite_fence_intersects(self):
        self.assertTrue(segments_intersect((5, 0), (5, 10), (0, 5), (10, 5)))

    def test_crossing_extension_beyond_fence_endpoint_does_not_intersect(self):
        self.assertFalse(segments_intersect((15, 0), (15, 10), (0, 5), (10, 5)))

    def test_path_must_cross_the_fence(self):
        self.assertFalse(segments_intersect((1, 1), (2, 2), (0, 5), (10, 5)))

    def test_touching_fence_endpoint_counts_as_intersection(self):
        self.assertTrue(segments_intersect((10, 5), (12, 7), (0, 5), (10, 5)))

    def test_collinear_overlap_counts_as_intersection(self):
        self.assertTrue(segments_intersect((3, 5), (8, 5), (0, 5), (10, 5)))

    def test_zero_length_fence_never_intersects(self):
        self.assertFalse(segments_intersect((0, 0), (10, 10), (5, 5), (5, 5)))


if __name__ == "__main__":
    unittest.main()
