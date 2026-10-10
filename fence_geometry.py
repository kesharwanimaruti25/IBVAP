def side_of_line(point, line_start, line_end):
    px, py = point
    ax, ay = line_start
    bx, by = line_end
    value = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def segments_intersect(first_start, first_end, second_start, second_end):
    if second_start == second_end:
        return False

    first_start_side = side_of_line(first_start, second_start, second_end)
    first_end_side = side_of_line(first_end, second_start, second_end)
    second_start_side = side_of_line(second_start, first_start, first_end)
    second_end_side = side_of_line(second_end, first_start, first_end)

    if first_start_side * first_end_side < 0 and second_start_side * second_end_side < 0:
        return True

    return (
        first_start_side == 0 and _within_bounds(first_start, second_start, second_end)
        or first_end_side == 0 and _within_bounds(first_end, second_start, second_end)
        or second_start_side == 0 and _within_bounds(second_start, first_start, first_end)
        or second_end_side == 0 and _within_bounds(second_end, first_start, first_end)
    )


def _within_bounds(point, segment_start, segment_end):
    px, py = point
    x1, y1 = segment_start
    x2, y2 = segment_end
    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)
