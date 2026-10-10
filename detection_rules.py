def is_sustained_approach(start_time, start_distance, current_time, current_distance,
                           sustain_seconds, minimum_closing_distance):
    return (
        current_time - start_time >= sustain_seconds
        and start_distance - current_distance >= minimum_closing_distance
    )


def toggle_friendly_track_at(x, y, tracked_person_boxes, friendly_tracks):
    matches = [
        ((x2 - x1) * (y2 - y1), track_id)
        for x1, y1, x2, y2, track_id in tracked_person_boxes
        if x1 <= x <= x2 and y1 <= y <= y2
    ]
    if not matches:
        return None, None
    _, track_id = min(matches)
    is_friendly = track_id not in friendly_tracks
    if is_friendly:
        friendly_tracks.add(track_id)
    else:
        friendly_tracks.remove(track_id)
    return track_id, is_friendly
