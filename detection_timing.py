def activity_clock(is_video_file, media_time_seconds, wall_time_seconds):
    return media_time_seconds if is_video_file else wall_time_seconds
