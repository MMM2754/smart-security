from storage.database import get_connection
conn = get_connection()

# Get unique completed videos
completed_entries = conn.execute('SELECT details FROM audit_trail WHERE action = "video_complete"').fetchall()
processed_video_names = set()  # Use set to get unique video names
for entry in completed_entries:
    video_name = entry[0].split(' | ')[0]
    processed_video_names.add(video_name)

videos_processed = len(processed_video_names)
videos_with_events = conn.execute('SELECT COUNT(DISTINCT source_video) FROM events').fetchone()[0]

print(f'Videos processed (unique): {videos_processed}')
print(f'Videos with events: {videos_with_events}')
print(f'Safe videos: {videos_processed - videos_with_events}')

print(f'\nProcessed videos: {processed_video_names}')

event_videos = conn.execute('SELECT DISTINCT source_video FROM events').fetchall()
event_video_names = [row[0] for row in event_videos]
safe_videos = list(processed_video_names - set(event_video_names))  # Set difference

print(f'Event videos: {event_video_names}')
print(f'Safe videos: {safe_videos}')

conn.close()