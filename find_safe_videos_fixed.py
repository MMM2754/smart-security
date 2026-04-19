from storage.database import get_connection
conn = get_connection()

# Get all events
events = conn.execute('SELECT source_video, alert_level, COUNT(*) as count FROM events GROUP BY source_video, alert_level').fetchall()

print('Events by video and level:')
for event in events:
    print(f'  {event["source_video"]}: {event["alert_level"]} ({event["count"]} events)')

# Get unique videos with events
with_events = conn.execute('SELECT DISTINCT source_video FROM events').fetchall()
event_videos = [row[0] for row in with_events]

print(f'\nVideos with events ({len(event_videos)}):')
for v in event_videos:
    print(f'  - {v}')

# Get all processed videos from audit trail
processed_entries = conn.execute('SELECT details FROM audit_trail WHERE action = "video_complete"').fetchall()
processed_videos = []
for entry in processed_entries:
    # Extract video name from details like "bbaf3s.mpg | 0 events | 4.2s"
    video_name = entry[0].split(' | ')[0]
    if video_name not in processed_videos:
        processed_videos.append(video_name)

print(f'\nAll processed videos ({len(processed_videos)}):')
for v in processed_videos:
    print(f'  - {v}')

# Find safe videos
safe_videos = [v for v in processed_videos if v not in event_videos]
print(f'\nSafe videos ({len(safe_videos)}):')
for v in safe_videos:
    print(f'  - {v}')

conn.close()