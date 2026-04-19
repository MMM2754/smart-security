from storage.database import get_connection
conn = get_connection()

# Get all processed videos
processed = conn.execute('SELECT DISTINCT source_video FROM audit_trail WHERE action = "video_complete"').fetchall()
processed_videos = [row[0] for row in processed]

# Get videos with events
with_events = conn.execute('SELECT DISTINCT source_video FROM events').fetchall()
event_videos = [row[0] for row in with_events]

# Find safe videos (processed but no events)
safe_videos = [v for v in processed_videos if v not in event_videos]

print('All processed videos:')
for v in processed_videos:
    print(f'  - {v}')

print(f'\nVideos with events ({len(event_videos)}):')
for v in event_videos:
    print(f'  - {v}')

print(f'\nSafe videos ({len(safe_videos)}):')
for v in safe_videos:
    print(f'  - {v}')

conn.close()