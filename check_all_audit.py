from storage.database import get_connection
conn = get_connection()

# Get all audit trail entries
all_audit = conn.execute('SELECT * FROM audit_trail ORDER BY timestamp').fetchall()

print('All audit trail entries:')
for entry in all_audit:
    print(f'  ID: {entry["id"]}, Timestamp: {entry["timestamp"]}, Action: {entry["action"]}, Details: {entry["details"]}')

# Get unique videos from all audit entries that mention videos
processed_videos = set()
for entry in all_audit:
    if entry["details"] and " | " in entry["details"]:
        video_name = entry["details"].split(' | ')[0]
        processed_videos.add(video_name)

print(f'\nUnique videos found in audit trail ({len(processed_videos)}):')
for v in sorted(processed_videos):
    print(f'  - {v}')

# Get videos with events
with_events = conn.execute('SELECT DISTINCT source_video FROM events').fetchall()
event_videos = [row[0] for row in with_events]

print(f'\nVideos with events ({len(event_videos)}):')
for v in event_videos:
    print(f'  - {v}')

# Find safe videos
safe_videos = [v for v in processed_videos if v not in event_videos]
print(f'\nSafe videos ({len(safe_videos)}):')
for v in safe_videos:
    print(f'  - {v}')

conn.close()