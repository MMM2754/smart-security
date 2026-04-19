from storage.database import get_connection
conn = get_connection()

# Get all audit trail entries
audit_entries = conn.execute('SELECT * FROM audit_trail WHERE action = "video_complete"').fetchall()

print('Audit trail entries for video_complete:')
for entry in audit_entries:
    print(f'  ID: {entry["id"]}, Timestamp: {entry["timestamp"]}, Action: {entry["action"]}, Details: {entry["details"]}, Source: {entry["source"]}')

# Get videos with events
with_events = conn.execute('SELECT DISTINCT source_video FROM events').fetchall()
event_videos = [row[0] for row in with_events]

print(f'\nVideos with events ({len(event_videos)}):')
for v in event_videos:
    print(f'  - {v}')

conn.close()