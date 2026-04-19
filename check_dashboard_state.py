from storage.database import get_connection
conn = get_connection()

# Get overall event counts by level
event_counts = conn.execute('SELECT alert_level, COUNT(*) as count FROM events GROUP BY alert_level').fetchall()

print('Current Alert Counts:')
for level, count in event_counts:
    print(f'  {level}: {count} alerts')

# Get videos processed
audit_entries = conn.execute('SELECT details FROM audit_trail WHERE action = "video_complete"').fetchall()
video_set = set()
for entry in audit_entries:
    details = entry[0]
    if ' | ' in details:
        video_name = details.split(' | ')[0]
        video_set.add(video_name)

video_list = list(video_set)

print(f'\nVideos processed: {len(video_list)}')
for video in video_list:
    print(f'  - {video}')

conn.close()