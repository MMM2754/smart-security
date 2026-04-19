from storage.database import get_connection
conn = get_connection()

# Get event counts by video
videos = conn.execute('SELECT DISTINCT source_video FROM events').fetchall()
video_names = [row[0] for row in videos]

print('Events per video:')
for video in video_names:
    count = conn.execute('SELECT COUNT(*) FROM events WHERE source_video = ?', (video,)).fetchone()[0]
    print(f'  {video}: {count} events')

# Get processing times
print('\nProcessing times:')
entries = conn.execute('SELECT details FROM audit_trail WHERE action = "video_complete"').fetchall()
for entry in entries:
    details = entry[0]
    parts = details.split(' | ')
    if len(parts) >= 3:
        video = parts[0]
        events = parts[1].split()[0]  # '1 events' -> '1'
        time_taken = parts[2]  # '135.3s'
        print(f'  {video}: {events} events, {time_taken}')

conn.close()