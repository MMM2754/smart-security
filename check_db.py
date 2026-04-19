from storage.database import get_connection
conn = get_connection()
videos_processed = conn.execute('SELECT COUNT(*) FROM audit_trail WHERE action = "video_complete"').fetchone()[0]
videos_with_events = conn.execute('SELECT COUNT(DISTINCT source_video) FROM events').fetchone()[0]
print(f'Videos processed: {videos_processed}')
print(f'Videos with events: {videos_with_events}')
print(f'Safe videos: {videos_processed - videos_with_events}')
conn.close()