from storage.database import get_connection
conn = get_connection()

# Check events from the latest bbaf3s.mpg run
events = conn.execute('SELECT event_type, alert_level FROM events WHERE source_video = "bbaf3s.mpg" ORDER BY timestamp DESC LIMIT 5').fetchall()

print('Latest events from bbaf3s.mpg:')
for event in events:
    print(f'  {event["event_type"]} - {event["alert_level"]}')

conn.close()