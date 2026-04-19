from storage.database import get_connection
conn = get_connection()

# Check what events were detected in the test video
events = conn.execute('SELECT event_type, alert_level, zone_id FROM events WHERE source_video = "test_red_alert.mp4"').fetchall()

print('Events detected in test_red_alert.mp4:')
for event in events:
    print(f'  {event["event_type"]} - {event["alert_level"]} - Zone: {event["zone_id"]}')

# Check most recent events
events = conn.execute('SELECT source_video, event_type, alert_level, zone_id FROM events ORDER BY timestamp DESC LIMIT 5').fetchall()

print('Most recent events:')
for event in events:
    print(f'  {event["source_video"]}: {event["event_type"]} - {event["alert_level"]} - Zone: {event["zone_id"]}')

conn.close()