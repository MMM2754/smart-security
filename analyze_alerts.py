from storage.database import get_connection
import json

conn = get_connection()

# Get recent events with descriptions
events = conn.execute('SELECT event_type, alert_level, description, manager_verdict FROM events ORDER BY timestamp DESC LIMIT 10').fetchall()

print('Recent Events Analysis:')
print('=' * 50)
for event in events:
    event_type = event['event_type']
    alert_level = event['alert_level']
    description = event['description']
    manager_verdict = event['manager_verdict']

    print(f'Event Type: {event_type}')
    print(f'Alert Level: {alert_level}')
    print(f'AI Description: {description}')

    if manager_verdict:
        try:
            verdict = json.loads(manager_verdict)
            print(f'Manager Reasoning: {verdict.get("reasoning", "N/A")}')
            print(f'Confidence: {verdict.get("confidence", "N/A")}')
        except:
            print(f'Manager Verdict: {manager_verdict[:100]}...')
    print('-' * 30)

conn.close()