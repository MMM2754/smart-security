from storage.database import get_connection
conn = get_connection()

# Get recent video processing times
entries = conn.execute('SELECT details FROM audit_trail WHERE action = "video_complete" ORDER BY timestamp DESC LIMIT 5').fetchall()

print('Recent video processing times:')
for entry in entries:
    details = entry[0]
    print(f'  {details}')

conn.close()