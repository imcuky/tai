import sqlite3

db_path = r'c:\Users\cathe\Desktop\tai\rag\file_rearrangement\rearrange\new database\CS 61A_metadata.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

print('Tables:', cur.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall())
print('\nFile table schema:')
for col in cur.execute('PRAGMA table_info(file)').fetchall():
    print(col)

print('\nSample data (first 5 files):')
for row in cur.execute('SELECT file_name, relative_path, description FROM file LIMIT 5').fetchall():
    print(f"File: {row[0]}")
    print(f"  Path: {row[1]}")
    print(f"  Description: {row[2][:100] if row[2] else 'None'}...")
    print()

# Check if relative_path contains the organization structure
print('\nChecking organization structure in paths:')
for row in cur.execute('SELECT relative_path FROM file WHERE relative_path LIKE "%/%/%" LIMIT 10').fetchall():
    print(f"  {row[0]}")

conn.close()
