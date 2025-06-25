import sqlite3
import os

db_path = 'data/stocks.db'

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print('Existing tables:', tables)
    
    # Check each table structure
    for table in tables:
        print(f"\n{table} structure:")
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[1]} {col[2]} {'NOT NULL' if col[3] else ''} {'PRIMARY KEY' if col[5] else ''}")
    
    conn.close()
else:
    print(f"Database {db_path} does not exist")
