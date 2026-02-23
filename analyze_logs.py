"""Analyze simulation logs."""
import sqlite3
from pathlib import Path

db_path = Path("data/simulation.db")
if not db_path.exists():
    print("No database found!")
    exit()

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"Tables: {tables}")

# Check behavior_log
try:
    cursor.execute('SELECT COUNT(*) FROM behavior_log')
    count = cursor.fetchone()[0]
    print(f"\nBehavior records: {count}")
    
    if count > 0:
        # Sample rows
        cursor.execute('SELECT step, energy, action, current_lifespan FROM behavior_log LIMIT 10')
        rows = cursor.fetchall()
        print("\nFirst 10 records (step, energy, action, lifespan):")
        for r in rows:
            print(f"  {r}")
        
        # Energy stats
        cursor.execute('SELECT MIN(energy), MAX(energy), AVG(energy) FROM behavior_log')
        stats = cursor.fetchone()
        print(f"\nEnergy stats:")
        print(f"  Min: {stats[0]:.1f}, Max: {stats[1]:.1f}, Avg: {stats[2]:.1f}")
        
        # Action distribution
        cursor.execute('SELECT action, COUNT(*) FROM behavior_log GROUP BY action')
        actions = cursor.fetchall()
        print(f"\nAction distribution:")
        action_names = ['Up', 'Down', 'Left', 'Right', 'Eat', 'Wait']
        for a, c in actions:
            print(f"  {action_names[a]}: {c}")
        
        # Deaths
        cursor.execute('SELECT COUNT(*) FROM behavior_log WHERE energy <= 0')
        deaths = cursor.fetchone()[0]
        print(f"\nDeaths (energy<=0): {deaths}")
        
except Exception as e:
    print(f"Error: {e}")

conn.close()
