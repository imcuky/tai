import sqlite3
import pandas as pd
import os
from pathlib import Path

def analyze_metadata_db(db_path):
    """Analyze the metadata.db structure and content"""
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Available tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Analyze the 'file' table structure
        print("\nFile table structure:")
        cursor.execute("PRAGMA table_info(file);")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # Check unique course codes
        print("\nAvailable course codes:")
        cursor.execute("SELECT DISTINCT course_code, COUNT(*) as count FROM file GROUP BY course_code;")
        courses = cursor.fetchall()
        for course, count in courses:
            print(f"  - {course}: {count} files")
        
        # Sample CS 61A data
        print("\nSample CS 61A files:")
        cursor.execute("SELECT file_name, lecture_number, topics FROM file WHERE course_code = 'CS 61A' LIMIT 5;")
        samples = cursor.fetchall()
        for sample in samples:
            print(f"  - {sample}")
            
    except sqlite3.Error as e:
        print(f"ERROR: Database error: {e}")
    finally:
        conn.close()

def filter_cs61a_data(source_db_path, target_db_path):
    """Create a new database with only CS 61A data"""
    
    if not os.path.exists(source_db_path):
        print(f"ERROR: Source database not found at: {source_db_path}")
        return False
    
    # Connect to source database
    source_conn = sqlite3.connect(source_db_path)
    source_cursor = source_conn.cursor()
    
    # Create target database
    target_conn = sqlite3.connect(target_db_path)
    target_cursor = target_conn.cursor()
    
    try:
        # Get the file table schema
        source_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='file';")
        schema = source_cursor.fetchone()
        
        if not schema:
            print("ERROR: 'file' table not found in source database")
            return False
        
        # Create the file table in target database
        target_cursor.execute(schema[0])
        
        # Copy CS 61A data
        source_cursor.execute("SELECT * FROM file WHERE course_code = 'CS 61A';")
        cs61a_data = source_cursor.fetchall()
        
        if not cs61a_data:
            print("ERROR: No CS 61A data found")
            return False
        
        # Get column count for placeholders
        source_cursor.execute("PRAGMA table_info(file);")
        columns = source_cursor.fetchall()
        placeholders = ','.join(['?' for _ in columns])
        
        # Insert CS 61A data
        target_cursor.executemany(f"INSERT INTO file VALUES ({placeholders})", cs61a_data)
        
        target_conn.commit()
        
        print(f"SUCCESS: Created filtered database with {len(cs61a_data)} CS 61A files")
        print(f"Saved to: {target_db_path}")
        
        return True
        
    except sqlite3.Error as e:
        print(f"ERROR: Database error: {e}")
        return False
    finally:
        source_conn.close()
        target_conn.close()

def export_cs61a_to_csv(db_path, csv_path):
    """Export CS 61A data to CSV for easier analysis"""
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Read CS 61A data into DataFrame
        df = pd.read_sql_query("SELECT * FROM file WHERE course_code = 'CS 61A'", conn)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        
        print(f"SUCCESS: Exported {len(df)} CS 61A records to CSV")
        print(f"Saved to: {csv_path}")
        
        # Show summary statistics
        print(f"\nCS 61A Data Summary:")
        print(f"  - Total files: {len(df)}")
        if 'lecture_number' in df.columns:
            print(f"  - Lecture range: {df['lecture_number'].min()} - {df['lecture_number'].max()}")
        if 'file_type' in df.columns:
            print(f"  - File types: {df['file_type'].value_counts().to_dict()}")
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"ERROR: Export error: {e}")
        return None

def find_metadata_db():
    """Find the metadata.db file in the project"""
    
    # Possible locations for metadata.db in the tai project
    possible_paths = [
        "../../metadata.db",                              # tai/metadata.db
        "../../ai_chatbot_backend/metadata.db",           # tai/ai_chatbot_backend/metadata.db
        "../../ai_chatbot_backend/data/metadata.db",      # tai/ai_chatbot_backend/data/metadata.db
        "../../../metadata.db",                           # in case of different structure
        "metadata.db"                                      # current directory
    ]
    
    print("Searching for metadata.db in tai project...")
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        print(f"Checking: {abs_path}")
        if os.path.exists(abs_path):
            print(f"FOUND: {abs_path}")
            return abs_path
    
    print("\nmetadata.db not found in expected locations.")
    print("Please check if the database exists by running:")
    print("  1. cd ../../ai_chatbot_backend")
    print("  2. make db-init  (or poetry run python scripts/initialize_db_and_files.py)")
    return None

if __name__ == "__main__":
    # Find the correct path to metadata.db
    metadata_db_path = find_metadata_db()
    
    if not metadata_db_path:
        print("\nERROR: Could not locate metadata.db")
        print("Please ensure the database has been created by running the backend initialization.")
        exit(1)
    
    # Output paths
    filtered_db_path = "./cs61a_metadata.db"
    csv_path = "./cs61a_files.csv"
    
    print(f"\nUsing database: {metadata_db_path}")
    print("Analyzing metadata database...")
    analyze_metadata_db(metadata_db_path)
    
    print("\nFiltering CS 61A data...")
    success = filter_cs61a_data(metadata_db_path, filtered_db_path)
    
    if success:
        print("\nExporting to CSV...")
        df = export_cs61a_to_csv(filtered_db_path, csv_path)
        
        if df is not None:
            print("\nDatabase filtering complete!")
            print(f"Filtered DB: {filtered_db_path}")
            print(f"CSV Export: {csv_path}")