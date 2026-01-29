"""
YouTube File Filter
===================
Filters YouTube video files from cs61a_metadata.db for lecture summary analysis.

This script queries the database for all files whose relative_path contains "youtube",
extracting video transcripts and metadata for lecture topic classification.
"""

import sqlite3
import pandas as pd
import os


def filter_youtube_files(db_path="cs61a_metadata.db", output_csv="cs61a_youtube_files.csv"):
    """
    Filter YouTube video files from the CS61A metadata database.
    
    Args:
        db_path: Path to the SQLite database file
        output_csv: Output CSV file path for filtered results
    
    Returns:
        pd.DataFrame: Filtered YouTube files, or None if no results
    """
    # Path resolution for DB - check multiple locations
    original_db_path = db_path
    if not os.path.exists(db_path):
        # Try parent directories
        for levels_up in range(1, 4):
            potential_path = os.path.join("../" * levels_up, original_db_path)
            if os.path.exists(potential_path):
                db_path = potential_path
                break
    
    if not os.path.exists(db_path):
        print(f"Error: Database file '{original_db_path}' not found.")
        print(f"Searched locations:")
        print(f"  - Current directory: {os.path.abspath(original_db_path)}")
        for levels_up in range(1, 4):
            print(f"  - {levels_up} level(s) up: {os.path.abspath(os.path.join('../' * levels_up, original_db_path))}")
        return None

    print(f"Connecting to database: {db_path}")
    print(f"Database absolute path: {os.path.abspath(db_path)}")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Filter records where "relative_path" contains "youtube"
        # Also check url field to catch any youtube content
        query = """
            SELECT * FROM file 
            WHERE lower(relative_path) LIKE '%youtube%' 
               OR lower(url) LIKE '%youtube%'
               OR lower(url) LIKE '%youtu.be%'
        """
        
        df = pd.read_sql_query(query, conn)
        print(f"Found {len(df)} YouTube video records.")
        
        if not df.empty:
            # Display sample of found files
            print("\nSample of YouTube files found:")
            for idx, row in df.head(3).iterrows():
                rel_path = row.get('relative_path', 'N/A')
                file_name = row.get('file_name', 'N/A')
                print(f"  - {file_name} ({rel_path})")
            
            if len(df) > 3:
                print(f"  ... and {len(df) - 3} more files")
            
            # Create output directory
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_script_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to CSV in output folder
            final_output_path = os.path.join(output_dir, os.path.basename(output_csv))
            df.to_csv(final_output_path, index=False)
            print(f"\nSaved filtered results to: {final_output_path}")
            print(f"Output absolute path: {os.path.abspath(final_output_path)}")
            
            return df
        else:
            print("No YouTube records found in database.")
            print("\nQuery used:")
            print(query)
            return None
            
    except Exception as e:
        print(f"Error executing query: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        conn.close()


if __name__ == "__main__":
    result = filter_youtube_files()
    if result is not None:
        print(f"\nSuccess! Filtered {len(result)} YouTube files.")
