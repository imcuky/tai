"""
Slide File Filter
=================
Filters lecture slide PDF files from cs61a_metadata.db for lecture summary analysis.

This script queries the database for all files whose relative_path contains "slides",
extracting slide content and metadata for lecture topic classification.
"""

import sqlite3
import pandas as pd
import os
import re


def filter_slide_files(db_path="cs61a_metadata.db", output_csv="cs61a_slide_files.csv"):
    """
    Filter lecture slide files from the CS61A metadata database.
    
    Args:
        db_path: Path to the SQLite database file
        output_csv: Output CSV file path for filtered results
    
    Returns:
        pd.DataFrame: Filtered slide files, or None if no results
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
        # Filter records where "relative_path" matches pattern /slidesXX/*.pdf
        # Looking for slide files like "slides01/...", "slides02/...", etc.
        query = """
            SELECT * FROM file 
            WHERE lower(relative_path) LIKE '%slides%'
               OR lower(file_path) LIKE '%slides%'
        """
        
        df = pd.read_sql_query(query, conn)
        print(f"Found {len(df)} slide file records.")
        
        if not df.empty:
            # Display sample of found files
            print("\nSample of slide files found:")
            for idx, row in df.head(5).iterrows():
                rel_path = row.get('relative_path', 'N/A')
                file_name = row.get('file_name', 'N/A')
                print(f"  - {file_name} ({rel_path})")
            
            if len(df) > 5:
                print(f"  ... and {len(df) - 5} more files")
            
            # Extract lecture numbers from paths
            def extract_lecture_number(path):
                """Extract lecture number from path like 'slides01' or 'lec01'"""
                if pd.isna(path):
                    return None
                # Match patterns like slides01, slides1, lec01, lec1
                match = re.search(r'(?:slides|lec)0*(\d+)', str(path), re.IGNORECASE)
                if match:
                    return int(match.group(1))
                return None
            
            df['lecture_number'] = df['relative_path'].apply(extract_lecture_number)
            
            # Show lecture number distribution
            if 'lecture_number' in df.columns and df['lecture_number'].notna().any():
                lec_counts = df['lecture_number'].value_counts().sort_index()
                print(f"\nLecture number distribution:")
                for lec_num, count in lec_counts.head(10).items():
                    print(f"  Lecture {int(lec_num)}: {count} files")
                if len(lec_counts) > 10:
                    print(f"  ... and {len(lec_counts) - 10} more lectures")
            
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
            print("No slide records found in database.")
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
    print("="*60)
    print("CS61A Slide File Filter")
    print("="*60)
    
    result = filter_slide_files()
    
    if result is not None:
        print("\n" + "="*60)
        print("Filter completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Filter failed or no results found.")
        print("="*60)
