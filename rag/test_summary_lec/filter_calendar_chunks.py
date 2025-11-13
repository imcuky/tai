import sqlite3
import pandas as pd
import os
from pathlib import Path

def find_metadata_db():
    """Find the metadata.db file in the project"""
    
    possible_paths = [
        "../../metadata.db",                              # tai/metadata.db
        "../../ai_chatbot_backend/metadata.db",           # tai/ai_chatbot_backend/metadata.db
        "../../ai_chatbot_backend/data/metadata.db",      # tai/ai_chatbot_backend/data/metadata.db
        "../../../metadata.db",                           # in case of different structure
        "metadata.db"                                      # current directory
    ]
    
    print("Searching for metadata.db...")
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"FOUND: {abs_path}")
            return abs_path
    
    print("metadata.db not found in expected locations.")
    return None

def analyze_chunks_table(db_path):
    """Analyze the chunks table structure"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if chunks table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks';")
        if not cursor.fetchone():
            print("ERROR: 'chunks' table not found")
            return
        
        # Analyze chunks table structure
        print("CHUNKS table structure:")
        cursor.execute("PRAGMA table_info(chunks);")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # Count total chunks
        cursor.execute("SELECT COUNT(*) FROM chunks;")
        total_chunks = cursor.fetchone()[0]
        print(f"\nTotal chunks: {total_chunks}")
        
        # Check available course codes
        cursor.execute("SELECT DISTINCT course_code, COUNT(*) FROM chunks GROUP BY course_code;")
        courses = cursor.fetchall()
        print("\nAvailable course codes in chunks:")
        for course, count in courses:
            print(f"  - {course}: {count} chunks")
        
        # Check reference_path patterns
        cursor.execute("SELECT DISTINCT reference_path FROM chunks WHERE reference_path LIKE '%Calendar%' LIMIT 10;")
        calendar_paths = cursor.fetchall()
        print("\nSample Calendar reference paths:")
        for path in calendar_paths:
            print(f"  - {path[0]}")
            
    except sqlite3.Error as e:
        print(f"ERROR: {e}")
    finally:
        conn.close()

def filter_calendar_chunks(db_path, course_code, output_csv=None):
    """Filter chunks by course code where reference_path contains 'Calendar'"""
    
    conn = sqlite3.connect(db_path)
    
    try:
        # SQL query to filter Calendar chunks for specific course
        query = """
        SELECT text,title,url,file_path,reference_path,course_name,course_code,chunk_index
        FROM chunks
        WHERE course_code = ? 
        AND reference_path LIKE '%> Calendar%'
        """
        
        print(f"\nFiltering chunks for course: {course_code}")
        print("Condition: reference_path contains 'Calendar'")
        
        df = pd.read_sql_query(query, conn, params=[course_code])
        
        if len(df) == 0:
            print(f"No Calendar chunks found for course: {course_code}")
            return None
        
        print(f"Found {len(df)} Calendar chunks for {course_code}")
        
        # Show summary
        print(f"\nSummary:")
        print(f"  - Total chunks: {len(df)}")
        print(f"  - Unique files: {df['file_uuid'].nunique() if 'file_uuid' in df.columns else 'N/A'}")
        print(f"  - Unique reference paths: {df['reference_path'].nunique()}")
        
        # Show sample reference paths
        print(f"\nUnique Calendar reference paths:")
        unique_paths = df['reference_path'].unique()
        for i, path in enumerate(unique_paths[:10]):  # Show first 10
            print(f"  {i+1}. {path}")
        if len(unique_paths) > 10:
            print(f"  ... and {len(unique_paths) - 10} more")
        
        df.drop('vector', axis=1, inplace=True, errors='ignore')  # Drop vector column if exists
        # Save to CSV if specified
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nSaved to: {output_csv}")
        
        return df
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    finally:
        conn.close()

def extract_calendar_text(df):
    """Extract and concatenate text from Calendar chunks"""
    
    if df is None or len(df) == 0:
        return None
    
    # Group by reference_path and concatenate text
    calendar_texts = {}
    
    for path in df['reference_path'].unique():
        path_chunks = df[df['reference_path'] == path].sort_values('chunk_uuid')
        
        if 'text' in path_chunks.columns:
            # Concatenate all chunk texts for this path
            full_text = ' '.join(path_chunks['text'].dropna().astype(str))
            calendar_texts[path] = full_text
        else:
            calendar_texts[path] = "No text column available"
    
    return calendar_texts

def main():
    """Main function to filter Calendar chunks by course code"""
    
    print("Calendar Chunks Filter")
    print("=" * 30)
    
    # Find database
    db_path = find_metadata_db()
    if not db_path:
        print("ERROR: Could not find metadata.db")
        return
    
    # Analyze chunks table
    print("\nAnalyzing chunks table...")
    analyze_chunks_table(db_path)
    
 
    course_code = "CS 61A"
    # Filter Calendar chunks
    output_csv = f"{course_code.replace(' ', '_').lower()}_calendar_chunks.csv"
    df = filter_calendar_chunks(db_path, course_code, output_csv)
    
    if df is not None:
        # Extract calendar texts
        print("\nExtracting calendar texts...")
        calendar_texts = extract_calendar_text(df)
        
        if calendar_texts:
            print(f"\nCalendar content preview:")
            for i, (path, text) in enumerate(calendar_texts.items()):
                print(f"\n{i+1}. {path}")
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"   {preview}")
                
                if i >= 2:  # Show only first 3
                    remaining = len(calendar_texts) - 3
                    if remaining > 0:
                        print(f"\n   ... and {remaining} more calendar files")
                    break
        
        print(f"\nResults saved to: {output_csv}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()