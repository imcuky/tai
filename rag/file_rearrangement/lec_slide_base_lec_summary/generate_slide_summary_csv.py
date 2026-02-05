"""
Slide-Based Lecture Summary Generator
======================================
Simplified pipeline that:
1. Filters slide files from cs61a_metadata.db
2. Aggregates files by lecture number  
3. Generates lecture summary CSV with file concepts and descriptions

This replaces the old filter_calendar_chunks.py + lec_info_csv.py workflow.
"""

import sqlite3
import pandas as pd
import os
import re
import json


def _safe_json_or_literal_load(s):
    """Parse JSON or Python literal string safely."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    if not isinstance(s, str):
        return s
    st = s.strip()
    if not st:
        return None
    try:
        return json.loads(st)
    except Exception:
        try:
            import ast
            return ast.literal_eval(st)
        except Exception:
            return None


def extract_lecture_number(path):
    """Extract lecture number from path like 'slides01', 'lec01', 'lecture/lec01/textbook' or 'disc01'"""
    if pd.isna(path):
        return None
    # Match patterns like slides01, slides1, lec01, lec1, lecture/lec01/textbook, disc01
    match = re.search(r'(?:slides|lec|disc)0*(\d+)', str(path), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def is_slide_file(path):
    """Check if path is a slide PDF file"""
    if pd.isna(path):
        return False
    path_lower = str(path).lower()
    return 'slides' in path_lower and '.pdf' in path_lower

def get_file_type(path):
    """Classify file type based on path for prioritization (slides > textbook > youtube)"""
    if pd.isna(path): 
        return 'other'
    path_lower = str(path).lower()
    
    if 'slides' in path_lower and '.pdf' in path_lower:
        return 'slide'
    elif 'textbook' in path_lower:
        return 'textbook'
    elif 'youtube' in path_lower:
        return 'youtube'
    return 'other'



def parse_sections_key_concepts(sections_raw):
    """Parse key_concepts from sections field."""
    parsed = _safe_json_or_literal_load(sections_raw)
    concepts = []
    seen = set()
    
    if isinstance(parsed, list):
        for sec in parsed:
            if isinstance(sec, dict):
                # Get key_concept
                kc = sec.get('key_concept')
                if isinstance(kc, str) and kc.strip():
                    val = kc.strip()
                    if val.lower() not in seen:
                        seen.add(val.lower())
                        concepts.append(val)
                elif isinstance(kc, list):
                    for item in kc:
                        if isinstance(item, str) and item.strip():
                            val = item.strip()
                            if val.lower() not in seen:
                                seen.add(val.lower())
                                concepts.append(val)
    
    return concepts


def generate_slide_lecture_summary(
    db_path="cs61a_metadata.db",
    output_csv="cs_61a_lecture_summary.csv"
):
    """
    Generate lecture summary CSV directly from database slide files.
    
    Args:
        db_path: Path to SQLite database
        output_csv: Output CSV file path
    
    Returns:
        pd.DataFrame: Lecture summary DataFrame, or None on error
    """
    # Path resolution for DB
    original_db_path = db_path
    if not os.path.exists(db_path):
        for levels_up in range(1, 4):
            potential_path = os.path.join("../" * levels_up, original_db_path)
            if os.path.exists(potential_path):
                db_path = potential_path
                break
    
    if not os.path.exists(db_path):
        print(f"Error: Database file '{original_db_path}' not found.")
        return None

    print(f"Connecting to database: {db_path}")
    print(f"Database absolute path: {os.path.abspath(db_path)}")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Filter files: Slides, Textbook, YouTube, and fallback to any lecture file
        # Prioritization: Slides > Textbook > YouTube > Other (Discussion/Misc)
        query = """
            SELECT file_name, relative_path, sections, description
            FROM file 
            WHERE lower(relative_path) LIKE '%slides%'
               OR lower(relative_path) LIKE '%lec%'
               OR lower(relative_path) LIKE '%youtube%'
               OR lower(relative_path) LIKE '%disc%'
        """
        
        try:
            df = pd.read_sql_query(query, conn)
        except Exception:
            # Fallback if specific columns fail
            query = """
                SELECT * FROM file 
                WHERE lower(relative_path) LIKE '%slides%'
                   OR lower(relative_path) LIKE '%lec%'
                   OR lower(relative_path) LIKE '%youtube%'
                   OR lower(relative_path) LIKE '%disc%'
            """
            df = pd.read_sql_query(query, conn)

        print(f"Found {len(df)} candidate records (broad search).")

        
        if df.empty:
            print("No matching files found in database.")
            return None
        
        # Extract lecture numbers
        df['lecture_number'] = df['relative_path'].apply(extract_lecture_number)
        df['file_type'] = df['relative_path'].apply(get_file_type)
        
        # Filter out files without lecture number
        df = df[df['lecture_number'].notna()].copy()
        print(f"Filtered to {len(df)} files with valid lecture numbers.")
        
        if df.empty:
            print("No files with valid lecture numbers found.")
            return None
        
        # Parse key concepts from sections
        df['key_concepts'] = df['sections'].apply(parse_sections_key_concepts)
        
        # Group by lecture number and apply prioritization
        lecture_groups = df.groupby('lecture_number')
        
        lecture_summaries = []
        
        for lec_num, group in lecture_groups:
            # Check for availability in priority order
            slide_files = group[group['file_type'] == 'slide']
            textbook_files = group[group['file_type'] == 'textbook']
            youtube_files = group[group['file_type'] == 'youtube']
            other_files = group[group['file_type'] == 'other']
            
            source_group = pd.DataFrame()
            source_type = "unknown"
            
            if not slide_files.empty:
                source_group = slide_files
                source_type = "slides"
            elif not textbook_files.empty:
                source_group = textbook_files
                source_type = "textbook"
            elif not youtube_files.empty:
                source_group = youtube_files
                source_type = "youtube"
            elif not other_files.empty:
                source_group = other_files
                source_type = "other/discussion"
            else:
                print(f"Lecture {int(lec_num)}: No valid source files found (skipped).")
                continue
            
            print(f"Lecture {int(lec_num)}: Using {source_type} ({len(source_group)} files)")
            
            # Aggregate key concepts from the selected source
            all_concepts = []

            for concepts_list in source_group['key_concepts']:
                if isinstance(concepts_list, list):
                    all_concepts.extend(concepts_list)
            
            # Deduplicate concepts
            unique_concepts = []
            seen = set()
            for concept in all_concepts:
                if concept.lower() not in seen:
                    seen.add(concept.lower())
                    unique_concepts.append(concept)
            
            # Build per-file concept and description maps
            file_concepts_map = {}
            file_descriptions_map = {}
            
            for _, row in source_group.iterrows():
                fname = row['file_name']
                concepts = row['key_concepts'] if isinstance(row['key_concepts'], list) else []
                desc = row['description'] if pd.notna(row['description']) else ""
                
                if concepts:
                    file_concepts_map[fname] = concepts
                if desc:
                    file_descriptions_map[fname] = [desc]
            
            # Create lecture summary record
            lecture_summaries.append({
                'lecture_number': int(lec_num),
                'topic': fname,  # Placeholder; actual topic to be filled by LLM
                'key_concepts': unique_concepts,
                'file_concepts_map': file_concepts_map,
                'file_descriptions_map': file_descriptions_map,
            })
        
        # Convert to DataFrame and sort by lecture number
        summary_df = pd.DataFrame(lecture_summaries).sort_values('lecture_number').reset_index(drop=True)
        
        print(f"\nGenerated summaries for {len(summary_df)} lectures.")
        print(f"Lecture range: {summary_df['lecture_number'].min()} to {summary_df['lecture_number'].max()}")
        
        # Create output directory
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV in output folder
        final_output_path = os.path.join(output_dir, os.path.basename(output_csv))
        summary_df.to_csv(final_output_path, index=False)
        
        print(f"\nSaved lecture summary to: {final_output_path}")
        print(f"Output absolute path: {os.path.abspath(final_output_path)}")
        
        # Display sample
        print("\nSample lecture summaries:")
        for _, row in summary_df.head(3).iterrows():
            lec_num = int(row['lecture_number'])
            concept_count = len(row['key_concepts']) if isinstance(row['key_concepts'], list) else 0
     

        return summary_df
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        conn.close()


if __name__ == "__main__":
    print("="*60)
    print("Slide-Based Lecture Summary Generator")
    print("="*60)
    
    result = generate_slide_lecture_summary()
    
    if result is not None:
        print("\n" + "="*60)
        print("Summary generation completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Summary generation failed.")
        print("="*60)
