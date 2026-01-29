# Calendar Lecture Info CSV Processing
# This script processes a CSV file containing calendar chunks for CS 61A,
# extracts structured lecture information, URL paths, and generates a lecture summary CSV
# by linking calendar entries to course files based on URL path segments.


import os
from dotenv import load_dotenv
from openai import OpenAI
import json

import pandas as pd
import re
import sqlite3

from urllib.parse import urlparse

def extract_calendar_info(text):
    """Extract structured information from a raw calendar chunk.

    Improvements:
      - Distinguish actual lecture date from preceding due-date annotations.
      - Derive a concise lecture title (e.g., 'Tail Calls', 'Interpreters') rather than the full cleaned text.
      - Lecture title is taken from the token immediately following the chosen lecture date token,
        truncated at the first '*' or '[' which usually begins resource listings.
    """
    if pd.isna(text) or not isinstance(text, str):
        return {"date": "", "topic": "", "cleaned_text": ""}

    # Keep original for parsing; build a cleaned version for fallback
    original = text
    cleaned = clean_calendar_text(original)

    # Split on pipe delimiters commonly used in calendar lines
    tokens = [t.strip() for t in original.split('|') if t.strip()]

    date_pattern = re.compile(r'^[A-Za-z]{3}\s+\d{1,2}/\d{1,2}$')
    lecture_date = ""
    lecture_title = ""

    # Identify candidate date tokens ignoring those that appear in a token with 'Due'
    # Some chunks may start with a due date line before the actual lecture line.
    candidate_indices = []
    for i, tok in enumerate(tokens):
        if 'due' in tok.lower():
            continue
        if date_pattern.match(tok):
            candidate_indices.append(i)

    # Heuristic: choose the last candidate date (handles due date appearing first)
    if candidate_indices:
        date_index = candidate_indices[-1]
        lecture_date = tokens[date_index]
        # Lecture title expected in the next token
        if date_index + 1 < len(tokens):
            title_token = tokens[date_index + 1]
            # Truncate at first resource delimiter
            cut_match = re.search(r'(\*|\[)', title_token)
            if cut_match:
                title_core = title_token[:cut_match.start()].strip()
            else:
                title_core = title_token.strip()
            # Normalize spacing and remove trailing punctuation
            lecture_title = re.sub(r'\s+', ' ', title_core).strip().strip(':').strip()

    # Fallbacks if parsing failed
    if not lecture_title:
        # Try extracting a capitalized phrase before first '*'
        star_split = original.split('*', 1)[0]
        # Remove leading pipes/spaces
        star_split = star_split.strip(' |')
        # Remove any leading date portion
        if lecture_date and star_split.startswith(lecture_date):
            star_split = star_split[len(lecture_date):].strip()
        # Trim at '[' if present
        bracket_pos = star_split.find('[')
        if bracket_pos != -1:
            star_split = star_split[:bracket_pos].strip()
        lecture_title = star_split.strip().strip(':')
        # If still empty, fallback to cleaned text (may be verbose)
        if not lecture_title:
            lecture_title = cleaned[:80]

    return {
        "date": lecture_date,
        "topic": lecture_title,
        "cleaned_text": cleaned
    }
def extract_url_paths(text):
    """Extract the last part of all URLs found in the text"""
    
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # Find all URLs in the text (both in markdown links and standalone)
    urls = []
    
    # Extract URLs from markdown links [text](url)
    markdown_urls = re.findall(r'\[([^\]]*)\]\(([^)]*)\)', text)
    for link_text, url in markdown_urls:
        urls.append(url.strip().strip('"'))
    
    # Extract standalone URLs
    standalone_urls = re.findall(r'https?://[^\s\)"\]]+', text)
    urls.extend(standalone_urls)
    
    # Extract the last directory/filename from each URL
    last_parts = []
    for url in urls:
        if url:
            try:
                # Parse the URL and get the path
                parsed = urlparse(url)
                path = parsed.path
                
                # Get the last part of the path
                if path:
                    last_part = path.split('/')[-1]
                    if last_part:  # Only add non-empty parts
                        last_parts.append(last_part)
                    else:
                        # If last part is empty, get the second to last
                        parts = [p for p in path.split('/') if p]
                        if parts:
                            last_parts.append(parts[-1])
            except:
                continue
    
    # Remove duplicates while preserving order
    unique_parts = []
    seen = set()
    for part in last_parts:
        if part not in seen:
            seen.add(part)
            unique_parts.append(part)
    
    return unique_parts

def _safe_json_or_literal_load(s):
    """Try JSON first, then Python literal to parse a list/dict from a string; return None on failure."""
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

def parse_sections_field(sections_raw):
    """Parse the 'sections' field which may be JSON, a Python literal, or already a list.
    Returns a list of dicts with potential keys like 'key_concept' and 'aspects'."""
    parsed = _safe_json_or_literal_load(sections_raw)
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []

def aggregate_concepts_and_aspects(sections_list):
    """Aggregate key_concepts (strings) and aspects (dicts with content/type) from sections list.
    Returns (key_concepts:list[str], aspects:list[dict], descriptions:list[str])."""
    key_concepts = []
    aspects = []
    descriptions = []
    seen_concepts = set()
    seen_aspect_tuples = set()
    seen_desc = set()

    for sec in sections_list or []:
        # key_concept may be str or list[str]
        kc_names = []
        if 'key_concept' in sec:
            kc = sec.get('key_concept')
            if isinstance(kc, str):
                val = kc.strip()
                if val:
                    kc_names.append(val)
            elif isinstance(kc, list):
                for item in kc:
                    if isinstance(item, str):
                        val = item.strip()
                        if val:
                            kc_names.append(val)

        # aspects is typically a list of dicts with keys 'content' and 'type'
        asp_list = sec.get('aspects') if isinstance(sec, dict) else None
        definition_content = None

        if isinstance(asp_list, list):
            for a in asp_list:
                if isinstance(a, dict):
                    content = str(a.get('content', '')).strip()
                    atype = str(a.get('type', '')).strip()
                    
                    # Check for definition to merge with key_concept
                    if not definition_content and atype.lower() == 'definition' and content:
                        definition_content = content

                    key = (content.lower(), atype.lower())
                    if content and key not in seen_aspect_tuples:
                        seen_aspect_tuples.add(key)
                        aspects.append({'content': content, 'type': atype})

        # Add key concepts (merged with definition if available)
        for name in kc_names:
            if definition_content:
                final_val = f"{name}: {definition_content}"
            else:
                final_val = name
            
            if final_val.lower() not in seen_concepts:
                seen_concepts.add(final_val.lower())
                key_concepts.append(final_val)

        # optional free-form description fields
        for desc_key in ('description', 'desc', 'summary'):
            dval = sec.get(desc_key) if isinstance(sec, dict) else None
            if isinstance(dval, str):
                dclean = dval.strip()
                if dclean and dclean.lower() not in seen_desc:
                    seen_desc.add(dclean.lower())
                    descriptions.append(dclean)

    return key_concepts, aspects, descriptions


def categorize_file(relative_path: str, file_name: str, url: str) -> str:
    """Infer a coarse category from relative path/file name.
    Examples: 'slides', 'lecture', 'video', 'reading', 'lab', 'homework', 'discussion',
    'project', 'exam', 'study-guide', 'support', 'other'.
    """
    rp = (relative_path or '').lower()
    fn = (file_name or '').lower()
    ur = (url or '').lower()
    path = ' '.join([rp, fn, ur])

    # High-confidence patterns
    if rp.startswith('cs 61a/support/study-guide') or 'study-guide' in path:
        return 'study-guide'
    if fn.endswith('.pdf') and ('slide' in path or 'lectur' in path or 'slides' in path):
        return 'slides'
    if 'youtube' in path or fn.endswith(('.mp4', '.mkv', '.webm')):
        return 'video'
    if any(x in path for x in ['lab', 'labs']):
        return 'lab'
    if any(x in path for x in ['homework', 'hw']):
        return 'homework'
    if any(x in path for x in ['discussion', 'disc']):
        return 'discussion'
    if any(x in path for x in ['project', 'proj']):
        return 'project'
    if any(x in path for x in ['exam', 'final', 'midterm']):
        return 'exam'
    if any(x in path for x in ['note', 'reading', 'handout']):
        return 'reading'
    if any(x in path for x in ['lecture']):
        return 'lecture'
    if any(x in path for x in ['support', 'admin', 'syllabus']):
        return 'support'
    return 'other'

def _prepare_file_match_keys(files_df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns to files_df for robust matching against calendar URL last parts."""
    df = files_df.copy()
    # Normalize strings to lower for matching
    for col in ['file_name', 'relative_path', 'url']:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('')
            df[col + '_low'] = df[col].str.lower()
        else:
            df[col] = ''
            df[col + '_low'] = ''

    # last part of relative_path and url path
    def last_path(p):
        if not p:
            return ''
        parts = [x for x in str(p).split('/') if x]
        return parts[-1].lower() if parts else ''

    df['rel_last'] = df['relative_path'].apply(last_path)
    df['url_last'] = df['url'].apply(last_path)
    df['file_last'] = df['file_name'].apply(last_path)
    return df

def match_files_for_calendar_row(files_df_idxed: pd.DataFrame, url_parts: list) -> pd.DataFrame:
    """Return subset of files that match any of the url_parts on file_last, rel_last, or url_last."""
    if not url_parts:
        return files_df_idxed.iloc[0:0]
    parts_low = [p.lower() for p in url_parts if isinstance(p, str) and p]
    if not parts_low:
        return files_df_idxed.iloc[0:0]

    mask = False
    for p in parts_low:
        cond = (
            (files_df_idxed['file_last'] == p) |
            (files_df_idxed['rel_last'] == p) |
            (files_df_idxed['url_last'] == p)
        )
        mask = cond if isinstance(mask, bool) else (mask | cond)
    return files_df_idxed[mask]

def _load_files_from_db_or_csv(files_db: str, files_csv: str) -> pd.DataFrame | None:
    """Prefer loading files data (uuid, file_name, relative_path, url, sections, description) from SQLite DB,
    otherwise from CSV. Returns DataFrame or None."""
    # Try DB first
    if files_db and os.path.exists(files_db):
        try:
            conn = sqlite3.connect(files_db)
            # Expect a table named 'file' with relevant columns
            query = """
                SELECT
                    uuid,
                    file_name,
                    relative_path,
                    url,
                    sections,
                    description
                FROM file
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Warning: failed to load from DB {files_db}: {e}")
    # Fallback to CSV
    if files_csv and os.path.exists(files_csv):
        try:
            df = pd.read_csv(files_csv)
            # Ensure required columns exist (create if missing)
            for col in ['uuid', 'file_name', 'relative_path', 'url', 'sections', 'description']:
                if col not in df.columns:
                    df[col] = ''
            return df
        except Exception as e:
            print(f"Warning: failed to load from CSV {files_csv}: {e}")
    return None

def generate_lecture_summaries(calendar_csv: str = "cs_61a_calendar_with_paths.csv",
                               files_db: str = "cs61a_metadata.db",
                               files_csv: str | None = None,
                               output_csv: str = "cs_61a_lecture_summary.csv"):
    """For each calendar row (lecture), find associated files by URL path segments and
    aggregate key_concepts and aspects from files' sections into a lecture-level summary."""
    if not os.path.exists(calendar_csv):
        print(f"Calendar CSV not found at {calendar_csv}. If you have cs_61a_calendar_chunks.csv, run URL extraction first.")
        return None
    cal_df = pd.read_csv(calendar_csv)
    # Determine default CSV path from current working directory if not provided
    if files_csv is None:
        files_csv = os.path.join(os.getcwd(), "cs61a_files.csv")

    files_df = _load_files_from_db_or_csv(files_db, files_csv)
    if files_df is None:
        print(f"Could not load file metadata from DB ({files_db}) or CSV ({files_csv}).")
        return None

    if 'text' not in cal_df.columns:
        print("Calendar CSV missing 'text' column")
        return None

    # Ensure we have URL paths; compute on the fly if missing
    if 'url_paths_list' in cal_df.columns:
        def parse_paths(x):
            v = _safe_json_or_literal_load(x)
            if isinstance(v, list):
                return [str(i) for i in v]
            # Fallback: split comma separated
            if isinstance(x, str) and x:
                return [s.strip() for s in x.split(',') if s.strip()]
            return []
        cal_df['url_parts'] = cal_df['url_paths_list'].apply(parse_paths)
    elif 'url_paths' in cal_df.columns:
        cal_df['url_parts'] = cal_df['url_paths'].fillna('').astype(str).apply(lambda s: [p.strip() for p in s.split(',') if p.strip()])
    else:
        # compute from text now
        cal_df['url_parts'] = cal_df['text'].apply(extract_url_paths)

    files_df_idx = _prepare_file_match_keys(files_df)

    results = []
    for _, row in cal_df.iterrows():
        url_parts = row.get('url_parts') or []
        matched = match_files_for_calendar_row(files_df_idx, url_parts)

        lecture_key_concepts = []
        lecture_aspects = []
        matched_file_ids = []
        matched_file_names = []
        slide_file_names = []
        per_file_concepts = {}
        per_file_aspects = {}
        per_file_descriptions = {}
        per_file_categories = {}

        if not matched.empty:
            # Prefer PDFs (slides) if present; else include all matched
            def is_pdf(fr):
                fn = str(fr.get('file_name', '')).lower()
                rl = str(fr.get('rel_last', '')).lower() if 'rel_last' in fr else ''
                ul = str(fr.get('url_last', '')).lower() if 'url_last' in fr else ''
                return fn.endswith('.pdf') or rl.endswith('.pdf') or ul.endswith('.pdf')

            matched_pdf = matched[matched.apply(is_pdf, axis=1)]
            rows_to_use = matched_pdf if not matched_pdf.empty else matched

            for __, frow in rows_to_use.iterrows():
                matched_file_ids.append(frow.get('uuid', ''))
                matched_file_names.append(frow.get('file_name', ''))
                if is_pdf(frow):
                    slide_file_names.append(frow.get('file_name', ''))
                sections_list = parse_sections_field(frow.get('sections'))
                kcs, asps, descs = aggregate_concepts_and_aspects(sections_list)
                lecture_key_concepts.extend(kcs)
                lecture_aspects.extend(asps)
                fname = frow.get('file_name', '') or frow.get('relative_path', '') or frow.get('url', '')
                if fname:
                    # Store per-file concepts (unique) and aspects (content strings unique)
                    existing_c = per_file_concepts.get(fname, [])
                    for c in kcs:
                        if c not in existing_c:
                            existing_c.append(c)
                    per_file_concepts[fname] = existing_c
                    existing_a = per_file_aspects.get(fname, [])
                    for a in asps:
                        content = a.get('content', '')
                        if content and content not in existing_a:
                            existing_a.append(content)
                    per_file_aspects[fname] = existing_a
                    # Store descriptions (from sections)
                    existing_d = per_file_descriptions.get(fname, [])
                    for d in descs:
                        if d not in existing_d:
                            existing_d.append(d)
                    # Also include file-level description if available
                    file_desc = str(frow.get('description', '') or '').strip()
                    if file_desc and file_desc not in existing_d:
                        existing_d.insert(0, file_desc)  # Prioritize file-level description
                    per_file_descriptions[fname] = existing_d
                    # Categorize file from relative path/name/url
                    per_file_categories[fname] = categorize_file(
                        frow.get('relative_path', ''), frow.get('file_name', ''), frow.get('url', '')
                    )

            # de-duplicate lecture-level lists
            seen_kc = set()
            uniq_kc = []
            for k in lecture_key_concepts:
                if k.lower() not in seen_kc:
                    seen_kc.add(k.lower())
                    uniq_kc.append(k)
            lecture_key_concepts = uniq_kc

            seen_as = set()
            uniq_as = []
            for a in lecture_aspects:
                key = (a.get('content', '').lower(), a.get('type', '').lower())
                if key not in seen_as:
                    seen_as.add(key)
                    uniq_as.append(a)
            lecture_aspects = uniq_as

        # Build lecture summary row
        summary_row = {
            'date': row.get('date', ''),
            'topic': row.get('topic', ''),
            'cleaned_text': row.get('cleaned_text', ''),
            'original_text': row.get('original_text', row.get('text', '')),
            'url_parts': json.dumps(url_parts, ensure_ascii=False),
            # 'matched_file_uuids': json.dumps([m for m in matched_file_ids if m], ensure_ascii=False),
            'matched_file_names': json.dumps([m for m in matched_file_names if m], ensure_ascii=False),
            'slide_files': json.dumps([m for m in slide_file_names if m], ensure_ascii=False),
            'key_concepts': json.dumps(lecture_key_concepts, ensure_ascii=False),
            # 'aspects': json.dumps(lecture_aspects, ensure_ascii=False),
            'file_concepts_map': json.dumps(per_file_concepts, ensure_ascii=False),
            # 'file_aspects_map': json.dumps(per_file_aspects, ensure_ascii=False),
            'file_descriptions_map': json.dumps(per_file_descriptions, ensure_ascii=False),
            # 'file_categories_map': json.dumps(per_file_categories, ensure_ascii=False)
        }

        # Optional concise textual summary
        if lecture_key_concepts:
            summary_row['lecture_summary'] = 'Key concepts: ' + '; '.join(lecture_key_concepts[:10])
        elif lecture_aspects:
            summary_row['lecture_summary'] = 'Aspects: ' + '; '.join([a.get('content', '') for a in lecture_aspects[:10]])
        else:
            summary_row['lecture_summary'] = ''

        results.append(summary_row)

    out_df = pd.DataFrame(results)
    
    # Create output directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    final_output_path = os.path.join(output_dir, os.path.basename(output_csv))
    out_df.to_csv(final_output_path, index=False)
    print(f"Lecture summaries saved to: {final_output_path}")
    print(f"Total lectures processed: {len(out_df)}")
    # print(f"Lectures with matched files: {(out_df['matched_file_uuids'].apply(lambda s: len(_safe_json_or_literal_load(s) or [])).astype(int) > 0).sum()}")
    return out_df


def process_calendar_with_urls(input_csv, output_csv):
    """Process the calendar CSV and extract URL paths"""
    
    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: {input_csv}")
        return None
    
    # Load the CSV
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    
    if 'text' not in df.columns:
        print("ERROR: 'text' column not found in CSV")
        return None
    
    # Show original first row example
    if len(df) > 0:
        print(f"\nOriginal first row text:")
        sample_text = df.iloc[0]['text']
        print(f"'{sample_text[:200]}...'")  # Show first 200 chars
    
    # Process each text entry
    processed_data = []
    
    for i, row in df.iterrows():
        original_text = row.get('text', '')
        
        # Extract calendar info (existing function)
        info = extract_calendar_info(original_text)
        
        # Extract URL paths
        url_paths = extract_url_paths(original_text)
        
        # Create new row with additional columns
        new_row = row.to_dict()
        new_row['original_text'] = original_text
        new_row['cleaned_text'] = info['cleaned_text']
        new_row['date'] = info['date']
        new_row['topic'] = info['topic']
        new_row['url_paths'] = ', '.join(url_paths)  # Join as comma-separated string
        new_row['url_paths_list'] = str(url_paths)   # Keep as list representation
        
        processed_data.append(new_row)
    
    # Create new DataFrame
    result_df = pd.DataFrame(processed_data)
    
    # Show processed first row example
    if len(result_df) > 0:
        print(f"\nProcessed first row:")
        print(f"Date: '{result_df.iloc[0]['date']}'")
        print(f"Topic: '{result_df.iloc[0]['topic']}'")
        print(f"URL Paths: '{result_df.iloc[0]['url_paths']}'")
    
    # Create output directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    final_output_path = os.path.join(output_dir, os.path.basename(output_csv))
    result_df.to_csv(final_output_path, index=False)
    print(f"\nProcessed data saved to: {final_output_path}")
    
    # Show summary statistics
    print(f"\nSummary:")
    print(f"  - Total rows processed: {len(result_df)}")
    print(f"  - Rows with URL paths: {result_df['url_paths'].apply(lambda x: len(x) > 0).sum()}")
    
    # Show sample URL paths found
    all_paths = []
    for paths in result_df['url_paths']:
        if paths:
            all_paths.extend(paths.split(', '))
    
    unique_paths = list(set(all_paths))
    print(f"\nSample URL paths found:")
    for path in sorted(unique_paths)[:15]:
        print(f"  - {path}")
    if len(unique_paths) > 15:
        print(f"  ... and {len(unique_paths) - 15} more")
    
    return result_df

def show_url_extraction_examples(input_csv, num_examples=3):
    """Show examples of URL path extraction"""
    
    df = pd.read_csv(input_csv)
    
    print("URL Path Extraction Examples:")
    print("=" * 60)
    
    for i in range(min(num_examples, len(df))):
        original = df.iloc[i]['text']
        url_paths = extract_url_paths(original)
        
        print(f"\nExample {i+1}:")
        print(f"ORIGINAL: {original[:100]}...")
        print(f"URL PATHS: {url_paths}")
        print("-" * 40)






def clean_calendar_text(text):
    """Clean calendar text by removing URLs and formatting"""
    
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    
    # Remove standalone URLs (http/https)
    text = re.sub(r'https?://[^\s\)]+', '', text)
    
    # Remove remaining parentheses that might be empty after URL removal
    text = re.sub(r'\(\s*\)', '', text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace and pipe characters
    text = text.strip().strip('|').strip()
    
    return text


def main():
    """Main function to process calendar chunks CSV with URL extraction"""
    
    print("Calendar Text Cleaner with URL Path Extraction")
    print("=" * 50)
    
    # File paths
    input_csv = "cs_61a_calendar_chunks.csv"
    output_csv = "cs_61a_calendar_with_paths.csv"
    
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"ERROR: {input_csv} not found")
        print("Please make sure the calendar chunks CSV file exists")
        return
    
    # Show URL extraction examples
    print("\nShowing URL path extraction examples...")
    show_url_extraction_examples(input_csv)
    
    # Process the CSV to include URL paths
    print(f"\nProcessing {input_csv}...")
    result_df = process_calendar_with_urls(input_csv, output_csv)
    
    # Always set lecture output path so later steps have a default
    lecture_output = "cs_61a_lecture_summary.csv"

    if result_df is not None:
        print(f"\nSUCCESS: Processed calendar data saved to {output_csv}")
        
        # Show column info
        print(f"\nOutput columns:")
        for col in result_df.columns:
            print(f"  - {col}")

        # After producing calendar with URL paths, generate lecture summaries by linking to cs61a_files.csv
        cal_with_paths = output_csv if os.path.exists(output_csv) else input_csv
        files_csv_path = "../../cs61a_files.csv"
        print("\nGenerating lecture summaries info by aggregating key concepts and aspects from associated files...")
        generate_lecture_summaries(calendar_csv=cal_with_paths, files_csv=files_csv_path, output_csv=lecture_output)

if __name__ == "__main__":
    #test_api()
    main()