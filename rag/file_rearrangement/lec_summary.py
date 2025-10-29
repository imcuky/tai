import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import pandas as pd
import re

import pandas as pd
import re
import os
import sqlite3

import pandas as pd
import re
import os
from urllib.parse import urlparse

def extract_calendar_info(text):
    """Extract structured information from calendar text"""
    
    if pd.isna(text) or not isinstance(text, str):
        return {"date": "", "topic": "", "cleaned_text": ""}
    
    # Clean the text first
    cleaned = clean_calendar_text(text)
    
    # Try to extract date pattern (like "Tue 6/24")
    date_match = re.search(r'([A-Za-z]{3}\s+\d{1,2}/\d{1,2})', text)
    date = date_match.group(1) if date_match else ""
    
    # Extract topic (text after date, before URLs/links)
    topic = ""
    if date:
        # Split by date and take the part after
        parts = text.split(date, 1)
        if len(parts) > 1:
            topic_part = parts[1]
            # Remove URLs and clean up
            topic = clean_calendar_text(topic_part).strip()
    else:
        topic = cleaned
    
    return {
        "date": date,
        "topic": topic,
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
    Returns (key_concepts:list[str], aspects:list[dict])."""
    key_concepts = []
    aspects = []
    seen_concepts = set()
    seen_aspect_tuples = set()

    for sec in sections_list or []:
        # key_concept may be str or list[str]
        if 'key_concept' in sec:
            kc = sec.get('key_concept')
            if isinstance(kc, str):
                val = kc.strip()
                if val and val.lower() not in seen_concepts:
                    seen_concepts.add(val.lower())
                    key_concepts.append(val)
            elif isinstance(kc, list):
                for item in kc:
                    if isinstance(item, str):
                        val = item.strip()
                        if val and val.lower() not in seen_concepts:
                            seen_concepts.add(val.lower())
                            key_concepts.append(val)

        # aspects is typically a list of dicts with keys 'content' and 'type'
        asp_list = sec.get('aspects') if isinstance(sec, dict) else None
        if isinstance(asp_list, list):
            for a in asp_list:
                if isinstance(a, dict):
                    content = str(a.get('content', '')).strip()
                    atype = str(a.get('type', '')).strip()
                    key = (content.lower(), atype.lower())
                    if content and key not in seen_aspect_tuples:
                        seen_aspect_tuples.add(key)
                        aspects.append({'content': content, 'type': atype})

    return key_concepts, aspects

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
    """Prefer loading files data (uuid, file_name, relative_path, url, sections) from SQLite DB,
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
                    sections
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
            for col in ['uuid', 'file_name', 'relative_path', 'url', 'sections']:
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
                kcs, asps = aggregate_concepts_and_aspects(sections_list)
                lecture_key_concepts.extend(kcs)
                lecture_aspects.extend(asps)

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
            'matched_file_uuids': json.dumps([m for m in matched_file_ids if m], ensure_ascii=False),
            'matched_file_names': json.dumps([m for m in matched_file_names if m], ensure_ascii=False),
            'slide_files': json.dumps([m for m in slide_file_names if m], ensure_ascii=False),
            'key_concepts': json.dumps(lecture_key_concepts, ensure_ascii=False),
            'aspects': json.dumps(lecture_aspects, ensure_ascii=False)
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
    out_df.to_csv(output_csv, index=False)
    print(f"Lecture summaries saved to: {output_csv}")
    print(f"Total lectures processed: {len(out_df)}")
    print(f"Lectures with matched files: {(out_df['matched_file_uuids'].apply(lambda s: len(_safe_json_or_literal_load(s) or [])).astype(int) > 0).sum()}")
    return out_df

def _heuristic_summary(topic: str, key_concepts: list, aspects: list) -> tuple[str, str]:
    """Generate a simple topic and summary if no API key is available or API fails."""
    topic_out = topic.strip() if isinstance(topic, str) else ''
    if not topic_out and key_concepts:
        topic_out = ', '.join(key_concepts[:3])
    # Build summary from key_concepts and aspects contents
    parts = []
    if key_concepts:
        parts.append("Key concepts: " + '; '.join(key_concepts[:6]))
    if aspects:
        contents = [a.get('content', '') for a in aspects if isinstance(a, dict) and a.get('content')]
        if contents:
            parts.append("Aspects: " + ' '.join(contents[:2]))
    summary_out = ' '.join(parts) if parts else (topic_out or 'Overview unavailable.')
    return topic_out or 'Topic', summary_out

def generate_openai_lecture_topics_json(summary_csv: str = "cs_61a_lecture_summary.csv",
                                        out_json: str = "cs_61a_lecture_topic_summaries.json",
                                        model: str = "gpt-4o-mini"):
    """Create a JSON mapping of "Lecture X" -> {topic, summary, date} using OpenAI; fallback heuristics if key missing."""
    if not os.path.exists(summary_csv):
        print(f"Lecture summary CSV not found: {summary_csv}")
        return None

    df = pd.read_csv(summary_csv)

    # Parse lists from JSON strings
    def parse_list(s):
        v = _safe_json_or_literal_load(s)
        return v if isinstance(v, list) else []

    df['key_concepts_list'] = df.get('key_concepts', '').apply(parse_list)
    df['aspects_list'] = df.get('aspects', '').apply(parse_list)
    df['slide_files_list'] = df.get('slide_files', '').apply(parse_list)

    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = None
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Warning: failed to init OpenAI client: {e}")
            client = None
    else:
        print("No OPENAI_API_KEY found; using heuristic summaries.")

    results = {}
    for idx, row in df.iterrows():
        lecture_key = f"Lecture {idx + 1}"
        date = str(row.get('date', '') or '')
        topic = str(row.get('topic', '') or '')
        cleaned_text = str(row.get('cleaned_text', '') or '')
        key_concepts = row.get('key_concepts_list') or []
        aspects = row.get('aspects_list') or []
        slide_files = row.get('slide_files_list') or []

        out_topic = ''
        out_summary = ''

        if client is not None:
            prompt = (
                "You will summarize a university lecture into a short topic and a concise 2-4 sentence summary.\n"
                "Return ONLY valid JSON with keys \"topic\" and \"summary\" (double-quoted JSON, no markdown, no comments).\n\n"
                f"Date: {date}\n"
                f"Existing topic (may be empty): {topic}\n"
                f"Key concepts: {json.dumps(key_concepts, ensure_ascii=False)}\n"
                f"Aspects: {json.dumps(aspects, ensure_ascii=False)}\n"
                f"Slide files (prioritize these): {json.dumps(slide_files, ensure_ascii=False)}\n"
                f"Context snippet: {cleaned_text[:500]}\n\n"
                "Rules:\n- Focus the summary on the slide materials when available; otherwise use other materials.\n- 'topic' should be a short title (3-8 words).\n- 'summary' should be 2-4 sentences describing what the lecture covers.\n"
            )
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Always output only compact, valid JSON object with double quotes: {\"topic\":\"...\",\"summary\":\"...\"}. No markdown."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=220,
                )
                content = (resp.choices[0].message.content or '').strip()
                # Try direct JSON parse; if fails, attempt to strip code fences
                try:
                    parsed = json.loads(content)
                except Exception:
                    import re as _re
                    # Strip code fences if present
                    if content.startswith("```"):
                        content = _re.sub(r"^```[a-zA-Z]*\n|```$", "", content).strip()
                    # Extract a JSON-looking object
                    m = _re.search(r"\{[\s\S]*\}", content)
                    raw_obj = m.group(0) if m else content
                    # Try json first
                    try:
                        parsed = json.loads(raw_obj)
                    except Exception:
                        # Try Python literal (handles single quotes)
                        try:
                            import ast as _ast
                            parsed = _ast.literal_eval(raw_obj)
                            if not isinstance(parsed, dict):
                                parsed = {}
                        except Exception:
                            # Last resort: naive single->double quote replacement
                            raw_fixed = raw_obj.replace("'", '"')
                            try:
                                parsed = json.loads(raw_fixed)
                            except Exception:
                                parsed = {}

                out_topic = str(parsed.get('topic', '')).strip()
                out_summary = str(parsed.get('summary', '')).strip()
            except Exception as e:
                print(f"OpenAI call failed for {lecture_key}: {e}")

        if not out_topic or not out_summary:
            out_topic, out_summary = _heuristic_summary(topic, key_concepts, aspects)

        results[lecture_key] = {
            "date": date,
            "topic": out_topic,
            "summary": out_summary,
        }

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Lecture topic summaries saved to: {out_json}")
    return results

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
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"\nProcessed data saved to: {output_csv}")
    
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





def test_openai_api():
    """Test OpenAI API connection and functionality"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    print(f"API Key found: {api_key[:10]}...{api_key[-4:]}")  # Show partial key for verification
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
        print("OpenAI client initialized successfully")
    except Exception as e:
        print(f"ERROR initializing OpenAI client: {e}")
        return False
    
    # Test 1: Simple completion
    print("\n" + "="*50)
    print("TEST 1: Simple Chat Completion")
    print("="*50)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello and tell me what 2+2 equals."}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        print("SUCCESS: API call completed")
        print(f"Model used: {response.model}")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"ERROR in API call: {e}")
        return False
    
    # Test 2: File categorization (similar to your use case)
    print("\n" + "="*50)
    print("TEST 2: File Categorization Test")
    print("="*50)
    
    test_files = [
        {"name": "lec01.pdf", "path": "lectures/lec01.pdf", "url": "course.com/lec01"},
        {"name": "hw01.py", "path": "homework/hw01.py", "url": "course.com/hw01"},
        {"name": "syllabus.html", "path": "admin/syllabus.html", "url": "course.com/syllabus"}
    ]
    
    for file_info in test_files:
        try:
            prompt = f"""
Categorize this CS 61A file into exactly one category: Lecture, Practice, or Support

File name: {file_info['name']}
Path: {file_info['path']}
URL: {file_info['url']}

Return only one word: Lecture, Practice, or Support
"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a file categorizer. Return only one word."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            category = response.choices[0].message.content.strip()
            print(f"File: {file_info['name']} -> Category: {category}")
            
        except Exception as e:
            print(f"ERROR categorizing {file_info['name']}: {e}")
    
    # Test 3: List available models
    print("\n" + "="*50)
    print("TEST 3: Available Models")
    print("="*50)
    
    try:
        models = client.models.list()
        gpt_models = [model.id for model in models.data if 'gpt' in model.id.lower()]
        print("Available GPT models:")
        for model in sorted(gpt_models)[:10]:  # Show first 10
            print(f"  - {model}")
        if len(gpt_models) > 10:
            print(f"  ... and {len(gpt_models) - 10} more")
            
    except Exception as e:
        print(f"ERROR listing models: {e}")
    
    print("\n" + "="*50)
    print("API TEST COMPLETED SUCCESSFULLY!")
    print("="*50)
    return True

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

# def main():
    
#     print(os.getcwd())
    
#     # Test the API
#     # success = test_openai_api()
    
#     # if success:
#     #     print("\nYour OpenAI API is working correctly!")
#     #     print("You can now use it in your file categorization script.")
#     # else:
#     #     print("\nAPI test failed. Please check your API key and try again.")

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
        print("\nGenerating lecture summaries by aggregating key concepts and aspects from associated files...")
        generate_lecture_summaries(calendar_csv=cal_with_paths, files_csv=files_csv_path, output_csv=lecture_output)

    # Generate OpenAI summaries JSON from the lecture summary CSV
    print("\nGenerating OpenAI-based lecture topic summaries (with heuristic fallback if no API key)...")
    generate_openai_lecture_topics_json(summary_csv=lecture_output,
                        out_json="cs_61a_lecture_topic_summaries.json",
                        model="gpt-4o-mini")

if __name__ == "__main__":
    main()