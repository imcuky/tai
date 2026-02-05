import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import json

import pandas as pd
import re
import sqlite3

from urllib.parse import urlparse


class LectureSummary(BaseModel):
    """Structured output model for lecture summaries."""
    lecture_topic: str
    lecture_summary: str

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



def generate_openai_lecture_topics_json(summary_csv: str = "cs_61a_lecture_summary.csv",
                                        out_json: str = "cs_61a_lecture_topic_summaries.json",
                                        model: str = "gpt-5-mini"):
    """Create a JSON mapping of "Lecture X" -> {topic, summary, date} using OpenAI; fallback heuristics if key missing."""
    if not os.path.exists(summary_csv):
        print(f"Lecture summary CSV not found: {summary_csv}")
        return None

    df = pd.read_csv(summary_csv)

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
        
        # Remove date from prompt per TODO; still retain for output
        # date = str(row.get('date', '') or '')
        lecnum_key = str(row.get('lecture_number', '') or '')
        lecture_key = f"Lecture {lecnum_key}" if lecnum_key else f"Lecture_{idx+1}"
        topic = str(row.get('topic', '') or '')
        file_concepts_map = row.get('file_concepts_map') or {}
        file_descriptions_map = row.get('file_descriptions_map') or {}
        out_topic = ''
        out_summary = ''

        if client is not None:
            # Use structured output with Pydantic model
            prompt = (
                "Summarize a CS course lecture using grouped key concepts and aspects.\n\n"
                f"Lecture topic hint from slides: {topic}\n\n"
                "Per-file Aggregated Concepts (key concepts and aspects, key concepts contains knowledge aspects topic thats cover inside this lecture):\n"
                f"{json.dumps(file_concepts_map, ensure_ascii=False, indent=2)}\n\n"
                "Per-file Descriptions (what is each file about):\n"
                f"{json.dumps(file_descriptions_map, ensure_ascii=False, indent=2)}\n\n"
                # "Rules:\n"
                # "- lecture_topic: 3-8 words describing the lecture's main focus.\n"
                # "- lecture_summary: 2-4 sentences; emphasize central ideas and conceptual progression.\n"
                # "- Avoid mentioning file names directly.\n"
                # "- Consider whether the file content matches the lecture topic.\n"
                # "- Focus on the educational concepts being taught.\n"
            )
            
            try:
                # Use structured output API for reliable responses
                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant that summarizes CS course lectures. Emphasize central ideas and conceptual progression. Focus on the educational concepts being taught. Keep the lecture summary extremely concise (max 2-3 sentences)."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        },
                    ],
                    response_format=LectureSummary,
                    # temperature=0.2, 
                    max_completion_tokens=2000, # Increased to allow for reasoning tokens + output
                )
                
                parsed_response = completion.choices[0].message.parsed
                if parsed_response:
                    out_topic = parsed_response.lecture_topic.strip()
                    out_summary = parsed_response.lecture_summary.strip()
                    try:
                        print(f"[+] {lecture_key}: {out_topic}")
                    except UnicodeEncodeError:
                        print(f"[+] {lecture_key}: {out_topic.encode('ascii', 'replace').decode('ascii')}")
                else:
                    print(f"[!] {lecture_key}: Structured output parsing failed/refused")
                    # Fallback to original topic
                    out_topic = topic

            except Exception as e:
                print(f"[X] {lecture_key}: OpenAI call failed - {e}")
                # Fallback to original topic
                break
                

        results[lecture_key] = {
            # "date": date,  # retained for downstream uses
            "lecture_topic": out_topic,
            "lecture_summary": out_summary,
        }

    # Create output directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    final_output_path = os.path.join(output_dir, os.path.basename(out_json))
    
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Lecture topic summaries saved to: {final_output_path}")
    return results







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
            model="gpt-5-mini-2025-08-07",
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
                model="gpt-5-mini-2025-08-07",
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

def test_api():
    
    # Test the API
    success = test_openai_api()
    
    if success:
        print("\nYour OpenAI API is working correctly!")
        print("You can now use it in your file categorization script.")
    else:
        print("\nAPI test failed. Please check your API key and try again.")


    

if __name__ == "__main__":
    #test_api()
    # Use the CSV from the output folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lecture_output = os.path.join(current_dir, "output", "cs_61a_lecture_summary.csv")
    
    generate_openai_lecture_topics_json(summary_csv=lecture_output,
                        out_json="cs_61a_lecture_topic_summaries.json",
                        model="gpt-5-mini")